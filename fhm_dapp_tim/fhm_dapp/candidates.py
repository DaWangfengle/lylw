"""候选准备: 从模型预测中提取成员检索支持及其锚点统计."""

from __future__ import annotations

import torch


def _seen_only_anchor_stat(
    value: torch.Tensor,
    seen_mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Accept a full-AP statistic or the already-cropped seen-only view."""
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [n_anchors, n_aps]")
    seen = seen_mask.to(device=value.device, dtype=torch.bool)
    seen_count = int(seen.sum().item())
    if value.shape[1] == seen_count:
        return value
    if value.shape[1] == seen.numel():
        return value[:, seen]
    raise ValueError(
        f"{name} AP width {value.shape[1]} matches neither the full "
        f"({seen.numel()}) nor seen-only ({seen_count}) AP space"
    )


@torch.no_grad()
def prepare_candidates(
    model,
    rss: torch.Tensor,
    anchor_detection_prob: torch.Tensor,
    anchor_rss_center: torch.Tensor,
    anchor_rss_scale: torch.Tensor | None,
    top_k: int,
    neural_scale: float,
    prototype_weight: float,
    posterior_temperature: float,
    candidate_building_conditioning: bool = True,
    include_measurement_stats: bool = True,
) -> dict[str, torch.Tensor]:
    """从 base model 预测中准备 DAPP 所需的候选张量.

    流程:
        1. 模型预测 Top-K 候选 + building.
        2. 可选地用 same_building 过滤候选权重.
        3. 提取候选的锚点统计 (detection_prob, rss_center) 和坐标.

    Args:
        model: AnchorStudentTModel (已加载 checkpoint).
        rss: [B, n_aps] query RSS.
        anchor_detection_prob: [n_anchors, n_aps_seen] 锚点检测概率;
            为兼容独立调用也接受完整 AP 空间.
        anchor_rss_center: [n_anchors, n_aps_seen] 锚点 RSS 中心.
        anchor_rss_scale: [n_anchors, n_aps_seen] 锚点 RSS 尺度.
        top_k: 当前成员的检索支持宽度. 单共识评估可先用较宽的
            ``member_top_k=L`` 检索，再在全局锚点空间截断为最终
            ``top_k=K``；此函数只负责前者.
        neural_scale: 神经网络 logit 缩放.
        prototype_weight: 原型距离权重.
        posterior_temperature: 后验温度.
        candidate_building_conditioning: 是否使用确定性 building
            预测条件化坐标后验和初始候选权重. 启用时若
            Top-K 内没有预测楼栋的候选, 完全回退到未条件化权重.
        include_measurement_stats: Whether to gather the three candidate--AP
            radio-memory tensors.  Consensus-first inference disables this
            for member-specific supports and gathers the certified shared
            memory only once after the common support has been formed.

    Returns:
        dict 含: baseline, radial_scale, candidate_indices, initial_weights,
        anchor_locations, component_locations, component_scales, query;
        detection_prob, rss_center and rss_scale are included only when
        ``include_measurement_stats`` is true.
    """
    prediction = model.predict(
        rss,
        top_k=top_k,
        neural_scale=neural_scale,
        prototype_distance_weight=prototype_weight,
        posterior_temperature=posterior_temperature,
        condition_building=candidate_building_conditioning,
        # Keep the building estimator fixed in the ablation: the switch only
        # removes its effect on candidate coordinates/weights.
        deterministic_building=True,
    )
    indices = prediction["candidate_indices"]
    embedding = prediction["embedding"]
    component_locations = prediction["component_locations"]
    component_scales = prediction["component_scales"]
    initial_weights = prediction["candidate_weights"]
    if candidate_building_conditioning:
        # 同楼过滤是可审计的候选先验. 当 Top-K 中不含预测楼栋
        # 时不产生全零后验, 而是保留原始候选分布.
        same_building = (
            model.anchor_buildings[indices]
            == prediction["building"].unsqueeze(1)
        )
        conditioned_weights = initial_weights * same_building
        has_same_building = same_building.any(dim=1, keepdim=True)
        initial_weights = torch.where(
            has_same_building,
            conditioned_weights / conditioned_weights.sum(
                dim=1, keepdim=True).clamp(min=1e-12),
            initial_weights,
        )
    result = {
        "baseline": prediction["coords"],
        "radial_scale": prediction["radial_scale"],
        "candidate_indices": indices,
        "predicted_building": prediction["building"],
        "predicted_floor": prediction["floor"],
        "candidate_floors": model.anchor_floors[indices],
        "initial_weights": initial_weights,
        "anchor_locations": model.anchor_coords[indices],
        "component_locations": component_locations,
        "component_scales": component_scales,
        "embedding": embedding,
        "query": rss[:, model.encoder.seen_ap_mask],
    }
    if include_measurement_stats:
        seen = model.encoder.seen_ap_mask
        detection_memory = _seen_only_anchor_stat(
            anchor_detection_prob, seen, name="anchor_detection_prob")
        center_memory = _seen_only_anchor_stat(
            anchor_rss_center, seen, name="anchor_rss_center")
        candidate_rss_center = center_memory[indices]
        if anchor_rss_scale is None:
            candidate_rss_scale = torch.full_like(candidate_rss_center, 0.15)
        else:
            scale_memory = _seen_only_anchor_stat(
                anchor_rss_scale, seen, name="anchor_rss_scale")
            candidate_rss_scale = scale_memory[indices]
        result.update({
            "detection_prob": detection_memory[indices],
            "rss_center": candidate_rss_center,
            "rss_scale": candidate_rss_scale,
        })
    return result


