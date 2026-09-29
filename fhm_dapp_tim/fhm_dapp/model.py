# 从 rf_anchor_model.py 内联，自包含 fhm_dapp_tim 实验
"""Reference-anchored probabilistic WiFi localization.

The model factorizes localization into anchor retrieval and a local metric
residual distribution. It intentionally avoids global coordinate diffusion:

    p(x | scan) = sum_j p(anchor_j | scan) p(x - anchor_j | scan, anchor_j)

Only visible APs are encoded. APs that were never observed in the training
partition share an out-of-vocabulary embedding.
"""

from __future__ import annotations

import copy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_lower_median(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    empty_value: float,
) -> torch.Tensor:
    """Deterministic ``nanmedian`` equivalent for masked row vectors.

    PyTorch's CUDA ``nanmedian(dim=...)`` is explicitly marked as lacking a
    deterministic implementation. Sorting masked values and gathering the
    lower middle order statistic has the same value convention, including the
    lower median for even counts, without invoking that CUDA kernel.
    """
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError("values and mask must be matching [batch,width] tensors")
    # nanmedian ignores NaNs but treats +/-infinity as ordered values.
    valid = mask & ~torch.isnan(values)
    ordered = torch.sort(
        torch.where(valid, values, torch.full_like(values, float("inf"))),
        dim=1,
        stable=True,
    ).values
    count = valid.sum(dim=1)
    lower_middle = ((count - 1).clamp(min=0) // 2).unsqueeze(1)
    median = ordered.gather(1, lower_middle).squeeze(1)
    return torch.where(
        count > 0, median, torch.full_like(median, float(empty_value))
    )


def _positive_affine_components(
    rss: torch.Tensor, visible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return equivariant location/scale pieces and the maximal invariant."""
    if rss.ndim != 2 or visible.shape != rss.shape or visible.dtype != torch.bool:
        raise ValueError("rss and visible must be matching [batch,aps] tensors")
    median = _masked_lower_median(rss, visible, empty_value=-1.0)
    absolute_deviation = torch.where(
        visible,
        (rss - median.unsqueeze(1)).abs(),
        torch.full_like(rss, float("nan")),
    )
    scale = torch.nanmean(absolute_deviation, dim=1)
    denominator = torch.where(
        torch.isfinite(scale) & (scale > 0.0),
        scale,
        torch.ones_like(scale),
    )
    invariant = torch.where(
        visible,
        (rss - median.unsqueeze(1)) / denominator.unsqueeze(1),
        torch.zeros_like(rss),
    )
    return median, absolute_deviation, scale, invariant


def positive_affine_maximal_invariant(
    rss: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    r"""Return ``h`` separating visibility-preserving positive-affine orbits.

    The joint statistic ``(visible, h)`` is maximal.  Mean absolute deviation
    about the visible median is used as the one-homogeneous scale.  Constant,
    singleton, and empty visible subvectors map to the zero representative of
    their respective visibility orbit.
    """
    return _positive_affine_components(rss, visible)[3]


def bounded_positive_affine_maximal_invariant(
    rss: torch.Tensor, visible: torch.Tensor
) -> torch.Tensor:
    r"""Injectively bound the visible maximal invariant to ``(-1, 1)``.

    ``h -> h / (1 + |h|)`` is one-to-one on the real line, so applying it
    coordinatewise preserves orbit separation while reserving ``-1`` for the
    missing-AP sentinel used by the marked radio memory.
    """
    invariant = positive_affine_maximal_invariant(rss, visible)
    bounded = invariant / (1.0 + invariant.abs())
    return torch.where(visible, bounded, torch.full_like(bounded, -1.0))


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(value: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(value, strength)


class SparseAPSetEncoder(nn.Module):
    """Encode a variable-size set of visible AP observations."""

    def __init__(
        self,
        n_aps: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_visible: int = 64,
        dropout: float = 0.1,
        visible_threshold: float = -0.75,
        seen_ap_mask: torch.Tensor | None = None,
        scan_feature_mode: str = "legacy",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if scan_feature_mode not in {
            "legacy",
            "robust",
            "invariant",
            "maximal_invariant",
            "bounded_maximal_invariant",
        }:
            raise ValueError(
                f"unknown scan feature mode: {scan_feature_mode}")

        self.n_aps = int(n_aps)
        self.d_model = int(d_model)
        self.max_visible = int(max_visible)
        self.visible_threshold = float(visible_threshold)
        self.scan_feature_mode = scan_feature_mode

        if seen_ap_mask is None:
            seen_ap_mask = torch.ones(n_aps, dtype=torch.bool)
        self.register_buffer(
            "seen_ap_mask", seen_ap_mask.to(dtype=torch.bool), persistent=True)

        self.ap_embedding = nn.Embedding(n_aps + 1, d_model)
        feature_count = {
            "legacy": 5,
            "robust": 7,
            "invariant": 4,
            "maximal_invariant": 4,
            "bounded_maximal_invariant": 4,
        }[scan_feature_mode]
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_count, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def _tokens(
        self, rss: torch.Tensor, ap_dropout: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rss.ndim != 2 or rss.shape[1] != self.n_aps:
            raise ValueError(
                f"expected RSS tensor [batch, {self.n_aps}], got {rss.shape}")

        visible = rss > self.visible_threshold
        if self.training and ap_dropout > 0.0:
            keep = torch.rand_like(rss) >= ap_dropout
            visible = visible & keep

        token_count = min(self.max_visible, self.n_aps)
        ranked_rss = rss.masked_fill(~visible, torch.finfo(rss.dtype).min)
        top_rss, top_ids = torch.topk(
            ranked_rss, k=token_count, dim=1, largest=True, sorted=True)
        valid = torch.gather(visible, 1, top_ids)

        neg_inf = torch.finfo(rss.dtype).min
        strongest = rss.masked_fill(~visible, neg_inf).max(dim=1).values
        strongest = torch.where(
            torch.isfinite(strongest), strongest, torch.full_like(strongest, -1.0))
        median, absolute_deviation, mean_deviation, maximal_full = (
            _positive_affine_components(rss, visible)
        )
        robust_scale = _masked_lower_median(
            absolute_deviation, visible, empty_value=float("nan")
        )
        robust_scale = torch.where(
            robust_scale > 1e-3, robust_scale, mean_deviation)
        robust_scale = torch.nan_to_num(
            robust_scale, nan=0.05).clamp(min=0.05)

        visible_count = visible.sum(dim=1).clamp(min=1)
        rank = torch.arange(
            token_count, device=rss.device, dtype=rss.dtype).unsqueeze(0)
        rank = rank / (visible_count.to(rss.dtype).unsqueeze(1) - 1).clamp(min=1)
        count_feature = torch.log1p(visible_count.to(rss.dtype))
        count_feature = count_feature / math.log1p(max(1, self.n_aps))
        count_feature = count_feature.unsqueeze(1).expand(-1, token_count)

        safe_rss = torch.where(valid, top_rss, torch.full_like(top_rss, -1.0))
        centered = safe_rss - median.unsqueeze(1)
        standardized = (
            centered / robust_scale.unsqueeze(1)).clamp(min=-6.0, max=6.0)
        strongest_standardized = standardized[:, :1]
        maximal_standardized = torch.gather(maximal_full, 1, top_ids)
        strongest_maximal = maximal_standardized[:, :1]
        bounded_maximal = maximal_standardized / (
            1.0 + maximal_standardized.abs()
        )
        strongest_bounded_maximal = bounded_maximal[:, :1]
        if self.scan_feature_mode == "legacy":
            features = torch.stack(
                [
                    safe_rss,
                    safe_rss - strongest.unsqueeze(1),
                    centered,
                    rank,
                    count_feature,
                ],
                dim=-1,
            )
        elif self.scan_feature_mode == "robust":
            spread = robust_scale.unsqueeze(1).expand(-1, token_count)
            features = torch.stack(
                [
                    safe_rss,
                    safe_rss - strongest.unsqueeze(1),
                    centered,
                    standardized,
                    rank,
                    count_feature,
                    spread,
                ],
                dim=-1,
            )
        elif self.scan_feature_mode == "invariant":
            features = torch.stack(
                [
                    standardized,
                    standardized - strongest_standardized,
                    rank,
                    count_feature,
                ],
                dim=-1,
            )
        elif self.scan_feature_mode == "maximal_invariant":
            # The raw maximal invariant is retained as the first channel.
            # Rank, count, and strongest-relative channels are deterministic
            # augmentations and therefore preserve both invariance and orbit
            # separation.  No non-injective clipping is applied.
            features = torch.stack(
                [
                    maximal_standardized,
                    maximal_standardized - strongest_maximal,
                    rank,
                    count_feature,
                ],
                dim=-1,
            )
        else:
            # The injective bounding map keeps maximal-invariant orbit
            # separation while sharing the radio-memory value range in V9.
            features = torch.stack(
                [
                    bounded_maximal,
                    bounded_maximal - strongest_bounded_maximal,
                    rank,
                    count_feature,
                ],
                dim=-1,
            )
        features = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))

        known = self.seen_ap_mask[top_ids]
        mapped_ids = torch.where(
            known & valid, top_ids, torch.full_like(top_ids, self.n_aps))
        tokens = self.ap_embedding(mapped_ids) + self.feature_encoder(features)
        padding_mask = ~valid

        # MultiheadAttention cannot process a row where every token is masked.
        no_visible = padding_mask.all(dim=1)
        if no_visible.any():
            padding_mask = padding_mask.clone()
            padding_mask[no_visible, 0] = False
            tokens = tokens.clone()
            tokens[no_visible, 0] = self.ap_embedding.weight[self.n_aps]
        return tokens, padding_mask

    def forward(self, rss: torch.Tensor, ap_dropout: float = 0.0) -> torch.Tensor:
        tokens, padding_mask = self._tokens(rss, ap_dropout=ap_dropout)
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        scores = self.pool_score(tokens).squeeze(-1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.output(pooled), dim=-1)


class AnchorStudentTModel(nn.Module):
    """Anchor retrieval with a mixture of local bivariate Student-t residuals."""

    def __init__(
        self,
        n_aps: int,
        anchor_coords: torch.Tensor,
        anchor_buildings: torch.Tensor,
        anchor_floors: torch.Tensor,
        seen_ap_mask: torch.Tensor,
        prototype_rss: torch.Tensor | None = None,
        ap_building_owner: torch.Tensor | None = None,
        n_devices: int = 0,
        physics_model: nn.Module | None = None,
        physics_rerank_weight: float = 0.0,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_visible: int = 64,
        dropout: float = 0.1,
        visible_threshold: float = -0.75,
        student_df: float = 4.0,
        max_residual_m: float = 30.0,
        min_scale_m: float = 0.25,
        residual_experts: int = 1,
        scan_feature_mode: str = "legacy",
        prototype_metric: str = "l2",
        anchor_detection_prob: torch.Tensor | None = None,
        anchor_rss_center: torch.Tensor | None = None,
        anchor_rss_scale: torch.Tensor | None = None,
        anchor_sample_count: torch.Tensor | None = None,
        candidate_reranker: bool = False,
        candidate_reranker_mode: str = "summary",
        reranker_anchor_decoder: bool = False,
        anchor_decoder_scale_m: float = 6.0,
        local_residual_radius_fraction: float = 0.0,
        local_residual_radius_min_m: float = 0.50,
        local_residual_radius_max_m: float = 8.0,
        local_residual_transform: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if student_df <= 2.0:
            raise ValueError("student_df must exceed 2 for finite variance")
        if residual_experts < 1:
            raise ValueError("residual_experts must be positive")
        if prototype_metric not in {"l2", "none"}:
            raise ValueError(f"unknown prototype metric: {prototype_metric}")
        if candidate_reranker_mode not in {"summary", "ap_pair"}:
            raise ValueError(
                f"unknown candidate reranker mode: {candidate_reranker_mode}")

        self.encoder = SparseAPSetEncoder(
            n_aps=n_aps,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_visible=max_visible,
            dropout=dropout,
            visible_threshold=visible_threshold,
            seen_ap_mask=seen_ap_mask,
            scan_feature_mode=scan_feature_mode,
        )
        self.prototype_metric = prototype_metric
        self.register_buffer(
            "anchor_coords", anchor_coords.to(dtype=torch.float32), persistent=True)
        self.register_buffer(
            "anchor_buildings", anchor_buildings.to(dtype=torch.long), persistent=True)
        self.register_buffer(
            "anchor_floors", anchor_floors.to(dtype=torch.long), persistent=True)
        if prototype_rss is None:
            prototype_rss = torch.empty(0, n_aps)
        if len(prototype_rss) not in (0, len(anchor_coords)):
            raise ValueError("prototype_rss must contain one row per anchor")
        self.register_buffer(
            "prototype_rss", prototype_rss.to(dtype=torch.float32), persistent=True)
        self.use_candidate_reranker = bool(candidate_reranker)
        self.candidate_reranker_mode = candidate_reranker_mode
        self.reranker_anchor_decoder = bool(reranker_anchor_decoder)
        self.anchor_decoder_scale_m = float(anchor_decoder_scale_m)
        if self.use_candidate_reranker:
            required_stats = (
                anchor_detection_prob,
                anchor_rss_center,
                anchor_rss_scale,
                anchor_sample_count,
            )
            if any(value is None for value in required_stats):
                raise ValueError(
                    "candidate reranking requires distributional anchor stats")
            expected_shape = prototype_rss.shape
            for name, value in (
                ("anchor_detection_prob", anchor_detection_prob),
                ("anchor_rss_center", anchor_rss_center),
                ("anchor_rss_scale", anchor_rss_scale),
            ):
                if value.shape != expected_shape:
                    raise ValueError(
                        f"{name} must have shape {tuple(expected_shape)}")
            if anchor_sample_count.shape != (len(anchor_coords),):
                raise ValueError(
                    "anchor_sample_count must contain one value per anchor")
            self.register_buffer(
                "anchor_detection_prob",
                anchor_detection_prob.to(dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "anchor_rss_center",
                anchor_rss_center.to(dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "anchor_rss_scale",
                anchor_rss_scale.to(dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "anchor_sample_count",
                anchor_sample_count.to(dtype=torch.float32),
                persistent=True,
            )
            evidence_features = (
                11 if self.candidate_reranker_mode == "summary" else 8)
            self.evidence_reranker = nn.Sequential(
                nn.Linear(evidence_features, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            nn.init.zeros_(self.evidence_reranker[-1].weight)
            nn.init.zeros_(self.evidence_reranker[-1].bias)
            if self.candidate_reranker_mode == "ap_pair":
                self.ap_pair_evidence = nn.Sequential(
                    nn.Linear(9, 24),
                    nn.LayerNorm(24),
                    nn.GELU(),
                    nn.Linear(24, 12),
                    nn.GELU(),
                    nn.Linear(12, 1),
                )
                self.ap_reliability_logits = nn.Parameter(
                    torch.full((n_aps,), 2.0))
            else:
                self.ap_pair_evidence = None
                self.register_parameter("ap_reliability_logits", None)
        else:
            self.evidence_reranker = None
            self.ap_pair_evidence = None
            self.register_parameter("ap_reliability_logits", None)
        if ap_building_owner is None:
            ap_building_owner = torch.zeros(n_aps, dtype=torch.long)
        self.register_buffer(
            "ap_building_owner",
            ap_building_owner.to(dtype=torch.long),
            persistent=True,
        )

        coord_mean = anchor_coords.float().mean(dim=0)
        coord_scale = anchor_coords.float().std(dim=0).clamp(min=1.0)
        self.register_buffer("anchor_coord_mean", coord_mean, persistent=True)
        self.register_buffer("anchor_coord_scale", coord_scale, persistent=True)

        self.anchor_keys = nn.Parameter(
            F.normalize(torch.randn(len(anchor_coords), d_model), dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_prototype_temperature = nn.Parameter(
            torch.tensor(math.log(0.20)))
        self.prototype_weight_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0))))
        self.device_head = None
        if n_devices > 1:
            self.device_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_devices),
            )
        self.physics_model = physics_model
        self.physics_rerank_weight = float(physics_rerank_weight)
        hidden = max(128, d_model)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model * 3 + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.student_df = float(student_df)
        self.max_residual_m = float(max_residual_m)
        self.min_scale_m = float(min_scale_m)
        radius_fraction = float(local_residual_radius_fraction)
        radius_min = float(local_residual_radius_min_m)
        radius_max = float(local_residual_radius_max_m)
        if not math.isfinite(radius_fraction) or radius_fraction < 0.0:
            raise ValueError(
                "local_residual_radius_fraction must be finite and nonnegative"
            )
        if (
            not math.isfinite(radius_min)
            or not math.isfinite(radius_max)
            or radius_min <= 0.0
            or radius_max < radius_min
        ):
            raise ValueError("local residual radius bounds are invalid")
        self.local_residual_radius_fraction = radius_fraction
        self.local_residual_radius_min_m = radius_min
        self.local_residual_radius_max_m = radius_max
        if local_residual_transform is None:
            residual_transform = torch.empty(0, 2, 2, dtype=torch.float32)
        else:
            residual_transform = torch.as_tensor(
                local_residual_transform, dtype=torch.float32
            )
            if residual_transform.shape != (len(anchor_coords), 2, 2):
                raise ValueError(
                    "local_residual_transform must have shape [anchors,2,2]"
                )
            if not bool(torch.isfinite(residual_transform).all()):
                raise ValueError("local_residual_transform must be finite")
            if radius_fraction > 0.0:
                raise ValueError(
                    "local residual radius and tensor transform are alternatives"
                )
        if radius_fraction > 0.0:
            pair_distance = torch.cdist(
                anchor_coords.to(dtype=torch.float32),
                anchor_coords.to(dtype=torch.float32),
            )
            same_stratum = (
                anchor_buildings[:, None] == anchor_buildings[None, :]
            ) & (anchor_floors[:, None] == anchor_floors[None, :])
            pair_distance = pair_distance.masked_fill(
                ~same_stratum, float("inf")
            )
            pair_distance.fill_diagonal_(float("inf"))
            nearest = pair_distance.amin(dim=1)
            if not bool(torch.isfinite(nearest).all()):
                global_distance = torch.cdist(
                    anchor_coords.to(dtype=torch.float32),
                    anchor_coords.to(dtype=torch.float32),
                )
                global_distance.fill_diagonal_(float("inf"))
                fallback = global_distance.amin(dim=1)
                nearest = torch.where(torch.isfinite(nearest), nearest, fallback)
            local_radius = (radius_fraction * nearest).clamp(
                min=radius_min, max=radius_max
            )
        else:
            local_radius = torch.empty(0, dtype=torch.float32)
        # The radius is a deterministic function of the registered anchor
        # dictionary and configuration, so it need not enlarge checkpoints.
        self.register_buffer(
            "local_residual_radius_m", local_radius, persistent=False
        )
        # V9 reconstructs this deterministic tensor from the registered anchor
        # geometry, keeping old checkpoints strictly loadable.
        self.register_buffer(
            "local_residual_transform", residual_transform, persistent=False
        )
        self.residual_experts = int(residual_experts)
        self._initialize_residual_head()
        self.extra_residual_heads = nn.ModuleList()
        if self.residual_experts > 1:
            for expert_id in range(1, self.residual_experts):
                head = copy.deepcopy(self.residual_head)
                angle = 2.0 * math.pi * expert_id / self.residual_experts
                offset = 1.5 / max(self.max_residual_m, 1e-6)
                head[-1].bias.data[0] = math.atanh(
                    max(-0.95, min(0.95, offset * math.cos(angle))))
                head[-1].bias.data[1] = math.atanh(
                    max(-0.95, min(0.95, offset * math.sin(angle))))
                self.extra_residual_heads.append(head)
            self.expert_gate = nn.Sequential(
                nn.Linear(d_model + 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.residual_experts),
            )
            self.expert_risk = nn.Sequential(
                nn.Linear(d_model + 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.residual_experts),
            )
            nn.init.zeros_(self.expert_gate[-1].weight)
            nn.init.zeros_(self.expert_gate[-1].bias)
            nn.init.zeros_(self.expert_risk[-1].weight)
            nn.init.constant_(
                self.expert_risk[-1].bias, math.log(math.expm1(8.0)))
        else:
            self.expert_gate = None
            self.expert_risk = None

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_coords.shape[0])

    def _initialize_residual_head(self) -> None:
        final = self.residual_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        initial_scale = 6.0 - self.min_scale_m
        final.bias.data[2:] = math.log(math.expm1(initial_scale))

    @torch.no_grad()
    def initialize_anchor_keys(
        self, prototype_rss: torch.Tensor, batch_size: int = 256
    ) -> None:
        """Initialize memory keys from robust anchor RSS prototypes."""
        was_training = self.training
        self.eval()
        keys = []
        for start in range(0, len(prototype_rss), batch_size):
            batch = prototype_rss[start:start + batch_size].to(
                self.anchor_keys.device)
            keys.append(self.encoder(batch))
        self.anchor_keys.copy_(torch.cat(keys, dim=0))
        self.train(was_training)

    @torch.no_grad()
    def initialize_experts_from_primary(
        self, directional_offset_m: float = 1.5
    ) -> None:
        """Clone a trained primary residual head into specialist heads."""
        if self.residual_experts <= 1:
            return
        for expert_id, head in enumerate(
            self.extra_residual_heads, start=1
        ):
            head.load_state_dict(self.residual_head.state_dict())
            angle = 2.0 * math.pi * expert_id / self.residual_experts
            normalized_offset = directional_offset_m / max(
                self.max_residual_m, 1e-6)
            head[-1].bias[0] += math.atanh(
                max(
                    -0.95,
                    min(0.95, normalized_offset * math.cos(angle)),
                )
            )
            head[-1].bias[1] += math.atanh(
                max(
                    -0.95,
                    min(0.95, normalized_offset * math.sin(angle)),
                )
            )

    def retrieval_logits_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        keys = F.normalize(self.anchor_keys, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * embedding @ keys.t()

    def retrieval_logits(
        self,
        rss: torch.Tensor,
        ap_dropout: float = 0.0,
        neural_scale: float | None = None,
        prototype_distance_weight: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(rss, ap_dropout=ap_dropout)
        neural_logits = self.retrieval_logits_from_embedding(embedding)
        logits = neural_logits
        if self.prototype_rss.numel() > 0 and self.prototype_metric == "l2":
            seen = self.encoder.seen_ap_mask
            query = rss[:, seen].float()
            prototypes = self.prototype_rss[:, seen].float()
            query_norm = query.square().mean(dim=1, keepdim=True)
            prototype_norm = prototypes.square().mean(dim=1).unsqueeze(0)
            cross = query @ prototypes.t() / max(1, int(seen.sum()))
            distance = (
                query_norm + prototype_norm - 2.0 * cross
            ).clamp(min=0.0).sqrt()
            if prototype_distance_weight is None:
                temperature = self.log_prototype_temperature.exp().clamp(
                    min=0.02, max=2.0)
                prototype_weight = (
                    F.softplus(self.prototype_weight_raw) / temperature)
            else:
                prototype_weight = distance.new_tensor(
                    prototype_distance_weight)
            if neural_scale is None:
                neural_scale = 1.0
            logits = neural_scale * neural_logits - prototype_weight * distance
        elif neural_scale is not None:
            logits = neural_scale * neural_logits
        return logits, embedding

    @staticmethod
    def _candidate_standardize(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt()
        return (centered / scale.clamp(min=1e-4)).clamp(min=-5.0, max=5.0)

    def candidate_evidence_features(
        self,
        rss: torch.Tensor,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Build device-aware evidence for a small retrieved anchor set."""
        if not self.use_candidate_reranker:
            raise RuntimeError("candidate reranker is disabled")
        seen = self.encoder.seen_ap_mask
        query = rss[:, seen].float()
        query_visible = query > -0.999
        query_mask = query_visible.float().unsqueeze(1)
        query_values = torch.where(
            query_visible, query, torch.zeros_like(query)).unsqueeze(1)

        detection_prob = self.anchor_detection_prob[:, seen][candidate_indices]
        center = self.anchor_rss_center[:, seen][candidate_indices]
        scale = self.anchor_rss_scale[:, seen][candidate_indices]
        probability = detection_prob.clamp(min=0.02, max=0.98)

        intersection = (query_mask * probability).sum(dim=2)
        query_count = query_mask.sum(dim=2).clamp(min=1.0)
        anchor_count = probability.sum(dim=2).clamp(min=1.0)
        union = query_count + anchor_count - intersection
        jaccard = intersection / union.clamp(min=1.0)
        query_coverage = intersection / query_count
        anchor_coverage = intersection / anchor_count

        present_log_support = (
            query_mask * probability.log()
        ).sum(dim=2) / query_count
        present_support = present_log_support.exp()
        missing_conflict = (
            (1.0 - query_mask) * probability
        ).sum(dim=2) / anchor_count

        shared_weight = query_mask * probability
        shared_mass = shared_weight.sum(dim=2).clamp(min=1e-4)
        query_mean = (
            shared_weight * query_values
        ).sum(dim=2) / shared_mass
        center_mean = (
            shared_weight * center
        ).sum(dim=2) / shared_mass
        query_delta = query_values - query_mean.unsqueeze(2)
        center_delta = center - center_mean.unsqueeze(2)
        covariance = (
            shared_weight * query_delta * center_delta
        ).sum(dim=2)
        query_variance = (
            shared_weight * query_delta.square()
        ).sum(dim=2)
        center_variance = (
            shared_weight * center_delta.square()
        ).sum(dim=2)
        correlation = covariance / (
            query_variance * center_variance
        ).clamp(min=1e-8).sqrt()
        sufficiently_shared = (
            query_mask * (probability >= 0.20)
        ).sum(dim=2) >= 3.0
        correlation = torch.where(
            sufficiently_shared,
            correlation.clamp(min=-1.0, max=1.0),
            torch.zeros_like(correlation),
        )
        affine_residual = 1.0 - correlation.square()
        standardized_residual = (
            shared_weight
            * (query_values - center).abs()
            / (scale + 0.05)
        ).sum(dim=2) / shared_mass
        standardized_residual = torch.tanh(
            torch.log1p(standardized_residual) / 2.0)

        neural_logits = torch.gather(
            self.retrieval_logits_from_embedding(embedding),
            1,
            candidate_indices,
        )
        sample_count = self.anchor_sample_count[candidate_indices]
        sample_count = (
            torch.log1p(sample_count) / math.log(32.0)
        ).clamp(max=1.5)
        return torch.stack(
            [
                self._candidate_standardize(coarse_logits) / 5.0,
                self._candidate_standardize(neural_logits) / 5.0,
                jaccard,
                query_coverage,
                anchor_coverage,
                present_support,
                missing_conflict,
                correlation,
                affine_residual,
                standardized_residual,
                sample_count,
            ],
            dim=2,
        )

    def ap_pair_evidence_features(
        self,
        rss: torch.Tensor,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate AP-specific query-anchor evidence without losing AP identity."""
        if self.ap_pair_evidence is None:
            raise RuntimeError("AP-pair evidence reranker is disabled")
        seen = self.encoder.seen_ap_mask
        query = rss[:, seen].float()
        query_visible = query > -0.999
        query_mask = query_visible.float().unsqueeze(1)
        query_values = torch.where(
            query_visible, query, torch.zeros_like(query)).unsqueeze(1)

        probability = self.anchor_detection_prob[:, seen][
            candidate_indices].clamp(min=0.02, max=0.98)
        center = self.anchor_rss_center[:, seen][candidate_indices]
        scale = self.anchor_rss_scale[:, seen][candidate_indices].clamp(
            min=0.03, max=1.0)
        signed_residual = (query_values - center) / (scale + 0.05)
        signed_residual = signed_residual.clamp(min=-8.0, max=8.0) / 4.0
        support = query_mask * probability
        present_conflict = query_mask * (1.0 - probability)
        missing_conflict = (1.0 - query_mask) * probability
        ap_features = torch.stack(
            [
                query_values.expand_as(center),
                center,
                signed_residual,
                signed_residual.abs(),
                query_mask.expand_as(center),
                probability,
                support,
                present_conflict,
                missing_conflict,
            ],
            dim=3,
        )
        ap_scores = self.ap_pair_evidence(ap_features).squeeze(-1)
        reliability = torch.sigmoid(
            self.ap_reliability_logits[seen]).view(1, 1, -1)
        evidence_mask = (
            query_mask + (probability >= 0.10).to(query_mask.dtype)
        ).clamp(max=1.0)
        evidence_weight = reliability * evidence_mask
        pair_score = torch.sum(
            evidence_weight * ap_scores, dim=2
        ) / evidence_weight.sum(dim=2).clamp(min=1.0)

        neural_logits = torch.gather(
            self.retrieval_logits_from_embedding(embedding),
            1,
            candidate_indices,
        )
        intersection = (query_mask * probability).sum(dim=2)
        query_count = query_mask.sum(dim=2).clamp(min=1.0)
        anchor_count = probability.sum(dim=2).clamp(min=1.0)
        union = query_count + anchor_count - intersection
        sample_count = (
            torch.log1p(self.anchor_sample_count[candidate_indices])
            / math.log(32.0)
        ).clamp(max=1.5)
        return torch.stack(
            [
                self._candidate_standardize(coarse_logits) / 5.0,
                self._candidate_standardize(neural_logits) / 5.0,
                pair_score,
                intersection / union.clamp(min=1.0),
                intersection / query_count,
                intersection / anchor_count,
                missing_conflict.mean(dim=2),
                sample_count,
            ],
            dim=2,
        )

    def candidate_reranker_parameters(self):
        if self.evidence_reranker is not None:
            yield from self.evidence_reranker.parameters()
        if self.ap_pair_evidence is not None:
            yield from self.ap_pair_evidence.parameters()
        if self.ap_reliability_logits is not None:
            yield self.ap_reliability_logits

    def train_candidate_reranker(self) -> None:
        if self.evidence_reranker is not None:
            self.evidence_reranker.train()
        if self.ap_pair_evidence is not None:
            self.ap_pair_evidence.train()

    def _candidate_components(
        self,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.reranker_anchor_decoder:
            return self._components(embedding, candidate_indices)
        location = self.anchor_coords[candidate_indices]
        scale = torch.full_like(location, self.anchor_decoder_scale_m)
        return location, scale

    def rerank_candidate_logits(
        self,
        rss: torch.Tensor,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_candidate_reranker:
            return coarse_logits
        if self.candidate_reranker_mode == "ap_pair":
            features = self.ap_pair_evidence_features(
                rss, embedding, candidate_indices, coarse_logits)
        else:
            features = self.candidate_evidence_features(
                rss, embedding, candidate_indices, coarse_logits)
        return coarse_logits + self.evidence_reranker(features).squeeze(-1)

    @staticmethod
    def augment_scan_style(
        rss: torch.Tensor,
        offset_std_db: float = 0.0,
        scale_std: float = 0.0,
    ) -> torch.Tensor:
        """Apply scan-level device-style perturbations to detected RSS only."""
        if offset_std_db <= 0.0 and scale_std <= 0.0:
            return rss
        visible = rss > -0.999
        values = torch.where(
            visible, rss, torch.full_like(rss, float("nan")))
        median = _masked_lower_median(
            values, visible, empty_value=-1.0
        ).unsqueeze(1)
        offset = torch.randn(
            len(rss), 1, device=rss.device, dtype=rss.dtype
        ) * (offset_std_db / 40.0)
        scale = 1.0 + torch.randn(
            len(rss), 1, device=rss.device, dtype=rss.dtype
        ) * scale_std
        scale = scale.clamp(0.75, 1.25)
        augmented = median + scale * (rss - median) + offset
        augmented = augmented.clamp(-0.99, 1.0)
        return torch.where(visible, augmented, torch.full_like(rss, -1.0))

    def _components(
        self, embedding: torch.Tensor, candidate_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        location, scale = self._components_from_head(
            self.residual_head, embedding, candidate_indices)
        return location, scale

    def _components_from_head(
        self,
        head: nn.Module,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = F.normalize(self.anchor_keys[candidate_indices], dim=-1)
        query = embedding.unsqueeze(1).expand_as(keys)
        anchor_xy = self.anchor_coords[candidate_indices]
        anchor_features = (
            anchor_xy - self.anchor_coord_mean) / self.anchor_coord_scale
        pair = torch.cat([query, keys, query - keys, anchor_features], dim=-1)
        raw = head(pair)
        if self.local_residual_transform.numel() > 0:
            transform = self.local_residual_transform[candidate_indices]
            direction = torch.tanh(raw[..., :2])
            norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
            direction = direction / torch.maximum(
                norm, torch.ones_like(norm)
            )
            residual = torch.einsum(
                "...ij,...j->...i", transform, direction
            )
        elif self.local_residual_radius_m.numel() > 0:
            radius = self.local_residual_radius_m[
                candidate_indices
            ].unsqueeze(-1)
            residual = torch.tanh(raw[..., :2]) * radius
        else:
            residual = torch.tanh(raw[..., :2]) * self.max_residual_m
        scale = F.softplus(raw[..., 2:]) + self.min_scale_m
        return anchor_xy + residual, scale

    def _expert_components(
        self,
        embedding: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        heads = [self.residual_head, *self.extra_residual_heads]
        components = [
            self._components_from_head(head, embedding, candidate_indices)
            for head in heads
        ]
        locations = torch.stack(
            [component[0] for component in components], dim=1)
        scales = torch.stack(
            [component[1] for component in components], dim=1)
        return locations, scales

    @staticmethod
    def _scan_style_features(rss: torch.Tensor) -> torch.Tensor:
        visible = rss > -0.999
        values = torch.where(
            visible, rss, torch.full_like(rss, float("nan")))
        median = _masked_lower_median(values, visible, empty_value=-1.0)
        maximum = torch.nan_to_num(
            torch.amax(values, dim=1), nan=-1.0)
        count = visible.float().mean(dim=1)
        return torch.stack([median, maximum, count], dim=1)

    def _expert_route(
        self,
        embedding: torch.Tensor,
        rss: torch.Tensor,
        risk_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        style = self._scan_style_features(rss)
        features = torch.cat([embedding, style], dim=1)
        gate_logits = self.expert_gate(features)
        predicted_risk = F.softplus(self.expert_risk(features)) + 0.1
        route_logits = (
            gate_logits
            - predicted_risk / max(float(risk_temperature), 1e-3)
        )
        return F.softmax(route_logits, dim=1), predicted_risk, gate_logits

    def student_log_prob(
        self,
        target: torch.Tensor,
        location: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Independent-axis Student-t log probability, summed over x and y."""
        df = self.student_df
        z = (target.unsqueeze(1) - location) / scale
        normalizer = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
        )
        log_prob = (
            normalizer
            - torch.log(scale)
            - 0.5 * (df + 1.0) * torch.log1p(z.square() / df)
        )
        return log_prob.sum(dim=-1)

    def training_loss(
        self,
        rss: torch.Tensor,
        target_coords: torch.Tensor,
        target_anchor: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        retrieval_weight: float = 0.5,
        hierarchy_weight: float = 1.0,
        self_target_weight: float = 0.0,
        coordinate_weight: float = 0.0,
        device_ids: torch.Tensor | None = None,
        domain_adversarial_weight: float = 0.0,
        rss_offset_aug_db: float = 0.0,
        rss_scale_aug: float = 0.0,
        physics_weight: float = 0.0,
        expert_route_weight: float = 0.0,
        expert_risk_weight: float = 0.0,
        expert_balance_weight: float = 0.0,
        expert_specialist_weight: float = 0.0,
        oracle_route_temperature: float = 2.0,
        risk_route_temperature: float = 2.0,
        environment_ids: torch.Tensor | None = None,
        environment_dro_weight: float = 0.0,
        environment_dro_temperature: float = 1.0,
        counterfactual_weight: float = 0.0,
        reranker_weight: float = 1.0,
        reranker_leave_anchor_out: bool = False,
        reranker_spatial_temperature_m: float = 4.0,
        episodic_reference_rss: torch.Tensor | None = None,
        episodic_reference_weight: float = 0.0,
        episodic_contrastive_weight: float = 0.0,
        episodic_consistency_weight: float = 0.0,
        episodic_temperature: float = 0.10,
        train_candidates: int = 32,
        ap_dropout: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        model_rss = self.augment_scan_style(
            rss,
            offset_std_db=rss_offset_aug_db,
            scale_std=rss_scale_aug,
        )
        raw_logits, embedding = self.retrieval_logits(
            model_rss, ap_dropout=ap_dropout)
        target_neighbors = neighbor_indices[target_anchor]
        spatial_weights = neighbor_weights[target_anchor]
        if self_target_weight > 0.0:
            self_weight = min(float(self_target_weight), 1.0)
            target_indices = torch.cat(
                [target_anchor.unsqueeze(1), target_neighbors], dim=1)
            target_weights = torch.cat(
                [
                    torch.full_like(
                        target_anchor.unsqueeze(1),
                        self_weight,
                        dtype=raw_logits.dtype,
                    ),
                    spatial_weights * (1.0 - self_weight),
                ],
                dim=1,
            )
            retrieval_logits = raw_logits
        else:
            target_indices = target_neighbors
            target_weights = spatial_weights
            retrieval_logits = raw_logits.clone()
            retrieval_logits.scatter_(
                1, target_anchor.unsqueeze(1), float("-inf"))

        log_probs = F.log_softmax(retrieval_logits, dim=1)
        retrieval_loss = -(
            torch.gather(log_probs, 1, target_indices) * target_weights
        ).sum(dim=1).mean()

        building_logits = self._marginal_logits(
            raw_logits, self.anchor_buildings)
        floor_logits = self._marginal_logits(raw_logits, self.anchor_floors)
        target_building = self.anchor_buildings[target_anchor]
        target_floor = self.anchor_floors[target_anchor]
        hierarchy_loss = (
            0.25 * F.cross_entropy(building_logits, target_building)
            + F.cross_entropy(floor_logits, target_floor)
        )

        if self.use_candidate_reranker and reranker_leave_anchor_out:
            candidate_count = min(train_candidates, self.n_anchors - 1)
            selection_scores = raw_logits.detach().clone()
            selection_scores.scatter_(
                1, target_anchor.unsqueeze(1), float("-inf"))
        else:
            candidate_count = min(
                max(train_candidates, target_indices.shape[1]),
                (
                    self.n_anchors
                    if self_target_weight > 0.0
                    else self.n_anchors - 1
                ),
            )
            selection_scores = retrieval_logits.detach().clone()
            forced = selection_scores.max(dim=1, keepdim=True).values + 1.0
            forced = forced + torch.arange(
                target_indices.shape[1],
                device=rss.device,
                dtype=selection_scores.dtype,
            ).unsqueeze(0) * 1e-3
            selection_scores.scatter_(1, target_indices, forced)
        candidate_indices = torch.topk(
            selection_scores, candidate_count, dim=1).indices

        coarse_candidate_logits = torch.gather(
            raw_logits, 1, candidate_indices)
        candidate_logits = self.rerank_candidate_logits(
            model_rss,
            embedding,
            candidate_indices,
            coarse_candidate_logits,
        )
        candidate_weights = F.softmax(candidate_logits, dim=1)
        reranker_loss = raw_logits.new_zeros(())
        if self.use_candidate_reranker:
            if reranker_leave_anchor_out:
                candidate_anchor_coords = self.anchor_coords[
                    candidate_indices]
                candidate_distances = torch.linalg.vector_norm(
                    candidate_anchor_coords - target_coords.unsqueeze(1),
                    dim=2,
                )
                target_building = self.anchor_buildings[target_anchor]
                same_building = (
                    self.anchor_buildings[candidate_indices]
                    == target_building.unsqueeze(1)
                )
                spatial_logits = (
                    -candidate_distances
                    / max(float(reranker_spatial_temperature_m), 1e-3)
                )
                has_same_building = same_building.any(
                    dim=1, keepdim=True)
                target_logits = torch.where(
                    has_same_building,
                    spatial_logits.masked_fill(
                        ~same_building, float("-inf")),
                    spatial_logits,
                )
                candidate_target_weights = F.softmax(target_logits, dim=1)
                if not self.reranker_anchor_decoder:
                    conditioned_weights = candidate_weights * same_building
                    conditioned_weights = (
                        conditioned_weights
                        / conditioned_weights.sum(
                            dim=1, keepdim=True).clamp(min=1e-12)
                    )
                    candidate_weights = torch.where(
                        has_same_building,
                        conditioned_weights,
                        candidate_weights,
                    )
            else:
                candidate_target_weights = torch.sum(
                    (
                        candidate_indices.unsqueeze(2)
                        == target_indices.unsqueeze(1)
                    ).to(raw_logits.dtype)
                    * target_weights.unsqueeze(1),
                    dim=2,
                )
            candidate_target_weights = (
                candidate_target_weights
                / candidate_target_weights.sum(
                    dim=1, keepdim=True).clamp(min=1e-12)
            )
            reranker_loss = -torch.sum(
                candidate_target_weights
                * F.log_softmax(candidate_logits, dim=1),
                dim=1,
            ).mean()
        route_loss = raw_logits.new_zeros(())
        risk_loss = raw_logits.new_zeros(())
        balance_loss = raw_logits.new_zeros(())
        specialist_loss = raw_logits.new_zeros(())
        environment_dro_loss = raw_logits.new_zeros(())
        counterfactual_loss = raw_logits.new_zeros(())
        episodic_reference_loss = raw_logits.new_zeros(())
        episodic_contrastive_loss = raw_logits.new_zeros(())
        episodic_consistency_loss = raw_logits.new_zeros(())
        if self.residual_experts == 1:
            location, scale = self._candidate_components(
                embedding, candidate_indices)
            component_lp = self.student_log_prob(
                target_coords, location, scale)
            mixture_lp = torch.logsumexp(
                torch.log(candidate_weights.clamp(min=1e-12))
                + component_lp,
                dim=1,
            )
            point_prediction = torch.sum(
                candidate_weights.unsqueeze(-1) * location, dim=1)
        else:
            locations, scales = self._expert_components(
                embedding, candidate_indices)
            expanded_target = target_coords.unsqueeze(1).expand(
                -1, self.residual_experts, -1)
            expert_component_lp = self.student_log_prob(
                expanded_target.reshape(-1, 2),
                locations.reshape(
                    -1, locations.shape[2], locations.shape[3]),
                scales.reshape(-1, scales.shape[2], scales.shape[3]),
            ).reshape(-1, self.residual_experts, locations.shape[2])
            expert_mixture_lp = torch.logsumexp(
                torch.log(candidate_weights.clamp(min=1e-12)).unsqueeze(1)
                + expert_component_lp,
                dim=2,
            )
            expert_points = torch.sum(
                candidate_weights[:, None, :, None] * locations, dim=2)
            expert_errors = torch.linalg.vector_norm(
                expert_points - target_coords.unsqueeze(1), dim=2)
            route_weights, predicted_risk, _ = self._expert_route(
                embedding, model_rss, risk_route_temperature)
            oracle_weights = F.softmax(
                -expert_errors.detach()
                / max(float(oracle_route_temperature), 1e-3),
                dim=1,
            )
            route_loss = -torch.sum(
                oracle_weights
                * torch.log(route_weights.clamp(min=1e-12)),
                dim=1,
            ).mean()
            risk_loss = F.smooth_l1_loss(
                torch.log1p(predicted_risk),
                torch.log1p(expert_errors.detach()),
            )
            mean_route = route_weights.mean(dim=0)
            balance_loss = torch.sum(
                mean_route
                * torch.log(
                    mean_route * self.residual_experts + 1e-12))
            specialist_loss = -torch.sum(
                oracle_weights * expert_mixture_lp, dim=1).mean()
            mixture_lp = torch.logsumexp(
                torch.log(route_weights.clamp(min=1e-12))
                + expert_mixture_lp,
                dim=1,
            )
            point_prediction = torch.sum(
                route_weights.unsqueeze(-1) * expert_points, dim=1)
        nll = -mixture_lp.mean()
        coordinate_errors = torch.linalg.vector_norm(
            point_prediction - target_coords, dim=1)
        coordinate_loss = coordinate_errors.mean()
        if (
            counterfactual_weight > 0.0
            and (rss_offset_aug_db > 0.0 or rss_scale_aug > 0.0)
        ):
            counterfactual_rss = self.augment_scan_style(
                rss,
                offset_std_db=rss_offset_aug_db,
                scale_std=rss_scale_aug,
            )
            counterfactual_logits, counterfactual_embedding = (
                self.retrieval_logits(
                    counterfactual_rss,
                    ap_dropout=ap_dropout,
                )
            )
            counterfactual_coarse_logits = torch.gather(
                counterfactual_logits, 1, candidate_indices)
            counterfactual_candidate_logits = self.rerank_candidate_logits(
                counterfactual_rss,
                counterfactual_embedding,
                candidate_indices,
                counterfactual_coarse_logits,
            )
            counterfactual_candidate_weights = F.softmax(
                counterfactual_candidate_logits, dim=1)
            if self.residual_experts == 1:
                counterfactual_locations, _ = self._candidate_components(
                    counterfactual_embedding, candidate_indices)
                counterfactual_point = torch.sum(
                    counterfactual_candidate_weights.unsqueeze(-1)
                    * counterfactual_locations,
                    dim=1,
                )
            else:
                counterfactual_locations, _ = self._expert_components(
                    counterfactual_embedding, candidate_indices)
                counterfactual_routes, _, _ = self._expert_route(
                    counterfactual_embedding,
                    counterfactual_rss,
                    risk_route_temperature,
                )
                counterfactual_expert_points = torch.sum(
                    counterfactual_candidate_weights[:, None, :, None]
                    * counterfactual_locations,
                    dim=2,
                )
                counterfactual_point = torch.sum(
                    counterfactual_routes.unsqueeze(-1)
                    * counterfactual_expert_points,
                    dim=1,
                )
            counterfactual_loss = F.smooth_l1_loss(
                counterfactual_point,
                point_prediction.detach(),
            )
        if (
            environment_ids is not None
            and environment_dro_weight > 0.0
        ):
            group_losses = torch.stack(
                [
                    coordinate_errors[environment_ids == environment_id].mean()
                    for environment_id in torch.unique(environment_ids)
                    if torch.any(environment_ids == environment_id)
                ]
            )
            temperature = max(float(environment_dro_temperature), 1e-3)
            environment_dro_loss = (
                temperature * torch.logsumexp(
                    group_losses / temperature, dim=0)
                - temperature * math.log(len(group_losses))
            )
        if (
            episodic_reference_rss is not None
            and (
                episodic_reference_weight > 0.0
                or episodic_contrastive_weight > 0.0
                or episodic_consistency_weight > 0.0
            )
        ):
            reference_logits, reference_embedding = self.retrieval_logits(
                episodic_reference_rss,
                ap_dropout=ap_dropout,
            )
            if episodic_reference_weight > 0.0:
                reference_log_probs = F.log_softmax(
                    reference_logits, dim=1)
                reference_retrieval = -(
                    torch.gather(
                        reference_log_probs, 1, target_indices)
                    * target_weights
                ).sum(dim=1).mean()
                reference_building_logits = self._marginal_logits(
                    reference_logits, self.anchor_buildings)
                reference_floor_logits = self._marginal_logits(
                    reference_logits, self.anchor_floors)
                reference_hierarchy = (
                    0.25 * F.cross_entropy(
                        reference_building_logits, target_building)
                    + F.cross_entropy(
                        reference_floor_logits, target_floor)
                )
                episodic_reference_loss = (
                    reference_retrieval + reference_hierarchy)
            if episodic_contrastive_weight > 0.0:
                temperature = max(float(episodic_temperature), 1e-3)
                similarity = (
                    embedding @ reference_embedding.t() / temperature)
                same_building = (
                    target_building.unsqueeze(1)
                    == target_building.unsqueeze(0)
                )
                positive = (
                    target_anchor.unsqueeze(1)
                    == target_anchor.unsqueeze(0)
                )
                forward_loss = (
                    torch.logsumexp(
                        similarity.masked_fill(
                            ~same_building, float("-inf")),
                        dim=1,
                    )
                    - torch.logsumexp(
                        similarity.masked_fill(
                            ~positive, float("-inf")),
                        dim=1,
                    )
                ).mean()
                reverse_similarity = similarity.t()
                reverse_loss = (
                    torch.logsumexp(
                        reverse_similarity.masked_fill(
                            ~same_building, float("-inf")),
                        dim=1,
                    )
                    - torch.logsumexp(
                        reverse_similarity.masked_fill(
                            ~positive, float("-inf")),
                        dim=1,
                    )
                ).mean()
                episodic_contrastive_loss = 0.5 * (
                    forward_loss + reverse_loss)
            if episodic_consistency_weight > 0.0:
                query_neural = self.retrieval_logits_from_embedding(
                    embedding)
                reference_neural = self.retrieval_logits_from_embedding(
                    reference_embedding)
                mean_logits = 0.5 * (
                    query_neural + reference_neural)
                mean_log_probs = F.log_softmax(mean_logits, dim=1)
                episodic_consistency_loss = 0.5 * (
                    F.kl_div(
                        mean_log_probs,
                        F.softmax(query_neural, dim=1),
                        reduction="batchmean",
                    )
                    + F.kl_div(
                        mean_log_probs,
                        F.softmax(reference_neural, dim=1),
                        reduction="batchmean",
                    )
                )
        domain_loss = nll.new_zeros(())
        if (
            self.device_head is not None
            and device_ids is not None
            and domain_adversarial_weight > 0.0
        ):
            device_logits = self.device_head(gradient_reverse(embedding))
            domain_loss = F.cross_entropy(device_logits, device_ids)

        physics_nll = nll.new_zeros(())
        if self.physics_model is not None and physics_weight > 0.0:
            physics_nll = self.physics_model.negative_log_likelihood(
                rss, target_coords)

        loss = (
            nll
            + retrieval_weight * retrieval_loss
            + hierarchy_weight * hierarchy_loss
            + coordinate_weight * coordinate_loss
            + domain_adversarial_weight * domain_loss
            + physics_weight * physics_nll
            + expert_route_weight * route_loss
            + expert_risk_weight * risk_loss
            + expert_balance_weight * balance_loss
            + expert_specialist_weight * specialist_loss
            + environment_dro_weight * environment_dro_loss
            + counterfactual_weight * counterfactual_loss
            + reranker_weight * reranker_loss
            + episodic_reference_weight * episodic_reference_loss
            + episodic_contrastive_weight * episodic_contrastive_loss
            + episodic_consistency_weight * episodic_consistency_loss
        )
        return {
            "loss": loss,
            "nll": nll.detach(),
            "retrieval_loss": retrieval_loss.detach(),
            "hierarchy_loss": hierarchy_loss.detach(),
            "coordinate_loss": coordinate_loss.detach(),
            "domain_loss": domain_loss.detach(),
            "physics_nll": physics_nll.detach(),
            "expert_route_loss": route_loss.detach(),
            "expert_risk_loss": risk_loss.detach(),
            "expert_balance_loss": balance_loss.detach(),
            "expert_specialist_loss": specialist_loss.detach(),
            "environment_dro_loss": environment_dro_loss.detach(),
            "counterfactual_loss": counterfactual_loss.detach(),
            "reranker_loss": reranker_loss.detach(),
            "episodic_reference_loss": episodic_reference_loss.detach(),
            "episodic_contrastive_loss": episodic_contrastive_loss.detach(),
            "episodic_consistency_loss": episodic_consistency_loss.detach(),
        }

    @staticmethod
    def _marginal_logits(
        anchor_logits: torch.Tensor, anchor_labels: torch.Tensor
    ) -> torch.Tensor:
        """Log-sum-exp anchor evidence into building or floor classes."""
        n_classes = int(anchor_labels.max().item()) + 1
        return torch.stack(
            [
                torch.logsumexp(
                    anchor_logits.masked_fill(
                        anchor_labels.unsqueeze(0) != class_id,
                        float("-inf"),
                    ),
                    dim=1,
                )
                for class_id in range(n_classes)
            ],
            dim=1,
        )

    def predict(
        self,
        rss: torch.Tensor,
        top_k: int = 32,
        target_coords: torch.Tensor | None = None,
        neural_scale: float | None = None,
        prototype_distance_weight: float | None = None,
        posterior_temperature: float = 1.0,
        condition_building: bool = False,
        deterministic_building: bool = False,
        risk_route_temperature: float = 2.0,
    ) -> dict[str, torch.Tensor]:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise TypeError("top_k must be an integer")
        candidate_count = int(top_k)
        if not 1 <= candidate_count <= self.n_anchors:
            raise ValueError(
                "top_k must satisfy 1 <= top_k <= n_anchors")
        logits, embedding = self.retrieval_logits(
            rss,
            neural_scale=neural_scale,
            prototype_distance_weight=prototype_distance_weight,
        )
        hierarchy_logits = logits
        if neural_scale is not None or prototype_distance_weight is not None:
            # Coordinate retrieval may be validation-calibrated, while the
            # hierarchy must retain the logits used by its training loss.
            hierarchy_logits, _ = self.retrieval_logits(rss)
        candidate_logits, candidate_indices = torch.topk(
            logits, candidate_count, dim=1)
        candidate_logits = self.rerank_candidate_logits(
            rss,
            embedding,
            candidate_indices,
            candidate_logits,
        )
        if self.residual_experts == 1:
            location, scale = self._candidate_components(
                embedding, candidate_indices)
            expert_locations = None
            expert_scales = None
            expert_weights = None
            predicted_expert_risk = None
        else:
            expert_locations, expert_scales = self._expert_components(
                embedding, candidate_indices)
            expert_weights, predicted_expert_risk, _ = self._expert_route(
                embedding, rss, risk_route_temperature)
            location = torch.sum(
                expert_weights[:, :, None, None] * expert_locations, dim=1)
            scale = torch.sqrt(torch.sum(
                expert_weights[:, :, None, None]
                * expert_scales.square(),
                dim=1,
            ))
        if (
            self.physics_model is not None
            and self.physics_rerank_weight != 0.0
        ):
            candidate_logits = (
                candidate_logits
                + self.physics_rerank_weight
                * self.physics_model.candidate_scores(rss, location)
            )
        weights = torch.softmax(
            candidate_logits / max(float(posterior_temperature), 1e-3),
            dim=1,
        )

        building_logits = self._marginal_logits(
            hierarchy_logits, self.anchor_buildings)
        floor_logits = self._marginal_logits(
            hierarchy_logits, self.anchor_floors)
        pred_building = building_logits.argmax(dim=1)
        pred_floor = floor_logits.argmax(dim=1)
        if deterministic_building:
            visible = (
                (rss > self.encoder.visible_threshold)
                & self.encoder.seen_ap_mask.unsqueeze(0)
            )
            signal = (rss + 1.0).clamp(min=0.0) * visible
            building_scores = torch.stack(
                [
                    signal[:, self.ap_building_owner == class_id].sum(dim=1)
                    for class_id in range(building_logits.shape[1])
                ],
                dim=1,
            )
            has_signal = building_scores.sum(dim=1) > 0.0
            voted_building = building_scores.argmax(dim=1)
            pred_building = torch.where(
                has_signal, voted_building, pred_building)

        # Horizontal coordinates are posterior expectations. Hard filtering by
        # an argmax floor would reintroduce cascade error into the coordinate
        # estimate and break the probabilistic factorization.
        coord_weights = weights
        if condition_building:
            same_building = (
                self.anchor_buildings[candidate_indices]
                == pred_building.unsqueeze(1)
            )
            conditioned = weights * same_building
            has_candidate = conditioned.sum(dim=1, keepdim=True) > 0.0
            conditioned = conditioned / conditioned.sum(
                dim=1, keepdim=True).clamp(min=1e-12)
            coord_weights = torch.where(
                has_candidate, conditioned, coord_weights)
        if self.residual_experts == 1:
            mean = torch.sum(
                coord_weights.unsqueeze(-1) * location, dim=1)
            component_variance = (
                scale.square() * self.student_df / (self.student_df - 2.0))
            radial_second_moment = torch.sum(
                coord_weights * (
                    (location - mean.unsqueeze(1)).square().sum(dim=-1)
                    + component_variance.sum(dim=-1)
                ),
                dim=1,
            )
        else:
            expert_points = torch.sum(
                coord_weights[:, None, :, None] * expert_locations, dim=2)
            mean = torch.sum(
                expert_weights.unsqueeze(-1) * expert_points, dim=1)
            component_variance = (
                expert_scales.square()
                * self.student_df / (self.student_df - 2.0)
            )
            radial_second_moment = torch.sum(
                expert_weights * torch.sum(
                    coord_weights[:, None, :] * (
                        (expert_locations - mean[:, None, None, :])
                        .square().sum(dim=-1)
                        + component_variance.sum(dim=-1)
                    ),
                    dim=2,
                ),
                dim=1,
            )
        radial_scale = radial_second_moment.clamp(min=1e-8).sqrt()

        result = {
            "coords": mean,
            "building": pred_building,
            "floor": pred_floor,
            "radial_scale": radial_scale,
            "candidate_indices": candidate_indices,
            "candidate_weights": weights,
            # Candidate preparation consumes these exact tensors.  Returning
            # them avoids a second encoder pass and a duplicate component
            # decode after ``predict``.
            "embedding": embedding,
            "component_locations": location,
            "component_scales": scale,
        }
        if expert_weights is not None:
            result["expert_weights"] = expert_weights
            result["predicted_expert_risk"] = predicted_expert_risk
        if target_coords is not None:
            if self.residual_experts == 1:
                component_lp = self.student_log_prob(
                    target_coords, location, scale)
                result["nll"] = -torch.logsumexp(
                    torch.log(weights.clamp(min=1e-12))
                    + component_lp,
                    dim=1,
                )
            else:
                expanded_target = target_coords.unsqueeze(1).expand(
                    -1, self.residual_experts, -1)
                component_lp = self.student_log_prob(
                    expanded_target.reshape(-1, 2),
                    expert_locations.reshape(
                        -1,
                        expert_locations.shape[2],
                        expert_locations.shape[3],
                    ),
                    expert_scales.reshape(
                        -1,
                        expert_scales.shape[2],
                        expert_scales.shape[3],
                    ),
                ).reshape(
                    -1,
                    self.residual_experts,
                    expert_locations.shape[2],
                )
                result["nll"] = -torch.logsumexp(
                    torch.log(expert_weights.clamp(min=1e-12)).unsqueeze(2)
                    + torch.log(weights.clamp(min=1e-12)).unsqueeze(1)
                    + component_lp,
                    dim=(1, 2),
                )
        return result


