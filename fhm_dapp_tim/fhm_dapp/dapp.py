"""DAPP 核心：AP 可靠性、平衡可见性损失、候选权重投影.

This is the DAPP inference core. It uses unlabeled query RSS consistency +
可见性一致性作为目标, 通过 KL 正则约束在神经网络先验附近,
梯度下降微调 Top-K 候选权重.
"""

from __future__ import annotations

from functools import lru_cache
import math

from scipy.special import stdtrit
import torch
import torch.nn.functional as F


def balanced_visibility_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reliability: torch.Tensor,
    positive_factor: float = 1.0,
    negative_factor: float = 1.0,
) -> torch.Tensor:
    """正负样本平衡的可见性 BCE, 按 AP 可靠性加权.

    Args:
        predicted: [B, n_aps] 预测的 AP 可见概率.
        target: [B, n_aps] 实际可见性 (0/1).
        reliability: [n_aps] 每个 AP 的可靠性权重.

    Returns:
        [B] 每条 query 的平衡可见性损失.
    """
    elementwise = F.binary_cross_entropy(
        predicted.clamp(min=1e-4, max=1.0 - 1e-4),
        target,
        reduction="none",
    )
    positive_weight = target * reliability
    negative_weight = (1.0 - target) * reliability
    positive = positive_weight.sum(dim=-1).clamp(min=1.0)
    negative = negative_weight.sum(dim=-1).clamp(min=1.0)
    positive_term = (elementwise * positive_weight).sum(dim=-1) / positive
    negative_term = (elementwise * negative_weight).sum(dim=-1) / negative
    total_factor = max(float(positive_factor) + float(negative_factor), 1e-8)
    return (
        float(positive_factor) * positive_term
        + float(negative_factor) * negative_term
    ) / total_factor


def anchor_ap_reliability(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
) -> torch.Tensor:
    """从锚点统计估计每个 AP 的定位可靠性.

    可靠性 = log1p(between_std / within_std), 衡量 AP 信号在锚点间的
    区分度. 区分度高的 AP 对定位更有用.

    Args:
        detection_prob: [n_anchors, n_aps] 每个 AP 在每个锚点的检测概率.
        rss_center: [n_anchors, n_aps] 条件 RSS 中心.
        rss_scale: [n_anchors, n_aps] 条件 RSS 尺度.

    Returns:
        [n_aps] 每个 AP 的可靠性权重, 范围 [0.25, 4.0].
    """
    probability = detection_prob.clamp(min=1e-3)
    mass = probability.sum(dim=0).clamp(min=1.0)
    mean = (probability * rss_center).sum(dim=0) / mass
    between = (
        probability * (rss_center - mean).square()
    ).sum(dim=0) / mass
    within = (probability * rss_scale).sum(dim=0) / mass
    information = torch.log1p(
        between.sqrt() / (within + 0.05))
    information = information / information.mean().clamp(min=1e-4)
    return information.clamp(min=0.25, max=4.0)


def anchor_ap_censored_threshold(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    base_threshold: float,
    quantile: float = 0.10,
    shrinkage: float = 0.50,
    min_threshold: float = -1.00,
    max_threshold: float = -0.50,
) -> torch.Tensor:
    """Estimate AP-specific RSS non-detection thresholds from anchors.

    The scalar final-model threshold treats all access points as having the same
    detection frontier. This helper derives a conservative per-AP frontier from
    the low RSS tail of anchor centers, then shrinks it toward the scalar
    threshold so sparse APs cannot dominate the censored likelihood.
    """
    q = min(max(float(quantile), 0.0), 1.0)
    shrink = min(max(float(shrinkage), 0.0), 1.0)
    thresholds = []
    base = torch.tensor(
        float(base_threshold),
        dtype=rss_center.dtype,
        device=rss_center.device,
    )
    for ap_index in range(rss_center.shape[1]):
        detected = detection_prob[:, ap_index] > 0.10
        values = rss_center[:, ap_index][detected]
        if values.numel() < 8:
            thresholds.append(base)
            continue
        empirical = torch.quantile(values, q)
        threshold = shrink * base + (1.0 - shrink) * empirical
        thresholds.append(threshold)
    return torch.stack(thresholds).clamp(
        min=float(min_threshold),
        max=float(max_threshold),
    )


def censored_non_detection_loss(
    predicted_detection: torch.Tensor,
    predicted_rss: torch.Tensor,
    query_mask: torch.Tensor,
    reliability: torch.Tensor,
    mode: str,
    detection_threshold: float,
    rss_threshold: float | torch.Tensor,
    rss_softness: float,
    contradiction_margin: float,
    initial_detection: torch.Tensor | None = None,
    initial_rss: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invisible-AP censored loss for DAPP variants.

    ``hinge`` reproduces the original final-model path. ``tobit`` treats an
    invisible AP as a censored non-detection event. ``contradiction_*`` penalizes
    only posterior moves that make an invisible AP less compatible with the
    initial radio-map posterior.
    """
    mode = mode.lower()
    if mode in {"none", "off"}:
        return torch.zeros(
            len(predicted_detection),
            dtype=predicted_detection.dtype,
            device=predicted_detection.device,
        )
    allowed = {
        "hinge",
        "tobit",
        "contradiction_hinge",
        "contradiction_tobit",
    }
    if mode not in allowed:
        raise ValueError(f"unknown censored_mode: {mode}")

    invisible_weight = (1.0 - query_mask) * reliability
    invisible_norm = invisible_weight.sum(dim=1).clamp(min=1.0)

    if mode == "hinge":
        detection_excess = torch.relu(
            predicted_detection - float(detection_threshold))
        rss_threshold_t = torch.as_tensor(
            rss_threshold,
            dtype=predicted_rss.dtype,
            device=predicted_rss.device,
        )
        rss_excess = torch.relu(predicted_rss - rss_threshold_t)
        element = (
            detection_excess.square()
            + predicted_detection.clamp(min=0.0, max=1.0)
            * rss_excess.square()
        )
    elif mode == "contradiction_hinge":
        if initial_detection is None or initial_rss is None:
            raise ValueError(
                "contradiction_hinge requires initial_detection and "
                "initial_rss")
        detection_excess = torch.relu(
            predicted_detection
            - initial_detection.detach()
            - float(contradiction_margin)
        )
        rss_target = torch.maximum(
            initial_rss.detach(),
            torch.as_tensor(
                rss_threshold,
                dtype=initial_rss.dtype,
                device=initial_rss.device,
            ).expand_as(initial_rss),
        )
        rss_excess = torch.relu(predicted_rss - rss_target)
        element = (
            detection_excess.square()
            + predicted_detection.clamp(min=0.0, max=1.0)
            * rss_excess.square()
        )
    else:
        softness = max(float(rss_softness), 1e-4)

        def _tobit_element(
            detection: torch.Tensor,
            rss_value: torch.Tensor,
        ) -> torch.Tensor:
            p_detect = detection.clamp(min=1e-4, max=1.0 - 1e-4)
            p_low = torch.sigmoid(
                (
                    torch.as_tensor(
                        rss_threshold,
                        dtype=rss_value.dtype,
                        device=rss_value.device,
                    )
                    - rss_value
                ) / softness)
            missing_prob = (1.0 - p_detect) + p_detect * p_low
            return -torch.log(missing_prob.clamp(min=1e-6))

        element = _tobit_element(predicted_detection, predicted_rss)
        if mode == "contradiction_tobit":
            if initial_detection is None or initial_rss is None:
                raise ValueError(
                    "contradiction_tobit requires initial_detection and "
                    "initial_rss")
            initial_element = _tobit_element(
                initial_detection.detach(),
                initial_rss.detach(),
            )
            element = torch.relu(
                element - initial_element - float(contradiction_margin)
            ).square()

    return (element * invisible_weight).sum(dim=1) / invisible_norm


def full_mixture_censored_loss(
    weights: torch.Tensor,
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    query_values: torch.Tensor,
    query_mask: torch.Tensor,
    reliability: torch.Tensor,
    rss_threshold: float | torch.Tensor,
    style_offset: torch.Tensor | None = None,
    style_scale: torch.Tensor | None = None,
    visibility_bias: torch.Tensor | None = None,
    visible_factor: float = 1.0,
    invisible_factor: float = 1.0,
) -> torch.Tensor:
    """Full top-K mixture likelihood for visible RSS and invisible AP events.

    For a visible AP, the likelihood marginalizes candidate anchors:
    sum_k w_k p(z=1|k) p(r|z=1,k).  For an invisible AP, the likelihood is:
    sum_k w_k [p(z=0|k) + p(z=1|k) P(r <= threshold|z=1,k)].
    """
    if rss_scale is None:
        raise ValueError("rss_scale is required for full mixture likelihood")
    p_detect = detection_prob.clamp(min=1e-4, max=1.0 - 1e-4)
    if visibility_bias is not None:
        p_detect = torch.sigmoid(torch.logit(p_detect) + visibility_bias.unsqueeze(1))
    scale = rss_scale.clamp(min=0.03)
    center = rss_center
    if style_scale is not None:
        center = center * style_scale.unsqueeze(1)
    if style_offset is not None:
        center = center + style_offset.unsqueeze(1)

    log_weights = torch.log(weights.clamp(min=1e-8)).unsqueeze(2)
    log_p_detect = torch.log(p_detect)
    residual = (query_values.unsqueeze(1) - center) / scale
    log_pdf = (
        -0.5 * residual.square()
        - torch.log(scale)
        - 0.5 * math.log(2.0 * math.pi)
    )
    visible_nll = -torch.logsumexp(
        log_weights + log_p_detect + log_pdf, dim=1)

    threshold = torch.as_tensor(
        rss_threshold,
        dtype=center.dtype,
        device=center.device,
    )
    z = (threshold - center) / scale
    p_below = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    missing_prob = (1.0 - p_detect) + p_detect * p_below
    invisible_nll = -torch.logsumexp(
        log_weights + torch.log(missing_prob.clamp(min=1e-8)), dim=1)

    reliability = reliability.expand_as(query_mask)
    visible_weight = query_mask * reliability
    invisible_weight = (1.0 - query_mask) * reliability
    visible_term = (
        visible_nll * visible_weight
    ).sum(dim=1) / visible_weight.sum(dim=1).clamp(min=1.0)
    invisible_term = (
        invisible_nll * invisible_weight
    ).sum(dim=1) / invisible_weight.sum(dim=1).clamp(min=1.0)
    total_factor = max(float(visible_factor) + float(invisible_factor), 1e-8)
    return (
        float(visible_factor) * visible_term
        + float(invisible_factor) * invisible_term
    ) / total_factor


def alpha_proximal_weights(
    prior: torch.Tensor,
    energy: torch.Tensor,
    prior_weight: float | torch.Tensor,
    alpha: float,
    bisection_steps: int = 80,
) -> torch.Tensor:
    r"""Solve an alpha-divergence proximal posterior update exactly.

    For ``0 < alpha <= 1`` this returns the unique minimizer on the
    probability simplex of

    .. math::
        \langle q,E\rangle + \lambda D_\alpha(p\Vert q),\qquad
        D_\alpha(p\Vert q)=
        \frac{1-\sum_k p_k^\alpha q_k^{1-\alpha}}
             {\alpha(1-\alpha)}.

    The continuous endpoint at ``alpha=1`` is used.  At ``alpha=0``,
    :math:`D_0(p\Vert q)=\mathrm{KL}(q\Vert p)` and the minimizer is the
    closed-form Gibbs posterior ``softmax(log(prior) - energy / lambda)``.

    For positive alpha, the KKT equations give

    .. math::
        q_k=p_k\left[\frac{\lambda}
        {\alpha(E_k+\eta)}\right]^{1/\alpha},

    where the scalar dual variable ``eta`` has a unique root.  The root is
    found independently for every batch row by a log-domain bisection.  Zero
    prior mass is preserved, so the update cannot invent unsupported anchors.

    Args:
        prior: ``[B, K]`` non-negative prior masses.
        energy: ``[B, K]`` finite anchor energies.
        prior_weight: Positive scalar or one positive value per batch row.
        alpha: Divergence order in ``[0, 1]``.
        bisection_steps: Fixed number of stable batched dual iterations.
    """
    if prior.ndim != 2 or energy.shape != prior.shape:
        raise ValueError("prior and energy must have the same [B, K] shape")
    if not prior.is_floating_point() or not energy.is_floating_point():
        raise TypeError("prior and energy must be floating-point tensors")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")
    if int(bisection_steps) <= 0:
        raise ValueError("bisection_steps must be positive")
    if not bool(torch.isfinite(prior).all()) or bool((prior < 0.0).any()):
        raise ValueError("prior must be finite and non-negative")
    if not bool(torch.isfinite(energy).all()):
        raise ValueError("energy must be finite")

    batch_size = prior.shape[0]
    row_mass = prior.sum(dim=1, keepdim=True)
    if bool((row_mass <= 0.0).any()):
        raise ValueError("every prior row must have positive mass")

    # Float64 dual arithmetic makes the bisection stable even when alpha is
    # small and the power 1 / alpha is sharp.  The result is cast back below.
    output_dtype = torch.promote_types(prior.dtype, energy.dtype)
    work_dtype = torch.float64
    p = prior.to(dtype=work_dtype)
    p = p / p.sum(dim=1, keepdim=True)
    e = energy.to(device=prior.device, dtype=work_dtype)

    lam = torch.as_tensor(
        prior_weight,
        device=prior.device,
        dtype=work_dtype,
    )
    if lam.ndim == 0:
        lam = lam.expand(batch_size).reshape(batch_size, 1)
    elif lam.numel() == batch_size:
        lam = lam.reshape(batch_size, 1)
    else:
        raise ValueError(
            "prior_weight must be scalar or contain one value per batch row")
    if not bool(torch.isfinite(lam).all()) or bool((lam <= 0.0).any()):
        raise ValueError("prior_weight must be finite and strictly positive")

    negative_infinity = torch.full_like(p, -torch.inf)
    log_p = torch.where(p > 0.0, torch.log(p), negative_infinity)
    if float(alpha) == 0.0:
        return torch.softmax(log_p - e / lam, dim=1).to(output_dtype)

    alpha_value = float(alpha)
    alpha_tensor = torch.as_tensor(
        alpha_value, device=prior.device, dtype=work_dtype)
    supported_energy = e.masked_fill(p <= 0.0, torch.inf)
    minimum_energy = supported_energy.amin(dim=1, keepdim=True)

    # eta must be greater than -min(E) on the prior support.  At the upper
    # endpoint every unnormalised mass is <= p_k, so it brackets the root.
    boundary = -minimum_energy
    lower = torch.nextafter(boundary, torch.full_like(boundary, torch.inf))
    upper = lam / alpha_tensor - minimum_energy
    log_numerator = torch.log(lam) - math.log(alpha_value)
    inverse_alpha = 1.0 / alpha_value

    for _ in range(int(bisection_steps)):
        midpoint = 0.5 * (lower + upper)
        positive_gap = (e + midpoint).clamp_min(torch.finfo(work_dtype).tiny)
        log_mass = (
            log_p
            + inverse_alpha * (log_numerator - torch.log(positive_gap))
        )
        mass_above_one = torch.logsumexp(
            log_mass, dim=1, keepdim=True
        ) > 0.0
        lower = torch.where(mass_above_one, midpoint, lower)
        upper = torch.where(mass_above_one, upper, midpoint)

    eta = 0.5 * (lower + upper)
    positive_gap = (e + eta).clamp_min(torch.finfo(work_dtype).tiny)
    log_q = (
        log_p
        + inverse_alpha * (log_numerator - torch.log(positive_gap))
    )
    # Softmax only removes the residual numerical root error; analytically the
    # KKT masses already sum to one.
    return torch.softmax(log_q, dim=1).to(output_dtype)


# If X follows a unit-scale Student-t distribution with nu degrees of freedom,
# median(|X|) = t_nu^{-1}(0.75).  Anchor ``rss_scale`` stores a MAD, so dividing
# by this quantile converts MAD to the matching Student-t scale parameter.  The
# literal t4 constant keeps the registered baseline numerically unchanged.
_T4_MAD_QUANTILE_75 = 0.7406970841282597
_GAUSSIAN_MAD_QUANTILE_75 = 0.6744897501960817


@lru_cache(maxsize=None)
def _student_t_mad_quantile_75(degrees_freedom: float) -> float:
    """Return the exact unit-scale Student-t MAD for a finite positive df."""
    try:
        nu = float(degrees_freedom)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Student-t degrees_freedom must be finite and positive"
        ) from error
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError(
            "Student-t degrees_freedom must be finite and positive")
    if nu == 4.0:
        return _T4_MAD_QUANTILE_75
    quantile = float(stdtrit(nu, 0.75))
    if not math.isfinite(quantile) or quantile <= 0.0:
        raise ValueError(
            "Student-t MAD quantile must be finite and positive")
    return quantile


def robust_hurdle_anchor_energy(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    query_values: torch.Tensor,
    query_mask: torch.Tensor,
    reliability: torch.Tensor,
    style_offset: float | torch.Tensor | None = None,
    style_scale: float | torch.Tensor | None = None,
    visible_factor: float = 1.0,
    invisible_factor: float = 1.0,
    distribution: str = "student_t",
    degrees_freedom: float = 4.0,
    score: str = "balanced_log",
    power_beta: float = 0.25,
    return_self_entropy_baseline: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    r"""Return a robust hurdle energy for every query--anchor pair.

    Each anchor is evaluated against *all* APs.  In the original log-score
    configuration a
    visible observation uses
    ``p_detect * StudentT_nu(rss | a * center + b, a * MAD / c_nu)``; an
    invisible observation uses the hurdle mass ``1 - p_detect``.  The legacy
    ``balanced_log`` score averages visible and invisible channels separately.
    ``joint_log`` instead uses one fixed AP normalization.  The robust
    ``density_power`` option is a strictly proper density-power score on the
    mixed hurdle measure (one atom for non-detection plus Lebesgue measure for
    visible RSS); it uses the same fixed AP normalization and remains bounded
    under gross distributional mismatch.  The optional Gaussian control uses
    its own exact
    MAD-to-scale constant, so both families retain the same robust-statistic
    semantics.

    ``style_scale=a`` is required to be positive and transforms both the RSS
    center and its dispersion.  This scale-equivariance is important: scaling
    only the center would spuriously change standardized residuals.

    Args:
        detection_prob: ``[B,K,A]`` or shared ``[K,A]`` detection propensity.
        rss_center: Matching detection-conditioned RSS medians.
        rss_scale: Matching detection-conditioned RSS MAD values.
        query_values: ``[B,A]`` observed RSS values (ignored when masked).
        query_mask: ``[B,A]`` visible indicators.
        reliability: ``[A]`` or ``[B,A]`` non-negative AP reliability.
        style_offset: Per-scan affine offset ``b``.
        style_scale: Per-scan positive affine scale ``a``.
        distribution: ``student_t`` for the final model or ``gaussian`` for
            a controlled tail-family ablation.
        degrees_freedom: Any finite positive Student-t degrees of freedom.
            The exact ``t_nu^{-1}(0.75)`` quantile converts the anchor MAD to
            the matching Student-t scale; the ``nu=4`` baseline retains its
            original literal conversion constant.
        score: ``balanced_log`` (legacy channel-balanced composite),
            ``joint_log`` (proper hurdle log score), or ``density_power``.
        power_beta: Positive density-power order when ``score`` is
            ``density_power``.  As beta tends to zero, anchor-dependent score
            differences converge to the joint hurdle log score.
        return_self_entropy_baseline: Also return the channel-balanced risk
            expected when the observation is generated by the same anchor.
            This is defined only for ``balanced_log`` and supplies an absolute
            discrepancy reference without location labels.
    """
    if query_values.ndim != 2 or query_mask.shape != query_values.shape:
        raise ValueError("query_values and query_mask must have shape [B, A]")
    if not query_values.is_floating_point():
        raise TypeError("query_values must be floating point")
    if not bool(torch.isfinite(query_values).all()):
        raise ValueError("query_values must be finite")
    if not bool(torch.isfinite(query_mask).all()) or bool(
        ((query_mask < 0.0) | (query_mask > 1.0)).any()
    ):
        raise ValueError("query_mask must be finite and lie in [0, 1]")

    if detection_prob.ndim == 2:
        detection_prob = detection_prob.unsqueeze(0)
    if rss_center.ndim == 2:
        rss_center = rss_center.unsqueeze(0)
    if rss_scale.ndim == 2:
        rss_scale = rss_scale.unsqueeze(0)
    if not (
        detection_prob.ndim == 3
        and detection_prob.shape == rss_center.shape == rss_scale.shape
    ):
        raise ValueError("anchor statistics must share shape [B,K,A] or [K,A]")

    batch_size, ap_count = query_values.shape
    if detection_prob.shape[0] not in {1, batch_size}:
        raise ValueError("anchor-statistics batch dimension is not broadcastable")
    if detection_prob.shape[2] != ap_count:
        raise ValueError("anchor statistics and queries must share AP count")

    dtype = torch.promote_types(query_values.dtype, rss_center.dtype)
    device = query_values.device
    p_detect = detection_prob.to(device=device, dtype=dtype).clamp(
        min=1e-6, max=1.0 - 1e-6)
    center = rss_center.to(device=device, dtype=dtype)
    mad = rss_scale.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(mad).all()):
        raise ValueError("rss_center and rss_scale must be finite")
    if bool((mad < 0.0).any()):
        raise ValueError("rss_scale (MAD) must be non-negative")

    def _per_scan_parameter(
        value: float | torch.Tensor | None,
        default: float,
        name: str,
    ) -> torch.Tensor:
        parameter = torch.as_tensor(
            default if value is None else value,
            device=device,
            dtype=dtype,
        )
        if parameter.ndim == 0:
            parameter = parameter.expand(batch_size)
        elif parameter.numel() == batch_size:
            parameter = parameter.reshape(batch_size)
        else:
            raise ValueError(f"{name} must be scalar or contain one value per scan")
        return parameter.reshape(batch_size, 1, 1)

    affine_scale = _per_scan_parameter(style_scale, 1.0, "style_scale")
    if not bool(torch.isfinite(affine_scale).all()) or bool(
        (affine_scale <= 0.0).any()
    ):
        raise ValueError("style_scale must be finite and strictly positive")
    affine_offset = _per_scan_parameter(style_offset, 0.0, "style_offset")
    if not bool(torch.isfinite(affine_offset).all()):
        raise ValueError("style_offset must be finite")

    distribution = distribution.lower()
    if distribution == "student_t":
        degrees_freedom = float(degrees_freedom)
        mad_quantile = _student_t_mad_quantile_75(degrees_freedom)
    elif distribution == "gaussian":
        mad_quantile = _GAUSSIAN_MAD_QUANTILE_75
    else:
        raise ValueError("distribution must be student_t or gaussian")

    calibrated_center = affine_scale * center + affine_offset
    calibrated_scale = (
        affine_scale * mad.clamp_min(1e-5) / mad_quantile
    ).clamp_min(1e-5)
    observation = query_values.to(dtype=dtype).unsqueeze(1)
    standardized = (observation - calibrated_center) / calibrated_scale

    if distribution == "student_t":
        log_t_normalizer = (
            math.lgamma((degrees_freedom + 1.0) / 2.0)
            - math.lgamma(degrees_freedom / 2.0)
            - 0.5 * math.log(degrees_freedom * math.pi)
        )
        log_conditional_density = (
            log_t_normalizer
            - torch.log(calibrated_scale)
            - 0.5 * (degrees_freedom + 1.0)
            * torch.log1p(standardized.square() / degrees_freedom)
        )
    elif distribution == "gaussian":
        log_conditional_density = (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(calibrated_scale)
            - 0.5 * standardized.square()
        )
    visible_nll = -torch.log(p_detect) - log_conditional_density
    invisible_nll = -torch.log1p(-p_detect)

    if reliability.ndim == 1:
        if reliability.numel() != ap_count:
            raise ValueError("reliability must share the query AP count")
        reliability_weight = reliability.reshape(1, 1, ap_count)
    elif reliability.ndim == 2 and reliability.shape in {
        (1, ap_count),
        (batch_size, ap_count),
    }:
        reliability_weight = reliability.unsqueeze(1)
    else:
        raise ValueError("reliability must have shape [A], [1,A], or [B,A]")
    reliability_weight = reliability_weight.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(reliability_weight).all()) or bool(
        (reliability_weight < 0.0).any()
    ):
        raise ValueError("reliability must be finite and non-negative")

    visible = query_mask.to(device=device, dtype=dtype).unsqueeze(1)
    visible_weight = visible * reliability_weight
    invisible_weight = (1.0 - visible) * reliability_weight
    visible_average = (
        visible_nll * visible_weight
    ).sum(dim=2) / visible_weight.sum(dim=2).clamp_min(1e-12)
    invisible_average = (
        invisible_nll * invisible_weight
    ).sum(dim=2) / invisible_weight.sum(dim=2).clamp_min(1e-12)

    visible_factor = float(visible_factor)
    invisible_factor = float(invisible_factor)
    if visible_factor < 0.0 or invisible_factor < 0.0:
        raise ValueError("hurdle channel factors must be non-negative")
    factor_sum = visible_factor + invisible_factor
    if factor_sum <= 0.0:
        raise ValueError("at least one hurdle channel factor must be positive")

    score = score.lower()
    if return_self_entropy_baseline and score != "balanced_log":
        raise ValueError(
            "self-entropy baseline is defined only for balanced_log"
        )
    if score == "joint_log":
        if not math.isclose(
            visible_factor, invisible_factor, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "joint_log requires equal visible and invisible factors"
            )
        per_ap = visible * visible_nll + (1.0 - visible) * invisible_nll
        return (
            per_ap * reliability_weight
        ).sum(dim=2) / reliability_weight.sum(dim=2).clamp_min(1e-12)

    if score == "density_power":
        if not math.isclose(
            visible_factor, invisible_factor, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "density_power requires equal visible and invisible factors"
            )
        beta = float(power_beta)
        if not math.isfinite(beta) or beta <= 0.0 or beta > 1.0:
            raise ValueError("power_beta must lie in (0, 1]")

        if distribution == "student_t":
            exponent = (
                (degrees_freedom + 1.0) * (1.0 + beta) - 1.0
            ) / 2.0
            log_beta_function = (
                math.lgamma(0.5)
                + math.lgamma(exponent)
                - math.lgamma(exponent + 0.5)
            )
            log_conditional_power_integral = (
                (1.0 + beta) * log_t_normalizer
                - beta * torch.log(calibrated_scale)
                + 0.5 * math.log(degrees_freedom)
                + log_beta_function
            )
        else:
            log_conditional_power_integral = (
                -0.5 * beta * math.log(2.0 * math.pi)
                - beta * torch.log(calibrated_scale)
                - 0.5 * math.log1p(beta)
            )
        conditional_power_integral = torch.exp(
            log_conditional_power_integral
        )
        hurdle_power_integral = (
            (1.0 - p_detect).pow(1.0 + beta)
            + p_detect.pow(1.0 + beta) * conditional_power_integral
        )
        observed_log_density = (
            visible * (torch.log(p_detect) + log_conditional_density)
            + (1.0 - visible) * torch.log1p(-p_detect)
        )
        per_ap = (
            hurdle_power_integral / (1.0 + beta)
            - torch.exp(beta * observed_log_density) / beta
        )
        return (
            per_ap * reliability_weight
        ).sum(dim=2) / reliability_weight.sum(dim=2).clamp_min(1e-12)

    if score != "balanced_log":
        raise ValueError(
            "score must be balanced_log, joint_log, or density_power"
        )
    balanced_energy = (
        visible_factor * visible_average
        + invisible_factor * invisible_average
    ) / factor_sum
    if not return_self_entropy_baseline:
        return balanced_energy

    if distribution == "student_t":
        # Differential entropy of a unit-scale Student-t distribution.  The
        # scale-dependent entropy is h(T_nu) + log(scale).
        nu = torch.as_tensor(
            degrees_freedom, device=device, dtype=dtype
        )
        unit_entropy = (
            0.5 * torch.log(nu)
            + torch.lgamma(0.5 * nu)
            + math.lgamma(0.5)
            - torch.lgamma(0.5 * (nu + 1.0))
            + 0.5 * (nu + 1.0)
            * (
                torch.digamma(0.5 * (nu + 1.0))
                - torch.digamma(0.5 * nu)
            )
        )
    else:
        unit_entropy = torch.as_tensor(
            0.5 * math.log(2.0 * math.pi * math.e),
            device=device,
            dtype=dtype,
        )
    conditional_entropy = unit_entropy + torch.log(calibrated_scale)
    visible_self_nll = -torch.log(p_detect) + conditional_entropy
    visible_self_average = (
        visible_self_nll * visible_weight
    ).sum(dim=2) / visible_weight.sum(dim=2).clamp_min(1e-12)
    invisible_self_average = (
        invisible_nll * invisible_weight
    ).sum(dim=2) / invisible_weight.sum(dim=2).clamp_min(1e-12)
    self_entropy_baseline = (
        visible_factor * visible_self_average
        + invisible_factor * invisible_self_average
    ) / factor_sum
    return balanced_energy, self_entropy_baseline


def project_alpha_proximal_hurdle(
    initial_weights: torch.Tensor,
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    query_values: torch.Tensor,
    query_mask: torch.Tensor,
    reliability: torch.Tensor,
    prior_weight: float | torch.Tensor,
    alpha: float = 0.25,
    steps: int = 20,
    learning_rate: float = 0.20,
    style_offset_max: float = 0.20,
    style_scale_max_log: float = 0.25,
    style_offset_weight: float = 0.10,
    style_scale_weight: float = 0.10,
    visible_factor: float = 1.0,
    invisible_factor: float = 1.0,
    hurdle_distribution: str = "student_t",
    hurdle_df: float = 4.0,
    hurdle_score: str = "balanced_log",
    hurdle_power_beta: float = 0.25,
    evidence_temperature_mode: str = "none",
    manifold_energy: torch.Tensor | None = None,
    manifold_weight: float = 1.0,
    return_calibration: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Alpha-proximal hurdle projection with only bounded scan calibration.

    Candidate logits are never optimized.  For each calibration iterate, the
    posterior is the exact alpha-proximal minimizer.  It is detached in the
    style step according to the envelope theorem, so gradients contain no
    approximation through the dual bisection.  An optional ``[B,K]`` manifold
    tensor enters directly as a linear anchor energy.
    """
    if initial_weights.ndim != 2:
        raise ValueError("initial_weights must have shape [B, K]")
    if not initial_weights.is_floating_point():
        raise TypeError("initial_weights must be floating point")
    if query_values.device != initial_weights.device:
        raise ValueError("queries and initial_weights must be on the same device")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if style_offset_max < 0.0 or style_scale_max_log < 0.0:
        raise ValueError("style bounds must be non-negative")
    if style_offset_weight < 0.0 or style_scale_weight < 0.0:
        raise ValueError("style regularization weights must be non-negative")
    if manifold_weight < 0.0:
        raise ValueError("manifold_weight must be non-negative")
    evidence_temperature_mode = evidence_temperature_mode.lower()
    if evidence_temperature_mode not in {
        "none", "visibility_ratio", "self_entropy_discrepancy"
    }:
        raise ValueError(
            "evidence_temperature_mode must be none, visibility_ratio, or "
            "self_entropy_discrepancy"
        )
    if (
        evidence_temperature_mode == "self_entropy_discrepancy"
        and hurdle_score.lower() != "balanced_log"
    ):
        raise ValueError(
            "self_entropy_discrepancy requires hurdle_score=balanced_log"
        )

    batch_size = initial_weights.shape[0]
    device = initial_weights.device
    dtype = initial_weights.dtype
    raw_offset = torch.zeros(
        batch_size,
        device=device,
        dtype=dtype,
        requires_grad=style_offset_max > 0.0,
    )
    raw_log_scale = torch.zeros(
        batch_size,
        device=device,
        dtype=dtype,
        requires_grad=style_scale_max_log > 0.0,
    )
    parameters = []
    if style_offset_max > 0.0:
        parameters.append(raw_offset)
    if style_scale_max_log > 0.0:
        parameters.append(raw_log_scale)

    if manifold_energy is not None:
        manifold_energy = manifold_energy.to(device=device, dtype=dtype)
        if manifold_energy.shape != initial_weights.shape:
            raise ValueError("manifold_energy must have shape [B, K]")
        if not bool(torch.isfinite(manifold_energy).all()):
            raise ValueError("manifold_energy must be finite")

    if evidence_temperature_mode in {"none", "self_entropy_discrepancy"}:
        evidence_temperature = torch.ones(
            batch_size, device=device, dtype=dtype
        )
    else:
        if detection_prob.ndim == 2:
            detection_for_temperature = detection_prob.unsqueeze(0)
        elif detection_prob.ndim == 3:
            detection_for_temperature = detection_prob
        else:
            raise ValueError(
                "detection_prob must have shape [K,A] or [B,K,A]"
            )
        detection_for_temperature = detection_for_temperature.to(
            device=device, dtype=dtype
        )
        expected_detection = torch.sum(
            initial_weights.unsqueeze(2) * detection_for_temperature,
            dim=1,
        )
        if reliability.ndim == 1:
            reliability_for_temperature = reliability.reshape(1, -1)
        elif reliability.ndim == 2:
            reliability_for_temperature = reliability
        else:
            raise ValueError(
                "reliability must have shape [A], [1,A], or [B,A]"
            )
        reliability_for_temperature = reliability_for_temperature.to(
            device=device, dtype=dtype
        )
        observed_mass = (
            query_mask.to(device=device, dtype=dtype)
            * reliability_for_temperature
        ).sum(dim=1)
        expected_mass = (
            expected_detection * reliability_for_temperature
        ).sum(dim=1)
        evidence_temperature = (
            observed_mass / expected_mass.clamp_min(1e-12)
        ).clamp(min=0.0, max=1.0)

    def _calibration() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if style_offset_max > 0.0:
            offset = float(style_offset_max) * torch.tanh(raw_offset)
        else:
            offset = torch.zeros_like(raw_offset)
        if style_scale_max_log > 0.0:
            log_scale = float(style_scale_max_log) * torch.tanh(raw_log_scale)
        else:
            log_scale = torch.zeros_like(raw_log_scale)
        return offset, log_scale, torch.exp(log_scale)

    def _energy(offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        anchor_energy = robust_hurdle_anchor_energy(
            detection_prob=detection_prob,
            rss_center=rss_center,
            rss_scale=rss_scale,
            query_values=query_values,
            query_mask=query_mask,
            reliability=reliability,
            style_offset=offset,
            style_scale=scale,
            visible_factor=visible_factor,
            invisible_factor=invisible_factor,
            distribution=hurdle_distribution,
            degrees_freedom=hurdle_df,
            score=hurdle_score,
            power_beta=hurdle_power_beta,
        )
        anchor_energy = anchor_energy * evidence_temperature.unsqueeze(1)
        if anchor_energy.shape != initial_weights.shape:
            raise ValueError("hurdle energy and initial_weights must share [B,K]")
        if manifold_energy is not None:
            anchor_energy = (
                anchor_energy + float(manifold_weight) * manifold_energy
            )
        return anchor_energy

    if parameters and steps > 0:
        optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
        for _ in range(int(steps)):
            offset, log_scale, scale = _calibration()
            anchor_energy = _energy(offset, scale)
            exact_weights = alpha_proximal_weights(
                initial_weights,
                anchor_energy.detach(),
                prior_weight,
                alpha,
            )
            # Envelope theorem: q*(style) is held fixed while differentiating
            # the value function with respect to the explicit style variables.
            loss = (
                (exact_weights * anchor_energy).sum(dim=1)
                + float(style_offset_weight) * offset.square()
                + float(style_scale_weight) * log_scale.square()
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    offset, log_scale, scale = _calibration()
    final_energy = _energy(offset, scale)
    if evidence_temperature_mode == "self_entropy_discrepancy":
        measurement_energy, self_entropy_baseline = (
            robust_hurdle_anchor_energy(
                detection_prob=detection_prob,
                rss_center=rss_center,
                rss_scale=rss_scale,
                query_values=query_values,
                query_mask=query_mask,
                reliability=reliability,
                style_offset=offset,
                style_scale=scale,
                visible_factor=visible_factor,
                invisible_factor=invisible_factor,
                distribution=hurdle_distribution,
                degrees_freedom=hurdle_df,
                score=hurdle_score,
                power_beta=hurdle_power_beta,
                return_self_entropy_baseline=True,
            )
        )
        supported_discrepancy = (
            measurement_energy - self_entropy_baseline
        ).masked_fill(initial_weights <= 0.0, torch.inf)
        best_discrepancy = supported_discrepancy.amin(dim=1)
        # One nat of excess risk corresponds to an e-fold reduction.  There
        # is no fitted threshold: zero is fixed by the model's own entropy.
        evidence_temperature = torch.exp(
            -torch.relu(best_discrepancy)
        ).to(dtype=dtype)
        final_energy = measurement_energy * evidence_temperature.unsqueeze(1)
        if manifold_energy is not None:
            final_energy = (
                final_energy + float(manifold_weight) * manifold_energy
            )
    projected = alpha_proximal_weights(
        initial_weights,
        final_energy.detach(),
        prior_weight,
        alpha,
    ).detach()
    if not return_calibration:
        return projected
    calibration = {
        "style_offset": offset.detach(),
        "style_log_scale": log_scale.detach(),
        "style_scale": scale.detach(),
        # Kept for diagnostics-schema compatibility.  The alpha-proximal
        # model deliberately has no trainable visibility-logit bias.
        "visibility_bias": torch.zeros_like(offset.detach()),
        "evidence_temperature": evidence_temperature.detach(),
    }
    return projected, calibration


def support_density_mask(
    anchor_buildings: torch.Tensor,
    anchor_floors: torch.Tensor,
    predicted_building: torch.Tensor,
    predicted_floor: torch.Tensor,
    scope: str,
) -> torch.Tensor | None:
    """Build a support mask for density regularization.

    Args:
        anchor_buildings: [M] support-anchor building ids.
        anchor_floors: [M] support-anchor floor ids.
        predicted_building: [B] query building prediction.
        predicted_floor: [B] query floor prediction.
        scope: one of {"all", "building", "floor"}.

    Returns:
        None for global support, otherwise [B, M] boolean mask. Floor scope
        falls back to same-building support if a predicted floor has no
        support anchors.
    """
    scope = scope.lower()
    if scope == "all":
        return None
    same_building = (
        anchor_buildings.to(predicted_building.device).unsqueeze(0)
        == predicted_building.unsqueeze(1)
    )
    if scope == "building":
        return same_building
    if scope == "floor":
        same_floor = (
            anchor_floors.to(predicted_floor.device).unsqueeze(0)
            == predicted_floor.unsqueeze(1)
        )
        same_bf = same_building & same_floor
        has_floor_support = same_bf.any(dim=1, keepdim=True)
        return torch.where(has_floor_support, same_bf, same_building)
    raise ValueError(f"unknown support density scope: {scope}")


def support_density_energy(
    mean_location: torch.Tensor,
    support_locations: torch.Tensor,
    support_mask: torch.Tensor | None,
    support_density_sigma: float,
    support_density_margin: float,
    support_density_mode: str,
) -> torch.Tensor:
    """Return a per-query support-density energy.

    `soft_distance` is the original A092 expected distance to a soft nearest
    support point. `kde_softmin` is closer to a KDE negative-log-density
    geometry: multiple nearby support points reduce the soft minimum, so larger
    sigma can cover space between sparse UJI reference points instead of simply
    pulling toward one nearest anchor.
    """
    support = support_locations.to(
        device=mean_location.device,
        dtype=mean_location.dtype,
    )
    sigma2 = max(float(support_density_sigma) ** 2, 1e-6)
    if support.dim() == 2:
        dist2 = torch.cdist(mean_location, support).square()
    elif support.dim() == 3:
        dist2 = torch.cdist(
            mean_location.unsqueeze(1), support).squeeze(1).square()
    else:
        raise ValueError(
            "support_locations must have shape [M,2] or [B,M,2]")

    density_dist2 = dist2
    if support_mask is not None:
        mask = support_mask.to(
            device=mean_location.device,
            dtype=torch.bool,
        )
        if mask.shape != dist2.shape:
            raise ValueError(
                "support_mask must have shape [B, M] matching "
                "support distances")
        fallback = torch.ones_like(mask)
        has_support = mask.any(dim=1, keepdim=True)
        mask = torch.where(has_support, mask, fallback)
        density_dist2 = dist2.masked_fill(~mask, 0.0)
        dist2 = dist2.masked_fill(~mask, float("inf"))

    mode = support_density_mode.lower()
    if mode == "soft_distance":
        local_support = torch.softmax(-0.5 * dist2 / sigma2, dim=1)
        support_distance2 = torch.sum(
            local_support * density_dist2, dim=1)
    elif mode == "kde_softmin":
        logits = -0.5 * dist2 / sigma2
        support_distance2 = (
            -2.0 * sigma2 * torch.logsumexp(logits, dim=1)
        ).clamp(min=0.0)
    else:
        raise ValueError(
            "support_density_mode must be soft_distance or kde_softmin")

    if support_density_margin > 0.0:
        support_distance = support_distance2.clamp(min=1e-8).sqrt()
        return (
            torch.relu(
                support_distance - float(support_density_margin)
            ).square()
            / 100.0
        )
    return support_distance2 / 100.0


def ambiguity_kl_scale(
    initial_weights: torch.Tensor,
    candidate_locations: torch.Tensor,
    radial_scale: torch.Tensor | None,
    ambiguity_weight: float,
    spread_pivot: float,
    radial_pivot: float,
    max_scale: float,
) -> torch.Tensor:
    """Per-scan KL multiplier for geometrically ambiguous Top-K posteriors.

    The scale is label-free and only strengthens the KL trust region. It uses
    the frozen Top-K posterior entropy, the posterior-weighted spatial spread,
    and optionally the model radial-scale estimate. High ambiguity means the
    RSS measurement objective is less identifiable, so posterior movement
    should be more conservative.
    """
    if ambiguity_weight <= 0.0:
        return torch.ones(
            (len(initial_weights),),
            dtype=initial_weights.dtype,
            device=initial_weights.device,
        )

    initial = initial_weights.clamp(min=1e-8)
    initial = initial / initial.sum(dim=1, keepdim=True).clamp(min=1e-8)
    entropy = -torch.sum(initial * torch.log(initial), dim=1)
    entropy_norm = entropy / math.log(max(initial.shape[1], 2))

    mean_location = torch.sum(
        initial.unsqueeze(2) * candidate_locations, dim=1)
    spread = torch.sqrt(torch.sum(
        initial
        * (candidate_locations - mean_location.unsqueeze(1))
        .square()
        .sum(dim=2),
        dim=1,
    ).clamp(min=0.0))
    spread_score = spread / max(float(spread_pivot), 1e-6)

    scores = [entropy_norm.clamp(min=0.0, max=2.0),
              spread_score.clamp(min=0.0, max=2.0)]
    if radial_scale is not None:
        radial = radial_scale.to(
            device=initial_weights.device,
            dtype=initial_weights.dtype,
        )
        scores.append(
            (radial / max(float(radial_pivot), 1e-6)).clamp(
                min=0.0, max=2.0))
    ambiguity = torch.stack(scores, dim=0).mean(dim=0)
    scale = 1.0 + float(ambiguity_weight) * ambiguity
    return scale.clamp(min=1.0, max=max(float(max_scale), 1.0))


def discriminative_ap_multiplier(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    initial_weights: torch.Tensor,
    query_mask: torch.Tensor,
    strength: float,
    rss_scale: float,
    detection_scale: float,
    min_multiplier: float,
    max_multiplier: float,
) -> torch.Tensor:
    """Sample-specific AP weights from Top-K candidate discriminativeness.

    The multiplier is label-free. APs whose predicted RSS or detection
    probability differs across the current Top-K candidate set receive larger
    weight because they can distinguish candidate locations. APs that are nearly
    constant over Top-K are downweighted after per-scan normalization because
    they can improve RSS reconstruction without resolving localization
    ambiguity.
    """
    if strength <= 0.0:
        return torch.ones_like(query_mask)

    initial = initial_weights.clamp(min=1e-8)
    initial = initial / initial.sum(dim=1, keepdim=True).clamp(min=1e-8)
    p_detect = detection_prob.clamp(min=1e-4, max=1.0 - 1e-4)

    detect_mean = torch.sum(initial.unsqueeze(2) * p_detect, dim=1)
    detect_var = torch.sum(
        initial.unsqueeze(2) * (p_detect - detect_mean.unsqueeze(1)).square(),
        dim=1,
    )
    detect_score = detect_var.clamp(min=0.0).sqrt() / max(
        float(detection_scale), 1e-6)

    visible_mass = torch.sum(
        initial.unsqueeze(2) * p_detect, dim=1).clamp(min=1e-4)
    rss_mean = torch.sum(
        initial.unsqueeze(2) * p_detect * rss_center, dim=1) / visible_mass
    rss_var = torch.sum(
        initial.unsqueeze(2)
        * p_detect
        * (rss_center - rss_mean.unsqueeze(1)).square(),
        dim=1,
    ) / visible_mass
    rss_score = rss_var.clamp(min=0.0).sqrt() / max(float(rss_scale), 1e-6)

    raw = 1.0 + float(strength) * torch.maximum(
        detect_score, rss_score).clamp(min=0.0, max=4.0)
    raw = raw.clamp(
        min=max(float(min_multiplier), 1e-4),
        max=max(float(max_multiplier), float(min_multiplier), 1e-4),
    )

    # Normalize per scan so the new term changes which APs dominate, not the
    # global RSS/visibility loss scale.
    active = ((query_mask > 0.0) | (detect_mean > 0.05)).to(raw.dtype)
    normalizer = (raw * active).sum(dim=1, keepdim=True) / active.sum(
        dim=1, keepdim=True).clamp(min=1.0)
    return raw / normalizer.clamp(min=1e-4)


def initial_support_manifold_energy(
    initial_weights: torch.Tensor,
    candidate_locations: torch.Tensor,
    sigma: float,
    margin: float,
) -> torch.Tensor:
    """Return the fixed per-anchor energy of the initial Top-K manifold.

    The energy is determined only by the frozen prior and candidate geometry.
    Consequently, any posterior manifold penalty is linear in its weights,
    which preserves the exact alpha-proximal update.
    """
    initial = initial_weights.clamp(min=1e-8)
    initial = initial / initial.sum(dim=1, keepdim=True).clamp(min=1e-8)
    dist2 = torch.cdist(candidate_locations, candidate_locations).square()
    sigma2 = max(float(sigma), 1e-6) ** 2
    log_support = torch.log(initial).unsqueeze(1) - 0.5 * dist2 / sigma2
    soft_distance2 = (-2.0 * sigma2 * torch.logsumexp(
        log_support, dim=2)).clamp(min=0.0)
    soft_distance = soft_distance2.sqrt()
    excess = torch.relu(soft_distance - max(float(margin), 0.0))
    return excess.square() / 100.0


def initial_support_manifold_loss(
    weights: torch.Tensor,
    initial_weights: torch.Tensor,
    candidate_locations: torch.Tensor,
    sigma: float,
    margin: float,
) -> torch.Tensor:
    """Penalize posterior mass far from the initial Top-K support manifold."""
    energy = initial_support_manifold_energy(
        initial_weights,
        candidate_locations,
        sigma,
        margin,
    )
    return torch.sum(weights * energy, dim=1)


def project_candidate_weights(
    query: torch.Tensor,
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    initial_weights: torch.Tensor,
    candidate_locations: torch.Tensor,
    visibility_weight: float,
    prior_weight: float,
    compactness_weight: float,
    style_offset_max: float,
    style_offset_weight: float,
    style_scale_max_log: float,
    style_scale_weight: float,
    visibility_bias_max: float,
    visibility_bias_weight: float,
    ap_reliability: torch.Tensor,
    steps: int,
    learning_rate: float,
    support_locations: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    support_density_weight: float = 0.0,
    support_density_sigma: float = 4.0,
    support_density_margin: float = 0.0,
    support_density_mode: str = "soft_distance",
    rss_centering: str = "none",
    rss_loss_weight: float = 1.0,
    visibility_positive_factor: float = 1.0,
    visibility_negative_factor: float = 1.0,
    mixture_censored_weight: float = 0.0,
    mixture_visible_factor: float = 1.0,
    mixture_invisible_factor: float = 1.0,
    adaptive_kl_alpha: float = 0.0,
    adaptive_kl_visible_pivot: float = 16.0,
    adaptive_kl_min_scale: float = 0.5,
    adaptive_kl_max_scale: float = 2.0,
    ambiguity_kl_weight: float = 0.0,
    ambiguity_kl_spread_pivot: float = 8.0,
    ambiguity_kl_radial_pivot: float = 8.0,
    ambiguity_kl_max_scale: float = 2.0,
    discriminative_ap_weight: float = 0.0,
    discriminative_ap_rss_scale: float = 0.15,
    discriminative_ap_detection_scale: float = 0.20,
    discriminative_ap_min: float = 0.25,
    discriminative_ap_max: float = 3.0,
    manifold_weight: float = 0.0,
    manifold_sigma: float = 6.0,
    manifold_margin: float = 2.0,
    censored_invisible_weight: float = 0.0,
    censored_detection_threshold: float = 0.10,
    censored_rss_threshold: float = -0.95,
    censored_mode: str = "hinge",
    censored_rss_softness: float = 0.10,
    censored_contradiction_margin: float = 0.0,
    candidate_floors: torch.Tensor | None = None,
    predicted_floor: torch.Tensor | None = None,
    floor_consistency_weight: float = 0.0,
    gate_mask: torch.Tensor | None = None,
    ambiguity_radial_scale: torch.Tensor | None = None,
    ap_censored_threshold: torch.Tensor | None = None,
    hierarchical_style: bool = False,
    style_residual_fraction: float = 1.0,
    style_group_ids: torch.Tensor | None = None,
    style_group_count: int = 1,
    return_calibration: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DAPP 核心: 无标签 test-time 投影微调候选权重.

    优化变量: logits (softmax 成权重) + 可选 style_offset.
    能量函数:
        E = RSS一致性 + visibility_weight * 可见性一致性
            + prior_weight * KL(w || w0)
            + compactness_weight * 空间紧凑性
            + style_offset_weight * style^2
            + support_density_weight * 支撑密度距离

    No-harm: support_density_weight=0 strictly recovers the final DAPP path.

    Args:
        query: [B, n_aps] query RSS (-1 表示不可见).
        detection_prob: [B, K, n_aps] 候选的 AP 检测概率.
        rss_center: [B, K, n_aps] 候选的条件 RSS 中心.
        rss_scale: [B, K, n_aps] 候选的条件 RSS 尺度.
        initial_weights: [B, K] 神经网络给的初始权重.
        candidate_locations: [B, K, 2] 候选坐标 (米).
        visibility_weight: 可见性损失权重.
        prior_weight: KL 先验权重.
        compactness_weight: 空间紧凑性权重.
        style_offset_max: style offset 的 tanh 截断上限.
        style_offset_weight: style offset 的 L2 权重.
        ap_reliability: [n_aps] AP 可靠性.
        steps: 梯度下降步数.
        learning_rate: Adam 学习率.
        support_locations: [M, 2] 或 [B, M, 2] 训练支撑坐标.
        support_mask: [B, M] 可选支撑 mask.
        support_density_weight: 最终预测点到训练支撑密度的惩罚权重.
        support_density_sigma: 支撑密度软近邻核宽度, 单位米.
        support_density_margin: OOD 安全半径, 单位米. 大于 0 时仅惩罚
            超过该半径的软近邻距离.
        support_density_mode: soft_distance 或 kde_softmin.
        rss_centering: none keeps the final RSS likelihood. visible_mean
            subtracts each row's visible-AP weighted mean from both query and
            predicted RSS before the SmoothL1 residual; this is diagnostic-only
            for testing scan-wise offset invariance.
        rss_loss_weight: multiplier for the moment RSS reconstruction loss.
        visibility_positive_factor/visibility_negative_factor: split the
            visibility BCE into detected-AP and non-detected-AP channels.
        mixture_censored_weight: optional full top-K censored likelihood weight.
        adaptive_kl_alpha: if positive, scale the KL trust-region weight by
            visible-AP count. Sparse scans receive stronger KL; denser scans
            receive weaker KL.
        ambiguity_kl_weight: if positive, further strengthens the KL
            trust-region on ambiguous Top-K posteriors using entropy, spatial
            spread and optional radial scale.
        censored_invisible_weight: optional hinge loss for invisible APs,
            treating missing APs as censored observations rather than exact
            RSS measurements.
        floor_consistency_weight: optional penalty on candidate probability
            mass outside the predicted floor.
        gate_mask: optional [B] boolean mask. False rows keep the initial
            posterior, implementing uncertainty-gated DAPP.
        return_calibration: when True, also return the optimized per-scan
            calibration variables for diagnostics.

    Returns:
        [B, K] 微调后的候选权重 (softmax 归一化).
    """
    query_visible = query > -0.999
    query_mask = query_visible.float()
    query_values = torch.where(
        query_visible, query, torch.zeros_like(query))
    rss_centering = rss_centering.lower()
    if rss_centering not in {"none", "visible_mean"}:
        raise ValueError("rss_centering must be none or visible_mean")
    initial = initial_weights.clamp(min=1e-8)
    initial_detection = torch.sum(
        initial.unsqueeze(2) * detection_prob, dim=1)
    initial_rss_mass = torch.sum(
        initial.unsqueeze(2) * detection_prob * rss_center, dim=1)
    initial_rss = initial_rss_mass / initial_detection.clamp(min=0.05)
    prior_weight_per_sample = torch.full(
        (len(query),),
        float(prior_weight),
        dtype=query.dtype,
        device=query.device,
    )
    if adaptive_kl_alpha > 0.0 and prior_weight > 0.0:
        visible_count = query_mask.sum(dim=1).clamp(min=1.0)
        kl_scale = (
            float(adaptive_kl_visible_pivot) / visible_count
        ).pow(float(adaptive_kl_alpha))
        kl_scale = kl_scale.clamp(
            min=float(adaptive_kl_min_scale),
            max=float(adaptive_kl_max_scale),
        )
        prior_weight_per_sample = prior_weight_per_sample * kl_scale
    prior_weight_per_sample = prior_weight_per_sample * ambiguity_kl_scale(
        initial,
        candidate_locations,
        ambiguity_radial_scale,
        ambiguity_kl_weight,
        ambiguity_kl_spread_pivot,
        ambiguity_kl_radial_pivot,
        ambiguity_kl_max_scale,
    )
    ap_multiplier = discriminative_ap_multiplier(
        detection_prob,
        rss_center,
        initial,
        query_mask,
        discriminative_ap_weight,
        discriminative_ap_rss_scale,
        discriminative_ap_detection_scale,
        discriminative_ap_min,
        discriminative_ap_max,
    )
    if gate_mask is not None:
        gate_mask = gate_mask.to(device=query.device, dtype=torch.bool)
        if gate_mask.shape != (len(query),):
            raise ValueError("gate_mask must have shape [B]")
    if ap_censored_threshold is not None:
        ap_censored_threshold = ap_censored_threshold.to(
            device=query.device,
            dtype=query.dtype,
        )
        if ap_censored_threshold.shape != (query.shape[1],):
            raise ValueError("ap_censored_threshold must have shape [n_aps]")
    if hierarchical_style:
        if style_group_ids is None:
            style_group_ids = torch.zeros(
                len(query), device=query.device, dtype=torch.long)
            style_group_count = 1
        else:
            style_group_ids = style_group_ids.to(
                device=query.device, dtype=torch.long)
            if style_group_ids.shape != (len(query),):
                raise ValueError("style_group_ids must have shape [B]")
            style_group_count = max(
                int(style_group_count),
                int(style_group_ids.max().item()) + 1,
            )
        style_group_count = max(int(style_group_count), 1)
    else:
        style_group_ids = torch.zeros(
            len(query), device=query.device, dtype=torch.long)
        style_group_count = 1
    effective_rss_threshold = (
        ap_censored_threshold
        if ap_censored_threshold is not None
        else censored_rss_threshold
    )
    if floor_consistency_weight > 0.0:
        if candidate_floors is None or predicted_floor is None:
            raise ValueError(
                "candidate_floors and predicted_floor are required when "
                "floor_consistency_weight > 0")
        floor_mismatch = (
            candidate_floors.to(query.device)
            != predicted_floor.to(query.device).unsqueeze(1)
        ).to(dtype=query.dtype)
    else:
        floor_mismatch = None
    logits = initial.log().detach().requires_grad_(True)
    raw_style_offset = torch.zeros(
        len(query), 1, device=query.device, requires_grad=True)
    raw_style_scale = torch.zeros(
        len(query), 1, device=query.device, requires_grad=True)
    raw_visibility_bias = torch.zeros(
        len(query), 1, device=query.device, requires_grad=True)
    raw_shared_style_offset = torch.zeros(
        style_group_count, 1, device=query.device, requires_grad=True)
    raw_shared_style_scale = torch.zeros(
        style_group_count, 1, device=query.device, requires_grad=True)
    raw_shared_visibility_bias = torch.zeros(
        style_group_count, 1, device=query.device, requires_grad=True)
    parameters = [logits]
    if style_offset_max > 0.0:
        parameters.append(raw_style_offset)
        if hierarchical_style:
            parameters.append(raw_shared_style_offset)
    if style_scale_max_log > 0.0:
        parameters.append(raw_style_scale)
        if hierarchical_style:
            parameters.append(raw_shared_style_scale)
    if visibility_bias_max > 0.0:
        parameters.append(raw_visibility_bias)
        if hierarchical_style:
            parameters.append(raw_shared_visibility_bias)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    residual_fraction = max(0.0, float(style_residual_fraction))

    for _ in range(steps):
        weights = torch.softmax(logits, dim=1)
        # RSS 一致性: 加权预测的 RSS 应接近 query RSS.
        if visibility_bias_max > 0.0:
            if hierarchical_style:
                shared_visibility_bias = (
                    torch.tanh(raw_shared_visibility_bias)
                    * float(visibility_bias_max)
                )[style_group_ids]
                residual_visibility_bias = (
                    torch.tanh(raw_visibility_bias)
                    * float(visibility_bias_max)
                    * residual_fraction
                )
                visibility_bias = torch.clamp(
                    shared_visibility_bias + residual_visibility_bias,
                    min=-float(visibility_bias_max),
                    max=float(visibility_bias_max),
                )
            else:
                shared_visibility_bias = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                visibility_bias = (
                    torch.tanh(raw_visibility_bias) * visibility_bias_max)
            calibrated_detection = torch.sigmoid(
                torch.logit(detection_prob.clamp(min=1e-4, max=1.0 - 1e-4))
                + visibility_bias.unsqueeze(1)
            )
        else:
            shared_visibility_bias = torch.zeros(
                1, 1, device=query.device, dtype=query.dtype)
            visibility_bias = torch.zeros_like(raw_visibility_bias)
            calibrated_detection = detection_prob
        if style_scale_max_log > 0.0:
            if hierarchical_style:
                shared_style_log_scale = (
                    torch.tanh(raw_shared_style_scale)
                    * float(style_scale_max_log)
                )[style_group_ids]
                residual_style_log_scale = (
                    torch.tanh(raw_style_scale)
                    * float(style_scale_max_log)
                    * residual_fraction
                )
                style_log_scale = torch.clamp(
                    shared_style_log_scale + residual_style_log_scale,
                    min=-float(style_scale_max_log),
                    max=float(style_scale_max_log),
                )
            else:
                shared_style_log_scale = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                style_log_scale = (
                    torch.tanh(raw_style_scale) * style_scale_max_log)
            style_scale = torch.exp(style_log_scale)
        else:
            shared_style_log_scale = torch.zeros(
                len(query), 1, device=query.device, dtype=query.dtype)
            style_log_scale = torch.zeros_like(raw_style_scale)
            style_scale = torch.ones_like(raw_style_scale)
        calibrated_rss_center = rss_center * style_scale.unsqueeze(1)
        predicted_detection = torch.sum(
            weights.unsqueeze(2) * calibrated_detection, dim=1)
        rss_mass = torch.sum(
            weights.unsqueeze(2)
            * calibrated_detection
            * calibrated_rss_center,
            dim=1,
        )
        predicted_rss = rss_mass / predicted_detection.clamp(min=0.05)
        if style_offset_max > 0.0:
            if hierarchical_style:
                shared_style_offset = (
                    torch.tanh(raw_shared_style_offset)
                    * float(style_offset_max)
                )[style_group_ids]
                residual_style_offset = (
                    torch.tanh(raw_style_offset)
                    * float(style_offset_max)
                    * residual_fraction
                )
                style_offset = torch.clamp(
                    shared_style_offset + residual_style_offset,
                    min=-float(style_offset_max),
                    max=float(style_offset_max),
                )
            else:
                shared_style_offset = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                style_offset = torch.tanh(raw_style_offset) * style_offset_max
        else:
            shared_style_offset = torch.zeros(
                len(query), 1, device=query.device, dtype=query.dtype)
            style_offset = torch.zeros_like(raw_style_offset)
        predicted_rss = predicted_rss + style_offset
        reliability = ap_reliability.unsqueeze(0) * ap_multiplier
        rss_weight = query_mask * reliability
        if rss_centering == "visible_mean":
            rss_weight_sum = rss_weight.sum(dim=1, keepdim=True).clamp(
                min=1.0)
            query_mean = (query_values * rss_weight).sum(
                dim=1, keepdim=True) / rss_weight_sum
            predicted_mean = (predicted_rss * rss_weight).sum(
                dim=1, keepdim=True) / rss_weight_sum
            residual_prediction = predicted_rss - predicted_mean
            residual_target = query_values - query_mean
        else:
            residual_prediction = predicted_rss
            residual_target = query_values
        rss_residual = F.smooth_l1_loss(
            residual_prediction,
            residual_target,
            reduction="none",
            beta=0.10,
        )
        rss_loss = (
            rss_residual * rss_weight
        ).sum(dim=1) / rss_weight.sum(dim=1).clamp(min=1.0)
        censored_loss = torch.zeros_like(rss_loss)
        if censored_invisible_weight > 0.0:
            censored_loss = censored_non_detection_loss(
                predicted_detection,
                predicted_rss,
                query_mask,
                reliability,
                censored_mode,
                censored_detection_threshold,
                effective_rss_threshold,
                censored_rss_softness,
                censored_contradiction_margin,
                initial_detection=initial_detection,
                initial_rss=initial_rss,
            )
        # 可见性一致性.
        visibility_loss = balanced_visibility_loss(
            predicted_detection,
            query_mask,
            reliability,
            visibility_positive_factor,
            visibility_negative_factor,
        )
        mixture_loss = torch.zeros_like(rss_loss)
        if mixture_censored_weight > 0.0:
            mixture_loss = full_mixture_censored_loss(
                weights,
                calibrated_detection,
                rss_center,
                rss_scale,
                query_values,
                query_mask,
                reliability,
                effective_rss_threshold,
                style_offset=style_offset,
                style_scale=style_scale,
                visibility_bias=None,
                visible_factor=mixture_visible_factor,
                invisible_factor=mixture_invisible_factor,
            )
        # KL 先验: 不要离神经网络初始权重太远.
        prior_loss = torch.sum(
            weights * (
                torch.log(weights.clamp(min=1e-8)) - torch.log(initial)
            ),
            dim=1,
        )
        # 空间紧凑性: 候选别太分散.
        mean_location = torch.sum(
            weights.unsqueeze(2) * candidate_locations, dim=1)
        compactness = torch.sum(
            weights
            * (candidate_locations - mean_location.unsqueeze(1))
            .square()
            .sum(dim=2),
            dim=1,
        ) / 100.0
        support_density = torch.zeros_like(compactness)
        if support_density_weight > 0.0:
            if support_locations is None:
                raise ValueError(
                    "support_locations is required when "
                    "support_density_weight > 0")
            support_density = support_density_energy(
                mean_location,
                support_locations,
                support_mask,
                support_density_sigma,
            support_density_margin,
            support_density_mode,
        )
        floor_loss = torch.zeros_like(prior_loss)
        if floor_mismatch is not None:
            floor_loss = torch.sum(weights * floor_mismatch, dim=1)
        manifold_loss = torch.zeros_like(prior_loss)
        if manifold_weight > 0.0:
            manifold_loss = initial_support_manifold_loss(
                weights,
                initial,
                candidate_locations,
                manifold_sigma,
                manifold_margin,
            )
        loss = (
            float(rss_loss_weight) * rss_loss
            + censored_invisible_weight * censored_loss
            + mixture_censored_weight * mixture_loss
            + visibility_weight * visibility_loss
            + prior_weight_per_sample * prior_loss
            + manifold_weight * manifold_loss
            + compactness_weight * compactness
            + style_offset_weight * style_offset.squeeze(1).square()
            + style_scale_weight * torch.log(style_scale).squeeze(1).square()
            + visibility_bias_weight * visibility_bias.squeeze(1).square()
            + support_density_weight * support_density
            + floor_consistency_weight * floor_loss
        ).mean()
        if hierarchical_style:
            loss = loss + (
                float(style_offset_weight)
                * shared_style_offset.squeeze(1).square().mean()
                + float(style_scale_weight)
                * shared_style_log_scale.squeeze(1).square().mean()
                + float(visibility_bias_weight)
                * shared_visibility_bias.squeeze(1).square().mean()
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    projected = torch.softmax(logits.detach(), dim=1)
    if gate_mask is not None:
        projected = torch.where(gate_mask.unsqueeze(1), projected, initial)
    if return_calibration:
        if style_offset_max > 0.0:
            if hierarchical_style:
                shared_style_offset = (
                    torch.tanh(raw_shared_style_offset.detach())
                    * float(style_offset_max)
                )[style_group_ids]
                residual_style_offset = (
                    torch.tanh(raw_style_offset.detach())
                    * float(style_offset_max)
                    * residual_fraction
                )
                style_offset = torch.clamp(
                    shared_style_offset + residual_style_offset,
                    min=-float(style_offset_max),
                    max=float(style_offset_max),
                )
            else:
                shared_style_offset = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                style_offset = (
                    torch.tanh(raw_style_offset.detach())
                    * float(style_offset_max)
                )
        else:
            shared_style_offset = torch.zeros(
                len(query), 1, device=query.device, dtype=query.dtype)
            style_offset = torch.zeros_like(raw_style_offset)
        if style_scale_max_log > 0.0:
            if hierarchical_style:
                shared_style_log_scale = (
                    torch.tanh(raw_shared_style_scale.detach())
                    * float(style_scale_max_log)
                )[style_group_ids]
                residual_style_log_scale = (
                    torch.tanh(raw_style_scale.detach())
                    * float(style_scale_max_log)
                    * residual_fraction
                )
                style_log_scale = torch.clamp(
                    shared_style_log_scale + residual_style_log_scale,
                    min=-float(style_scale_max_log),
                    max=float(style_scale_max_log),
                )
            else:
                shared_style_log_scale = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                style_log_scale = (
                    torch.tanh(raw_style_scale.detach())
                    * float(style_scale_max_log)
                )
            style_scale = torch.exp(style_log_scale)
        else:
            shared_style_log_scale = torch.zeros(
                len(query), 1, device=query.device, dtype=query.dtype)
            style_scale = torch.ones_like(raw_style_scale)
            style_log_scale = torch.zeros_like(raw_style_scale)
        if visibility_bias_max > 0.0:
            if hierarchical_style:
                shared_visibility_bias = (
                    torch.tanh(raw_shared_visibility_bias.detach())
                    * float(visibility_bias_max)
                )[style_group_ids]
                residual_visibility_bias = (
                    torch.tanh(raw_visibility_bias.detach())
                    * float(visibility_bias_max)
                    * residual_fraction
                )
                visibility_bias = torch.clamp(
                    shared_visibility_bias + residual_visibility_bias,
                    min=-float(visibility_bias_max),
                    max=float(visibility_bias_max),
                )
            else:
                shared_visibility_bias = torch.zeros(
                    len(query), 1, device=query.device, dtype=query.dtype)
                visibility_bias = (
                    torch.tanh(raw_visibility_bias.detach())
                    * float(visibility_bias_max)
                )
        else:
            shared_visibility_bias = torch.zeros(
                len(query), 1, device=query.device, dtype=query.dtype)
            visibility_bias = torch.zeros_like(raw_visibility_bias)
        calibration = {
            "style_offset": style_offset.squeeze(1),
            "style_scale": style_scale.squeeze(1),
            "style_log_scale": style_log_scale.squeeze(1),
            "visibility_bias": visibility_bias.squeeze(1),
            "shared_style_offset": shared_style_offset.squeeze(1),
            "shared_style_log_scale": shared_style_log_scale.squeeze(1),
            "shared_visibility_bias": shared_visibility_bias.squeeze(1),
            "style_group_id": style_group_ids,
        }
        return projected, calibration
    return projected


def project_joint_candidate_weights(
    queries: list[torch.Tensor],
    detection_probs: list[torch.Tensor],
    rss_centers: list[torch.Tensor],
    rss_scales: list[torch.Tensor],
    initial_weights_list: list[torch.Tensor],
    candidate_locations_list: list[torch.Tensor],
    ap_reliabilities: list[torch.Tensor],
    visibility_weight: float,
    prior_weight: float,
    compactness_weight: float,
    style_offset_max: float,
    style_offset_weight: float,
    style_scale_max_log: float,
    style_scale_weight: float,
    visibility_bias_max: float,
    visibility_bias_weight: float,
    steps: int,
    learning_rate: float,
    joint_ensemble_weight: float,
    precision_weights: torch.Tensor | None = None,
    support_locations_list: list[torch.Tensor] | None = None,
    support_masks: list[torch.Tensor | None] | None = None,
    support_density_weight: float = 0.0,
    support_density_sigma: float = 4.0,
    support_density_margin: float = 0.0,
    support_density_mode: str = "soft_distance",
    rss_centering: str = "none",
    rss_loss_weight: float = 1.0,
    visibility_positive_factor: float = 1.0,
    visibility_negative_factor: float = 1.0,
    mixture_censored_weight: float = 0.0,
    mixture_visible_factor: float = 1.0,
    mixture_invisible_factor: float = 1.0,
    adaptive_kl_alpha: float = 0.0,
    adaptive_kl_visible_pivot: float = 16.0,
    adaptive_kl_min_scale: float = 0.5,
    adaptive_kl_max_scale: float = 2.0,
    ambiguity_kl_weight: float = 0.0,
    ambiguity_kl_spread_pivot: float = 8.0,
    ambiguity_kl_radial_pivot: float = 8.0,
    ambiguity_kl_max_scale: float = 2.0,
    discriminative_ap_weight: float = 0.0,
    discriminative_ap_rss_scale: float = 0.15,
    discriminative_ap_detection_scale: float = 0.20,
    discriminative_ap_min: float = 0.25,
    discriminative_ap_max: float = 3.0,
    manifold_weight: float = 0.0,
    manifold_sigma: float = 6.0,
    manifold_margin: float = 2.0,
    censored_invisible_weight: float = 0.0,
    censored_detection_threshold: float = 0.10,
    censored_rss_threshold: float = -0.95,
    censored_mode: str = "hinge",
    censored_rss_softness: float = 0.10,
    censored_contradiction_margin: float = 0.0,
    candidate_floors_list: list[torch.Tensor] | None = None,
    predicted_floors: list[torch.Tensor] | None = None,
    floor_consistency_weight: float = 0.0,
    gate_masks: list[torch.Tensor | None] | None = None,
    radial_scales_list: list[torch.Tensor] | None = None,
    return_calibration: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Joint ensemble posterior projection.

    Unlike ``ensemble_consensus_weight`` in ``ProjectedEnsemble``, this function
    does not shrink coordinates after independent DAPP. It optimizes the
    candidate-posterior logits of all ensemble members in one objective and
    adds a coordinate-disagreement energy between the projected member means.
    """
    member_count = len(initial_weights_list)
    if not (
        len(queries)
        == len(detection_probs)
        == len(rss_centers)
        == len(rss_scales)
        == len(candidate_locations_list)
        == len(ap_reliabilities)
        == member_count
    ):
        raise ValueError("joint projection lists must have the same length")
    if member_count == 0:
        return []
    if joint_ensemble_weight <= 0.0:
        raise ValueError("joint_ensemble_weight must be positive")
    rss_centering = rss_centering.lower()
    if rss_centering not in {"none", "visible_mean"}:
        raise ValueError("rss_centering must be none or visible_mean")

    batch_size = len(queries[0])
    device = queries[0].device
    dtype = queries[0].dtype
    if precision_weights is None:
        precision_weights = torch.full(
            (member_count, batch_size),
            1.0 / member_count,
            device=device,
            dtype=dtype,
        )
    else:
        precision_weights = precision_weights.to(device=device, dtype=dtype)
        if precision_weights.shape != (member_count, batch_size):
            raise ValueError(
                "precision_weights must have shape [members, batch]")
        precision_weights = precision_weights / precision_weights.sum(
            dim=0, keepdim=True).clamp(min=1e-8)

    support_locations_list = support_locations_list or [None] * member_count
    support_masks = support_masks or [None] * member_count
    gate_masks = gate_masks or [None] * member_count
    radial_scales_list = radial_scales_list or [None] * member_count
    if candidate_floors_list is None:
        candidate_floors_list = [None] * member_count
    if predicted_floors is None:
        predicted_floors = [None] * member_count

    initials = [w.clamp(min=1e-8) for w in initial_weights_list]
    initial_detections = []
    initial_rsses = []
    for initial, detection_prob, rss_center in zip(
        initials, detection_probs, rss_centers
    ):
        initial_detection = torch.sum(
            initial.unsqueeze(2) * detection_prob, dim=1)
        initial_rss_mass = torch.sum(
            initial.unsqueeze(2) * detection_prob * rss_center, dim=1)
        initial_detections.append(initial_detection)
        initial_rsses.append(
            initial_rss_mass / initial_detection.clamp(min=0.05))
    logits = [
        initial.log().detach().requires_grad_(True)
        for initial in initials
    ]
    raw_style_offsets = [
        torch.zeros(batch_size, 1, device=device, requires_grad=True)
        for _ in range(member_count)
    ]
    raw_style_scales = [
        torch.zeros(batch_size, 1, device=device, requires_grad=True)
        for _ in range(member_count)
    ]
    raw_visibility_biases = [
        torch.zeros(batch_size, 1, device=device, requires_grad=True)
        for _ in range(member_count)
    ]
    parameters = list(logits)
    if style_offset_max > 0.0:
        parameters.extend(raw_style_offsets)
    if style_scale_max_log > 0.0:
        parameters.extend(raw_style_scales)
    if visibility_bias_max > 0.0:
        parameters.extend(raw_visibility_biases)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    query_masks = []
    query_values = []
    prior_weights = []
    ap_multipliers = []
    floor_mismatches = []
    for member_index, query in enumerate(queries):
        visible = query > -0.999
        query_mask = visible.float()
        query_masks.append(query_mask)
        query_values.append(torch.where(visible, query, torch.zeros_like(query)))
        per_sample_prior = torch.full(
            (batch_size,),
            float(prior_weight),
            dtype=query.dtype,
            device=query.device,
        )
        if adaptive_kl_alpha > 0.0 and prior_weight > 0.0:
            visible_count = query_mask.sum(dim=1).clamp(min=1.0)
            kl_scale = (
                float(adaptive_kl_visible_pivot) / visible_count
            ).pow(float(adaptive_kl_alpha))
            kl_scale = kl_scale.clamp(
                min=float(adaptive_kl_min_scale),
                max=float(adaptive_kl_max_scale),
            )
            per_sample_prior = per_sample_prior * kl_scale
        per_sample_prior = per_sample_prior * ambiguity_kl_scale(
            initials[member_index],
            candidate_locations_list[member_index],
            radial_scales_list[member_index],
            ambiguity_kl_weight,
            ambiguity_kl_spread_pivot,
            ambiguity_kl_radial_pivot,
            ambiguity_kl_max_scale,
        )
        prior_weights.append(per_sample_prior)
        ap_multipliers.append(discriminative_ap_multiplier(
            detection_probs[member_index],
            rss_centers[member_index],
            initials[member_index],
            query_mask,
            discriminative_ap_weight,
            discriminative_ap_rss_scale,
            discriminative_ap_detection_scale,
            discriminative_ap_min,
            discriminative_ap_max,
        ))

        if floor_consistency_weight > 0.0:
            candidate_floors = candidate_floors_list[member_index]
            predicted_floor = predicted_floors[member_index]
            if candidate_floors is None or predicted_floor is None:
                raise ValueError(
                    "candidate floors and predicted floors are required when "
                    "floor_consistency_weight > 0")
            floor_mismatches.append(
                (
                    candidate_floors.to(query.device)
                    != predicted_floor.to(query.device).unsqueeze(1)
                ).to(dtype=query.dtype)
            )
        else:
            floor_mismatches.append(None)

        gate_mask = gate_masks[member_index]
        if gate_mask is not None and gate_mask.shape != (batch_size,):
            raise ValueError("each gate mask must have shape [B]")

    for _ in range(steps):
        member_losses = []
        member_locations = []
        for member_index in range(member_count):
            query = queries[member_index]
            query_mask = query_masks[member_index]
            query_value = query_values[member_index]
            reliability = (
                ap_reliabilities[member_index].unsqueeze(0)
                * ap_multipliers[member_index]
            )
            weights = torch.softmax(logits[member_index], dim=1)
            detection_prob = detection_probs[member_index]
            rss_center = rss_centers[member_index]
            rss_scale = rss_scales[member_index]

            if visibility_bias_max > 0.0:
                visibility_bias = (
                    torch.tanh(raw_visibility_biases[member_index])
                    * float(visibility_bias_max)
                )
                calibrated_detection = torch.sigmoid(
                    torch.logit(
                        detection_prob.clamp(min=1e-4, max=1.0 - 1e-4)
                    )
                    + visibility_bias.unsqueeze(1)
                )
            else:
                visibility_bias = torch.zeros_like(
                    raw_visibility_biases[member_index])
                calibrated_detection = detection_prob
            if style_scale_max_log > 0.0:
                style_scale = torch.exp(
                    torch.tanh(raw_style_scales[member_index])
                    * float(style_scale_max_log)
                )
            else:
                style_scale = torch.ones_like(raw_style_scales[member_index])
            calibrated_rss_center = rss_center * style_scale.unsqueeze(1)
            predicted_detection = torch.sum(
                weights.unsqueeze(2) * calibrated_detection, dim=1)
            rss_mass = torch.sum(
                weights.unsqueeze(2)
                * calibrated_detection
                * calibrated_rss_center,
                dim=1,
            )
            predicted_rss = rss_mass / predicted_detection.clamp(min=0.05)
            if style_offset_max > 0.0:
                style_offset = (
                    torch.tanh(raw_style_offsets[member_index])
                    * float(style_offset_max)
                )
            else:
                style_offset = torch.zeros_like(
                    raw_style_offsets[member_index])
            predicted_rss = predicted_rss + style_offset

            rss_weight = query_mask * reliability
            if rss_centering == "visible_mean":
                rss_weight_sum = rss_weight.sum(
                    dim=1, keepdim=True).clamp(min=1.0)
                query_mean = (query_value * rss_weight).sum(
                    dim=1, keepdim=True) / rss_weight_sum
                predicted_mean = (predicted_rss * rss_weight).sum(
                    dim=1, keepdim=True) / rss_weight_sum
                residual_prediction = predicted_rss - predicted_mean
                residual_target = query_value - query_mean
            else:
                residual_prediction = predicted_rss
                residual_target = query_value
            rss_residual = F.smooth_l1_loss(
                residual_prediction,
                residual_target,
                reduction="none",
                beta=0.10,
            )
            rss_loss = (
                rss_residual * rss_weight
            ).sum(dim=1) / rss_weight.sum(dim=1).clamp(min=1.0)

            censored_loss = torch.zeros_like(rss_loss)
            if censored_invisible_weight > 0.0:
                censored_loss = censored_non_detection_loss(
                    predicted_detection,
                    predicted_rss,
                    query_mask,
                    reliability,
                    censored_mode,
                    censored_detection_threshold,
                    censored_rss_threshold,
                    censored_rss_softness,
                    censored_contradiction_margin,
                    initial_detection=initial_detections[member_index],
                    initial_rss=initial_rsses[member_index],
                )

            visibility_loss = balanced_visibility_loss(
                predicted_detection,
                query_mask,
                reliability,
                visibility_positive_factor,
                visibility_negative_factor,
            )
            mixture_loss = torch.zeros_like(rss_loss)
            if mixture_censored_weight > 0.0:
                mixture_loss = full_mixture_censored_loss(
                    weights,
                    calibrated_detection,
                    rss_center,
                    rss_scale,
                    query_value,
                    query_mask,
                    reliability,
                    censored_rss_threshold,
                    style_offset=style_offset,
                    style_scale=style_scale,
                    visibility_bias=None,
                    visible_factor=mixture_visible_factor,
                    invisible_factor=mixture_invisible_factor,
                )
            prior_loss = torch.sum(
                weights * (
                    torch.log(weights.clamp(min=1e-8))
                    - torch.log(initials[member_index])
                ),
                dim=1,
            )
            mean_location = torch.sum(
                weights.unsqueeze(2)
                * candidate_locations_list[member_index],
                dim=1,
            )
            member_locations.append(mean_location)
            compactness = torch.sum(
                weights
                * (
                    candidate_locations_list[member_index]
                    - mean_location.unsqueeze(1)
                ).square().sum(dim=2),
                dim=1,
            ) / 100.0
            support_density = torch.zeros_like(compactness)
            if support_density_weight > 0.0:
                support_locations = support_locations_list[member_index]
                if support_locations is None:
                    raise ValueError(
                        "support_locations are required when "
                        "support_density_weight > 0")
                support_density = support_density_energy(
                    mean_location,
                    support_locations,
                    support_masks[member_index],
                    support_density_sigma,
                    support_density_margin,
                    support_density_mode,
                )
            floor_loss = torch.zeros_like(prior_loss)
            floor_mismatch = floor_mismatches[member_index]
            if floor_mismatch is not None:
                floor_loss = torch.sum(weights * floor_mismatch, dim=1)
            manifold_loss = torch.zeros_like(prior_loss)
            if manifold_weight > 0.0:
                manifold_loss = initial_support_manifold_loss(
                    weights,
                    initials[member_index],
                    candidate_locations_list[member_index],
                    manifold_sigma,
                    manifold_margin,
                )

            member_loss = (
                float(rss_loss_weight) * rss_loss
                + censored_invisible_weight * censored_loss
                + mixture_censored_weight * mixture_loss
                + visibility_weight * visibility_loss
                + prior_weights[member_index] * prior_loss
                + manifold_weight * manifold_loss
                + compactness_weight * compactness
                + style_offset_weight * style_offset.squeeze(1).square()
                + style_scale_weight * torch.log(style_scale).squeeze(1).square()
                + visibility_bias_weight * visibility_bias.squeeze(1).square()
                + support_density_weight * support_density
                + floor_consistency_weight * floor_loss
            )
            member_losses.append(member_loss)

        location_stack = torch.stack(member_locations, dim=0)
        consensus_center = torch.sum(
            precision_weights.unsqueeze(2) * location_stack, dim=0)
        disagreement = torch.sum(
            precision_weights
            * (location_stack - consensus_center.unsqueeze(0))
            .square()
            .sum(dim=2),
            dim=0,
        ) / 100.0
        member_loss_stack = torch.stack(member_losses, dim=0).mean(dim=0)
        loss = (
            member_loss_stack
            + float(joint_ensemble_weight) * disagreement
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    projected = [torch.softmax(member_logits.detach(), dim=1)
                 for member_logits in logits]
    for member_index, gate_mask in enumerate(gate_masks):
        if gate_mask is None:
            continue
        projected[member_index] = torch.where(
            gate_mask.to(device=device, dtype=torch.bool).unsqueeze(1),
            projected[member_index],
            initials[member_index],
        )
    if not return_calibration:
        return projected

    calibration = {
        "style_offset": [],
        "style_scale": [],
        "style_log_scale": [],
        "visibility_bias": [],
    }
    for member_index in range(member_count):
        if style_offset_max > 0.0:
            style_offset = (
                torch.tanh(raw_style_offsets[member_index].detach())
                * float(style_offset_max)
            )
        else:
            style_offset = torch.zeros_like(raw_style_offsets[member_index])
        if style_scale_max_log > 0.0:
            style_log_scale = (
                torch.tanh(raw_style_scales[member_index].detach())
                * float(style_scale_max_log)
            )
            style_scale = torch.exp(style_log_scale)
        else:
            style_scale = torch.ones_like(raw_style_scales[member_index])
            style_log_scale = torch.zeros_like(raw_style_scales[member_index])
        if visibility_bias_max > 0.0:
            visibility_bias = (
                torch.tanh(raw_visibility_biases[member_index].detach())
                * float(visibility_bias_max)
            )
        else:
            visibility_bias = torch.zeros_like(raw_visibility_biases[member_index])
        calibration["style_offset"].append(style_offset.squeeze(1))
        calibration["style_scale"].append(style_scale.squeeze(1))
        calibration["style_log_scale"].append(style_log_scale.squeeze(1))
        calibration["visibility_bias"].append(visibility_bias.squeeze(1))
    return projected, calibration


def project_entropy_min_weights(
    initial_weights: torch.Tensor,
    prior_weight: float,
    steps: int,
    learning_rate: float,
) -> torch.Tensor:
    """TENT-style entropy minimization baseline for anchor weights.

    This baseline adapts only the candidate posterior logits at test time. It
    has no RSS reconstruction term and no visibility likelihood; the only
    stabilizer is KL to the frozen model posterior. It is intentionally weaker
    than DAPP and exists as a protocol-matched generic test-time adaptation
    control.
    """
    initial = initial_weights.clamp(min=1e-8)
    logits = initial.log().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    kl_weight = float(prior_weight)
    for _ in range(steps):
        weights = torch.softmax(logits, dim=1)
        entropy = -torch.sum(
            weights * torch.log(weights.clamp(min=1e-8)),
            dim=1,
        )
        prior_loss = torch.sum(
            weights * (
                torch.log(weights.clamp(min=1e-8)) - torch.log(initial)
            ),
            dim=1,
        )
        loss = (entropy + kl_weight * prior_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.softmax(logits.detach(), dim=1)


