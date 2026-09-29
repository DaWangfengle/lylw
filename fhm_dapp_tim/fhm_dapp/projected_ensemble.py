"""Final AC-RHAP alpha-consensus robust-hurdle evaluation.

流程:
    1. 加载 5 个 seed 的 checkpoint
    2. 按全局锚点身份对齐后验并计算 alpha 共识
    3. 在身份认证的单一共享 radio memory 上提取候选测量统计
    4. 在共识后验上执行一次匹配 alpha-prox 并解码一次坐标
    5. 配对 bootstrap 检验 projected vs consensus prior

This is the paper-ready implementation used by scripts/reproduce_main.ps1.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.anchor import (
    build_anchor_memory,
    grouped_protocol_split,
    load_anchor_checkpoint,
    raw_coords,
    resolve_path,
)
from fhm_dapp_tim.fhm_dapp.candidates import prepare_candidates
from fhm_dapp_tim.fhm_dapp.consensus import (
    alpha_consensus_prior,
    weighted_component_mean,
)
from fhm_dapp_tim.fhm_dapp.dapp import (
    anchor_ap_censored_threshold,
    anchor_ap_reliability,
    initial_support_manifold_energy,
    project_alpha_proximal_hurdle,
    project_entropy_min_weights,
    project_candidate_weights,
    project_joint_candidate_weights,
    support_density_energy,
    support_density_mask,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.ensemble import (
    huber_precision_center,
    paired_bootstrap,
    weighted_geometric_median,
)
from fhm_dapp_tim.fhm_dapp.metrics import coordinate_metrics
from fhm_dapp_tim.fhm_dapp.uncertainty import (
    covariance_ellipse_area,
    mahalanobis_scores,
    student_t_mixture_moments,
)


_CONSENSUS_CANDIDATE_KEYS = (
    "baseline",
    "radial_scale",
    "candidate_indices",
    "predicted_building",
    "predicted_floor",
    "initial_weights",
    "embedding",
)


SERIALIZED_CONFIGURATION_SCHEMA_VERSION = 2

# These are the only runtime values serialized for the joint/single-consensus
# path.  Derived identity/readout fields are appended by the single builder
# below; callers cannot silently add an unregistered configuration key.
_SINGLE_CONSENSUS_CONFIGURATION_VALUE_FIELDS = (
    "variant_name",
    "top_k",
    "member_top_k",
    "neural_scale",
    "prototype_weight",
    "posterior_temperature",
    "candidate_building_conditioning",
    "visibility_weight",
    "prior_weight",
    "compactness_weight",
    "style_offset_max",
    "style_offset_weight",
    "style_scale_max_log",
    "style_scale_weight",
    "visibility_bias_max",
    "visibility_bias_weight",
    "reliability_power",
    "member_weight_mode",
    "measurement_pool_mode",
    "support_density_weight",
    "support_density_sigma",
    "support_density_margin",
    "support_density_scope",
    "support_density_mode",
    "support_density_step",
    "support_ood_precision_scale",
    "support_ood_precision_power",
    "support_ood_sigma",
    "support_ood_scope",
    "support_ood_mode",
    "rss_centering",
    "rss_loss_weight",
    "visibility_positive_factor",
    "visibility_negative_factor",
    "mixture_censored_weight",
    "mixture_visible_factor",
    "mixture_invisible_factor",
    "projection_mode",
    "prior_alpha",
    "hurdle_distribution",
    "hurdle_df",
    "hurdle_score",
    "hurdle_power_beta",
    "evidence_temperature_mode",
    "rss_global_offset_db",
    "rss_dropout_prob",
    "rss_dropout_seed",
    "adaptive_kl_alpha",
    "adaptive_kl_visible_pivot",
    "adaptive_kl_min_scale",
    "adaptive_kl_max_scale",
    "ambiguity_kl_weight",
    "ambiguity_kl_spread_pivot",
    "ambiguity_kl_radial_pivot",
    "ambiguity_kl_max_scale",
    "discriminative_ap_weight",
    "discriminative_ap_rss_scale",
    "discriminative_ap_detection_scale",
    "discriminative_ap_min",
    "discriminative_ap_max",
    "manifold_weight",
    "manifold_sigma",
    "manifold_margin",
    "ap_censored_threshold_mode",
    "ap_censored_threshold_quantile",
    "ap_censored_threshold_shrinkage",
    "ap_censored_threshold_min",
    "ap_censored_threshold_max",
    "hierarchical_style",
    "style_residual_fraction",
    "style_group_count",
    "style_group_mode",
    "censored_invisible_weight",
    "censored_detection_threshold",
    "censored_rss_threshold",
    "censored_mode",
    "censored_rss_softness",
    "censored_contradiction_margin",
    "floor_consistency_weight",
    "gate_entropy_min",
    "gate_spread_min",
    "gate_visible_max",
    "gate_radial_scale_min",
    "ensemble_consensus_weight",
    "joint_ensemble_weight",
    "consensus_prior_mode",
    "consensus_prior_epsilon",
    "consensus_prior_alpha",
    "steps",
    "learning_rate",
)

_INDEPENDENT_CONFIGURATION_VALUE_FIELDS = (
    "variant_name",
    "top_k",
    "member_top_k",
    "neural_scale",
    "prototype_weight",
    "posterior_temperature",
    "candidate_building_conditioning",
    "visibility_weight",
    "prior_weight",
    "compactness_weight",
    "style_offset_max",
    "style_offset_weight",
    "reliability_power",
    "support_density_weight",
    "support_density_sigma",
    "support_density_margin",
    "support_density_scope",
    "support_density_mode",
    "support_density_step",
    "support_ood_precision_scale",
    "support_ood_precision_power",
    "support_ood_sigma",
    "support_ood_scope",
    "support_ood_mode",
    "rss_centering",
    "rss_loss_weight",
    "visibility_positive_factor",
    "visibility_negative_factor",
    "mixture_censored_weight",
    "mixture_visible_factor",
    "mixture_invisible_factor",
    "projection_mode",
    "adaptive_kl_alpha",
    "adaptive_kl_visible_pivot",
    "adaptive_kl_min_scale",
    "adaptive_kl_max_scale",
    "ambiguity_kl_weight",
    "ambiguity_kl_spread_pivot",
    "ambiguity_kl_radial_pivot",
    "ambiguity_kl_max_scale",
    "discriminative_ap_weight",
    "discriminative_ap_rss_scale",
    "discriminative_ap_detection_scale",
    "discriminative_ap_min",
    "discriminative_ap_max",
    "manifold_weight",
    "manifold_sigma",
    "manifold_margin",
    "censored_invisible_weight",
    "censored_detection_threshold",
    "censored_rss_threshold",
    "censored_mode",
    "censored_rss_softness",
    "censored_contradiction_margin",
    "floor_consistency_weight",
    "gate_entropy_min",
    "gate_spread_min",
    "gate_visible_max",
    "gate_radial_scale_min",
    "risk_interpolation_mode",
    "risk_interpolation_pivot",
    "risk_interpolation_temperature",
    "risk_interpolation_min_gamma",
    "risk_interpolation_max_gamma",
    "risk_interpolation_entropy_pivot",
    "risk_interpolation_diameter_pivot",
    "risk_interpolation_visible_pivot",
    "risk_interpolation_entropy_weight",
    "risk_interpolation_diameter_weight",
    "risk_interpolation_visible_weight",
    "ensemble_consensus_weight",
    "joint_ensemble_weight",
    "steps",
    "learning_rate",
)

SINGLE_CONSENSUS_SERIALIZED_CONFIGURATION_FIELDS = (
    "configuration_schema_version",
    "configuration_pipeline_mode",
    *_SINGLE_CONSENSUS_CONFIGURATION_VALUE_FIELDS,
    "effective_measurement_pool_mode",
    "measurement_diagnostics_apply_affine",
    "measurement_pool",
    "decoder",
    "compute_predictive_distribution",
    "coordinate_predictive_distribution",
    "ensemble",
)

INDEPENDENT_SERIALIZED_CONFIGURATION_FIELDS = (
    "configuration_schema_version",
    "configuration_pipeline_mode",
    *_INDEPENDENT_CONFIGURATION_VALUE_FIELDS,
    "decoder",
    "compute_predictive_distribution",
    "coordinate_predictive_distribution",
    "ensemble",
)


def _compact_consensus_candidates(
    candidates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Retain only tensors used after member-wise consensus preparation.

    In shared-memory mode, member-specific ``[B,K,A]`` detection/RSS tensors
    are never constructed.  Historical pooling modes still discard them here
    and re-gather on the common support.  The final path retains one certified
    ``O(N*A)`` radio memory, separate from per-batch workspace.
    """
    missing = [key for key in _CONSENSUS_CANDIDATE_KEYS if key not in candidates]
    if missing:
        raise KeyError(
            "consensus candidate state is missing: " + ", ".join(missing))
    return {key: candidates[key] for key in _CONSENSUS_CANDIDATE_KEYS}


def _resolve_predictive_distribution_flag(
    requested: bool | None,
    *,
    include_diagnostics: bool,
    single_consensus: bool,
) -> bool:
    """Resolve the optional reliability readout without changing point decode.

    Projection diagnostics and the predictive reliability readout are
    deliberately separate output roles.  Ordinary point evidence therefore
    remains readout-free even when per-row projection diagnostics are stored;
    the uncertainty stage must request the distribution explicitly.
    """
    if requested is None:
        del include_diagnostics
        return False
    enabled = bool(requested)
    if enabled and not single_consensus:
        raise ValueError(
            "compute_predictive_distribution requires "
            "consensus_prior_mode != none"
        )
    return enabled


def _synchronize_timing_device(device: torch.device) -> None:
    """Close asynchronous CUDA work at point-inference timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


_POINT_INFERENCE_TIMING_EXCLUDES = [
    "target-dependent coordinate and hierarchy metrics",
    "paired bootstrap",
    "predictive-distribution reliability readout",
    "NLL/Mahalanobis/Spearman statistics",
    "alternative-decoder diagnostics",
    "diagnostic and JSON aggregation",
]


_ALPHA_PROX_LEGACY_POINT_NEUTRALS: dict[str, object] = {
    "visibility_weight": 0.0,
    "compactness_weight": 0.0,
    "visibility_bias_max": 0.0,
    "visibility_bias_weight": 0.0,
    "rss_centering": "none",
    "rss_loss_weight": 0.0,
    "visibility_positive_factor": 1.0,
    "visibility_negative_factor": 1.0,
    "mixture_censored_weight": 0.0,
    "support_density_weight": 0.0,
    "adaptive_kl_alpha": 0.0,
    "ambiguity_kl_weight": 0.0,
    "discriminative_ap_weight": 0.0,
    "censored_invisible_weight": 0.0,
    "floor_consistency_weight": 0.0,
    "hierarchical_style": False,
    "risk_interpolation_mode": "none",
    "ensemble_consensus_weight": 0.0,
}

_ALPHA_PROX_LEGACY_COMPANION_DEFAULTS: dict[str, object] = {
    "support_density_sigma": 4.0,
    "support_density_margin": 0.0,
    "support_density_scope": "all",
    "support_density_mode": "soft_distance",
    "support_density_step": 1.0,
    "adaptive_kl_visible_pivot": 16.0,
    "adaptive_kl_min_scale": 1.0,
    "adaptive_kl_max_scale": 1.0,
    "ambiguity_kl_spread_pivot": 8.0,
    "ambiguity_kl_radial_pivot": 8.0,
    "ambiguity_kl_max_scale": 2.0,
    "discriminative_ap_rss_scale": 0.15,
    "discriminative_ap_detection_scale": 0.20,
    "discriminative_ap_min": 0.25,
    "discriminative_ap_max": 3.0,
    "censored_mode": "hinge",
    "censored_rss_softness": 0.10,
    "censored_contradiction_margin": 0.0,
    "style_residual_fraction": 1.0,
    "style_group_count": 1,
    "style_group_mode": "batch",
    "risk_interpolation_pivot": 8.0,
    "risk_interpolation_temperature": 2.0,
    "risk_interpolation_min_gamma": 0.0,
    "risk_interpolation_max_gamma": 1.0,
    "risk_interpolation_entropy_pivot": 2.0,
    "risk_interpolation_diameter_pivot": 80.0,
    "risk_interpolation_visible_pivot": 16.0,
    "risk_interpolation_entropy_weight": 1.0,
    "risk_interpolation_diameter_weight": 1.0,
    "risk_interpolation_visible_weight": 1.0,
    "ap_censored_threshold_quantile": 0.10,
    "ap_censored_threshold_shrinkage": 0.50,
    "ap_censored_threshold_min": -1.00,
    "ap_censored_threshold_max": -0.50,
}

_ALPHA_PROX_DIAGNOSTIC_ONLY_DEFAULTS: dict[str, object] = {
    "support_ood_precision_scale": 0.0,
    "support_ood_precision_power": 2.0,
    "support_ood_sigma": 4.0,
    "support_ood_scope": "building",
    "support_ood_mode": "kde_softmin",
    "censored_detection_threshold": 0.10,
    "censored_rss_threshold": -0.95,
}

_ALPHA_PROX_GATE_OPTIONS = (
    "gate_entropy_min",
    "gate_spread_min",
    "gate_visible_max",
    "gate_radial_scale_min",
)


def _option_differs(actual: object, expected: object) -> bool:
    """Compare public option values without treating bool as an integer."""
    if isinstance(expected, bool):
        return not isinstance(actual, bool) or actual is not expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return True
        return not bool(np.isclose(
            float(actual), float(expected), rtol=0.0, atol=1e-12))
    return actual != expected


def _validate_alpha_proximal_public_options(
    projection_mode: str,
    options: dict[str, object],
) -> None:
    """Reject or disclose options that alpha-prox does not consume.

    The public API historically exposed controls from several DAPP branches.
    Silently accepting an active legacy control under ``alpha_prox_hurdle``
    makes a serialized configuration appear more complex than the executed
    model.  Point-model conflicts therefore fail closed.  Options that only
    alter diagnostic readouts, or companions whose controller is neutral, are
    retained for backward compatibility but generate one explicit warning.
    """
    if projection_mode.lower() != "alpha_prox_hurdle":
        return

    missing = [
        key for key in (
            *_ALPHA_PROX_LEGACY_POINT_NEUTRALS,
            *_ALPHA_PROX_LEGACY_COMPANION_DEFAULTS,
            *_ALPHA_PROX_DIAGNOSTIC_ONLY_DEFAULTS,
            *_ALPHA_PROX_GATE_OPTIONS,
            "hurdle_distribution",
            "hurdle_df",
            "hurdle_score",
            "hurdle_power_beta",
        )
        if key not in options
    ]
    if missing:
        raise KeyError(
            "alpha-prox option audit is missing: " + ", ".join(missing))

    conflicts = [
        key
        for key, neutral in _ALPHA_PROX_LEGACY_POINT_NEUTRALS.items()
        if _option_differs(options[key], neutral)
    ]
    conflicts.extend(
        key
        for key in _ALPHA_PROX_GATE_OPTIONS
        if float(options[key]) >= 0.0
    )
    if conflicts:
        raise ValueError(
            "alpha_prox_hurdle does not consume these active legacy point-"
            "localization options: " + ", ".join(sorted(conflicts))
        )

    diagnostic_only = [
        key
        for key, default in _ALPHA_PROX_DIAGNOSTIC_ONLY_DEFAULTS.items()
        if _option_differs(options[key], default)
    ]
    if diagnostic_only:
        warnings.warn(
            "alpha_prox_hurdle treats these options as diagnostic-only; they "
            "do not alter the decoded point coordinate: "
            + ", ".join(sorted(diagnostic_only)),
            UserWarning,
            stacklevel=2,
        )

    companions = [
        key
        for key, default in _ALPHA_PROX_LEGACY_COMPANION_DEFAULTS.items()
        if _option_differs(options[key], default)
    ]
    if (
        str(options["hurdle_score"]).lower() != "density_power"
        and _option_differs(options["hurdle_power_beta"], 0.25)
    ):
        companions.append("hurdle_power_beta")
    if (
        str(options["hurdle_distribution"]).lower() == "gaussian"
        and _option_differs(options["hurdle_df"], 4.0)
    ):
        companions.append("hurdle_df")
    if companions:
        warnings.warn(
            "alpha_prox_hurdle ignores these inactive legacy/dependent "
            "options under the selected controller settings: "
            + ", ".join(sorted(companions)),
            UserWarning,
            stacklevel=2,
        )


def _point_inference_timing_payload(
    batch_seconds: list[float],
    *,
    device: torch.device,
    pipeline_mode: str,
    include_diagnostics: bool,
    compute_predictive_distribution: bool,
) -> dict[str, object]:
    """Build one auditable timing schema for both localization pipelines."""
    scopes = {
        "single_consensus": (
            "online RSS batch transfer -> candidate construction -> member "
            "alpha consensus -> one certified shared-memory gather -> one "
            "projection -> coordinate decode"
        ),
        "independent_members": (
            "online RSS batches -> per-member candidate construction -> E "
            "independent projections -> precision-weighted Huber coordinate "
            "fusion decode"
        ),
    }
    if pipeline_mode not in scopes:
        raise ValueError(f"unsupported timing pipeline mode: {pipeline_mode}")
    normalized = [float(value) for value in batch_seconds]
    eligible = bool(
        normalized
        and not include_diagnostics
        and not compute_predictive_distribution
    )
    return {
        "point_inference_seconds": (
            float(sum(normalized)) if normalized else None),
        "point_inference_batch_seconds": normalized,
        "scope": scopes[pipeline_mode],
        "excludes": list(_POINT_INFERENCE_TIMING_EXCLUDES),
        "pipeline_mode": pipeline_mode,
        "paper_point_latency_eligible": eligible,
        "cuda_synchronized_boundaries": device.type == "cuda",
    }


def _member_weight_tensor(
    radial_scales: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Return normalized label-free member weights ``[M,B]``."""
    if radial_scales.ndim != 2 or radial_scales.numel() == 0:
        raise ValueError("radial_scales must have non-empty shape [M,B]")
    if not bool(torch.isfinite(radial_scales).all()) or bool(
        (radial_scales <= 0.0).any()
    ):
        raise ValueError("radial_scales must be finite and positive")
    normalized_mode = mode.lower()
    if normalized_mode == "inverse_scale":
        weights = radial_scales.clamp(min=1e-6).square().reciprocal()
    elif normalized_mode == "uniform":
        weights = torch.ones_like(radial_scales)
    else:
        raise ValueError(
            "member_weight_mode must be inverse_scale or uniform")
    return weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8)


def _resolve_measurement_pool(mode: str, projection_mode: str) -> str:
    """Resolve the explicit pool or the backward-compatible automatic mode."""
    normalized = mode.lower()
    if normalized not in {"auto", "shared", "w2", "moment_match"}:
        raise ValueError(
            "measurement_pool_mode must be auto, shared, w2 or moment_match")
    if normalized == "auto":
        return (
            "w2"
            if projection_mode == "alpha_prox_hurdle"
            else "moment_match"
        )
    return normalized


def _resolve_member_top_k(
    top_k: int,
    member_top_k: int | None,
    *,
    single_consensus: bool,
    anchor_count: int | None = None,
) -> int:
    """Resolve member retrieval width ``L`` under the ``K <= L`` contract.

    ``top_k`` is the final common support width ``K`` consumed by the
    consensus projection and the robust-hurdle update. ``member_top_k`` is
    the per-member retrieval width ``L`` used only before single-consensus
    pooling.  The ``None`` sentinel is deliberately resolved to ``K`` so the
    public default executes the historical graph and remains numerically
    identical.  Wider retrieval is rejected for every non-consensus branch.
    """
    if not isinstance(single_consensus, bool):
        raise TypeError("single_consensus must be boolean")

    def strict_positive_integer(value: object, name: str) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{name} must be positive")
        return resolved

    final_width = strict_positive_integer(top_k, "top_k")
    member_width = (
        final_width
        if member_top_k is None
        else strict_positive_integer(member_top_k, "member_top_k")
    )
    if member_width < final_width:
        raise ValueError("candidate widths must satisfy top_k <= member_top_k")
    if not single_consensus and member_width != final_width:
        raise ValueError(
            "member_top_k may exceed top_k only for single-consensus "
            "evaluation"
        )
    if anchor_count is not None:
        resolved_anchor_count = strict_positive_integer(
            anchor_count, "anchor_count")
        if member_width > resolved_anchor_count:
            raise ValueError(
                "candidate widths must satisfy top_k <= member_top_k <= "
                "anchor_count"
            )
    return member_width


def _configuration_values(
    options: Mapping[str, object],
    fields: tuple[str, ...],
) -> dict[str, object]:
    missing = [field for field in fields if field not in options]
    if missing:
        raise KeyError(
            "serialized configuration is missing runtime values: "
            + ", ".join(missing)
        )
    return {field: options[field] for field in fields}


def build_serialized_configuration(
    options: Mapping[str, object],
    *,
    pipeline_mode: str,
    compute_predictive_distribution: bool,
    effective_measurement_pool: str | None = None,
) -> dict[str, object]:
    """Build the one closed-world configuration serialized in evidence.

    Both evaluation branches and the reproduction verifier call this helper.
    The exact field sets are public constants, so an extra, missing, or
    branch-inappropriate key cannot be silently ignored by evidence reuse.
    """
    if pipeline_mode not in {
        "single_consensus", "joint_projection", "independent_members"
    }:
        raise ValueError(
            "pipeline_mode must be single_consensus, joint_projection, or "
            "independent_members"
        )
    if not isinstance(compute_predictive_distribution, bool):
        raise TypeError("compute_predictive_distribution must be boolean")

    missing_widths = [
        name for name in ("top_k", "member_top_k") if name not in options
    ]
    if missing_widths:
        raise KeyError(
            "serialized configuration is missing runtime values: "
            + ", ".join(missing_widths)
        )
    resolved_options = dict(options)
    resolved_options["member_top_k"] = _resolve_member_top_k(
        options["top_k"],
        options["member_top_k"],
        single_consensus=pipeline_mode == "single_consensus",
    )

    if pipeline_mode in {"single_consensus", "joint_projection"}:
        configuration = {
            "configuration_schema_version": (
                SERIALIZED_CONFIGURATION_SCHEMA_VERSION),
            "configuration_pipeline_mode": pipeline_mode,
            **_configuration_values(
                resolved_options,
                _SINGLE_CONSENSUS_CONFIGURATION_VALUE_FIELDS,
            ),
        }
        consensus_mode = str(resolved_options["consensus_prior_mode"]).lower()
        single_consensus = consensus_mode != "none"
        if single_consensus != (pipeline_mode == "single_consensus"):
            raise ValueError(
                "configuration pipeline mode disagrees with "
                "consensus_prior_mode"
            )
        projection_mode = str(resolved_options["projection_mode"]).lower()
        resolved_pool = (
            _resolve_measurement_pool(
                str(options["measurement_pool_mode"]), projection_mode)
            if effective_measurement_pool is None
            else str(effective_measurement_pool)
        )
        if resolved_pool not in {"shared", "w2", "moment_match"}:
            raise ValueError("invalid effective measurement pool")
        configuration.update({
            "effective_measurement_pool_mode": resolved_pool,
            "measurement_diagnostics_apply_affine": True,
            "measurement_pool": {
                "shared": "shared_identity_certified_radio_memory",
                "w2": "visibility-conditioned_w2_location_scale",
                "moment_match": "visibility-conditioned_second_moment",
            }[resolved_pool],
            "decoder": "Student-t component",
            "compute_predictive_distribution": (
                compute_predictive_distribution),
            "coordinate_predictive_distribution": (
                "member_anchor_product_student_t_mixture"
                if single_consensus else "not_available"
            ),
            "ensemble": (
                "alpha consensus + alpha-proximal robust hurdle"
                if projection_mode == "alpha_prox_hurdle"
                else (
                    "alpha anchor consensus + single DAPP"
                    if single_consensus
                    else "canonical Huber precision center"
                )
            ),
        })
        expected_fields = SINGLE_CONSENSUS_SERIALIZED_CONFIGURATION_FIELDS
    else:
        configuration = {
            "configuration_schema_version": (
                SERIALIZED_CONFIGURATION_SCHEMA_VERSION),
            "configuration_pipeline_mode": pipeline_mode,
            **_configuration_values(
                resolved_options,
                _INDEPENDENT_CONFIGURATION_VALUE_FIELDS,
            ),
            "decoder": "Student-t component",
            "compute_predictive_distribution": (
                compute_predictive_distribution),
            "coordinate_predictive_distribution": "not_available",
            "ensemble": "canonical Huber precision center",
        }
        expected_fields = INDEPENDENT_SERIALIZED_CONFIGURATION_FIELDS

    if set(configuration) != set(expected_fields):
        raise RuntimeError(
            "internal serialized-configuration schema drift detected")
    return configuration


def serialized_configuration_schema_ready(
    configuration: object,
) -> bool:
    """Return true only for one of the two exact serialized field schemas."""
    if not isinstance(configuration, dict):
        return False
    if configuration.get("configuration_schema_version") != (
        SERIALIZED_CONFIGURATION_SCHEMA_VERSION
    ):
        return False
    mode = configuration.get("configuration_pipeline_mode")
    expected = (
        SINGLE_CONSENSUS_SERIALIZED_CONFIGURATION_FIELDS
        if mode in {"single_consensus", "joint_projection"}
        else INDEPENDENT_SERIALIZED_CONFIGURATION_FIELDS
        if mode == "independent_members"
        else ()
    )
    return bool(expected and set(configuration) == set(expected))


def _powered_ap_reliability(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    power: float,
) -> torch.Tensor:
    """Skip the deleted AP-reliability component on the exact zero path."""
    if not np.isfinite(float(power)) or float(power) < 0.0:
        raise ValueError("reliability_power must be finite and nonnegative")
    if float(power) == 0.0:
        return torch.ones(
            detection_prob.shape[1],
            dtype=detection_prob.dtype,
            device=detection_prob.device,
        )
    return anchor_ap_reliability(
        detection_prob, rss_center, rss_scale).pow(float(power))


def _validate_checkpoint_alignment(checkpoints: list[tuple]) -> None:
    """Fail closed unless integer anchor/AP identities agree across members."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    reference_model, reference_checkpoint = checkpoints[0]
    reference_coords = reference_model.anchor_coords.detach().cpu()
    reference_buildings = reference_model.anchor_buildings.detach().cpu()
    reference_floors = reference_model.anchor_floors.detach().cpu()
    reference_seen = reference_model.encoder.seen_ap_mask.detach().cpu()
    if reference_coords.ndim != 2 or reference_coords.shape[-1] != 2:
        raise ValueError("checkpoint 0 anchor coordinates must have shape [K,2]")
    anchor_count = int(reference_coords.shape[0])
    if reference_buildings.shape != (anchor_count,):
        raise ValueError(
            "checkpoint 0 anchor buildings do not match the anchor count")
    if reference_floors.shape != (anchor_count,):
        raise ValueError(
            "checkpoint 0 anchor floors do not match the anchor count")
    if reference_seen.ndim != 1:
        raise ValueError("checkpoint 0 seen_ap_mask must be one-dimensional")
    if not bool(torch.isfinite(reference_coords).all()):
        raise ValueError("checkpoint 0 anchor coordinates are not finite")
    reference_stats = reference_checkpoint.get("stats", {})
    reference_ap_order = tuple(reference_stats.get("wap_cols", ()))
    if not reference_ap_order:
        raise ValueError("checkpoint 0 does not record AP feature order")
    if len(reference_ap_order) != int(reference_seen.numel()):
        raise ValueError(
            "checkpoint 0 AP feature order and seen_ap_mask disagree")
    reference_origin = np.asarray(
        reference_checkpoint.get("coord_origin"), dtype=np.float64)
    if reference_origin.ndim != 1 or not np.isfinite(reference_origin).all():
        raise ValueError("checkpoint 0 coordinate origin is missing or invalid")

    for member_index, (model, checkpoint) in enumerate(checkpoints[1:], 1):
        coords = model.anchor_coords.detach().cpu()
        buildings = model.anchor_buildings.detach().cpu()
        floors = model.anchor_floors.detach().cpu()
        seen = model.encoder.seen_ap_mask.detach().cpu()
        if coords.shape != reference_coords.shape:
            raise ValueError(
                f"checkpoint {member_index} uses a different anchor count")
        if not torch.allclose(
            coords, reference_coords, atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                f"checkpoint {member_index} anchor coordinates/order differ")
        if not torch.equal(buildings, reference_buildings):
            raise ValueError(
                f"checkpoint {member_index} anchor building/order differ")
        if not torch.equal(floors, reference_floors):
            raise ValueError(
                f"checkpoint {member_index} anchor floor/order differ")
        if not torch.equal(seen, reference_seen):
            raise ValueError(
                f"checkpoint {member_index} seen AP mask/order differs")
        ap_order = tuple(checkpoint.get("stats", {}).get("wap_cols", ()))
        if ap_order != reference_ap_order:
            raise ValueError(
                f"checkpoint {member_index} AP feature order differs")
        origin = np.asarray(checkpoint.get("coord_origin"), dtype=np.float64)
        if origin.shape != reference_origin.shape or not np.allclose(
            origin, reference_origin, atol=1e-8, rtol=0.0
        ):
            raise ValueError(
                f"checkpoint {member_index} uses different coordinates")


def _hash_named_arrays(
    values: list[tuple[str, np.ndarray | torch.Tensor]],
) -> str:
    """Hash typed array payloads without lossy JSON float conversion."""
    digest = hashlib.sha256()
    for name, value in values:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
        else:
            array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError(f"{name} cannot use an object dtype")
        header = json.dumps(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _selection_contract(
    selected: np.ndarray,
    test_partition: str,
) -> dict[str, object]:
    """Content-address ordered rows and their registered derivation."""
    if test_partition in {
        "selection_holdout", "validation", "calibration"
    }:
        partition_source = "reference_checkpoint_training_split"
        partition_derivation = (
            "grouped_protocol_split_from_reference_checkpoint_config")
    elif test_partition == "all":
        partition_source = "registered_evaluation_file"
        partition_derivation = "all_rows_in_file_order"
    elif test_partition == "held_out_phone":
        partition_source = "registered_phone_holdout"
        partition_derivation = "phone_id_equals_held_out_phone"
    else:
        raise ValueError(f"unsupported selection-contract partition: {test_partition}")
    indices = np.asarray(selected)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("selected row indices must be one-dimensional integers")
    indices = np.ascontiguousarray(indices, dtype=np.dtype("<i8"))
    rows_sha256 = _hash_named_arrays([
        ("selected_row_indices", indices),
    ])
    payload = {
        "test_partition": str(test_partition),
        "partition_source": partition_source,
        "partition_derivation": partition_derivation,
        "rows": int(indices.size),
        "selected_row_indices_dtype": "int64-le",
        "selected_row_indices_sha256": rows_sha256,
    }
    payload["selection_contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _validate_shared_training_indices(
    member_indices: list[np.ndarray],
) -> np.ndarray:
    """Return the common row indices or fail closed on any member mismatch."""
    if not member_indices:
        raise ValueError("shared measurement memory requires member indices")
    normalized = []
    for member_index, values in enumerate(member_indices):
        array = np.asarray(values)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"checkpoint {member_index} training indices must be 1-D "
                "integers"
            )
        array = np.ascontiguousarray(array, dtype=np.int64)
        if bool((array < 0).any()):
            raise ValueError(
                f"checkpoint {member_index} training indices are negative")
        normalized.append(array)
    reference = normalized[0]
    for member_index, values in enumerate(normalized[1:], 1):
        if not np.array_equal(values, reference):
            raise ValueError(
                f"checkpoint {member_index} training indices differ; "
                "shared measurement memory is invalid"
            )
    return reference.copy()


def _shared_memory_contract(
    training_indices: np.ndarray,
    shared_memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seen_mask: torch.Tensor,
) -> dict[str, object]:
    """Validate and hash the exact shared radio-memory sufficient statistic."""
    if len(shared_memory) != 3:
        raise ValueError("shared memory must contain detection, center and scale")
    detection, center, scale = shared_memory
    if not (
        detection.ndim == 2
        and detection.shape == center.shape == scale.shape
    ):
        raise ValueError("shared radio-memory tensors must share shape [N,A]")
    seen = seen_mask.detach().cpu().to(dtype=torch.bool)
    if seen.shape != (detection.shape[1],):
        raise ValueError("seen AP mask does not match shared radio memory")
    if not bool(torch.isfinite(detection).all()) or bool(
        ((detection <= 0.0) | (detection >= 1.0)).any()
    ):
        raise ValueError("shared detection probabilities must lie in (0,1)")
    if not bool(torch.isfinite(center).all()):
        raise ValueError("shared RSS centers must be finite")
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0.0).any()):
        raise ValueError("shared RSS scales must be finite and positive")
    indices = _validate_shared_training_indices([
        np.asarray(training_indices)
    ])
    index_hash = _hash_named_arrays([("training_indices", indices)])
    memory_hash = _hash_named_arrays([
        ("training_indices", indices),
        ("detection_probability", detection),
        ("rss_center", center),
        ("rss_scale", scale),
        ("seen_ap_mask", seen),
    ])
    return {
        "shared_measurement_memory": True,
        "shared_measurement_memory_status": (
            "single shared radio memory; all member training indices "
            "hard-validated"
        ),
        "training_indices_sha256": index_hash,
        "shared_measurement_memory_sha256": memory_hash,
    }


def _gather_shared_measurement_memory(
    shared_memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the already-cropped shared memory on the consensus support."""
    detection, center, scale = shared_memory
    if not (
        detection.ndim == 2
        and detection.shape == center.shape == scale.shape
    ):
        raise ValueError("shared radio-memory tensors must share shape [N,A]")
    selected_detection = detection[candidate_indices]
    selected_center = center[candidate_indices]
    selected_scale = scale[candidate_indices]
    return selected_detection, selected_center, selected_scale


def _checkpoint_identity_contract(
    checkpoints: list[tuple],
    *,
    training_indices: np.ndarray | None = None,
    shared_memory: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Return auditable hashes after ``_validate_checkpoint_alignment``."""
    model, checkpoint = checkpoints[0]
    anchor_payload = {
        "coordinates_m": model.anchor_coords.detach().cpu().tolist(),
        "building": model.anchor_buildings.detach().cpu().tolist(),
        "floor": model.anchor_floors.detach().cpu().tolist(),
        "coordinate_origin_m": np.asarray(
            checkpoint["coord_origin"], dtype=np.float64).tolist(),
    }
    ap_payload = {
        "feature_order": list(checkpoint["stats"]["wap_cols"]),
        "seen_mask": model.encoder.seen_ap_mask.detach().cpu().tolist(),
    }

    def digest(value: dict) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    contract = {
        "status": "all ensemble members hard-validated against member 0",
        "members": int(len(checkpoints)),
        "anchors": int(model.anchor_coords.shape[0]),
        "ap_features": int(model.encoder.seen_ap_mask.numel()),
        "anchor_dictionary_sha256": digest(anchor_payload),
        "ap_feature_order_sha256": digest(ap_payload),
    }
    if (training_indices is None) != (shared_memory is None):
        raise ValueError(
            "training indices and shared memory must be supplied together")
    if training_indices is not None and shared_memory is not None:
        contract.update(_shared_memory_contract(
            training_indices,
            shared_memory,
            model.encoder.seen_ap_mask,
        ))
    return contract


def _rank01(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=values.dtype)
    if len(values) <= 1:
        ranks.fill_(0.0)
        return ranks
    ranks[order] = torch.linspace(
        0.0,
        1.0,
        steps=len(values),
        device=values.device,
        dtype=values.dtype,
    )
    return ranks


def style_cluster_ids(
    query: torch.Tensor,
    group_count: int,
    mode: str,
) -> tuple[torch.Tensor | None, int]:
    """Build unlabeled RSS-style groups inside an evaluation batch."""
    groups = max(int(group_count), 1)
    mode = mode.lower()
    if groups <= 1 or mode == "batch":
        return None, 1
    visible = query > -0.999
    visible_count = visible.sum(dim=1).to(dtype=query.dtype).clamp(min=1.0)
    query_values = torch.where(visible, query, torch.zeros_like(query))
    visible_mean = query_values.sum(dim=1) / visible_count
    if mode == "visible_mean":
        score = visible_mean
    elif mode == "visible_count":
        score = visible_count
    elif mode == "visible_mean_count":
        score = _rank01(visible_mean) + _rank01(visible_count)
    else:
        raise ValueError(
            "style_group_mode must be batch, visible_mean, visible_count or "
            "visible_mean_count")
    order = torch.argsort(score)
    bins = torch.div(
        torch.arange(len(query), device=query.device) * groups,
        max(len(query), 1),
        rounding_mode="floor",
    ).clamp(max=groups - 1)
    group_ids = torch.empty(len(query), device=query.device, dtype=torch.long)
    group_ids[order] = bins.to(dtype=torch.long)
    return group_ids, groups


def _leave_one_phone_out_split(
    coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phones: np.ndarray,
    timestamps: np.ndarray,
    held_out_phone: int,
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic leave-one-phone-out split used by the final audit."""
    held_out = np.flatnonzero(phones == held_out_phone).astype(np.int64)
    development = np.flatnonzero(phones != held_out_phone).astype(np.int64)
    if len(held_out) == 0:
        raise ValueError(f"held-out phone {held_out_phone} has no rows")
    train_local, validation_local, calibration_local = grouped_protocol_split(
        coords[development],
        building[development],
        floor[development],
        validation_fraction,
        calibration_fraction,
        seed,
        timestamps=timestamps[development],
        strategy=strategy,
    )
    return (
        development[train_local],
        development[validation_local],
        development[calibration_local],
        held_out,
    )


def _selection_protocol_partitions(
    coords: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    timestamps: np.ndarray,
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
    strategy: str,
) -> dict[str, np.ndarray]:
    """Map the grouped split to explicit point-selection/evaluation roles.

    The first return from :func:`grouped_protocol_split` is deliberately named
    ``selection_holdout`` here.  For internal model development it is the
    reference-checkpoint training subset and remains disjoint from the grouped
    validation and calibration subsets.
    """
    selection_holdout, validation, calibration = grouped_protocol_split(
        coords,
        building,
        floor,
        validation_fraction,
        calibration_fraction,
        seed,
        timestamps=timestamps,
        strategy=strategy,
    )
    return {
        "selection_holdout": selection_holdout,
        "validation": validation,
        "calibration": calibration,
    }


class ProjectedEnsemble:
    """Multi-member alpha-consensus and single-DAPP localizer.

    用法:
        ensemble = ProjectedEnsemble(
            checkpoints=["seed42.pt", "seed43.pt", ...],
            anchor_data="UJIndoorLoc/trainingData.csv",
            device="auto",
        )
        result = ensemble.evaluate(
            test_data="UJIndoorLoc/validationData.csv",
        )
        print(result["projected_consensus"]["mae"])
    """

    def __init__(
        self,
        checkpoints: list[str | Path],
        anchor_data: str | Path,
        device: str = "auto",
        dataset: str = "auto",
        held_out_phone: int | None = None,
    ) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto"
            else "cpu"
        )
        self.dataset = dataset
        self.held_out_phone = held_out_phone

        # 加载所有 checkpoint.
        self.checkpoints = [
            load_anchor_checkpoint(resolve_path(str(path)), self.device)
            for path in checkpoints
        ]
        _validate_checkpoint_alignment(self.checkpoints)
        self.reference = self.checkpoints[0][1]
        self.checkpoint_paths = [str(path) for path in checkpoints]

        # Reconstruct one empirical radio memory.  Ensemble diversity comes
        # from the frozen predictors, not from duplicating this deterministic
        # sufficient statistic.  Fail closed if any checkpoint would select a
        # different radio-map training row.
        anchor_rss, anchor_coords_norm, anchor_building_t, anchor_floor_t, \
            _anchor_space_t, _anchor_user_t, anchor_phone_t, \
            anchor_timestamp_t, _anchor_stats = load_dataset(
                resolve_path(str(anchor_data)),
                dataset_type=self.dataset,
                reference_stats=self.reference["stats"],
            )
        anchor_xy = (
            raw_coords(
                anchor_coords_norm,
                anchor_building_t,
                self.reference["stats"],
            )
            - self.reference["coord_origin"]
        )
        member_train_indices = []
        for _model, checkpoint in self.checkpoints:
            config = checkpoint["config"]
            if config.get("train_all", False):
                if self.held_out_phone is not None:
                    raise ValueError(
                        "held_out_phone cannot be used with train_all "
                        "checkpoints")
                train_idx = np.arange(len(anchor_rss), dtype=np.int64)
            elif self.held_out_phone is not None:
                train_idx, _, _, _ = _leave_one_phone_out_split(
                    anchor_xy,
                    anchor_building_t.numpy(),
                    anchor_floor_t.numpy(),
                    anchor_phone_t.numpy(),
                    anchor_timestamp_t.numpy(),
                    self.held_out_phone,
                    config.get("validation_fraction", 0.10),
                    config.get("calibration_fraction", 0.10),
                    checkpoint["seed"],
                    config.get("split_strategy", "temporal"),
                )
            else:
                train_idx, _, _ = grouped_protocol_split(
                    anchor_xy,
                    anchor_building_t.numpy(),
                    anchor_floor_t.numpy(),
                    config.get("validation_fraction", 0.10),
                    config.get("calibration_fraction", 0.10),
                    checkpoint["seed"],
                    timestamps=anchor_timestamp_t.numpy(),
                    strategy=config.get("split_strategy", "temporal"),
                )
            member_train_indices.append(train_idx)
        train_idx = _validate_shared_training_indices(member_train_indices)
        self.training_indices = train_idx
        memory, _ = build_anchor_memory(
            anchor_rss,
            anchor_xy,
            anchor_building_t.numpy(),
            anchor_floor_t.numpy(),
            train_idx,
        )
        for member_index, (model, _checkpoint) in enumerate(self.checkpoints):
            if not (
                torch.equal(
                    memory["building"], model.anchor_buildings.cpu())
                and torch.equal(
                    memory["floor"], model.anchor_floors.cpu())
                and torch.allclose(
                    memory["coords"],
                    model.anchor_coords.cpu(),
                    atol=1e-4)
            ):
                raise ValueError(
                    "shared radio memory does not match checkpoint "
                    f"{member_index}"
                )
        full_anchor_memory = (
            memory["detection_prob"],
            memory["rss_center"],
            memory["rss_scale"],
        )
        # Hash the complete sufficient statistic before reducing its physical
        # online representation to the AP coordinates every member shares.
        self.identity_contract = _checkpoint_identity_contract(
            self.checkpoints,
            training_indices=self.training_indices,
            shared_memory=full_anchor_memory,
        )
        common_seen = (
            self.checkpoints[0][0].encoder.seen_ap_mask.detach().cpu())
        self.anchor_memory = tuple(
            value[:, common_seen].to(self.device)
            for value in full_anchor_memory
        )
        # Compatibility for the historical independent and pooling paths:
        # every list element is the exact same tuple/tensor object, not a copy.
        self.anchor_memories = [self.anchor_memory] * len(self.checkpoints)

    def _evaluate_joint_dapp(
        self,
        *,
        test_data: str | Path,
        test_partition: str,
        effective_held_phone: int | None,
        rss: torch.Tensor,
        target: np.ndarray,
        building_t: torch.Tensor,
        floor_t: torch.Tensor,
        space_t: torch.Tensor,
        user_t: torch.Tensor,
        phone_t: torch.Tensor,
        timestamp_t: torch.Tensor,
        selected: np.ndarray,
        batch: int,
        top_k: int,
        member_top_k: int,
        neural_scale: float,
        prototype_weight: float,
        posterior_temperature: float,
        candidate_building_conditioning: bool,
        visibility_weight: float,
        prior_weight: float,
        prior_alpha: float,
        compactness_weight: float,
        style_offset_max: float,
        style_offset_weight: float,
        style_scale_max_log: float,
        style_scale_weight: float,
        visibility_bias_max: float,
        visibility_bias_weight: float,
        reliability_power: float,
        member_weight_mode: str,
        measurement_pool_mode: str,
        support_density_weight: float,
        support_density_sigma: float,
        support_density_margin: float,
        support_density_scope: str,
        support_density_mode: str,
        support_density_step: float,
        support_ood_precision_scale: float,
        support_ood_precision_power: float,
        support_ood_sigma: float,
        support_ood_scope: str,
        support_ood_mode: str,
        rss_centering: str,
        rss_loss_weight: float,
        visibility_positive_factor: float,
        visibility_negative_factor: float,
        mixture_censored_weight: float,
        mixture_visible_factor: float,
        mixture_invisible_factor: float,
        hurdle_distribution: str,
        hurdle_df: float,
        hurdle_score: str,
        hurdle_power_beta: float,
        evidence_temperature_mode: str,
        projection_mode: str,
        steps: int,
        learning_rate: float,
        include_diagnostics: bool,
        compute_predictive_distribution: bool,
        rss_global_offset_db: float,
        rss_dropout_prob: float,
        rss_dropout_seed: int,
        variant_name: str,
        adaptive_kl_alpha: float,
        adaptive_kl_visible_pivot: float,
        adaptive_kl_min_scale: float,
        adaptive_kl_max_scale: float,
        ambiguity_kl_weight: float,
        ambiguity_kl_spread_pivot: float,
        ambiguity_kl_radial_pivot: float,
        ambiguity_kl_max_scale: float,
        discriminative_ap_weight: float,
        discriminative_ap_rss_scale: float,
        discriminative_ap_detection_scale: float,
        discriminative_ap_min: float,
        discriminative_ap_max: float,
        manifold_weight: float,
        manifold_sigma: float,
        manifold_margin: float,
        censored_invisible_weight: float,
        censored_detection_threshold: float,
        censored_rss_threshold: float,
        censored_mode: str,
        censored_rss_softness: float,
        censored_contradiction_margin: float,
        floor_consistency_weight: float,
        gate_entropy_min: float,
        gate_spread_min: float,
        gate_visible_max: float,
        gate_radial_scale_min: float,
        ensemble_consensus_weight: float,
        joint_ensemble_weight: float,
        consensus_prior_mode: str,
        consensus_prior_epsilon: float,
        consensus_prior_alpha: float,
        ap_censored_threshold_mode: str,
        ap_censored_threshold_quantile: float,
        ap_censored_threshold_shrinkage: float,
        ap_censored_threshold_min: float,
        ap_censored_threshold_max: float,
        hierarchical_style: bool,
        style_residual_fraction: float,
        style_group_count: int,
        style_group_mode: str,
    ) -> dict:
        single_consensus = consensus_prior_mode != "none"
        anchor_count = int(
            self.checkpoints[0][0].anchor_coords.shape[0])
        member_top_k = _resolve_member_top_k(
            top_k,
            member_top_k,
            single_consensus=single_consensus,
            anchor_count=anchor_count,
        )
        member_weight_mode = member_weight_mode.lower()
        if member_weight_mode not in {"inverse_scale", "uniform"}:
            raise ValueError(
                "member_weight_mode must be inverse_scale or uniform")
        measurement_pool_mode = measurement_pool_mode.lower()
        effective_measurement_pool = _resolve_measurement_pool(
            measurement_pool_mode, projection_mode)
        if single_consensus:
            if projection_mode not in {
                "dapp",
                "entropy_min",
                "alpha_prox_hurdle",
            }:
                raise ValueError(
                    "consensus projection requires dapp, entropy_min or "
                    "alpha_prox_hurdle")
        elif projection_mode != "dapp":
            raise ValueError("joint ensemble projection requires DAPP")
        if projection_mode == "alpha_prox_hurdle":
            if not single_consensus:
                raise ValueError(
                    "alpha_prox_hurdle requires a single consensus prior")
            if prior_weight <= 0.0:
                raise ValueError(
                    "alpha_prox_hurdle requires positive prior_weight")
            if not 0.0 <= prior_alpha <= 1.0:
                raise ValueError("prior_alpha must be in [0, 1]")
            if hurdle_distribution not in {"gaussian", "student_t"}:
                raise ValueError(
                    "hurdle_distribution must be gaussian or student_t")
            if hurdle_distribution == "student_t" and (
                not np.isfinite(float(hurdle_df)) or float(hurdle_df) <= 0.0
            ):
                raise ValueError(
                    "Student-t hurdle_df must be finite and positive")
            if hurdle_score not in {
                "balanced_log",
                "joint_log",
                "density_power",
            }:
                raise ValueError(
                    "hurdle_score must be balanced_log, joint_log, or "
                    "density_power"
                )
            if hurdle_score == "density_power" and not (
                0.0 < float(hurdle_power_beta) <= 1.0
            ):
                raise ValueError(
                    "hurdle_power_beta must lie in (0, 1]"
                )
            if evidence_temperature_mode not in {
                "none",
                "visibility_ratio",
                "self_entropy_discrepancy",
            }:
                raise ValueError(
                    "evidence_temperature_mode must be none, "
                    "visibility_ratio, or self_entropy_discrepancy"
                )
        if consensus_prior_mode not in {"none", "geometric", "power"}:
            raise ValueError(
                "consensus_prior_mode must be none, geometric or power")
        if not 0.0 <= consensus_prior_alpha <= 1.0:
            raise ValueError("consensus_prior_alpha must be in [0, 1]")
        if single_consensus and joint_ensemble_weight > 0.0:
            raise ValueError(
                "consensus prior and joint disagreement modes are exclusive")
        if not single_consensus and joint_ensemble_weight <= 0.0:
            raise ValueError("joint_ensemble_weight must be positive")
        if single_consensus and ap_censored_threshold_mode != "none":
            raise ValueError(
                "consensus prior currently requires AP threshold mode none")
        if support_density_weight > 0.0 and support_density_step < 1.0:
            raise ValueError(
                "joint ensemble projection does not support partial "
                "support_density_step blending")

        coordinate_dfs = {
            float(model.student_df) for model, _ in self.checkpoints
        }
        if compute_predictive_distribution and len(coordinate_dfs) != 1:
            raise ValueError(
                "predictive distribution requires a common coordinate "
                "Student-t degree of freedom across members")
        coordinate_df = next(iter(coordinate_dfs))

        member_count = len(self.checkpoints)
        result_member_count = 1 if single_consensus else member_count
        per_member = [
            {
                "baseline": [],
                "projected": [],
                "scales": [],
                "ood_factors": [],
                "ood_distances": [],
                "pred_buildings": [],
                "pred_floors": [],
                "projection_kls": [],
                "projection_shifts": [],
                "candidate_diameters": [],
                "pinsker_shift_bounds": [],
                "visible_counts": [],
                "initial_entropies": [],
                "projected_entropies": [],
                "initial_visibility_brier": [],
                "projected_visibility_brier": [],
                "initial_visible_rss_mae_db": [],
                "projected_visible_rss_mae_db": [],
                "initial_visible_rss_rmse_db": [],
                "projected_visible_rss_rmse_db": [],
                "initial_censored_violation_rate": [],
                "projected_censored_violation_rate": [],
                "initial_censored_penalty": [],
                "projected_censored_penalty": [],
                "style_offsets": [],
                "style_scales": [],
                "style_log_scales": [],
                "visibility_biases": [],
                "evidence_temperatures": [],
            }
            for _ in range(result_member_count)
        ]
        consensus_candidate_indices_batches = []
        consensus_candidate_locations_batches = []
        consensus_initial_weights_batches = []
        consensus_projected_weights_batches = []
        consensus_candidate_oracle_batches = []
        predictive_mean_batches = []
        predictive_covariance_batches = []
        predictive_within_covariance_batches = []
        predictive_between_candidate_covariance_batches = []
        predictive_between_member_covariance_batches = []
        predictive_root_trace_batches = []
        predictive_log_prob_batches = []
        consensus_member_weight_batches = []
        point_inference_batch_seconds = []

        for start in range(0, len(rss), batch):
            # The paper latency scope starts with the online RSS batch.  It
            # excludes label-based metrics, bootstrap, reliability readout,
            # and serialization, all of which occur after the decode boundary.
            _synchronize_timing_device(self.device)
            point_inference_start = perf_counter()
            query = rss[start:start + batch].to(self.device)
            consensus_query = None
            candidates_list = []
            reliabilities = []
            support_masks = []
            gate_masks = []
            shared_ap_reliability = None
            if single_consensus and effective_measurement_pool == "shared":
                shared_detection, shared_center, shared_scale = self.anchor_memory
                shared_ap_reliability = _powered_ap_reliability(
                    shared_detection,
                    shared_center,
                    shared_scale,
                    reliability_power,
                )
            for (model, _), (
                anchor_detection_prob,
                anchor_rss_center,
                anchor_rss_scale,
            ) in zip(self.checkpoints, self.anchor_memories):
                if shared_ap_reliability is None:
                    ap_reliability = _powered_ap_reliability(
                        anchor_detection_prob,
                        anchor_rss_center,
                        anchor_rss_scale,
                        reliability_power,
                    )
                else:
                    ap_reliability = shared_ap_reliability
                model.eval()
                candidates = prepare_candidates(
                    model,
                    query,
                    anchor_detection_prob,
                    anchor_rss_center,
                    anchor_rss_scale,
                    member_top_k,
                    neural_scale,
                    prototype_weight,
                    posterior_temperature,
                    candidate_building_conditioning=(
                        candidate_building_conditioning),
                    include_measurement_stats=not (
                        single_consensus
                        and effective_measurement_pool == "shared"
                    ),
                )
                gate_mask = None
                if (
                    gate_entropy_min >= 0.0
                    or gate_spread_min >= 0.0
                    or gate_visible_max >= 0.0
                    or gate_radial_scale_min >= 0.0
                ):
                    gate_terms = []
                    if gate_entropy_min >= 0.0:
                        initial_for_gate = (
                            candidates["initial_weights"].clamp(min=1e-8))
                        initial_entropy = -torch.sum(
                            initial_for_gate * torch.log(initial_for_gate),
                            dim=1,
                        )
                        gate_terms.append(
                            initial_entropy >= float(gate_entropy_min))
                    if gate_spread_min >= 0.0:
                        pairwise_for_gate = torch.cdist(
                            candidates["component_locations"],
                            candidates["component_locations"],
                        )
                        candidate_spread = pairwise_for_gate.amax(dim=(1, 2))
                        gate_terms.append(
                            candidate_spread >= float(gate_spread_min))
                    if gate_visible_max >= 0.0:
                        visible_count_for_gate = (
                            candidates["query"] > -0.999).sum(dim=1)
                        gate_terms.append(
                            visible_count_for_gate
                            <= float(gate_visible_max))
                    if gate_radial_scale_min >= 0.0:
                        gate_terms.append(
                            candidates["radial_scale"]
                            >= float(gate_radial_scale_min))
                    gate_mask = torch.stack(gate_terms, dim=0).any(dim=0)
                support_mask = None
                if not single_consensus:
                    support_mask = support_density_mask(
                        model.anchor_buildings,
                        model.anchor_floors,
                        candidates["predicted_building"],
                        candidates["predicted_floor"],
                        support_density_scope,
                    )
                reliabilities.append(ap_reliability)
                if single_consensus:
                    if consensus_query is None:
                        consensus_query = candidates["query"]
                    candidates = _compact_consensus_candidates(candidates)
                    candidates_list.append((model, candidates))
                else:
                    candidates_list.append((model, candidates))
                    assert support_mask is not None
                    support_masks.append(support_mask)
                    gate_masks.append(gate_mask)

            precision_weights = _member_weight_tensor(
                torch.stack([
                    candidates["radial_scale"]
                    for _, candidates in candidates_list
                ], dim=0),
                member_weight_mode,
            )
            if single_consensus:
                if consensus_query is None:
                    raise RuntimeError("missing shared consensus query")
                with torch.no_grad():
                    initial_stack = torch.stack([
                        c["initial_weights"] for _, c in candidates_list
                    ], dim=0)
                    index_stack = torch.stack([
                        c["candidate_indices"] for _, c in candidates_list
                    ], dim=0)
                    consensus_indices, consensus_initial = (
                        alpha_consensus_prior(
                            initial_stack,
                            index_stack,
                            precision_weights,
                            anchor_count,
                            top_k,
                            epsilon=consensus_prior_epsilon,
                            alpha=(
                                0.0
                                if consensus_prior_mode == "geometric"
                                else consensus_prior_alpha
                            ),
                        )
                    )

                    location_members = []
                    coordinate_scale_members = []
                    detection_sum = None
                    rss_first_sum = None
                    rss_scale_sum = None
                    rss_second_sum = None
                    for member_index, ((model, candidates), memory) in enumerate(
                        zip(candidates_list, self.anchor_memories)
                    ):
                        locations, coordinate_scales = model._components(
                            candidates["embedding"], consensus_indices)
                        detection_memory, center_memory, scale_memory = memory
                        location_members.append(locations)
                        coordinate_scale_members.append(coordinate_scales)
                        if effective_measurement_pool == "shared":
                            continue
                        member_detection = detection_memory[consensus_indices]
                        member_center = center_memory[consensus_indices]
                        member_scale = scale_memory[consensus_indices]
                        member_precision = precision_weights[
                            member_index, :, None, None]
                        weighted_detection = (
                            member_precision * member_detection)
                        first = weighted_detection * member_center
                        scale_first = (
                            weighted_detection * member_scale
                            if effective_measurement_pool == "w2"
                            else None
                        )
                        second = (
                            weighted_detection * (
                                member_scale.square()
                                + member_center.square()
                            )
                            if effective_measurement_pool == "moment_match"
                            else None
                        )
                        if detection_sum is None:
                            detection_sum = weighted_detection
                            rss_first_sum = first
                            rss_scale_sum = scale_first
                            rss_second_sum = second
                        else:
                            detection_sum = detection_sum + weighted_detection
                            rss_first_sum = rss_first_sum + first
                            if effective_measurement_pool == "w2":
                                assert rss_scale_sum is not None
                                assert scale_first is not None
                                rss_scale_sum = rss_scale_sum + scale_first
                            else:
                                assert rss_second_sum is not None
                                assert second is not None
                                rss_second_sum = rss_second_sum + second
                        # Historical regression pools retain only streaming
                        # accumulators.  The final shared path never enters
                        # this member-specific [B,K,A] branch.
                        del (
                            member_detection,
                            member_center,
                            member_scale,
                            weighted_detection,
                            first,
                            scale_first,
                            second,
                        )

                    consensus_locations = weighted_component_mean(
                        torch.stack(location_members, dim=0),
                        precision_weights,
                    )
                    if effective_measurement_pool == "shared":
                        (
                            consensus_detection,
                            consensus_rss_center,
                            consensus_rss_scale,
                        ) = _gather_shared_measurement_memory(
                            self.anchor_memory,
                            consensus_indices,
                        )
                    else:
                        assert detection_sum is not None
                        assert rss_first_sum is not None
                        consensus_detection = detection_sum
                        detected_mass = consensus_detection.clamp(min=1e-6)
                        consensus_rss_center = rss_first_sum / detected_mass
                    if effective_measurement_pool == "w2":
                        # Exact 1-D W2 barycenter of the common robust
                        # location--scale family, conditioned on visibility.
                        assert rss_scale_sum is not None
                        consensus_rss_scale = (
                            rss_scale_sum / detected_mass
                        ).clamp(min=1e-6)
                    elif effective_measurement_pool == "moment_match":
                        assert rss_second_sum is not None
                        consensus_rss_scale = torch.sqrt(
                            (
                                rss_second_sum / detected_mass
                                - consensus_rss_center.square()
                            ).clamp(min=1e-6)
                        )
                    consensus_baseline = torch.sum(
                        consensus_initial.unsqueeze(2) * consensus_locations,
                        dim=1,
                    )
                    baseline_stack = torch.stack([
                        c["baseline"] for _, c in candidates_list
                    ], dim=0)
                    scale_stack = torch.stack([
                        c["radial_scale"] for _, c in candidates_list
                    ], dim=0)
                    between_member = (
                        baseline_stack - consensus_baseline.unsqueeze(0)
                    ).square().sum(dim=2)
                    consensus_radial_scale = torch.sqrt(torch.sum(
                        precision_weights
                        * (scale_stack.square() + between_member),
                        dim=0,
                    ).clamp(min=1e-6))
                    consensus_building = torch.mode(torch.stack([
                        c["predicted_building"] for _, c in candidates_list
                    ], dim=0), dim=0).values
                    consensus_floor = torch.mode(torch.stack([
                        c["predicted_floor"] for _, c in candidates_list
                    ], dim=0), dim=0).values
                    first_model = self.checkpoints[0][0]
                    consensus_candidate_floors = (
                        first_model.anchor_floors[consensus_indices])
                    consensus_support_mask = support_density_mask(
                        first_model.anchor_buildings,
                        first_model.anchor_floors,
                        consensus_building,
                        consensus_floor,
                        support_density_scope,
                    )
                    consensus_reliability = (
                        shared_ap_reliability
                        if shared_ap_reliability is not None
                        else torch.stack(reliabilities, dim=0).mean(dim=0)
                    )
                    gate_mask = None
                    if (
                        gate_entropy_min >= 0.0
                        or gate_spread_min >= 0.0
                        or gate_visible_max >= 0.0
                        or gate_radial_scale_min >= 0.0
                    ):
                        gate_terms = []
                        if gate_entropy_min >= 0.0:
                            entropy = -torch.sum(
                                consensus_initial.clamp(min=1e-8)
                                * torch.log(
                                    consensus_initial.clamp(min=1e-8)),
                                dim=1,
                            )
                            gate_terms.append(
                                entropy >= float(gate_entropy_min))
                        if gate_spread_min >= 0.0:
                            spread = torch.cdist(
                                consensus_locations,
                                consensus_locations,
                            ).amax(dim=(1, 2))
                            gate_terms.append(
                                spread >= float(gate_spread_min))
                        if gate_visible_max >= 0.0:
                            visible_count = (
                                consensus_query > -0.999).sum(dim=1)
                            gate_terms.append(
                                visible_count <= float(gate_visible_max))
                        if gate_radial_scale_min >= 0.0:
                            gate_terms.append(
                                consensus_radial_scale
                                >= float(gate_radial_scale_min))
                        gate_mask = torch.stack(
                            gate_terms, dim=0).any(dim=0)
                    style_group_ids_t, effective_style_group_count = (
                        style_cluster_ids(
                            consensus_query,
                            style_group_count,
                            style_group_mode,
                        )
                    )

                with torch.enable_grad():
                    if projection_mode == "entropy_min":
                        consensus_projected_weights = (
                            project_entropy_min_weights(
                                consensus_initial,
                                prior_weight,
                                steps,
                                learning_rate,
                            )
                        )
                        if include_diagnostics:
                            zeros = torch.zeros(
                                len(consensus_query),
                                device=consensus_query.device,
                                dtype=consensus_query.dtype,
                            )
                            consensus_calibration = {
                                "style_offset": zeros,
                                "style_scale": torch.ones_like(zeros),
                                "style_log_scale": zeros,
                                "visibility_bias": zeros,
                            }
                        else:
                            consensus_calibration = None
                    elif projection_mode == "alpha_prox_hurdle":
                        hurdle_mask = consensus_query > -0.999
                        hurdle_values = torch.where(
                            hurdle_mask,
                            consensus_query,
                            torch.zeros_like(consensus_query),
                        )
                        manifold_energy = None
                        if manifold_weight > 0.0:
                            manifold_energy = initial_support_manifold_energy(
                                consensus_initial,
                                consensus_locations,
                                manifold_sigma,
                                manifold_margin,
                            )
                        consensus_projection_result = (
                            project_alpha_proximal_hurdle(
                                initial_weights=consensus_initial,
                                detection_prob=consensus_detection,
                                rss_center=consensus_rss_center,
                                rss_scale=consensus_rss_scale,
                                query_values=hurdle_values,
                                query_mask=hurdle_mask,
                                reliability=consensus_reliability,
                                prior_weight=prior_weight,
                                alpha=prior_alpha,
                                steps=steps,
                                learning_rate=learning_rate,
                                style_offset_max=style_offset_max,
                                style_scale_max_log=style_scale_max_log,
                                style_offset_weight=style_offset_weight,
                                style_scale_weight=style_scale_weight,
                                visible_factor=mixture_visible_factor,
                                invisible_factor=mixture_invisible_factor,
                                hurdle_distribution=hurdle_distribution,
                                hurdle_df=hurdle_df,
                                hurdle_score=hurdle_score,
                                hurdle_power_beta=hurdle_power_beta,
                                evidence_temperature_mode=(
                                    evidence_temperature_mode
                                ),
                                manifold_energy=manifold_energy,
                                manifold_weight=manifold_weight,
                                return_calibration=include_diagnostics,
                            )
                        )
                        if include_diagnostics:
                            (
                                consensus_projected_weights,
                                consensus_calibration,
                            ) = consensus_projection_result
                        else:
                            consensus_projected_weights = (
                                consensus_projection_result)
                            consensus_calibration = None
                    else:
                        consensus_projection_result = project_candidate_weights(
                            consensus_query,
                            consensus_detection,
                            consensus_rss_center,
                            consensus_rss_scale,
                            consensus_initial,
                            consensus_locations,
                            visibility_weight,
                            prior_weight,
                            compactness_weight,
                            style_offset_max,
                            style_offset_weight,
                            style_scale_max_log,
                            style_scale_weight,
                            visibility_bias_max,
                            visibility_bias_weight,
                            consensus_reliability,
                            steps,
                            learning_rate,
                            first_model.anchor_coords,
                            consensus_support_mask,
                            support_density_weight,
                            support_density_sigma,
                            support_density_margin,
                            support_density_mode,
                            rss_centering,
                            rss_loss_weight=rss_loss_weight,
                            visibility_positive_factor=(
                                visibility_positive_factor),
                            visibility_negative_factor=(
                                visibility_negative_factor),
                            mixture_censored_weight=mixture_censored_weight,
                            mixture_visible_factor=mixture_visible_factor,
                            mixture_invisible_factor=mixture_invisible_factor,
                            adaptive_kl_alpha=adaptive_kl_alpha,
                            adaptive_kl_visible_pivot=adaptive_kl_visible_pivot,
                            adaptive_kl_min_scale=adaptive_kl_min_scale,
                            adaptive_kl_max_scale=adaptive_kl_max_scale,
                            ambiguity_kl_weight=ambiguity_kl_weight,
                            ambiguity_kl_spread_pivot=(
                                ambiguity_kl_spread_pivot),
                            ambiguity_kl_radial_pivot=(
                                ambiguity_kl_radial_pivot),
                            ambiguity_kl_max_scale=ambiguity_kl_max_scale,
                            discriminative_ap_weight=discriminative_ap_weight,
                            discriminative_ap_rss_scale=(
                                discriminative_ap_rss_scale),
                            discriminative_ap_detection_scale=(
                                discriminative_ap_detection_scale),
                            discriminative_ap_min=discriminative_ap_min,
                            discriminative_ap_max=discriminative_ap_max,
                            manifold_weight=manifold_weight,
                            manifold_sigma=manifold_sigma,
                            manifold_margin=manifold_margin,
                            hierarchical_style=hierarchical_style,
                            style_residual_fraction=style_residual_fraction,
                            style_group_ids=style_group_ids_t,
                            style_group_count=effective_style_group_count,
                            censored_invisible_weight=censored_invisible_weight,
                            censored_detection_threshold=(
                                censored_detection_threshold),
                            censored_rss_threshold=censored_rss_threshold,
                            censored_mode=censored_mode,
                            censored_rss_softness=censored_rss_softness,
                            censored_contradiction_margin=(
                                censored_contradiction_margin),
                            candidate_floors=consensus_candidate_floors,
                            predicted_floor=consensus_floor,
                            floor_consistency_weight=floor_consistency_weight,
                            gate_mask=gate_mask,
                            ambiguity_radial_scale=consensus_radial_scale,
                            return_calibration=include_diagnostics,
                        )
                        if include_diagnostics:
                            (
                                consensus_projected_weights,
                                consensus_calibration,
                            ) = consensus_projection_result
                        else:
                            consensus_projected_weights = (
                                consensus_projection_result)
                            consensus_calibration = None
                consensus_coordinates = torch.sum(
                    consensus_projected_weights.unsqueeze(2)
                    * consensus_locations,
                    dim=1,
                )
                _synchronize_timing_device(self.device)
                point_inference_batch_seconds.append(
                    perf_counter() - point_inference_start)

                if include_diagnostics:
                    consensus_member_weight_batches.append(
                        precision_weights.detach().cpu().numpy())
                batch_target = None
                if compute_predictive_distribution or include_diagnostics:
                    batch_target = torch.as_tensor(
                        target[start:start + len(query)],
                        device=consensus_locations.device,
                        dtype=consensus_locations.dtype,
                    )
                if compute_predictive_distribution:
                    assert batch_target is not None
                    predictive = student_t_mixture_moments(
                        torch.stack(location_members, dim=0),
                        torch.stack(coordinate_scale_members, dim=0),
                        precision_weights,
                        consensus_projected_weights,
                        coordinate_df,
                        batch_target,
                    )
                    if not torch.allclose(
                        predictive.mean,
                        consensus_coordinates,
                        atol=2e-5,
                        rtol=2e-5,
                    ):
                        raise RuntimeError(
                            "predictive mixture mean does not match "
                            "AC-RHAP decode")
                    predictive_mean_batches.append(
                        predictive.mean.detach().cpu().numpy())
                    predictive_covariance_batches.append(
                        predictive.covariance.detach().cpu().numpy())
                    predictive_within_covariance_batches.append(
                        predictive.within_covariance.detach().cpu().numpy())
                    predictive_between_candidate_covariance_batches.append(
                        predictive.between_candidate_covariance.detach()
                        .cpu().numpy())
                    predictive_between_member_covariance_batches.append(
                        predictive.between_member_covariance.detach()
                        .cpu().numpy())
                    predictive_root_trace_batches.append(
                        predictive.root_trace_scale.detach().cpu().numpy())
                    assert predictive.log_prob is not None
                    predictive_log_prob_batches.append(
                        predictive.log_prob.detach().cpu().numpy())
                if include_diagnostics:
                    assert batch_target is not None
                    initial = consensus_initial.clamp(min=1e-8)
                    projected_weight = consensus_projected_weights.clamp(
                        min=1e-8)
                    projection_kl = torch.sum(
                        projected_weight
                        * (torch.log(projected_weight) - torch.log(initial)),
                        dim=1,
                    )
                    initial_entropy = -torch.sum(
                        initial * torch.log(initial), dim=1)
                    projected_entropy = -torch.sum(
                        projected_weight * torch.log(projected_weight), dim=1)
                    projection_shift = torch.linalg.norm(
                        consensus_coordinates - consensus_baseline, dim=1)
                    candidate_diameter = torch.cdist(
                        consensus_locations,
                        consensus_locations,
                    ).amax(dim=(1, 2))
                    pinsker_shift_bound = candidate_diameter * torch.sqrt(
                        0.5 * projection_kl.clamp(min=0.0))
                    visible_count = (consensus_query > -0.999).sum(dim=1)
                    visible_mask = consensus_query > -0.999

                    def _consensus_measurement_terms(
                        posterior_weights: torch.Tensor,
                        style_offset: torch.Tensor | None = None,
                        style_scale: torch.Tensor | None = None,
                    ) -> dict[str, torch.Tensor]:
                        predicted_detection = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * consensus_detection,
                            dim=1,
                        )
                        diagnostic_center = consensus_rss_center
                        if style_scale is not None:
                            diagnostic_center = (
                                style_scale[:, None, None]
                                * diagnostic_center
                            )
                        if style_offset is not None:
                            diagnostic_center = (
                                diagnostic_center
                                + style_offset[:, None, None]
                            )
                        rss_mass = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * consensus_detection
                            * diagnostic_center,
                            dim=1,
                        )
                        predicted_rss = (
                            rss_mass / predicted_detection.clamp(min=0.05))
                        visible_float = visible_mask.float()
                        invisible_mask = ~visible_mask
                        invisible_float = invisible_mask.float()
                        visible_den = visible_float.sum(dim=1).clamp(min=1.0)
                        invisible_den = invisible_float.sum(dim=1).clamp(
                            min=1.0)
                        visibility_brier = (
                            predicted_detection - visible_float
                        ).square().mean(dim=1)
                        visible_rss_error = (
                            predicted_rss - consensus_query) * visible_float
                        visible_rss_mae_db = (
                            visible_rss_error.abs().sum(dim=1)
                            / visible_den
                            * 40.0
                        )
                        visible_rss_rmse_db = torch.sqrt(
                            visible_rss_error.square().sum(dim=1)
                            / visible_den
                        ) * 40.0
                        censored_violation = (
                            (predicted_detection
                             > float(censored_detection_threshold))
                            & (predicted_rss > float(censored_rss_threshold))
                            & invisible_mask
                        )
                        censored_violation_rate = (
                            censored_violation.float().sum(dim=1)
                            / invisible_den
                        )
                        censored_penalty = (
                            torch.relu(
                                predicted_detection
                                - float(censored_detection_threshold)
                            ).square()
                            + predicted_detection.clamp(0.0, 1.0)
                            * torch.relu(
                                predicted_rss
                                - float(censored_rss_threshold)
                            ).square()
                        )
                        censored_penalty = (
                            (censored_penalty * invisible_float).sum(dim=1)
                            / invisible_den
                        )
                        return {
                            "visibility_brier": visibility_brier,
                            "visible_rss_mae_db": visible_rss_mae_db,
                            "visible_rss_rmse_db": visible_rss_rmse_db,
                            "censored_violation_rate": (
                                censored_violation_rate),
                            "censored_penalty": censored_penalty,
                        }

                    initial_measurement = _consensus_measurement_terms(initial)
                    projected_measurement = _consensus_measurement_terms(
                        projected_weight,
                        style_offset=consensus_calibration["style_offset"],
                        style_scale=consensus_calibration["style_scale"],
                    )
                    per_member[0]["projection_kls"].append(
                        projection_kl.detach().cpu().numpy())
                    per_member[0]["projection_shifts"].append(
                        projection_shift.detach().cpu().numpy())
                    per_member[0]["candidate_diameters"].append(
                        candidate_diameter.detach().cpu().numpy())
                    per_member[0]["pinsker_shift_bounds"].append(
                        pinsker_shift_bound.detach().cpu().numpy())
                    per_member[0]["visible_counts"].append(
                        visible_count.detach().cpu().numpy())
                    per_member[0]["initial_entropies"].append(
                        initial_entropy.detach().cpu().numpy())
                    per_member[0]["projected_entropies"].append(
                        projected_entropy.detach().cpu().numpy())
                    for key, destination in (
                        ("style_offset", "style_offsets"),
                        ("style_scale", "style_scales"),
                        ("style_log_scale", "style_log_scales"),
                        ("visibility_bias", "visibility_biases"),
                    ):
                        per_member[0][destination].append(
                            consensus_calibration[key].detach().cpu().numpy())
                    per_member[0]["evidence_temperatures"].append(
                        consensus_calibration.get(
                            "evidence_temperature",
                            torch.ones_like(
                                consensus_calibration["style_offset"]
                            ),
                        ).detach().cpu().numpy()
                    )
                    for prefix, measurement in (
                        ("initial", initial_measurement),
                        ("projected", projected_measurement),
                    ):
                        for name in (
                            "visibility_brier",
                            "visible_rss_mae_db",
                            "visible_rss_rmse_db",
                            "censored_violation_rate",
                            "censored_penalty",
                        ):
                            per_member[0][f"{prefix}_{name}"].append(
                                measurement[name].detach().cpu().numpy())
                    candidate_oracle_error = torch.linalg.vector_norm(
                        consensus_locations - batch_target.unsqueeze(1),
                        dim=2,
                    ).amin(dim=1)
                    consensus_candidate_indices_batches.append(
                        consensus_indices.detach().cpu().numpy())
                    consensus_candidate_locations_batches.append(
                        consensus_locations.detach().cpu().numpy())
                    consensus_initial_weights_batches.append(
                        consensus_initial.detach().cpu().numpy())
                    consensus_projected_weights_batches.append(
                        consensus_projected_weights.detach().cpu().numpy())
                    consensus_candidate_oracle_batches.append(
                        candidate_oracle_error.detach().cpu().numpy())
                if (
                    include_diagnostics
                    or (
                        support_ood_precision_scale > 0.0
                        and support_ood_precision_power > 0.0
                    )
                ):
                    ood_mask = support_density_mask(
                        first_model.anchor_buildings,
                        first_model.anchor_floors,
                        consensus_building,
                        consensus_floor,
                        support_ood_scope,
                    )
                    ood_energy = support_density_energy(
                        consensus_coordinates,
                        first_model.anchor_coords,
                        ood_mask,
                        support_ood_sigma,
                        0.0,
                        support_ood_mode,
                    )
                    ood_distance = (ood_energy * 100.0).clamp(
                        min=0.0).sqrt()
                else:
                    ood_distance = torch.zeros_like(consensus_radial_scale)
                if (
                    support_ood_precision_scale > 0.0
                    and support_ood_precision_power > 0.0
                ):
                    ood_scale = max(
                        float(support_ood_precision_scale), 1e-6)
                    factor = 1.0 / (
                        1.0
                        + (ood_distance / ood_scale).pow(
                            float(support_ood_precision_power))
                    )
                else:
                    factor = torch.ones_like(consensus_radial_scale)
                per_member[0]["baseline"].append(
                    consensus_baseline.detach().cpu().numpy())
                per_member[0]["projected"].append(
                    consensus_coordinates.detach().cpu().numpy())
                per_member[0]["scales"].append(
                    consensus_radial_scale.detach().cpu().numpy())
                per_member[0]["ood_factors"].append(
                    factor.detach().cpu().numpy())
                per_member[0]["ood_distances"].append(
                    ood_distance.detach().cpu().numpy())
                per_member[0]["pred_buildings"].append(
                    consensus_building.detach().cpu().numpy())
                per_member[0]["pred_floors"].append(
                    consensus_floor.detach().cpu().numpy())
                continue
            with torch.enable_grad():
                projection_result = project_joint_candidate_weights(
                    [c["query"] for _, c in candidates_list],
                    [c["detection_prob"] for _, c in candidates_list],
                    [c["rss_center"] for _, c in candidates_list],
                    [c["rss_scale"] for _, c in candidates_list],
                    [c["initial_weights"] for _, c in candidates_list],
                    [c["component_locations"] for _, c in candidates_list],
                    reliabilities,
                    visibility_weight,
                    prior_weight,
                    compactness_weight,
                    style_offset_max,
                    style_offset_weight,
                    style_scale_max_log,
                    style_scale_weight,
                    visibility_bias_max,
                    visibility_bias_weight,
                    steps,
                    learning_rate,
                    joint_ensemble_weight,
                    precision_weights=precision_weights,
                    support_locations_list=[
                        model.anchor_coords for model, _ in candidates_list
                    ],
                    support_masks=support_masks,
                    support_density_weight=support_density_weight,
                    support_density_sigma=support_density_sigma,
                    support_density_margin=support_density_margin,
                    support_density_mode=support_density_mode,
                    rss_centering=rss_centering,
                    rss_loss_weight=rss_loss_weight,
                    visibility_positive_factor=visibility_positive_factor,
                    visibility_negative_factor=visibility_negative_factor,
                    mixture_censored_weight=mixture_censored_weight,
                    mixture_visible_factor=mixture_visible_factor,
                    mixture_invisible_factor=mixture_invisible_factor,
                    adaptive_kl_alpha=adaptive_kl_alpha,
                    adaptive_kl_visible_pivot=adaptive_kl_visible_pivot,
                    adaptive_kl_min_scale=adaptive_kl_min_scale,
                    adaptive_kl_max_scale=adaptive_kl_max_scale,
                    ambiguity_kl_weight=ambiguity_kl_weight,
                    ambiguity_kl_spread_pivot=ambiguity_kl_spread_pivot,
                    ambiguity_kl_radial_pivot=ambiguity_kl_radial_pivot,
                    ambiguity_kl_max_scale=ambiguity_kl_max_scale,
                    discriminative_ap_weight=discriminative_ap_weight,
                    discriminative_ap_rss_scale=discriminative_ap_rss_scale,
                    discriminative_ap_detection_scale=(
                        discriminative_ap_detection_scale),
                    discriminative_ap_min=discriminative_ap_min,
                    discriminative_ap_max=discriminative_ap_max,
                    manifold_weight=manifold_weight,
                    manifold_sigma=manifold_sigma,
                    manifold_margin=manifold_margin,
                    censored_invisible_weight=censored_invisible_weight,
                    censored_detection_threshold=censored_detection_threshold,
                    censored_rss_threshold=censored_rss_threshold,
                    censored_mode=censored_mode,
                    censored_rss_softness=censored_rss_softness,
                    censored_contradiction_margin=(
                        censored_contradiction_margin),
                    candidate_floors_list=[
                        c["candidate_floors"] for _, c in candidates_list
                    ],
                    predicted_floors=[
                        c["predicted_floor"] for _, c in candidates_list
                    ],
                    floor_consistency_weight=floor_consistency_weight,
                    gate_masks=gate_masks,
                    radial_scales_list=[
                        c["radial_scale"] for _, c in candidates_list
                    ],
                    return_calibration=include_diagnostics,
                )
                if include_diagnostics:
                    weights_list, calibration_values = projection_result
                else:
                    weights_list = projection_result
                    calibration_values = None

            for member_index, ((model, candidates), weights) in enumerate(
                zip(candidates_list, weights_list)
            ):
                coordinates = torch.sum(
                    weights.unsqueeze(2)
                    * candidates["component_locations"],
                    dim=1,
                )
                if include_diagnostics:
                    initial = candidates["initial_weights"].clamp(min=1e-8)
                    projected_weight = weights.clamp(min=1e-8)
                    projection_kl = torch.sum(
                        projected_weight
                        * (torch.log(projected_weight) - torch.log(initial)),
                        dim=1,
                    )
                    initial_entropy = -torch.sum(
                        initial * torch.log(initial), dim=1)
                    projected_entropy = -torch.sum(
                        projected_weight * torch.log(projected_weight), dim=1)
                    projection_shift = torch.linalg.norm(
                        coordinates - candidates["baseline"], dim=1)
                    pairwise_candidate_distance = torch.cdist(
                        candidates["component_locations"],
                        candidates["component_locations"],
                    )
                    candidate_diameter = pairwise_candidate_distance.amax(
                        dim=(1, 2))
                    pinsker_shift_bound = candidate_diameter * torch.sqrt(
                        0.5 * projection_kl.clamp(min=0.0))
                    visible_count = (candidates["query"] > -0.999).sum(dim=1)
                    visible_mask = candidates["query"] > -0.999

                    def _measurement_terms(
                        posterior_weights: torch.Tensor,
                    ) -> dict[str, torch.Tensor]:
                        predicted_detection = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * candidates["detection_prob"],
                            dim=1,
                        )
                        rss_mass = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * candidates["detection_prob"]
                            * candidates["rss_center"],
                            dim=1,
                        )
                        predicted_rss = (
                            rss_mass
                            / predicted_detection.clamp(min=0.05)
                        )
                        visible_float = visible_mask.float()
                        invisible_mask = ~visible_mask
                        invisible_float = invisible_mask.float()
                        visible_den = visible_float.sum(dim=1).clamp(min=1.0)
                        invisible_den = (
                            invisible_float.sum(dim=1).clamp(min=1.0))
                        visibility_brier = (
                            predicted_detection - visible_float
                        ).square().mean(dim=1)
                        visible_rss_error = (
                            predicted_rss - candidates["query"]
                        ) * visible_float
                        visible_rss_mae_db = (
                            visible_rss_error.abs().sum(dim=1)
                            / visible_den
                            * 40.0
                        )
                        visible_rss_rmse_db = torch.sqrt(
                            visible_rss_error.square().sum(dim=1)
                            / visible_den
                        ) * 40.0
                        censored_violation = (
                            (predicted_detection
                             > float(censored_detection_threshold))
                            & (predicted_rss > float(censored_rss_threshold))
                            & invisible_mask
                        )
                        censored_violation_rate = (
                            censored_violation.float().sum(dim=1)
                            / invisible_den
                        )
                        censored_penalty = (
                            torch.relu(
                                predicted_detection
                                - float(censored_detection_threshold)
                            ).square()
                            + predicted_detection.clamp(0.0, 1.0)
                            * torch.relu(
                                predicted_rss
                                - float(censored_rss_threshold)
                            ).square()
                        )
                        censored_penalty = (
                            (censored_penalty * invisible_float).sum(dim=1)
                            / invisible_den
                        )
                        return {
                            "visibility_brier": visibility_brier,
                            "visible_rss_mae_db": visible_rss_mae_db,
                            "visible_rss_rmse_db": visible_rss_rmse_db,
                            "censored_violation_rate": (
                                censored_violation_rate),
                            "censored_penalty": censored_penalty,
                        }

                    initial_measurement = _measurement_terms(initial)
                    projected_measurement = _measurement_terms(projected_weight)
                    per_member[member_index]["projection_kls"].append(
                        projection_kl.detach().cpu().numpy())
                    per_member[member_index]["projection_shifts"].append(
                        projection_shift.detach().cpu().numpy())
                    per_member[member_index]["candidate_diameters"].append(
                        candidate_diameter.detach().cpu().numpy())
                    per_member[member_index]["pinsker_shift_bounds"].append(
                        pinsker_shift_bound.detach().cpu().numpy())
                    per_member[member_index]["visible_counts"].append(
                        visible_count.detach().cpu().numpy())
                    per_member[member_index]["initial_entropies"].append(
                        initial_entropy.detach().cpu().numpy())
                    per_member[member_index]["projected_entropies"].append(
                        projected_entropy.detach().cpu().numpy())
                    if calibration_values is not None:
                        per_member[member_index]["style_offsets"].append(
                            calibration_values["style_offset"][member_index]
                            .detach().cpu().numpy())
                        per_member[member_index]["style_scales"].append(
                            calibration_values["style_scale"][member_index]
                            .detach().cpu().numpy())
                        per_member[member_index]["style_log_scales"].append(
                            calibration_values["style_log_scale"][member_index]
                            .detach().cpu().numpy())
                        per_member[member_index]["visibility_biases"].append(
                            calibration_values["visibility_bias"][member_index]
                            .detach().cpu().numpy())
                    for prefix, measurement in (
                        ("initial", initial_measurement),
                        ("projected", projected_measurement),
                    ):
                        per_member[member_index][
                            f"{prefix}_visibility_brier"].append(
                                measurement[
                                    "visibility_brier"].detach().cpu().numpy())
                        per_member[member_index][
                            f"{prefix}_visible_rss_mae_db"].append(
                                measurement[
                                    "visible_rss_mae_db"].detach().cpu().numpy())
                        per_member[member_index][
                            f"{prefix}_visible_rss_rmse_db"].append(
                                measurement[
                                    "visible_rss_rmse_db"].detach()
                                    .cpu().numpy())
                        per_member[member_index][
                            f"{prefix}_censored_violation_rate"].append(
                                measurement[
                                    "censored_violation_rate"].detach()
                                    .cpu().numpy())
                        per_member[member_index][
                            f"{prefix}_censored_penalty"].append(
                                measurement[
                                    "censored_penalty"].detach().cpu().numpy())

                if (
                    include_diagnostics
                    or (
                        support_ood_precision_scale > 0.0
                        and support_ood_precision_power > 0.0
                    )
                ):
                    ood_mask = support_density_mask(
                        model.anchor_buildings,
                        model.anchor_floors,
                        candidates["predicted_building"],
                        candidates["predicted_floor"],
                        support_ood_scope,
                    )
                    ood_energy = support_density_energy(
                        coordinates,
                        model.anchor_coords,
                        ood_mask,
                        support_ood_sigma,
                        0.0,
                        support_ood_mode,
                    )
                    ood_distance = (ood_energy * 100.0).clamp(
                        min=0.0).sqrt()
                else:
                    ood_distance = torch.zeros(
                        len(query),
                        device=coordinates.device,
                        dtype=coordinates.dtype,
                    )
                if (
                    support_ood_precision_scale > 0.0
                    and support_ood_precision_power > 0.0
                ):
                    scale = max(float(support_ood_precision_scale), 1e-6)
                    factor = 1.0 / (
                        1.0
                        + (ood_distance / scale).pow(
                            float(support_ood_precision_power))
                    )
                else:
                    factor = torch.ones(
                        len(query),
                        device=coordinates.device,
                        dtype=coordinates.dtype,
                    )

                per_member[member_index]["baseline"].append(
                    candidates["baseline"].cpu().numpy())
                per_member[member_index]["projected"].append(
                    coordinates.cpu().numpy())
                per_member[member_index]["scales"].append(
                    candidates["radial_scale"].cpu().numpy())
                per_member[member_index]["ood_factors"].append(
                    factor.detach().cpu().numpy())
                per_member[member_index]["ood_distances"].append(
                    ood_distance.detach().cpu().numpy())
                per_member[member_index]["pred_buildings"].append(
                    candidates["predicted_building"].cpu().numpy())
                per_member[member_index]["pred_floors"].append(
                    candidates["predicted_floor"].cpu().numpy())

        baseline_members_np = np.stack([
            np.concatenate(values["baseline"]) for values in per_member
        ]).astype(np.float64)
        projected_members_np = np.stack([
            np.concatenate(values["projected"]) for values in per_member
        ]).astype(np.float64)
        member_scales_np = np.stack([
            np.concatenate(values["scales"]) for values in per_member
        ]).astype(np.float64)
        member_ood_factors_np = np.stack([
            np.concatenate(values["ood_factors"]) for values in per_member
        ]).astype(np.float64)
        member_ood_distances_np = np.stack([
            np.concatenate(values["ood_distances"]) for values in per_member
        ]).astype(np.float64)
        member_pred_buildings_np = np.stack([
            np.concatenate(values["pred_buildings"]) for values in per_member
        ])
        member_pred_floors_np = np.stack([
            np.concatenate(values["pred_floors"]) for values in per_member
        ])

        from scipy import stats as _scipy_stats
        ensemble_building, _ = _scipy_stats.mode(
            member_pred_buildings_np, axis=0, keepdims=False)
        ensemble_floor, _ = _scipy_stats.mode(
            member_pred_floors_np, axis=0, keepdims=False)
        gt_building = building_t[selected].numpy()
        gt_floor = floor_t[selected].numpy()
        building_acc = float((ensemble_building == gt_building).mean())
        floor_acc = float((ensemble_floor == gt_floor).mean())
        if member_weight_mode == "uniform":
            precision = np.ones_like(member_scales_np)
        else:
            precision = 1.0 / np.maximum(member_scales_np, 1e-6) ** 2
        precision *= np.clip(member_ood_factors_np, 1e-6, 1.0)
        precision /= precision.sum(axis=0, keepdims=True)
        if ensemble_consensus_weight > 0.0:
            consensus_step = float(np.clip(
                ensemble_consensus_weight, 0.0, 1.0))
            consensus_center = np.sum(
                projected_members_np * precision[..., None], axis=0)
            projected_members_np = (
                (1.0 - consensus_step) * projected_members_np
                + consensus_step * consensus_center[None, :, :]
            )

        if single_consensus:
            # The final path has exactly one consensus coordinate.  Assign it
            # directly: no coordinate-level Huber/geometric fusion is run.
            baseline_huber = baseline_members_np[0]
            projected_huber = projected_members_np[0]
            baseline_geometric_median = baseline_huber
            projected_geometric_median = projected_huber
        else:
            # Backward-compatible independent-member ablation.
            baseline_huber = huber_precision_center(
                baseline_members_np, precision)
            projected_huber = huber_precision_center(
                projected_members_np, precision)
            baseline_geometric_median = weighted_geometric_median(
                baseline_members_np, precision)
            projected_geometric_median = weighted_geometric_median(
                projected_members_np, precision)

        baseline_metrics, baseline_errors = coordinate_metrics(
            baseline_huber, target)
        projected_metrics, projected_errors = coordinate_metrics(
            projected_huber, target)
        baseline_geometric_metrics, baseline_geometric_errors = (
            coordinate_metrics(baseline_geometric_median, target)
        )
        projected_geometric_metrics, projected_geometric_errors = (
            coordinate_metrics(projected_geometric_median, target)
        )

        predictive_distribution = None
        predictive_mean_np = None
        predictive_covariance_np = None
        predictive_within_covariance_np = None
        predictive_between_candidate_covariance_np = None
        predictive_between_member_covariance_np = None
        predictive_root_trace_np = None
        predictive_nll_np = None
        predictive_mahalanobis_np = None
        predictive_base_area_np = None
        if compute_predictive_distribution:
            predictive_mean_np = np.concatenate(
                predictive_mean_batches).astype(np.float64)
            predictive_covariance_np = np.concatenate(
                predictive_covariance_batches).astype(np.float64)
            predictive_within_covariance_np = np.concatenate(
                predictive_within_covariance_batches).astype(np.float64)
            predictive_between_candidate_covariance_np = np.concatenate(
                predictive_between_candidate_covariance_batches
            ).astype(np.float64)
            predictive_between_member_covariance_np = np.concatenate(
                predictive_between_member_covariance_batches
            ).astype(np.float64)
            predictive_root_trace_np = np.concatenate(
                predictive_root_trace_batches).astype(np.float64)
            predictive_nll_np = -np.concatenate(
                predictive_log_prob_batches).astype(np.float64)
            predictive_mahalanobis_np = mahalanobis_scores(
                target, predictive_mean_np, predictive_covariance_np)
            predictive_base_area_np = covariance_ellipse_area(
                predictive_covariance_np, 1.0)
            root_trace_rank = _scipy_stats.spearmanr(
                predictive_root_trace_np, projected_errors).statistic
            area_rank = _scipy_stats.spearmanr(
                predictive_base_area_np, projected_errors).statistic
            root_trace_rank = (
                float(root_trace_rank) if np.isfinite(root_trace_rank) else 0.0)
            area_rank = float(area_rank) if np.isfinite(area_rank) else 0.0
            total_trace = np.trace(
                predictive_covariance_np, axis1=1, axis2=2)
            within_trace = np.trace(
                predictive_within_covariance_np, axis1=1, axis2=2)
            between_candidate_trace = np.trace(
                predictive_between_candidate_covariance_np, axis1=1, axis2=2)
            between_member_trace = np.trace(
                predictive_between_member_covariance_np, axis1=1, axis2=2)
            aggregate_trace = max(float(total_trace.sum()), 1e-12)
            predictive_distribution = {
                "family": (
                    "member-by-anchor finite mixture of independent-axis "
                    "Student-t components"),
                "degrees_freedom": float(coordinate_df),
                "mixture_weights": (
                    "member_precision_times_alpha_prox_anchor"
                    if member_weight_mode == "inverse_scale"
                    else "uniform_member_times_alpha_prox_anchor"
                ),
                "member_weight_mode": member_weight_mode,
                "member_count": int(member_count),
                "point_decode": "exact_mixture_mean",
                "covariance": (
                    "exact finite-mixture second moment; not an elliptical "
                    "distributional assumption"),
                "covariance_decomposition": {
                    "identity": (
                        "total = within_component + between_candidate + "
                        "between_member"),
                    "mean_trace_m2": {
                        "total": float(total_trace.mean()),
                        "within_component": float(within_trace.mean()),
                        "between_candidate": float(
                            between_candidate_trace.mean()),
                        "between_member": float(between_member_trace.mean()),
                    },
                    "aggregate_trace_fraction": {
                        "within_component": float(
                            within_trace.sum() / aggregate_trace),
                        "between_candidate": float(
                            between_candidate_trace.sum() / aggregate_trace),
                        "between_member": float(
                            between_member_trace.sum() / aggregate_trace),
                    },
                },
                "mean_nll_nats": float(predictive_nll_np.mean()),
                "median_nll_nats": float(np.median(predictive_nll_np)),
                "p90_nll_nats": float(np.percentile(predictive_nll_np, 90)),
                "mean_root_trace_scale_m": float(
                    predictive_root_trace_np.mean()),
                "median_root_trace_scale_m": float(
                    np.median(predictive_root_trace_np)),
                "p90_root_trace_scale_m": float(
                    np.percentile(predictive_root_trace_np, 90)),
                "mean_unit_ellipse_area_m2": float(
                    predictive_base_area_np.mean()),
                "median_unit_ellipse_area_m2": float(
                    np.median(predictive_base_area_np)),
                "mean_mahalanobis_score": float(
                    predictive_mahalanobis_np.mean()),
                "spearman_root_trace_scale_vs_error": float(
                    root_trace_rank),
                "spearman_area_vs_error": float(area_rank),
            }

        projection_diagnostics = None
        measurement_diagnostics = None
        diagnostic_arrays = {}
        if include_diagnostics:
            member_projection_kls_np = np.stack([
                np.concatenate(values["projection_kls"])
                for values in per_member
            ]).astype(np.float64)
            member_projection_shifts_np = np.stack([
                np.concatenate(values["projection_shifts"])
                for values in per_member
            ]).astype(np.float64)
            member_candidate_diameters_np = np.stack([
                np.concatenate(values["candidate_diameters"])
                for values in per_member
            ]).astype(np.float64)
            member_pinsker_shift_bounds_np = np.stack([
                np.concatenate(values["pinsker_shift_bounds"])
                for values in per_member
            ]).astype(np.float64)
            member_visible_counts_np = np.stack([
                np.concatenate(values["visible_counts"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_entropies_np = np.stack([
                np.concatenate(values["initial_entropies"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_entropies_np = np.stack([
                np.concatenate(values["projected_entropies"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_visibility_brier_np = np.stack([
                np.concatenate(values["initial_visibility_brier"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_visibility_brier_np = np.stack([
                np.concatenate(values["projected_visibility_brier"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_visible_rss_mae_db_np = np.stack([
                np.concatenate(values["initial_visible_rss_mae_db"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_visible_rss_mae_db_np = np.stack([
                np.concatenate(values["projected_visible_rss_mae_db"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_visible_rss_rmse_db_np = np.stack([
                np.concatenate(values["initial_visible_rss_rmse_db"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_visible_rss_rmse_db_np = np.stack([
                np.concatenate(values["projected_visible_rss_rmse_db"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_censored_violation_rate_np = np.stack([
                np.concatenate(values["initial_censored_violation_rate"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_censored_violation_rate_np = np.stack([
                np.concatenate(values["projected_censored_violation_rate"])
                for values in per_member
            ]).astype(np.float64)
            member_initial_censored_penalty_np = np.stack([
                np.concatenate(values["initial_censored_penalty"])
                for values in per_member
            ]).astype(np.float64)
            member_projected_censored_penalty_np = np.stack([
                np.concatenate(values["projected_censored_penalty"])
                for values in per_member
            ]).astype(np.float64)
            member_style_offsets_np = np.stack([
                np.concatenate(values["style_offsets"])
                for values in per_member
            ]).astype(np.float64)
            member_style_scales_np = np.stack([
                np.concatenate(values["style_scales"])
                for values in per_member
            ]).astype(np.float64)
            member_style_log_scales_np = np.stack([
                np.concatenate(values["style_log_scales"])
                for values in per_member
            ]).astype(np.float64)
            member_visibility_biases_np = np.stack([
                np.concatenate(values["visibility_biases"])
                for values in per_member
            ]).astype(np.float64)
            member_evidence_temperatures_np = np.stack([
                np.concatenate(values["evidence_temperatures"])
                for values in per_member
            ]).astype(np.float64)

            sample_kl = member_projection_kls_np.mean(axis=0)
            sample_shift = member_projection_shifts_np.mean(axis=0)
            sample_candidate_diameter = (
                member_candidate_diameters_np.mean(axis=0))
            sample_pinsker_shift_bound = (
                member_pinsker_shift_bounds_np.mean(axis=0))
            sample_visible = member_visible_counts_np.mean(axis=0)
            sample_entropy_delta = (
                member_projected_entropies_np - member_initial_entropies_np
            ).mean(axis=0)
            sample_style_offset = member_style_offsets_np.mean(axis=0)
            sample_style_scale = member_style_scales_np.mean(axis=0)
            sample_style_log_scale = member_style_log_scales_np.mean(axis=0)
            sample_visibility_bias = member_visibility_biases_np.mean(axis=0)
            sample_evidence_temperature = (
                member_evidence_temperatures_np.mean(axis=0)
            )

            def _summarize(indices: np.ndarray) -> dict:
                return {
                    "n": int(len(indices)),
                    "visible_ap_mean": float(sample_visible[indices].mean()),
                    "base_mae": float(baseline_errors[indices].mean()),
                    "dapp_mae": float(projected_errors[indices].mean()),
                    "delta_mae": float(
                        projected_errors[indices].mean()
                        - baseline_errors[indices].mean()),
                    "kl_mean": float(sample_kl[indices].mean()),
                    "kl_p90": float(np.percentile(sample_kl[indices], 90)),
                    "shift_mean_m": float(sample_shift[indices].mean()),
                    "shift_p90_m": float(
                        np.percentile(sample_shift[indices], 90)),
                    "candidate_diameter_mean_m": float(
                        sample_candidate_diameter[indices].mean()),
                    "candidate_diameter_p90_m": float(np.percentile(
                        sample_candidate_diameter[indices], 90)),
                    "pinsker_shift_bound_mean_m": float(
                        sample_pinsker_shift_bound[indices].mean()),
                    "pinsker_shift_bound_p90_m": float(np.percentile(
                        sample_pinsker_shift_bound[indices], 90)),
                    "entropy_delta_mean": float(
                        sample_entropy_delta[indices].mean()),
                    "style_offset_mean": float(
                        sample_style_offset[indices].mean()),
                    "style_offset_abs_mean": float(
                        np.abs(sample_style_offset[indices]).mean()),
                    "style_scale_mean": float(
                        sample_style_scale[indices].mean()),
                    "style_log_scale_mean": float(
                        sample_style_log_scale[indices].mean()),
                    "visibility_bias_mean": float(
                        sample_visibility_bias[indices].mean()),
                    "visibility_bias_abs_mean": float(
                        np.abs(sample_visibility_bias[indices]).mean()),
                    "evidence_temperature_mean": float(
                        sample_evidence_temperature[indices].mean()),
                    "evidence_temperature_p10": float(
                        np.percentile(
                            sample_evidence_temperature[indices], 10
                        )
                    ),
                }

            order = np.argsort(sample_visible)
            split_indices = np.array_split(order, 3)
            labels = ["low_visible", "mid_visible", "high_visible"]
            projection_diagnostics = {
                "description": (
                    "Projection diagnostics; delta_mae is DAPP minus its "
                    "pre-projection baseline, so negative is better."
                ),
                "overall": _summarize(np.arange(len(sample_visible))),
                "visible_ap_terciles": {
                    label: _summarize(indices)
                    for label, indices in zip(labels, split_indices)
                },
            }

            def _metric_pair(initial_values: np.ndarray,
                             projected_values: np.ndarray) -> dict:
                initial_mean = float(initial_values.mean())
                projected_mean = float(projected_values.mean())
                return {
                    "initial": initial_mean,
                    "projected": projected_mean,
                    "delta": projected_mean - initial_mean,
                }

            measurement_diagnostics = {
                "description": (
                    "AP-level measurement diagnostics averaged over ensemble "
                    "members; deltas are projected minus initial posterior."
                ),
                "visibility_brier": _metric_pair(
                    member_initial_visibility_brier_np,
                    member_projected_visibility_brier_np,
                ),
                "visible_rss_mae_db": _metric_pair(
                    member_initial_visible_rss_mae_db_np,
                    member_projected_visible_rss_mae_db_np,
                ),
                "visible_rss_rmse_db": _metric_pair(
                    member_initial_visible_rss_rmse_db_np,
                    member_projected_visible_rss_rmse_db_np,
                ),
                "censored_violation_rate": _metric_pair(
                    member_initial_censored_violation_rate_np,
                    member_projected_censored_violation_rate_np,
                ),
                "censored_penalty": _metric_pair(
                    member_initial_censored_penalty_np,
                    member_projected_censored_penalty_np,
                ),
                "calibration": {
                    "style_offset_mean": float(sample_style_offset.mean()),
                    "style_offset_abs_mean": float(
                        np.abs(sample_style_offset).mean()),
                    "style_offset_p10": float(
                        np.percentile(sample_style_offset, 10)),
                    "style_offset_p90": float(
                        np.percentile(sample_style_offset, 90)),
                    "style_offset_db_mean": float(
                        40.0 * sample_style_offset.mean()),
                    "style_offset_db_abs_mean": float(
                        40.0 * np.abs(sample_style_offset).mean()),
                    "style_scale_mean": float(sample_style_scale.mean()),
                    "style_scale_p10": float(
                        np.percentile(sample_style_scale, 10)),
                    "style_scale_p90": float(
                        np.percentile(sample_style_scale, 90)),
                    "style_log_scale_mean": float(
                        sample_style_log_scale.mean()),
                    "style_log_scale_abs_mean": float(
                        np.abs(sample_style_log_scale).mean()),
                    "visibility_bias_mean": float(
                        sample_visibility_bias.mean()),
                    "visibility_bias_abs_mean": float(
                        np.abs(sample_visibility_bias).mean()),
                    "visibility_bias_p10": float(
                        np.percentile(sample_visibility_bias, 10)),
                    "visibility_bias_p90": float(
                        np.percentile(sample_visibility_bias, 90)),
                    "evidence_temperature_mean": float(
                        sample_evidence_temperature.mean()),
                    "evidence_temperature_p10": float(
                        np.percentile(sample_evidence_temperature, 10)),
                    "evidence_temperature_p90": float(
                        np.percentile(sample_evidence_temperature, 90)),
                },
            }
            diagnostic_arrays = {
                "member_projection_kl": member_projection_kls_np.tolist(),
                "member_projection_shift_m": (
                    member_projection_shifts_np.tolist()),
                "member_candidate_diameter_m": (
                    member_candidate_diameters_np.tolist()),
                "member_pinsker_shift_bound_m": (
                    member_pinsker_shift_bounds_np.tolist()),
                "member_visible_ap_count": member_visible_counts_np.tolist(),
                "member_initial_entropy": member_initial_entropies_np.tolist(),
                "member_projected_entropy": (
                    member_projected_entropies_np.tolist()),
                "member_initial_visibility_brier": (
                    member_initial_visibility_brier_np.tolist()),
                "member_projected_visibility_brier": (
                    member_projected_visibility_brier_np.tolist()),
                "member_initial_visible_rss_mae_db": (
                    member_initial_visible_rss_mae_db_np.tolist()),
                "member_projected_visible_rss_mae_db": (
                    member_projected_visible_rss_mae_db_np.tolist()),
                "member_initial_visible_rss_rmse_db": (
                    member_initial_visible_rss_rmse_db_np.tolist()),
                "member_projected_visible_rss_rmse_db": (
                    member_projected_visible_rss_rmse_db_np.tolist()),
                "member_initial_censored_violation_rate": (
                    member_initial_censored_violation_rate_np.tolist()),
                "member_projected_censored_violation_rate": (
                    member_projected_censored_violation_rate_np.tolist()),
                "member_initial_censored_penalty": (
                    member_initial_censored_penalty_np.tolist()),
                "member_projected_censored_penalty": (
                    member_projected_censored_penalty_np.tolist()),
                "member_style_offset": member_style_offsets_np.tolist(),
                "member_style_scale": member_style_scales_np.tolist(),
                "member_style_log_scale": (
                    member_style_log_scales_np.tolist()),
                "member_visibility_bias": (
                    member_visibility_biases_np.tolist()),
                "member_evidence_temperature": (
                    member_evidence_temperatures_np.tolist()),
            }
            if single_consensus:
                consensus_member_weights_np = np.concatenate(
                    consensus_member_weight_batches,
                    axis=1,
                ).astype(np.float64)
                consensus_candidate_indices_np = np.concatenate(
                    consensus_candidate_indices_batches).astype(np.int64)
                consensus_candidate_locations_np = np.concatenate(
                    consensus_candidate_locations_batches).astype(np.float64)
                consensus_initial_weights_np = np.concatenate(
                    consensus_initial_weights_batches).astype(np.float64)
                consensus_projected_weights_np = np.concatenate(
                    consensus_projected_weights_batches).astype(np.float64)
                consensus_candidate_oracle_np = np.concatenate(
                    consensus_candidate_oracle_batches).astype(np.float64)
                diagnostic_arrays.update({
                    "consensus_member_weights": (
                        consensus_member_weights_np.tolist()),
                    "member_candidate_indices": [
                        consensus_candidate_indices_np.tolist()],
                    "member_candidate_locations_m": [
                        consensus_candidate_locations_np.tolist()],
                    "member_initial_candidate_weights": [
                        consensus_initial_weights_np.tolist()],
                    "member_projected_candidate_weights": [
                        consensus_projected_weights_np.tolist()],
                    "member_candidate_oracle_min_error_m": [
                        consensus_candidate_oracle_np.tolist()],
                })

        member_results = []
        member_projected_errors = []
        member_baseline_errors = []
        for index, (baseline, projected) in enumerate(zip(
            baseline_members_np, projected_members_np
        )):
            baseline_member_metrics, baseline_member_errors = (
                coordinate_metrics(baseline, target))
            projected_member_metrics, projected_member_errors = (
                coordinate_metrics(projected, target))
            member_baseline_errors.append(baseline_member_errors)
            member_projected_errors.append(projected_member_errors)
            member_results.append({
                "checkpoint": (
                    "alpha_consensus(all_checkpoints)"
                    if single_consensus
                    else self.checkpoint_paths[index]
                ),
                "baseline": baseline_member_metrics,
                "projected": projected_member_metrics,
            })

        return {
            "protocol": (
                "model and point-localizer configuration fixed on designated "
                "internal splits before benchmark evaluation; benchmark "
                "labels are evaluation-only for point localization"
            ),
            "checkpoints": self.checkpoint_paths,
            "global_identity_contract": self.identity_contract,
            "test_data": str(test_data),
            "test_partition": test_partition,
            "held_out_phone": effective_held_phone,
            "rows": int(len(selected)),
            "selection_contract": _selection_contract(
                selected, test_partition),
            "device": str(self.device),
            "timing": _point_inference_timing_payload(
                point_inference_batch_seconds if single_consensus else [],
                device=self.device,
                pipeline_mode=(
                    "single_consensus"
                    if single_consensus else "independent_members"),
                include_diagnostics=include_diagnostics,
                compute_predictive_distribution=(
                    compute_predictive_distribution),
            ),
            "configuration": build_serialized_configuration(
                locals(),
                pipeline_mode=(
                    "single_consensus"
                    if single_consensus else "joint_projection"
                ),
                compute_predictive_distribution=(
                    compute_predictive_distribution),
                effective_measurement_pool=effective_measurement_pool,
            ),
            "members": member_results,
            **(
                {
                    "baseline_consensus": baseline_metrics,
                    "projected_consensus": projected_metrics,
                    "paired_bootstrap_consensus": paired_bootstrap(
                        projected_errors, baseline_errors
                    ),
                }
                if single_consensus
                else {}
            ),
            # Legacy aliases retained for existing evidence readers.
            "baseline_huber": baseline_metrics,
            "projected_huber": projected_metrics,
            "baseline_weighted_geometric_median": baseline_geometric_metrics,
            "projected_weighted_geometric_median": (
                projected_geometric_metrics),
            "paired_bootstrap_huber": paired_bootstrap(
                projected_errors, baseline_errors),
            "paired_bootstrap_weighted_geometric_median": paired_bootstrap(
                projected_geometric_errors, baseline_geometric_errors),
            "errors_m": {
                **(
                    {
                        "baseline_consensus": baseline_errors.tolist(),
                        "projected_consensus": projected_errors.tolist(),
                    }
                    if single_consensus
                    else {}
                ),
                "baseline_huber": baseline_errors.tolist(),
                "projected_huber": projected_errors.tolist(),
                "baseline_weighted_geometric_median": (
                    baseline_geometric_errors.tolist()),
                "projected_weighted_geometric_median": (
                    projected_geometric_errors.tolist()),
            },
            "building_acc": building_acc,
            "floor_acc": floor_acc,
            "projection_diagnostics": projection_diagnostics,
            "measurement_diagnostics": measurement_diagnostics,
            "predictive_distribution": predictive_distribution,
            "diagnostics": (
                {
                    "member_baseline_errors_m": [
                        values.tolist() for values in member_baseline_errors
                    ],
                    "member_projected_errors_m": [
                        values.tolist() for values in member_projected_errors
                    ],
                    "member_baseline_xy_m": baseline_members_np.tolist(),
                    "member_projected_xy_m": projected_members_np.tolist(),
                    **(
                        {
                            "baseline_consensus_xy_m": baseline_huber.tolist(),
                            "projected_consensus_xy_m": projected_huber.tolist(),
                        }
                        if single_consensus
                        else {}
                    ),
                    "baseline_huber_xy_m": baseline_huber.tolist(),
                    "projected_huber_xy_m": projected_huber.tolist(),
                    "baseline_weighted_geometric_median_xy_m": (
                        baseline_geometric_median.tolist()),
                    "projected_weighted_geometric_median_xy_m": (
                        projected_geometric_median.tolist()),
                    "member_radial_scale_m": member_scales_np.tolist(),
                    **(
                        {
                            "predictive_mean_xy_m": (
                                predictive_mean_np.tolist()),
                            "predictive_covariance_m2": (
                                predictive_covariance_np.tolist()),
                            "predictive_within_covariance_m2": (
                                predictive_within_covariance_np.tolist()),
                            "predictive_between_candidate_covariance_m2": (
                                predictive_between_candidate_covariance_np.tolist()),
                            "predictive_between_member_covariance_m2": (
                                predictive_between_member_covariance_np.tolist()),
                            "predictive_root_trace_scale_m": (
                                predictive_root_trace_np.tolist()),
                            "predictive_nll_nats": predictive_nll_np.tolist(),
                            "predictive_mahalanobis_score": (
                                predictive_mahalanobis_np.tolist()),
                            "predictive_unit_ellipse_area_m2": (
                                predictive_base_area_np.tolist()),
                        }
                        if compute_predictive_distribution
                        else {}
                    ),
                    "target_xy_m": target.tolist(),
                    "building_id": building_t[selected].numpy().tolist(),
                    "floor_id": floor_t[selected].numpy().tolist(),
                    "space_id": space_t[selected].numpy().tolist(),
                    "user_id": user_t[selected].numpy().tolist(),
                    "phone_id": phone_t[selected].numpy().tolist(),
                    "timestamp": timestamp_t[selected].numpy().tolist(),
                    "precision": (
                        consensus_member_weights_np.tolist()
                        if single_consensus else precision.tolist()
                    ),
                    "precision_role": (
                        "five-member consensus rho"
                        if single_consensus
                        else "coordinate-fusion precision"
                    ),
                    "member_support_ood_distance_m": (
                        member_ood_distances_np.tolist()),
                    "member_ood_precision_factor": (
                        member_ood_factors_np.tolist()),
                    "member_pred_building": (
                        member_pred_buildings_np.tolist()),
                    "member_pred_floor": (
                        member_pred_floors_np.tolist()),
                    "ensemble_pred_building": (
                        ensemble_building.tolist()),
                    "ensemble_pred_floor": (
                        ensemble_floor.tolist()),
                    **diagnostic_arrays,
                }
                if include_diagnostics
                else None
            ),
        }

    def evaluate(
        self,
        test_data: str | Path,
        batch: int = 128,
        top_k: int = 24,
        member_top_k: int | None = None,
        neural_scale: float = 0.75,
        prototype_weight: float = 8.0,
        posterior_temperature: float = 0.75,
        candidate_building_conditioning: bool = False,
        visibility_weight: float = 0.0,
        prior_weight: float = 0.50,
        prior_alpha: float = 0.25,
        compactness_weight: float = 0.0,
        style_offset_max: float = 0.20,
        style_offset_weight: float = 0.10,
        style_scale_max_log: float = 0.25,
        style_scale_weight: float = 0.10,
        visibility_bias_max: float = 0.0,
        visibility_bias_weight: float = 0.0,
        reliability_power: float = 0.0,
        member_weight_mode: str = "inverse_scale",
        measurement_pool_mode: str = "shared",
        support_density_weight: float = 0.0,
        support_density_sigma: float = 4.0,
        support_density_margin: float = 0.0,
        support_density_scope: str = "all",
        support_density_mode: str = "soft_distance",
        support_density_step: float = 1.0,
        support_ood_precision_scale: float = 0.0,
        support_ood_precision_power: float = 2.0,
        support_ood_sigma: float = 4.0,
        support_ood_scope: str = "building",
        support_ood_mode: str = "kde_softmin",
        rss_centering: str = "none",
        rss_loss_weight: float = 0.0,
        visibility_positive_factor: float = 1.0,
        visibility_negative_factor: float = 1.0,
        mixture_censored_weight: float = 0.0,
        mixture_visible_factor: float = 1.0,
        mixture_invisible_factor: float = 1.0,
        hurdle_distribution: str = "student_t",
        hurdle_df: float = 4.0,
        hurdle_score: str = "balanced_log",
        hurdle_power_beta: float = 0.25,
        evidence_temperature_mode: str = "none",
        projection_mode: str = "alpha_prox_hurdle",
        steps: int = 10,
        learning_rate: float = 0.2,
        test_partition: str = "all",
        held_out_phone: int | None = None,
        include_diagnostics: bool = False,
        compute_predictive_distribution: bool | None = None,
        rss_global_offset_db: float = 0.0,
        rss_dropout_prob: float = 0.0,
        rss_dropout_seed: int = 20260625,
        variant_name: str = "ac_rhap_final",
        adaptive_kl_alpha: float = 0.0,
        adaptive_kl_visible_pivot: float = 16.0,
        adaptive_kl_min_scale: float = 1.0,
        adaptive_kl_max_scale: float = 1.0,
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
        manifold_sigma: float = 8.0,
        manifold_margin: float = 6.0,
        ap_censored_threshold_mode: str = "none",
        ap_censored_threshold_quantile: float = 0.10,
        ap_censored_threshold_shrinkage: float = 0.50,
        ap_censored_threshold_min: float = -1.00,
        ap_censored_threshold_max: float = -0.50,
        hierarchical_style: bool = False,
        style_residual_fraction: float = 1.0,
        style_group_count: int = 1,
        style_group_mode: str = "batch",
        censored_invisible_weight: float = 0.0,
        censored_detection_threshold: float = 0.10,
        censored_rss_threshold: float = -0.95,
        censored_mode: str = "hinge",
        censored_rss_softness: float = 0.10,
        censored_contradiction_margin: float = 0.0,
        floor_consistency_weight: float = 0.0,
        gate_entropy_min: float = -1.0,
        gate_spread_min: float = -1.0,
        gate_visible_max: float = -1.0,
        gate_radial_scale_min: float = -1.0,
        risk_interpolation_mode: str = "none",
        risk_interpolation_pivot: float = 8.0,
        risk_interpolation_temperature: float = 2.0,
        risk_interpolation_min_gamma: float = 0.0,
        risk_interpolation_max_gamma: float = 1.0,
        risk_interpolation_entropy_pivot: float = 2.0,
        risk_interpolation_diameter_pivot: float = 80.0,
        risk_interpolation_visible_pivot: float = 16.0,
        risk_interpolation_entropy_weight: float = 1.0,
        risk_interpolation_diameter_weight: float = 1.0,
        risk_interpolation_visible_weight: float = 1.0,
        ensemble_consensus_weight: float = 0.0,
        joint_ensemble_weight: float = 0.0,
        consensus_prior_mode: str = "power",
        consensus_prior_epsilon: float = 1e-6,
        consensus_prior_alpha: float = 0.25,
    ) -> dict:
        """在测试数据上运行完整 AC-RHAP 评估.

        Args:
            test_data: 测试数据 CSV 路径.
            test_partition: all/selection_holdout/validation/calibration/
                held_out_phone. selection_holdout is the first return of the
                grouped protocol and is disjoint from the two point-model
                selection subsets; all is descriptive only for UJI.
            batch: 批大小.
            top_k: 最终共识与投影支持宽度 K (final default: 24).
            member_top_k: 每个成员进入共识前的检索宽度 L. ``None``
                严格解析为 L=K，保持历史默认数值不变；只有单共识路径
                允许 K<L.
            neural_scale: 神经网络 logit 缩放.
            prototype_weight: 原型距离权重.
            posterior_temperature: 后验温度.
            candidate_building_conditioning: optional legacy ablation that
                conditions the candidate posterior on a predicted building;
                the final point localizer leaves it disabled.
            visibility_weight: DAPP 可见性损失权重.
            prior_weight: DAPP/proximal 先验权重.
            prior_alpha: alpha_prox_hurdle 的散度阶数; 0 是
                KL/Gibbs 端点, 最终候选为 0.25.
            compactness_weight: DAPP 空间紧凑性权重.
            style_offset_max: DAPP style offset 上限.
            style_offset_weight: DAPP style offset L2 权重.
            reliability_power: AP 可靠性幂次.
            support_density_weight: 最终预测点到训练支撑密度的惩罚权重.
            support_density_sigma: 支撑密度软近邻核宽度, 单位米.
            support_density_margin: OOD 安全半径, 单位米.
            support_density_scope: 支撑范围, all/building/floor.
            support_density_mode: soft_distance 或 kde_softmin.
            support_density_step: convex step from original DAPP weights to
                support-regularized weights.
            support_ood_precision_scale: if positive, downweight each
                ensemble member by support OOD risk before Huber fusion.
            support_ood_precision_power: exponent for the OOD precision
                multiplier.
            support_ood_sigma: support-density kernel width for risk.
            support_ood_scope: support range for risk, all/building/floor.
            support_ood_mode: risk energy mode, soft_distance/kde_softmin.
            rss_centering: DAPP RSS residual mode. none is the final protocol;
                visible_mean is an A101 diagnostic scan-wise offset-invariant
                residual.
            rss_loss_weight: multiplier for moment RSS reconstruction.
            visibility_positive_factor/visibility_negative_factor: split the
                visibility BCE into detected-AP and non-detected-AP channels.
            mixture_censored_weight: optional legacy soft-censoring weight.
            hurdle_distribution/hurdle_df: shared-anchor hurdle conditional
                RSS family and fixed-per-configuration Student-t degrees of
                freedom; each Student-t choice uses its exact MAD-to-scale
                quantile.
            hurdle_score/hurdle_power_beta: measurement score and the robust
                density-power order.  ``balanced_log`` preserves the original
                channel-balanced surrogate; ``joint_log`` and
                ``density_power`` use a fixed AP normalization.
            evidence_temperature_mode: ``visibility_ratio`` shrinks the
                generalized measurement risk when observed reliable AP mass
                falls below the consensus-prior expected visible mass.
            projection_mode: alpha_prox_hurdle is the final shared-memory
                robust-hurdle loss with an exact alpha-proximal posterior;
                dapp and entropy_min are legacy/control objectives.
            steps: DAPP 梯度下降步数.
            learning_rate: DAPP Adam 学习率.
            include_diagnostics: include per-row member errors and OOD
                distances for exploratory analysis.
            compute_predictive_distribution: optional reliability readout.
                ``None`` preserves diagnostic compatibility (enabled only for
                a single-consensus diagnostic run); ``False`` is the pure
                point-localization path and does not alter its coordinates.
            rss_global_offset_db: simulated domain shift. If non-zero, applies a
                global RSS offset in dB to visible APs only (invisible APs at
                -110 dBm remain unchanged). Used for domain-shift sensitivity
                analysis.
            rss_dropout_prob: simulated non-detection probability applied to
                visible APs at test time. Dropped APs are encoded as invisible.
            rss_dropout_seed: deterministic RNG seed for RSS dropout audits.
            variant_name: metadata label for DAPP variant runs.
            adaptive_kl_alpha: adaptive KL exponent. 0 disables adaptive KL.
            censored_invisible_weight: additional censored invisible-AP loss.
            censored_mode: invisible-AP loss mode: hinge, tobit,
                contradiction_hinge or contradiction_tobit.
            floor_consistency_weight: penalty on candidate mass outside the
                predicted floor.
            gate_entropy_min/gate_spread_min/gate_visible_max: if any is set
                to a non-negative value, DAPP is applied only to rows matching
                at least one uncertainty gate.
            ensemble_consensus_weight: optional post-DAPP shrinkage of member
                coordinates toward the precision-weighted ensemble consensus
                before Huber fusion. 0 disables the variant.
            joint_ensemble_weight: if positive, jointly optimizes all ensemble
                member candidate posteriors with a coordinate-disagreement
                penalty during DAPP itself. This is different from post-DAPP
                consensus shrinkage.

        Returns:
            dict 含 members, baseline_huber, projected_huber,
            geometric_median, paired_bootstrap, errors_m.
        """
        # 加载测试数据.
        (
            rss_all,
            coords_norm,
            building_t,
            floor_t,
            space_t,
            user_t,
            phone_t,
            timestamp_t,
            _stats,
        ) = load_dataset(
            resolve_path(str(test_data)),
            dataset_type=self.dataset,
            reference_stats=self.reference["stats"],
        )

        # Simulated domain shift: apply global RSS offset in dB to visible APs.
        if rss_global_offset_db != 0.0:
            rss_np = rss_all.numpy()
            rss_dbm = (rss_np + 1.0) * 40.0 - 110.0
            visible = rss_np > -0.999
            rss_dbm[visible] = rss_dbm[visible] + rss_global_offset_db
            rss_dbm = np.clip(rss_dbm, -110.0, -30.0)
            rss_all = torch.from_numpy(
                (rss_dbm + 110.0) / 40.0 - 1.0).to(rss_all.dtype)
        if rss_dropout_prob > 0.0:
            if not 0.0 <= rss_dropout_prob <= 1.0:
                raise ValueError("rss_dropout_prob must be in [0, 1]")
            rss_np = rss_all.numpy()
            visible = rss_np > -0.999
            rng = np.random.default_rng(int(rss_dropout_seed))
            drop_mask = visible & (
                rng.random(rss_np.shape) < float(rss_dropout_prob))
            rss_np[drop_mask] = -1.0
            rss_all = torch.from_numpy(rss_np).to(rss_all.dtype)

        target_all = (
            raw_coords(coords_norm, building_t, self.reference["stats"])
            - self.reference["coord_origin"]
        )
        effective_held_phone = (
            self.held_out_phone if held_out_phone is None else held_out_phone)
        if test_partition == "all":
            selected = np.arange(len(rss_all), dtype=np.int64)
        elif test_partition in {
            "selection_holdout", "validation", "calibration"
        }:
            config = self.reference["config"]
            partitions = _selection_protocol_partitions(
                target_all,
                building_t.numpy(),
                floor_t.numpy(),
                timestamp_t.numpy(),
                config.get("validation_fraction", 0.10),
                config.get("calibration_fraction", 0.10),
                self.reference["seed"],
                config.get("split_strategy", "temporal"),
            )
            selected = partitions[test_partition]
        elif test_partition == "held_out_phone":
            if effective_held_phone is None:
                raise ValueError(
                    "test_partition=held_out_phone requires held_out_phone")
            selected = np.flatnonzero(
                phone_t.numpy() == int(effective_held_phone)).astype(np.int64)
            if len(selected) == 0:
                raise ValueError(
                    f"held-out phone {effective_held_phone} has no rows")
        else:
            raise ValueError(
                "test_partition must be all, selection_holdout, validation, "
                "calibration or held_out_phone")
        rss = rss_all[selected]
        target = target_all[selected]
        projection_mode = projection_mode.lower()
        if projection_mode not in {
            "dapp",
            "entropy_min",
            "alpha_prox_hurdle",
        }:
            raise ValueError(
                "projection_mode must be dapp, entropy_min or "
                "alpha_prox_hurdle")
        hurdle_distribution = hurdle_distribution.lower()
        if hurdle_distribution not in {"gaussian", "student_t"}:
            raise ValueError(
                "hurdle_distribution must be gaussian or student_t")
        if (
            projection_mode == "alpha_prox_hurdle"
            and hurdle_distribution == "student_t"
            and (
                not np.isfinite(float(hurdle_df))
                or float(hurdle_df) <= 0.0
            )
        ):
            raise ValueError(
                "Student-t hurdle_df must be finite and positive")
        hurdle_score = hurdle_score.lower()
        if hurdle_score not in {
            "balanced_log",
            "joint_log",
            "density_power",
        }:
            raise ValueError(
                "hurdle_score must be balanced_log, joint_log, or "
                "density_power"
            )
        if hurdle_score == "density_power" and not (
            0.0 < float(hurdle_power_beta) <= 1.0
        ):
            raise ValueError("hurdle_power_beta must lie in (0, 1]")
        evidence_temperature_mode = evidence_temperature_mode.lower()
        if evidence_temperature_mode not in {
            "none", "visibility_ratio", "self_entropy_discrepancy"
        }:
            raise ValueError(
                "evidence_temperature_mode must be none, visibility_ratio, "
                "or self_entropy_discrepancy"
            )
        if not 0.0 <= float(prior_alpha) <= 1.0:
            raise ValueError("prior_alpha must be in [0, 1]")
        ap_censored_threshold_mode = ap_censored_threshold_mode.lower()
        if ap_censored_threshold_mode not in {"none", "anchor_quantile"}:
            raise ValueError(
                "ap_censored_threshold_mode must be none or anchor_quantile")
        risk_interpolation_mode = risk_interpolation_mode.lower()
        if risk_interpolation_mode not in {
            "none",
            "radial_sigmoid",
            "uncertainty_sigmoid",
        }:
            raise ValueError(
                "risk_interpolation_mode must be none, radial_sigmoid or "
                "uncertainty_sigmoid")
        style_group_mode = style_group_mode.lower()
        if style_group_mode not in {
            "batch",
            "visible_mean",
            "visible_count",
            "visible_mean_count",
        }:
            raise ValueError(
                "style_group_mode must be batch, visible_mean, "
                "visible_count or visible_mean_count")
        consensus_prior_mode = consensus_prior_mode.lower()
        if consensus_prior_mode not in {"none", "geometric", "power"}:
            raise ValueError(
                "consensus_prior_mode must be none, geometric or power")
        member_top_k = _resolve_member_top_k(
            top_k,
            member_top_k,
            single_consensus=consensus_prior_mode != "none",
            anchor_count=int(
                self.checkpoints[0][0].anchor_coords.shape[0]),
        )
        public_options = locals().copy()
        _validate_alpha_proximal_public_options(
            projection_mode,
            public_options,
        )
        resolved_compute_predictive_distribution = (
            _resolve_predictive_distribution_flag(
                compute_predictive_distribution,
                include_diagnostics=include_diagnostics,
                single_consensus=consensus_prior_mode != "none",
            )
        )
        if consensus_prior_epsilon <= 0.0:
            raise ValueError("consensus_prior_epsilon must be positive")
        if not 0.0 <= consensus_prior_alpha <= 1.0:
            raise ValueError("consensus_prior_alpha must be in [0, 1]")
        if (
            projection_mode == "alpha_prox_hurdle"
            and consensus_prior_mode == "none"
        ):
            raise ValueError(
                "alpha_prox_hurdle requires consensus_prior_mode != none")
        if joint_ensemble_weight > 0.0 or consensus_prior_mode != "none":
            return self._evaluate_joint_dapp(
                test_data=test_data,
                test_partition=test_partition,
                effective_held_phone=effective_held_phone,
                rss=rss,
                target=target,
                building_t=building_t,
                floor_t=floor_t,
                space_t=space_t,
                user_t=user_t,
                phone_t=phone_t,
                timestamp_t=timestamp_t,
                selected=selected,
                batch=batch,
                top_k=top_k,
                member_top_k=member_top_k,
                neural_scale=neural_scale,
                prototype_weight=prototype_weight,
                posterior_temperature=posterior_temperature,
                candidate_building_conditioning=(
                    candidate_building_conditioning),
                visibility_weight=visibility_weight,
                prior_weight=prior_weight,
                prior_alpha=prior_alpha,
                compactness_weight=compactness_weight,
                style_offset_max=style_offset_max,
                style_offset_weight=style_offset_weight,
                style_scale_max_log=style_scale_max_log,
                style_scale_weight=style_scale_weight,
                visibility_bias_max=visibility_bias_max,
                visibility_bias_weight=visibility_bias_weight,
                reliability_power=reliability_power,
                member_weight_mode=member_weight_mode,
                measurement_pool_mode=measurement_pool_mode,
                support_density_weight=support_density_weight,
                support_density_sigma=support_density_sigma,
                support_density_margin=support_density_margin,
                support_density_scope=support_density_scope,
                support_density_mode=support_density_mode,
                support_density_step=support_density_step,
                support_ood_precision_scale=support_ood_precision_scale,
                support_ood_precision_power=support_ood_precision_power,
                support_ood_sigma=support_ood_sigma,
                support_ood_scope=support_ood_scope,
                support_ood_mode=support_ood_mode,
                rss_centering=rss_centering,
                rss_loss_weight=rss_loss_weight,
                visibility_positive_factor=visibility_positive_factor,
                visibility_negative_factor=visibility_negative_factor,
                mixture_censored_weight=mixture_censored_weight,
                mixture_visible_factor=mixture_visible_factor,
                mixture_invisible_factor=mixture_invisible_factor,
                hurdle_distribution=hurdle_distribution,
                hurdle_df=hurdle_df,
                hurdle_score=hurdle_score,
                hurdle_power_beta=hurdle_power_beta,
                evidence_temperature_mode=evidence_temperature_mode,
                projection_mode=projection_mode,
                steps=steps,
                learning_rate=learning_rate,
                include_diagnostics=include_diagnostics,
                compute_predictive_distribution=(
                    resolved_compute_predictive_distribution),
                rss_global_offset_db=rss_global_offset_db,
                rss_dropout_prob=rss_dropout_prob,
                rss_dropout_seed=rss_dropout_seed,
                variant_name=variant_name,
                adaptive_kl_alpha=adaptive_kl_alpha,
                adaptive_kl_visible_pivot=adaptive_kl_visible_pivot,
                adaptive_kl_min_scale=adaptive_kl_min_scale,
                adaptive_kl_max_scale=adaptive_kl_max_scale,
                ambiguity_kl_weight=ambiguity_kl_weight,
                ambiguity_kl_spread_pivot=ambiguity_kl_spread_pivot,
                ambiguity_kl_radial_pivot=ambiguity_kl_radial_pivot,
                ambiguity_kl_max_scale=ambiguity_kl_max_scale,
                discriminative_ap_weight=discriminative_ap_weight,
                discriminative_ap_rss_scale=discriminative_ap_rss_scale,
                discriminative_ap_detection_scale=(
                    discriminative_ap_detection_scale),
                discriminative_ap_min=discriminative_ap_min,
                discriminative_ap_max=discriminative_ap_max,
                manifold_weight=manifold_weight,
                manifold_sigma=manifold_sigma,
                manifold_margin=manifold_margin,
                censored_invisible_weight=censored_invisible_weight,
                censored_detection_threshold=censored_detection_threshold,
                censored_rss_threshold=censored_rss_threshold,
                censored_mode=censored_mode,
                censored_rss_softness=censored_rss_softness,
                censored_contradiction_margin=censored_contradiction_margin,
                floor_consistency_weight=floor_consistency_weight,
                gate_entropy_min=gate_entropy_min,
                gate_spread_min=gate_spread_min,
                gate_visible_max=gate_visible_max,
                gate_radial_scale_min=gate_radial_scale_min,
                ensemble_consensus_weight=ensemble_consensus_weight,
                joint_ensemble_weight=joint_ensemble_weight,
                consensus_prior_mode=consensus_prior_mode,
                consensus_prior_epsilon=consensus_prior_epsilon,
                consensus_prior_alpha=consensus_prior_alpha,
                ap_censored_threshold_mode=ap_censored_threshold_mode,
                ap_censored_threshold_quantile=(
                    ap_censored_threshold_quantile),
                ap_censored_threshold_shrinkage=(
                    ap_censored_threshold_shrinkage),
                ap_censored_threshold_min=ap_censored_threshold_min,
                ap_censored_threshold_max=ap_censored_threshold_max,
                hierarchical_style=hierarchical_style,
                style_residual_fraction=style_residual_fraction,
                style_group_count=style_group_count,
                style_group_mode=style_group_mode,
            )

        # 每个 checkpoint 独立跑 DAPP.
        baseline_members = []
        projected_members = []
        member_scales = []
        member_ood_factors = []
        member_ood_distances = []
        member_pred_buildings = []
        member_pred_floors = []
        member_candidate_indices = []
        member_candidate_locations = []
        member_initial_candidate_weights = []
        member_projected_candidate_weights = []
        member_projection_kls = []
        member_projection_shifts = []
        member_candidate_diameters = []
        member_candidate_oracle_min_errors = []
        member_pinsker_shift_bounds = []
        member_visible_counts = []
        member_initial_entropies = []
        member_projected_entropies = []
        member_initial_visibility_brier = []
        member_projected_visibility_brier = []
        member_initial_visible_rss_mae_db = []
        member_projected_visible_rss_mae_db = []
        member_initial_visible_rss_rmse_db = []
        member_projected_visible_rss_rmse_db = []
        member_initial_censored_violation_rate = []
        member_projected_censored_violation_rate = []
        member_initial_censored_penalty = []
        member_projected_censored_penalty = []
        member_style_offsets = []
        member_style_scales = []
        member_style_log_scales = []
        member_visibility_biases = []
        member_shared_style_offsets = []
        member_shared_style_log_scales = []
        member_shared_visibility_biases = []
        independent_point_inference_batch_seconds: list[float] = []
        independent_point_inference_start = None
        if (
            not include_diagnostics
            and not resolved_compute_predictive_distribution
        ):
            # This branch evaluates every member independently.  Its paper
            # timing therefore spans all E projections and ends at the final
            # Huber coordinate fusion, before any label-dependent audit.
            _synchronize_timing_device(self.device)
            independent_point_inference_start = perf_counter()
        for (model, _), (
            anchor_detection_prob,
            anchor_rss_center,
            anchor_rss_scale,
        ) in zip(self.checkpoints, self.anchor_memories):
            ap_reliability = _powered_ap_reliability(
                anchor_detection_prob,
                anchor_rss_center,
                anchor_rss_scale,
                reliability_power,
            )
            if ap_censored_threshold_mode == "anchor_quantile":
                ap_censored_threshold = anchor_ap_censored_threshold(
                    anchor_detection_prob,
                    anchor_rss_center,
                    censored_rss_threshold,
                    quantile=ap_censored_threshold_quantile,
                    shrinkage=ap_censored_threshold_shrinkage,
                    min_threshold=ap_censored_threshold_min,
                    max_threshold=ap_censored_threshold_max,
                )
            else:
                ap_censored_threshold = None
            model.eval()
            baseline = []
            projected = []
            scales = []
            ood_factors = []
            ood_distances = []
            pred_buildings = []
            pred_floors = []
            candidate_indices = []
            candidate_locations = []
            initial_candidate_weights = []
            projected_candidate_weights = []
            projection_kls = []
            projection_shifts = []
            candidate_diameters = []
            candidate_oracle_min_errors = []
            pinsker_shift_bounds = []
            visible_counts = []
            initial_entropies = []
            projected_entropies = []
            initial_visibility_brier = []
            projected_visibility_brier = []
            initial_visible_rss_mae_db = []
            projected_visible_rss_mae_db = []
            initial_visible_rss_rmse_db = []
            projected_visible_rss_rmse_db = []
            initial_censored_violation_rate = []
            projected_censored_violation_rate = []
            initial_censored_penalty = []
            projected_censored_penalty = []
            style_offsets = []
            style_scales = []
            style_log_scales = []
            visibility_biases = []
            shared_style_offsets = []
            shared_style_log_scales = []
            shared_visibility_biases = []
            for start in range(0, len(rss), batch):
                query = rss[start:start + batch].to(self.device)
                candidates = prepare_candidates(
                    model,
                    query,
                    anchor_detection_prob,
                    anchor_rss_center,
                    anchor_rss_scale,
                    top_k,
                    neural_scale,
                    prototype_weight,
                    posterior_temperature,
                    candidate_building_conditioning=(
                        candidate_building_conditioning),
                )
                gate_mask = None
                if (
                    gate_entropy_min >= 0.0
                    or gate_spread_min >= 0.0
                    or gate_visible_max >= 0.0
                ):
                    gate_terms = []
                    if gate_entropy_min >= 0.0:
                        initial_for_gate = (
                            candidates["initial_weights"].clamp(min=1e-8))
                        initial_entropy = -torch.sum(
                            initial_for_gate * torch.log(initial_for_gate),
                            dim=1,
                        )
                        gate_terms.append(
                            initial_entropy >= float(gate_entropy_min))
                    if gate_spread_min >= 0.0:
                        pairwise_for_gate = torch.cdist(
                            candidates["component_locations"],
                            candidates["component_locations"],
                        )
                        candidate_spread = pairwise_for_gate.amax(dim=(1, 2))
                        gate_terms.append(
                            candidate_spread >= float(gate_spread_min))
                    if gate_visible_max >= 0.0:
                        visible_count_for_gate = (
                            candidates["query"] > -0.999).sum(dim=1)
                        gate_terms.append(
                            visible_count_for_gate
                            <= float(gate_visible_max))
                    if gate_radial_scale_min >= 0.0:
                        gate_terms.append(
                            candidates["radial_scale"]
                            >= float(gate_radial_scale_min))
                    gate_mask = torch.stack(gate_terms, dim=0).any(dim=0)
                style_group_ids_t, effective_style_group_count = (
                    style_cluster_ids(
                        candidates["query"],
                        style_group_count,
                        style_group_mode,
                    )
                )
                base_projected_weights = None
                density_step = float(support_density_step)
                if (
                    projection_mode == "dapp"
                    and density_step < 1.0
                    and support_density_weight > 0.0
                ):
                    with torch.enable_grad():
                        base_projected_weights = project_candidate_weights(
                            candidates["query"],
                            candidates["detection_prob"],
                            candidates["rss_center"],
                            candidates["rss_scale"],
                            candidates["initial_weights"],
                            candidates["component_locations"],
                            visibility_weight,
                            prior_weight,
                            compactness_weight,
                            style_offset_max,
                            style_offset_weight,
                            style_scale_max_log,
                            style_scale_weight,
                            visibility_bias_max,
                            visibility_bias_weight,
                            ap_reliability,
                            steps,
                            learning_rate,
                            rss_centering=rss_centering,
                            rss_loss_weight=rss_loss_weight,
                            visibility_positive_factor=(
                                visibility_positive_factor),
                            visibility_negative_factor=(
                                visibility_negative_factor),
                            mixture_censored_weight=mixture_censored_weight,
                            mixture_visible_factor=mixture_visible_factor,
                            mixture_invisible_factor=mixture_invisible_factor,
                            adaptive_kl_alpha=adaptive_kl_alpha,
                            adaptive_kl_visible_pivot=(
                                adaptive_kl_visible_pivot),
                            adaptive_kl_min_scale=adaptive_kl_min_scale,
                            adaptive_kl_max_scale=adaptive_kl_max_scale,
                            ambiguity_kl_weight=ambiguity_kl_weight,
                            ambiguity_kl_spread_pivot=(
                                ambiguity_kl_spread_pivot),
                            ambiguity_kl_radial_pivot=(
                                ambiguity_kl_radial_pivot),
                            ambiguity_kl_max_scale=ambiguity_kl_max_scale,
                            discriminative_ap_weight=(
                                discriminative_ap_weight),
                            discriminative_ap_rss_scale=(
                                discriminative_ap_rss_scale),
                            discriminative_ap_detection_scale=(
                                discriminative_ap_detection_scale),
                            discriminative_ap_min=discriminative_ap_min,
                            discriminative_ap_max=discriminative_ap_max,
                            manifold_weight=manifold_weight,
                            manifold_sigma=manifold_sigma,
                            manifold_margin=manifold_margin,
                            hierarchical_style=hierarchical_style,
                            style_residual_fraction=style_residual_fraction,
                            style_group_ids=style_group_ids_t,
                            style_group_count=effective_style_group_count,
                            censored_invisible_weight=(
                                censored_invisible_weight),
                            censored_detection_threshold=(
                                censored_detection_threshold),
                            censored_rss_threshold=censored_rss_threshold,
                            censored_mode=censored_mode,
                            censored_rss_softness=censored_rss_softness,
                            censored_contradiction_margin=(
                                censored_contradiction_margin),
                            candidate_floors=candidates["candidate_floors"],
                            predicted_floor=candidates["predicted_floor"],
                            floor_consistency_weight=(
                                floor_consistency_weight),
                            gate_mask=gate_mask,
                            ambiguity_radial_scale=candidates["radial_scale"],
                            ap_censored_threshold=ap_censored_threshold,
                        )
                with torch.enable_grad():
                    calibration_values = None
                    if projection_mode == "dapp":
                        support_mask = support_density_mask(
                            model.anchor_buildings,
                            model.anchor_floors,
                            candidates["predicted_building"],
                            candidates["predicted_floor"],
                            support_density_scope,
                        )
                        projection_result = project_candidate_weights(
                            candidates["query"],
                            candidates["detection_prob"],
                            candidates["rss_center"],
                            candidates["rss_scale"],
                            candidates["initial_weights"],
                            candidates["component_locations"],
                            visibility_weight,
                            prior_weight,
                            compactness_weight,
                            style_offset_max,
                            style_offset_weight,
                            style_scale_max_log,
                            style_scale_weight,
                            visibility_bias_max,
                            visibility_bias_weight,
                            ap_reliability,
                            steps,
                            learning_rate,
                            model.anchor_coords,
                            support_mask,
                            support_density_weight,
                            support_density_sigma,
                            support_density_margin,
                            support_density_mode,
                            rss_centering,
                            rss_loss_weight=rss_loss_weight,
                            visibility_positive_factor=(
                                visibility_positive_factor),
                            visibility_negative_factor=(
                                visibility_negative_factor),
                            mixture_censored_weight=mixture_censored_weight,
                            mixture_visible_factor=mixture_visible_factor,
                            mixture_invisible_factor=mixture_invisible_factor,
                            adaptive_kl_alpha=adaptive_kl_alpha,
                            adaptive_kl_visible_pivot=(
                                adaptive_kl_visible_pivot),
                            adaptive_kl_min_scale=adaptive_kl_min_scale,
                            adaptive_kl_max_scale=adaptive_kl_max_scale,
                            ambiguity_kl_weight=ambiguity_kl_weight,
                            ambiguity_kl_spread_pivot=(
                                ambiguity_kl_spread_pivot),
                            ambiguity_kl_radial_pivot=(
                                ambiguity_kl_radial_pivot),
                            ambiguity_kl_max_scale=ambiguity_kl_max_scale,
                            discriminative_ap_weight=(
                                discriminative_ap_weight),
                            discriminative_ap_rss_scale=(
                                discriminative_ap_rss_scale),
                            discriminative_ap_detection_scale=(
                                discriminative_ap_detection_scale),
                            discriminative_ap_min=discriminative_ap_min,
                            discriminative_ap_max=discriminative_ap_max,
                            manifold_weight=manifold_weight,
                            manifold_sigma=manifold_sigma,
                            manifold_margin=manifold_margin,
                            hierarchical_style=hierarchical_style,
                            style_residual_fraction=style_residual_fraction,
                            style_group_ids=style_group_ids_t,
                            style_group_count=effective_style_group_count,
                            censored_invisible_weight=(
                                censored_invisible_weight),
                            censored_detection_threshold=(
                                censored_detection_threshold),
                            censored_rss_threshold=censored_rss_threshold,
                            censored_mode=censored_mode,
                            censored_rss_softness=censored_rss_softness,
                            censored_contradiction_margin=(
                                censored_contradiction_margin),
                            candidate_floors=candidates["candidate_floors"],
                            predicted_floor=candidates["predicted_floor"],
                            floor_consistency_weight=(
                                floor_consistency_weight),
                            gate_mask=gate_mask,
                            ambiguity_radial_scale=candidates["radial_scale"],
                            ap_censored_threshold=ap_censored_threshold,
                            return_calibration=include_diagnostics,
                        )
                        if include_diagnostics:
                            weights, calibration_values = projection_result
                        else:
                            weights = projection_result
                    else:
                        weights = project_entropy_min_weights(
                            candidates["initial_weights"],
                            prior_weight,
                            steps,
                            learning_rate,
                        )
                if (
                    projection_mode == "dapp"
                    and risk_interpolation_mode != "none"
                ):
                    temperature = max(
                        float(risk_interpolation_temperature), 1e-6)
                    if risk_interpolation_mode == "radial_sigmoid":
                        risk_score = (
                            candidates["radial_scale"]
                            - float(risk_interpolation_pivot)
                        )
                    else:
                        initial_for_risk = (
                            candidates["initial_weights"].clamp(min=1e-8))
                        initial_entropy_for_risk = -torch.sum(
                            initial_for_risk * torch.log(initial_for_risk),
                            dim=1,
                        )
                        pairwise_for_risk = torch.cdist(
                            candidates["component_locations"],
                            candidates["component_locations"],
                        )
                        candidate_diameter_for_risk = pairwise_for_risk.amax(
                            dim=(1, 2))
                        visible_count_for_risk = (
                            candidates["query"] > -0.999).sum(dim=1).to(
                                dtype=candidates["radial_scale"].dtype)
                        risk_score = (
                            (
                                candidates["radial_scale"]
                                - float(risk_interpolation_pivot)
                            )
                            + float(risk_interpolation_entropy_weight)
                            * (
                                initial_entropy_for_risk
                                - float(risk_interpolation_entropy_pivot)
                            )
                            + float(risk_interpolation_diameter_weight)
                            * (
                                candidate_diameter_for_risk
                                - float(risk_interpolation_diameter_pivot)
                            )
                            / max(float(risk_interpolation_diameter_pivot), 1e-6)
                            * float(risk_interpolation_pivot)
                            + float(risk_interpolation_visible_weight)
                            * (
                                float(risk_interpolation_visible_pivot)
                                - visible_count_for_risk
                            )
                            / max(float(risk_interpolation_visible_pivot), 1e-6)
                            * float(risk_interpolation_pivot)
                        )
                    gamma = torch.sigmoid(risk_score / temperature)
                    gamma = torch.clamp(
                        gamma,
                        min=float(risk_interpolation_min_gamma),
                        max=float(risk_interpolation_max_gamma),
                    )
                    weights = (
                        (1.0 - gamma.unsqueeze(1))
                        * candidates["initial_weights"]
                        + gamma.unsqueeze(1) * weights
                    )
                if (
                    projection_mode == "dapp"
                    and density_step < 1.0
                    and support_density_weight > 0.0
                ):
                    weights = (
                        (1.0 - density_step) * base_projected_weights
                        + density_step * weights
                    )
                coordinates = torch.sum(
                    weights.unsqueeze(2)
                    * candidates["component_locations"],
                    dim=1,
                )
                if include_diagnostics:
                    initial = candidates["initial_weights"].clamp(min=1e-8)
                    projected_weight = weights.clamp(min=1e-8)
                    projection_kl = torch.sum(
                        projected_weight
                        * (torch.log(projected_weight) - torch.log(initial)),
                        dim=1,
                    )
                    initial_entropy = -torch.sum(
                        initial * torch.log(initial), dim=1)
                    projected_entropy = -torch.sum(
                        projected_weight * torch.log(projected_weight), dim=1)
                    projection_shift = torch.linalg.norm(
                        coordinates - candidates["baseline"], dim=1)
                    pairwise_candidate_distance = torch.cdist(
                        candidates["component_locations"],
                        candidates["component_locations"],
                    )
                    candidate_diameter = pairwise_candidate_distance.amax(
                        dim=(1, 2))
                    batch_target = torch.as_tensor(
                        target[start:start + len(query)],
                        device=candidates["component_locations"].device,
                        dtype=candidates["component_locations"].dtype,
                    )
                    candidate_oracle_min_error = torch.linalg.vector_norm(
                        candidates["component_locations"]
                        - batch_target.unsqueeze(1),
                        dim=2,
                    ).amin(dim=1)
                    pinsker_shift_bound = candidate_diameter * torch.sqrt(
                        0.5 * projection_kl.clamp(min=0.0))
                    visible_count = (candidates["query"] > -0.999).sum(dim=1)
                    visible_mask = candidates["query"] > -0.999

                    def _measurement_terms(
                        posterior_weights: torch.Tensor,
                    ) -> dict[str, torch.Tensor]:
                        predicted_detection = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * candidates["detection_prob"],
                            dim=1,
                        )
                        rss_mass = torch.sum(
                            posterior_weights.unsqueeze(2)
                            * candidates["detection_prob"]
                            * candidates["rss_center"],
                            dim=1,
                        )
                        predicted_rss = (
                            rss_mass
                            / predicted_detection.clamp(min=0.05)
                        )
                        visible_float = visible_mask.float()
                        invisible_mask = ~visible_mask
                        invisible_float = invisible_mask.float()
                        visible_den = visible_float.sum(dim=1).clamp(min=1.0)
                        invisible_den = (
                            invisible_float.sum(dim=1).clamp(min=1.0))
                        visibility_brier = (
                            predicted_detection - visible_float
                        ).square().mean(dim=1)
                        visible_rss_error = (
                            predicted_rss - candidates["query"]
                        ) * visible_float
                        visible_rss_mae_db = (
                            visible_rss_error.abs().sum(dim=1)
                            / visible_den
                            * 40.0
                        )
                        visible_rss_rmse_db = torch.sqrt(
                            visible_rss_error.square().sum(dim=1)
                            / visible_den
                        ) * 40.0
                        censored_violation = (
                            (predicted_detection
                             > float(censored_detection_threshold))
                            & (predicted_rss > float(censored_rss_threshold))
                            & invisible_mask
                        )
                        censored_violation_rate = (
                            censored_violation.float().sum(dim=1)
                            / invisible_den
                        )
                        censored_penalty = (
                            torch.relu(
                                predicted_detection
                                - float(censored_detection_threshold)
                            ).square()
                            + predicted_detection.clamp(0.0, 1.0)
                            * torch.relu(
                                predicted_rss
                                - float(censored_rss_threshold)
                            ).square()
                        )
                        censored_penalty = (
                            (censored_penalty * invisible_float).sum(dim=1)
                            / invisible_den
                        )
                        return {
                            "visibility_brier": visibility_brier,
                            "visible_rss_mae_db": visible_rss_mae_db,
                            "visible_rss_rmse_db": visible_rss_rmse_db,
                            "censored_violation_rate": (
                                censored_violation_rate),
                            "censored_penalty": censored_penalty,
                        }

                    initial_measurement = _measurement_terms(initial)
                    projected_measurement = _measurement_terms(projected_weight)
                    candidate_indices.append(
                        candidates["candidate_indices"].detach().cpu().numpy())
                    candidate_locations.append(
                        candidates["component_locations"].detach().cpu().numpy())
                    initial_candidate_weights.append(
                        initial.detach().cpu().numpy())
                    projected_candidate_weights.append(
                        projected_weight.detach().cpu().numpy())
                    projection_kls.append(projection_kl.detach().cpu().numpy())
                    projection_shifts.append(
                        projection_shift.detach().cpu().numpy())
                    candidate_diameters.append(
                        candidate_diameter.detach().cpu().numpy())
                    candidate_oracle_min_errors.append(
                        candidate_oracle_min_error.detach().cpu().numpy())
                    pinsker_shift_bounds.append(
                        pinsker_shift_bound.detach().cpu().numpy())
                    visible_counts.append(visible_count.detach().cpu().numpy())
                    initial_entropies.append(
                        initial_entropy.detach().cpu().numpy())
                    projected_entropies.append(
                        projected_entropy.detach().cpu().numpy())
                    initial_visibility_brier.append(
                        initial_measurement["visibility_brier"]
                        .detach().cpu().numpy())
                    projected_visibility_brier.append(
                        projected_measurement["visibility_brier"]
                        .detach().cpu().numpy())
                    initial_visible_rss_mae_db.append(
                        initial_measurement["visible_rss_mae_db"]
                        .detach().cpu().numpy())
                    projected_visible_rss_mae_db.append(
                        projected_measurement["visible_rss_mae_db"]
                        .detach().cpu().numpy())
                    initial_visible_rss_rmse_db.append(
                        initial_measurement["visible_rss_rmse_db"]
                        .detach().cpu().numpy())
                    projected_visible_rss_rmse_db.append(
                        projected_measurement["visible_rss_rmse_db"]
                        .detach().cpu().numpy())
                    initial_censored_violation_rate.append(
                        initial_measurement["censored_violation_rate"]
                        .detach().cpu().numpy())
                    projected_censored_violation_rate.append(
                        projected_measurement["censored_violation_rate"]
                        .detach().cpu().numpy())
                    initial_censored_penalty.append(
                        initial_measurement["censored_penalty"]
                        .detach().cpu().numpy())
                    projected_censored_penalty.append(
                        projected_measurement["censored_penalty"]
                        .detach().cpu().numpy())
                    if calibration_values is not None:
                        style_offsets.append(
                            calibration_values["style_offset"]
                            .detach().cpu().numpy())
                        style_scales.append(
                            calibration_values["style_scale"]
                            .detach().cpu().numpy())
                        style_log_scales.append(
                            calibration_values["style_log_scale"]
                            .detach().cpu().numpy())
                        visibility_biases.append(
                            calibration_values["visibility_bias"]
                            .detach().cpu().numpy())
                        if "shared_style_offset" in calibration_values:
                            shared_style_offsets.append(
                                calibration_values["shared_style_offset"]
                                .detach().cpu().numpy())
                            shared_style_log_scales.append(
                                calibration_values["shared_style_log_scale"]
                                .detach().cpu().numpy())
                            shared_visibility_biases.append(
                                calibration_values["shared_visibility_bias"]
                                .detach().cpu().numpy())
                if (
                    include_diagnostics
                    or (
                        support_ood_precision_scale > 0.0
                        and support_ood_precision_power > 0.0
                    )
                ):
                    ood_mask = support_density_mask(
                        model.anchor_buildings,
                        model.anchor_floors,
                        candidates["predicted_building"],
                        candidates["predicted_floor"],
                        support_ood_scope,
                    )
                    ood_energy = support_density_energy(
                        coordinates,
                        model.anchor_coords,
                        ood_mask,
                        support_ood_sigma,
                        0.0,
                        support_ood_mode,
                    )
                    ood_distance = (ood_energy * 100.0).clamp(
                        min=0.0).sqrt()
                else:
                    ood_distance = torch.zeros(
                        len(query),
                        device=coordinates.device,
                        dtype=coordinates.dtype,
                    )
                if (
                    support_ood_precision_scale > 0.0
                    and support_ood_precision_power > 0.0
                ):
                    scale = max(float(support_ood_precision_scale), 1e-6)
                    factor = 1.0 / (
                        1.0
                        + (ood_distance / scale).pow(
                            float(support_ood_precision_power))
                    )
                else:
                    factor = torch.ones(
                        len(query),
                        device=coordinates.device,
                        dtype=coordinates.dtype,
                    )
                baseline.append(candidates["baseline"].cpu().numpy())
                projected.append(coordinates.cpu().numpy())
                scales.append(candidates["radial_scale"].cpu().numpy())
                ood_factors.append(factor.detach().cpu().numpy())
                ood_distances.append(ood_distance.detach().cpu().numpy())
                pred_buildings.append(candidates["predicted_building"].cpu().numpy())
                pred_floors.append(candidates["predicted_floor"].cpu().numpy())
            baseline_members.append(np.concatenate(baseline))
            projected_members.append(np.concatenate(projected))
            member_scales.append(np.concatenate(scales))
            member_ood_factors.append(np.concatenate(ood_factors))
            member_ood_distances.append(np.concatenate(ood_distances))
            member_pred_buildings.append(np.concatenate(pred_buildings))
            member_pred_floors.append(np.concatenate(pred_floors))
            if include_diagnostics:
                member_candidate_indices.append(
                    np.concatenate(candidate_indices))
                member_candidate_locations.append(
                    np.concatenate(candidate_locations))
                member_initial_candidate_weights.append(
                    np.concatenate(initial_candidate_weights))
                member_projected_candidate_weights.append(
                    np.concatenate(projected_candidate_weights))
                member_projection_kls.append(np.concatenate(projection_kls))
                member_projection_shifts.append(
                    np.concatenate(projection_shifts))
                member_candidate_diameters.append(
                    np.concatenate(candidate_diameters))
                member_candidate_oracle_min_errors.append(
                    np.concatenate(candidate_oracle_min_errors))
                member_pinsker_shift_bounds.append(
                    np.concatenate(pinsker_shift_bounds))
                member_visible_counts.append(np.concatenate(visible_counts))
                member_initial_entropies.append(
                    np.concatenate(initial_entropies))
                member_projected_entropies.append(
                    np.concatenate(projected_entropies))
                member_initial_visibility_brier.append(
                    np.concatenate(initial_visibility_brier))
                member_projected_visibility_brier.append(
                    np.concatenate(projected_visibility_brier))
                member_initial_visible_rss_mae_db.append(
                    np.concatenate(initial_visible_rss_mae_db))
                member_projected_visible_rss_mae_db.append(
                    np.concatenate(projected_visible_rss_mae_db))
                member_initial_visible_rss_rmse_db.append(
                    np.concatenate(initial_visible_rss_rmse_db))
                member_projected_visible_rss_rmse_db.append(
                    np.concatenate(projected_visible_rss_rmse_db))
                member_initial_censored_violation_rate.append(
                    np.concatenate(initial_censored_violation_rate))
                member_projected_censored_violation_rate.append(
                    np.concatenate(projected_censored_violation_rate))
                member_initial_censored_penalty.append(
                    np.concatenate(initial_censored_penalty))
                member_projected_censored_penalty.append(
                    np.concatenate(projected_censored_penalty))
                if style_offsets:
                    member_style_offsets.append(np.concatenate(style_offsets))
                    member_style_scales.append(np.concatenate(style_scales))
                    member_style_log_scales.append(
                        np.concatenate(style_log_scales))
                    member_visibility_biases.append(
                        np.concatenate(visibility_biases))
                    if shared_style_offsets:
                        member_shared_style_offsets.append(
                            np.concatenate(shared_style_offsets))
                        member_shared_style_log_scales.append(
                            np.concatenate(shared_style_log_scales))
                        member_shared_visibility_biases.append(
                            np.concatenate(shared_visibility_biases))

        # 聚合.
        baseline_members_np = np.stack(baseline_members).astype(np.float64)
        projected_members_np = np.stack(projected_members).astype(np.float64)
        member_scales_np = np.stack(member_scales).astype(np.float64)
        member_ood_factors_np = np.stack(member_ood_factors).astype(np.float64)
        member_ood_distances_np = np.stack(member_ood_distances).astype(
            np.float64)

        precision = 1.0 / np.maximum(member_scales_np, 1e-6) ** 2
        precision *= np.clip(member_ood_factors_np, 1e-6, 1.0)
        precision /= precision.sum(axis=0, keepdims=True)
        if ensemble_consensus_weight > 0.0:
            consensus_step = float(np.clip(
                ensemble_consensus_weight, 0.0, 1.0))
            consensus_center = np.sum(
                projected_members_np * precision[..., None], axis=0)
            projected_members_np = (
                (1.0 - consensus_step) * projected_members_np
                + consensus_step * consensus_center[None, :, :]
            )

        baseline_huber = huber_precision_center(
            baseline_members_np, precision)
        projected_huber = huber_precision_center(
            projected_members_np, precision)
        if independent_point_inference_start is not None:
            _synchronize_timing_device(self.device)
            independent_point_inference_batch_seconds.append(
                perf_counter() - independent_point_inference_start)

        # Alternative decoder and hierarchy accuracy are offline audits, not
        # part of the primary online point-localization latency.
        baseline_geometric_median = weighted_geometric_median(
            baseline_members_np, precision)
        projected_geometric_median = weighted_geometric_median(
            projected_members_np, precision)

        member_pred_buildings_np = np.stack(member_pred_buildings)
        member_pred_floors_np = np.stack(member_pred_floors)
        from scipy import stats as _scipy_stats
        ensemble_building, _ = _scipy_stats.mode(
            member_pred_buildings_np, axis=0, keepdims=False)
        ensemble_floor, _ = _scipy_stats.mode(
            member_pred_floors_np, axis=0, keepdims=False)
        gt_building = building_t[selected].numpy()
        gt_floor = floor_t[selected].numpy()
        building_acc = float((ensemble_building == gt_building).mean())
        floor_acc = float((ensemble_floor == gt_floor).mean())

        baseline_metrics, baseline_errors = coordinate_metrics(
            baseline_huber, target)
        projected_metrics, projected_errors = coordinate_metrics(
            projected_huber, target)
        baseline_geometric_metrics, baseline_geometric_errors = (
            coordinate_metrics(baseline_geometric_median, target)
        )
        projected_geometric_metrics, projected_geometric_errors = (
            coordinate_metrics(projected_geometric_median, target)
        )

        projection_diagnostics = None
        diagnostic_arrays = {}
        if include_diagnostics:
            member_candidate_indices_np = np.stack(
                member_candidate_indices).astype(np.int64)
            member_candidate_locations_np = np.stack(
                member_candidate_locations).astype(np.float64)
            member_initial_candidate_weights_np = np.stack(
                member_initial_candidate_weights).astype(np.float64)
            member_projected_candidate_weights_np = np.stack(
                member_projected_candidate_weights).astype(np.float64)
            member_projection_kls_np = np.stack(
                member_projection_kls).astype(np.float64)
            member_projection_shifts_np = np.stack(
                member_projection_shifts).astype(np.float64)
            member_candidate_diameters_np = np.stack(
                member_candidate_diameters).astype(np.float64)
            member_candidate_oracle_min_errors_np = np.stack(
                member_candidate_oracle_min_errors).astype(np.float64)
            member_pinsker_shift_bounds_np = np.stack(
                member_pinsker_shift_bounds).astype(np.float64)
            member_visible_counts_np = np.stack(
                member_visible_counts).astype(np.float64)
            member_initial_entropies_np = np.stack(
                member_initial_entropies).astype(np.float64)
            member_projected_entropies_np = np.stack(
                member_projected_entropies).astype(np.float64)
            member_initial_visibility_brier_np = np.stack(
                member_initial_visibility_brier).astype(np.float64)
            member_projected_visibility_brier_np = np.stack(
                member_projected_visibility_brier).astype(np.float64)
            member_initial_visible_rss_mae_db_np = np.stack(
                member_initial_visible_rss_mae_db).astype(np.float64)
            member_projected_visible_rss_mae_db_np = np.stack(
                member_projected_visible_rss_mae_db).astype(np.float64)
            member_initial_visible_rss_rmse_db_np = np.stack(
                member_initial_visible_rss_rmse_db).astype(np.float64)
            member_projected_visible_rss_rmse_db_np = np.stack(
                member_projected_visible_rss_rmse_db).astype(np.float64)
            member_initial_censored_violation_rate_np = np.stack(
                member_initial_censored_violation_rate).astype(np.float64)
            member_projected_censored_violation_rate_np = np.stack(
                member_projected_censored_violation_rate).astype(np.float64)
            member_initial_censored_penalty_np = np.stack(
                member_initial_censored_penalty).astype(np.float64)
            member_projected_censored_penalty_np = np.stack(
                member_projected_censored_penalty).astype(np.float64)
            member_style_offsets_np = np.stack(
                member_style_offsets).astype(np.float64)
            member_style_scales_np = np.stack(
                member_style_scales).astype(np.float64)
            member_style_log_scales_np = np.stack(
                member_style_log_scales).astype(np.float64)
            member_visibility_biases_np = np.stack(
                member_visibility_biases).astype(np.float64)
            if member_shared_style_offsets:
                member_shared_style_offsets_np = np.stack(
                    member_shared_style_offsets).astype(np.float64)
                member_shared_style_log_scales_np = np.stack(
                    member_shared_style_log_scales).astype(np.float64)
                member_shared_visibility_biases_np = np.stack(
                    member_shared_visibility_biases).astype(np.float64)
            else:
                member_shared_style_offsets_np = np.zeros_like(
                    member_style_offsets_np)
                member_shared_style_log_scales_np = np.zeros_like(
                    member_style_log_scales_np)
                member_shared_visibility_biases_np = np.zeros_like(
                    member_visibility_biases_np)

            sample_kl = member_projection_kls_np.mean(axis=0)
            sample_shift = member_projection_shifts_np.mean(axis=0)
            sample_candidate_diameter = (
                member_candidate_diameters_np.mean(axis=0))
            sample_candidate_oracle_min_error = (
                member_candidate_oracle_min_errors_np.min(axis=0))
            sample_pinsker_shift_bound = (
                member_pinsker_shift_bounds_np.mean(axis=0))
            sample_visible = member_visible_counts_np.mean(axis=0)
            sample_entropy_delta = (
                member_projected_entropies_np - member_initial_entropies_np
            ).mean(axis=0)
            sample_style_offset = member_style_offsets_np.mean(axis=0)
            sample_style_scale = member_style_scales_np.mean(axis=0)
            sample_style_log_scale = member_style_log_scales_np.mean(axis=0)
            sample_visibility_bias = member_visibility_biases_np.mean(axis=0)
            sample_shared_style_offset = (
                member_shared_style_offsets_np.mean(axis=0))
            sample_shared_style_log_scale = (
                member_shared_style_log_scales_np.mean(axis=0))
            sample_shared_visibility_bias = (
                member_shared_visibility_biases_np.mean(axis=0))

            def _summarize(indices: np.ndarray) -> dict:
                return {
                    "n": int(len(indices)),
                    "visible_ap_mean": float(sample_visible[indices].mean()),
                    "base_mae": float(baseline_errors[indices].mean()),
                    "dapp_mae": float(projected_errors[indices].mean()),
                    "delta_mae": float(
                        projected_errors[indices].mean()
                        - baseline_errors[indices].mean()),
                    "kl_mean": float(sample_kl[indices].mean()),
                    "kl_p90": float(np.percentile(sample_kl[indices], 90)),
                    "shift_mean_m": float(sample_shift[indices].mean()),
                    "shift_p90_m": float(np.percentile(sample_shift[indices], 90)),
                    "candidate_diameter_mean_m": float(
                        sample_candidate_diameter[indices].mean()),
                    "candidate_diameter_p90_m": float(np.percentile(
                        sample_candidate_diameter[indices], 90)),
                    "candidate_union_oracle_mae_m": float(
                        sample_candidate_oracle_min_error[indices].mean()),
                    "candidate_union_within_5m_rate": float(
                        (sample_candidate_oracle_min_error[indices] <= 5.0).mean()),
                    "pinsker_shift_bound_mean_m": float(
                        sample_pinsker_shift_bound[indices].mean()),
                    "pinsker_shift_bound_p90_m": float(np.percentile(
                        sample_pinsker_shift_bound[indices], 90)),
                    "entropy_delta_mean": float(
                        sample_entropy_delta[indices].mean()),
                    "style_offset_abs_mean": float(
                        np.abs(sample_style_offset[indices]).mean()),
                    "style_scale_mean": float(
                        sample_style_scale[indices].mean()),
                    "visibility_bias_abs_mean": float(
                        np.abs(sample_visibility_bias[indices]).mean()),
                    "shared_style_offset_abs_mean": float(
                        np.abs(sample_shared_style_offset[indices]).mean()),
                    "shared_style_log_scale_abs_mean": float(
                        np.abs(sample_shared_style_log_scale[indices]).mean()),
                    "shared_visibility_bias_abs_mean": float(
                        np.abs(sample_shared_visibility_bias[indices]).mean()),
                }

            order = np.argsort(sample_visible)
            split_indices = np.array_split(order, 3)
            labels = ["low_visible", "mid_visible", "high_visible"]
            projection_diagnostics = {
                "description": (
                    "Projection diagnostics; delta_mae is DAPP minus its "
                    "pre-projection baseline, so negative is better."
                ),
                "overall": _summarize(np.arange(len(sample_visible))),
                "visible_ap_terciles": {
                    label: _summarize(indices)
                    for label, indices in zip(labels, split_indices)
                },
            }
            diagnostic_arrays = {
                "member_candidate_indices": (
                    member_candidate_indices_np.tolist()),
                "member_candidate_locations_m": (
                    member_candidate_locations_np.tolist()),
                "member_initial_candidate_weights": (
                    member_initial_candidate_weights_np.tolist()),
                "member_projected_candidate_weights": (
                    member_projected_candidate_weights_np.tolist()),
                "member_projection_kl": member_projection_kls_np.tolist(),
                "member_projection_shift_m": (
                    member_projection_shifts_np.tolist()),
                "member_candidate_diameter_m": (
                    member_candidate_diameters_np.tolist()),
                "member_candidate_oracle_min_error_m": (
                    member_candidate_oracle_min_errors_np.tolist()),
                "member_pinsker_shift_bound_m": (
                    member_pinsker_shift_bounds_np.tolist()),
                "member_visible_ap_count": member_visible_counts_np.tolist(),
                "member_initial_entropy": member_initial_entropies_np.tolist(),
                "member_projected_entropy": (
                    member_projected_entropies_np.tolist()),
                "member_initial_visibility_brier": (
                    member_initial_visibility_brier_np.tolist()),
                "member_projected_visibility_brier": (
                    member_projected_visibility_brier_np.tolist()),
                "member_initial_visible_rss_mae_db": (
                    member_initial_visible_rss_mae_db_np.tolist()),
                "member_projected_visible_rss_mae_db": (
                    member_projected_visible_rss_mae_db_np.tolist()),
                "member_initial_visible_rss_rmse_db": (
                    member_initial_visible_rss_rmse_db_np.tolist()),
                "member_projected_visible_rss_rmse_db": (
                    member_projected_visible_rss_rmse_db_np.tolist()),
                "member_initial_censored_violation_rate": (
                    member_initial_censored_violation_rate_np.tolist()),
                "member_projected_censored_violation_rate": (
                    member_projected_censored_violation_rate_np.tolist()),
                "member_initial_censored_penalty": (
                    member_initial_censored_penalty_np.tolist()),
                "member_projected_censored_penalty": (
                    member_projected_censored_penalty_np.tolist()),
                "member_style_offset": member_style_offsets_np.tolist(),
                "member_style_scale": member_style_scales_np.tolist(),
                "member_style_log_scale": (
                    member_style_log_scales_np.tolist()),
                "member_visibility_bias": (
                    member_visibility_biases_np.tolist()),
                "member_shared_style_offset": (
                    member_shared_style_offsets_np.tolist()),
                "member_shared_style_log_scale": (
                    member_shared_style_log_scales_np.tolist()),
                "member_shared_visibility_bias": (
                    member_shared_visibility_biases_np.tolist()),
            }

        # 每个成员的单独指标.
        member_results = []
        member_projected_errors = []
        member_baseline_errors = []
        for index, (baseline, projected) in enumerate(zip(
            baseline_members_np, projected_members_np
        )):
            baseline_member_metrics, baseline_member_errors = coordinate_metrics(
                baseline, target)
            projected_member_metrics, projected_member_errors = coordinate_metrics(
                projected, target)
            member_baseline_errors.append(baseline_member_errors)
            member_projected_errors.append(projected_member_errors)
            member_results.append({
                "checkpoint": self.checkpoint_paths[index],
                "baseline": baseline_member_metrics,
                "projected": projected_member_metrics,
            })

        return {
            "protocol": (
                "model and point-localizer configuration fixed on designated "
                "internal splits before benchmark evaluation; benchmark "
                "labels are evaluation-only for point localization"
            ),
            "checkpoints": self.checkpoint_paths,
            "global_identity_contract": self.identity_contract,
            "test_data": str(test_data),
            "test_partition": test_partition,
            "held_out_phone": effective_held_phone,
            "rows": int(len(selected)),
            "selection_contract": _selection_contract(
                selected, test_partition),
            "device": str(self.device),
            "timing": _point_inference_timing_payload(
                independent_point_inference_batch_seconds,
                device=self.device,
                pipeline_mode="independent_members",
                include_diagnostics=include_diagnostics,
                compute_predictive_distribution=(
                    resolved_compute_predictive_distribution),
            ),
            "configuration": build_serialized_configuration(
                locals(),
                pipeline_mode="independent_members",
                compute_predictive_distribution=(
                    resolved_compute_predictive_distribution),
            ),
            "members": member_results,
            "baseline_huber": baseline_metrics,
            "projected_huber": projected_metrics,
            "baseline_weighted_geometric_median": baseline_geometric_metrics,
            "projected_weighted_geometric_median": (
                projected_geometric_metrics),
            "paired_bootstrap_huber": paired_bootstrap(
                projected_errors, baseline_errors),
            "paired_bootstrap_weighted_geometric_median": paired_bootstrap(
                projected_geometric_errors, baseline_geometric_errors),
            "errors_m": {
                "baseline_huber": baseline_errors.tolist(),
                "projected_huber": projected_errors.tolist(),
                "baseline_weighted_geometric_median": (
                    baseline_geometric_errors.tolist()),
                "projected_weighted_geometric_median": (
                    projected_geometric_errors.tolist()),
            },
            "building_acc": building_acc,
            "floor_acc": floor_acc,
            "projection_diagnostics": projection_diagnostics,
            "diagnostics": (
                {
                    "member_baseline_errors_m": [
                        values.tolist() for values in member_baseline_errors
                    ],
                    "member_projected_errors_m": [
                        values.tolist() for values in member_projected_errors
                    ],
                    "member_baseline_xy_m": baseline_members_np.tolist(),
                    "member_projected_xy_m": projected_members_np.tolist(),
                    "baseline_huber_xy_m": baseline_huber.tolist(),
                    "projected_huber_xy_m": projected_huber.tolist(),
                    "baseline_weighted_geometric_median_xy_m": (
                        baseline_geometric_median.tolist()),
                    "projected_weighted_geometric_median_xy_m": (
                        projected_geometric_median.tolist()),
                    "member_radial_scale_m": member_scales_np.tolist(),
                    "target_xy_m": target.tolist(),
                    "building_id": building_t[selected].numpy().tolist(),
                    "floor_id": floor_t[selected].numpy().tolist(),
                    "space_id": space_t[selected].numpy().tolist(),
                    "user_id": user_t[selected].numpy().tolist(),
                    "phone_id": phone_t[selected].numpy().tolist(),
                    "timestamp": timestamp_t[selected].numpy().tolist(),
                    "precision": precision.tolist(),
                    "member_support_ood_distance_m": (
                        member_ood_distances_np.tolist()),
                    "member_ood_precision_factor": (
                        member_ood_factors_np.tolist()),
                    "member_pred_building": (
                        member_pred_buildings_np.tolist()),
                    "member_pred_floor": (
                        member_pred_floors_np.tolist()),
                    "ensemble_pred_building": (
                        ensemble_building.tolist()),
                    "ensemble_pred_floor": (
                        ensemble_floor.tolist()),
                    **diagnostic_arrays,
                }
                if include_diagnostics
                else None
            ),
        }


