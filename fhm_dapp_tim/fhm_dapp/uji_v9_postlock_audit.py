"""Post-lock uncertainty and atomic attribution for the V9 UJI experiment.

This module can only read the already frozen outer cache and result.  It does
not train, select, or alter the V9 model.  All variants retain the frozen
``(K, lambda)`` and common random numbers so their coordinate-group losses are
paired.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    marked_energy_on_support,
    simultaneous_paired_upper_bounds,
    topk_prior_projection,
    uniform_member_anchor_prior,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v9_experiment import (
    BOOTSTRAP_REPETITIONS,
    DEFAULT_EVIDENCE,
    DEFAULT_FREEZE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    PROTOCOL_VERSION,
    STUDENT_DF,
    _configuration_metrics,
    _marked_energy_for_projection,
    load_cache,
    load_registry,
    load_selection_freeze,
)


DEFAULT_AUDIT = DEFAULT_EVIDENCE / "outer_test_postlock_audit_v9_uji.json"
AUDIT_SEED = 9811
AUDIT_VARIANTS = (
    "front_only",
    "single_member",
    "visibility_event_only",
    "visible_mark_only",
    "no_nondetection",
    "no_local_offset",
)


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int = 10000,
    seed: int = AUDIT_SEED,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Deterministic coordinate-group percentile interval for a mean."""

    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise ValueError("values must be a finite vector with at least two groups")
    count = int(repetitions)
    if count <= 0:
        raise ValueError("repetitions must be positive")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    rng = np.random.default_rng(int(seed))
    means = np.empty(count, dtype=np.float64)
    batch = 1000
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        indices = rng.integers(0, len(sample), size=(stop - start, len(sample)))
        means[start:stop] = sample[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(sample.mean()),
        "standard_error": float(sample.std(ddof=1) / np.sqrt(len(sample))),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": level,
        "bootstrap_repetitions": count,
        "seed": int(seed),
        "coordinate_groups": int(len(sample)),
    }


def _marked_variant(
    cache: dict,
    projection: Mapping[str, torch.Tensor | int],
    *,
    variant: str,
    device: torch.device,
    batch_size: int = 96,
) -> torch.Tensor:
    """Return an exact additive marked-energy ablation on frozen support."""

    if variant not in {"visibility_event_only", "visible_mark_only", "no_nondetection"}:
        raise ValueError(f"unknown marked variant: {variant}")
    indices = projection["anchor_indices"]
    if not isinstance(indices, torch.Tensor):
        raise TypeError("projection misses anchor indices")
    fields = {key: value.to(device) for key, value in cache["radio_fields"].items()}
    if variant == "visibility_event_only":
        # The conditional density is identical for every anchor; only the
        # Bernoulli detection/non-detection event can change posterior odds.
        fields["rss_center"] = torch.zeros_like(fields["rss_center"])
        fields["rss_scale"] = torch.ones_like(fields["rss_scale"])
    elif variant == "visible_mark_only":
        # p=1/2 makes both Bernoulli terms anchor-constant while retaining the
        # conditional Student-t mark with its original one-half coefficient.
        fields["detection_probability"] = torch.full_like(
            fields["detection_probability"], 0.5
        )

    output = []
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        energy = marked_energy_on_support(
            cache["query"][start:stop].to(device),
            indices[start:stop].to(device),
            fields,
            degrees_freedom=STUDENT_DF,
            visible_factor=1.0,
            invisible_factor=(0.0 if variant == "no_nondetection" else 1.0),
        )
        if variant == "no_nondetection":
            # Preserve the full model's 1/2 coefficient for the visible
            # channel instead of silently rescaling lambda.
            energy = 0.5 * energy
        output.append(energy.detach().cpu())
    return torch.cat(output)


def _public_metrics(metrics: dict) -> dict[str, float | int]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, torch.Tensor)
    }


def _load_locked_result(path: str | Path, *, freeze: dict, registry: dict) -> dict:
    result_path = Path(path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 outer result")
    if result.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("outer result registry mismatch")
    if result.get("freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("outer result freeze mismatch")
    value = copy.deepcopy(result)
    stored = value.pop("result_sha256", None)
    if stored != _canonical_json_sha256(value):
        raise ValueError("outer result self-hash failed")
    if result.get("outer_test_access_number") != 1:
        raise ValueError("post-lock audit requires the unique outer evaluation")
    return result


def audit_outer(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    result_path: str | Path = DEFAULT_RESULT,
    output: str | Path = DEFAULT_AUDIT,
    device_name: str = "cuda",
) -> dict:
    """Audit uncertainty and prespecified atoms without making a new choice."""

    registry = load_registry(registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry)
    result = _load_locked_result(result_path, freeze=freeze, registry=registry)
    selected = freeze["selected"]
    cache_path = Path(result["outer_cache"])
    if _sha256_file(cache_path) != result["outer_cache_sha256"]:
        raise ValueError("outer cache changed after the unique evaluation")
    cache = load_cache(
        cache_path,
        registry=registry,
        expected_role="outer_test",
        expected_candidate=selected["candidate"],
    )
    E = int(selected["E"])
    K = int(selected["K"])
    lambda_value = float(selected["lambda"])
    device = torch.device(device_name)

    full_joint = uniform_member_anchor_prior(cache["member_logits"], members=E)
    projection = topk_prior_projection(full_joint, top_k=K)
    full_energy, _ = _marked_energy_for_projection(
        cache,
        projection,
        candidate=selected["candidate"],
        device=device,
    )
    full = _configuration_metrics(
        cache,
        projection,
        full_energy,
        members=E,
        lambda_value=lambda_value,
        device=device,
    )
    for name in (
        "coordinate_macro_energy_score_2d_m",
        "coordinate_macro_mae_2d_m",
        "coordinate_macro_mae_3d_m",
    ):
        if not np.isclose(float(full[name]), float(result["metrics"][name]), atol=1e-6):
            raise RuntimeError(f"locked full-model reproduction failed for {name}")

    variants: dict[str, dict] = {}
    front_only = _configuration_metrics(
        cache,
        projection,
        torch.zeros_like(full_energy),
        members=E,
        lambda_value=lambda_value,
        device=device,
    )
    variants["front_only"] = front_only

    one_joint = uniform_member_anchor_prior(cache["member_logits"], members=1)
    one_projection = topk_prior_projection(one_joint, top_k=K)
    one_energy, _ = _marked_energy_for_projection(
        cache,
        one_projection,
        candidate=selected["candidate"],
        device=device,
    )
    variants["single_member"] = _configuration_metrics(
        cache,
        one_projection,
        one_energy,
        members=1,
        lambda_value=lambda_value,
        device=device,
    )

    for name in ("visibility_event_only", "visible_mark_only", "no_nondetection"):
        variants[name] = _configuration_metrics(
            cache,
            projection,
            _marked_variant(cache, projection, variant=name, device=device),
            members=E,
            lambda_value=lambda_value,
            device=device,
        )

    anchor_cache = dict(cache)
    anchor_cache["member_locations"] = cache["anchor_coords"].view(
        1, 1, -1, 2
    ).expand(E, len(cache["target_xy"]), -1, -1)
    variants["no_local_offset"] = _configuration_metrics(
        anchor_cache,
        projection,
        full_energy,
        members=E,
        lambda_value=lambda_value,
        device=device,
    )

    full_energy_group = full["group_energy_score_2d_m"].numpy()
    full_mae_group = full["group_mae_2d_m"].numpy()
    ordered = list(AUDIT_VARIANTS)
    energy_pairs = simultaneous_paired_upper_bounds(
        full_energy_group,
        [variants[name]["group_energy_score_2d_m"].numpy() for name in ordered],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=AUDIT_SEED + 1,
    )
    mae_pairs = simultaneous_paired_upper_bounds(
        full_mae_group,
        [variants[name]["group_mae_2d_m"].numpy() for name in ordered],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=AUDIT_SEED + 2,
    )
    energy_pairs["difference_order"] = [f"full_minus_{name}" for name in ordered]
    mae_pairs["difference_order"] = [f"full_minus_{name}" for name in ordered]

    group_mae_3d = full["group_mae_3d_m"].numpy()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "post_lock_attribution_not_model_selection",
        "registry_sha256": registry["registry_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "outer_result": str(Path(result_path).resolve()),
        "outer_result_sha256": _sha256_file(result_path),
        "outer_result_self_hash": result["result_sha256"],
        "outer_cache_sha256": result["outer_cache_sha256"],
        "selected": selected,
        "full_model_reproduced": True,
        "full_model": _public_metrics(full),
        "coordinate_group_uncertainty": {
            "energy_score_2d_m": bootstrap_mean_interval(
                full_energy_group, seed=AUDIT_SEED
            ),
            "mae_2d_m": bootstrap_mean_interval(
                full_mae_group, seed=AUDIT_SEED + 10
            ),
            "mae_3d_m": bootstrap_mean_interval(
                group_mae_3d, seed=AUDIT_SEED + 20
            ),
        },
        "coordinate_group_mae_quantiles_m": {
            str(q): float(np.quantile(full_mae_group, q))
            for q in (0.50, 0.75, 0.90, 0.95)
        },
        "variants": {name: _public_metrics(variants[name]) for name in ordered},
        "paired_full_minus_variant_energy": energy_pairs,
        "paired_full_minus_variant_mae": mae_pairs,
        "interpretation_contract": (
            "Variants are post-lock atomic attribution only; they cannot alter "
            "the frozen model or support a second outer-test selection."
        ),
    }
    payload["audit_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = audit_outer(
        registry_path=args.registry,
        freeze_path=args.freeze,
        result_path=args.result,
        output=args.output,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
