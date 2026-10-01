"""V18.4 pre-model decoder diagnostic on the frozen V18.3 posterior.

This module is intentionally diagnostic, not a V18.4 training implementation.
It reads only the already-open V18.3 selection checkpoint plus the V18.3
early-stop and calibration role shards. It never reads confirmation or the
official validation benchmark.

Four point decisions are compared on exactly the same frozen posterior:

1. posterior mean;
2. MAP component location;
3. Top-8 renormalized posterior mean;
4. full-support weighted geometric median.

The probabilistic distribution is unchanged; NLL and Energy Score are therefore
not reinterpreted as decoder metrics.

The diagnostic has a preregistered mechanism decision rule:
- geometric-median mechanism supported if it improves coordinate-macro MAE by
  >=10% on BOTH early-stop and calibration, with no role worsening by >0.25 m;
- tail-dilution mechanism supported if Top-8 mean meets the same rule while the
  geometric median does not;
- otherwise posterior-learning is treated as the dominant remaining problem.

No decoder is selected by looking for the numerically best post-hoc result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as v183
from fhm_dapp_tim.fhm_dapp.v18_3_single import v183_predict


PROBE_VERSION = "v18_4_decoder_mechanism_probe_v1_20261001"
DECODERS = (
    "posterior_mean",
    "map_component",
    "top8_mean",
    "geometric_median",
)
TOPK_DIAGNOSTIC = 8
GEOMETRIC_MEDIAN_MAX_ITER = 128
GEOMETRIC_MEDIAN_TOLERANCE_M = 1e-5
MECHANISM_RELATIVE_IMPROVEMENT = 0.10
MECHANISM_MAX_ROLE_WORSENING_M = 0.25

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "evidence" / "v18_3" / "registry_v18_3.json"
DEFAULT_CHECKPOINT = ROOT / "models" / "v18_3" / "selection.pt"
DEFAULT_FREEZE = ROOT / "evidence" / "v18_3" / "selection_freeze_v18_3.json"
DEFAULT_CONFIRMATION_LOCK = (
    ROOT / "evidence" / "v18_3" / "confirmation_consumed.lock"
)
DEFAULT_CONFIRMATION_RESULT = (
    ROOT / "evidence" / "v18_3" / "confirmation_result_v18_3.json"
)
DEFAULT_OUTPUT = ROOT / "evidence" / "v18_4" / "decoder_probe_v18_3.json"
DEFAULT_MARKDOWN = ROOT / "evidence" / "v18_4" / "decoder_probe_v18_3.md"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [batch,support]")
    if bool((weights < 0.0).any()):
        raise ValueError("weights cannot be negative")
    total = weights.sum(dim=1, keepdim=True)
    if bool((total <= 0.0).any()):
        raise ValueError("every row must have positive posterior mass")
    return weights / total


def posterior_mean(
    weights: torch.Tensor,
    locations: torch.Tensor,
) -> torch.Tensor:
    weights = _normalize_weights(weights)
    if locations.ndim != 3 or locations.shape[:2] != weights.shape:
        raise ValueError("locations must have shape [batch,support,2]")
    return torch.sum(weights.unsqueeze(2) * locations, dim=1)


def map_component(
    weights: torch.Tensor,
    locations: torch.Tensor,
) -> torch.Tensor:
    weights = _normalize_weights(weights)
    if locations.ndim != 3 or locations.shape[:2] != weights.shape:
        raise ValueError("locations must have shape [batch,support,2]")
    index = weights.argmax(dim=1)
    row = torch.arange(len(weights), device=weights.device)
    return locations[row, index]


def topk_mean(
    weights: torch.Tensor,
    locations: torch.Tensor,
    *,
    k: int = TOPK_DIAGNOSTIC,
) -> torch.Tensor:
    weights = _normalize_weights(weights)
    if locations.ndim != 3 or locations.shape[:2] != weights.shape:
        raise ValueError("locations must have shape [batch,support,2]")
    if int(k) <= 0:
        raise ValueError("k must be positive")
    width = min(int(k), weights.shape[1])
    top_weight, top_index = torch.topk(weights, width, dim=1)
    gather_index = top_index.unsqueeze(2).expand(-1, -1, locations.shape[2])
    top_location = torch.gather(locations, 1, gather_index)
    top_weight = top_weight / top_weight.sum(dim=1, keepdim=True).clamp(
        min=1e-12
    )
    return torch.sum(top_weight.unsqueeze(2) * top_location, dim=1)


def _weighted_geometric_median_row(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_iter: int,
    tolerance: float,
) -> torch.Tensor:
    """Modified Weiszfeld solver for one weighted 2-D support."""

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [support,2]")
    if weights.ndim != 1 or len(weights) != len(points):
        raise ValueError("weights must have shape [support]")
    weights = weights / weights.sum().clamp(min=1e-12)
    current = torch.sum(weights.unsqueeze(1) * points, dim=0)
    eps = max(float(tolerance) * 0.1, 1e-9)

    for _ in range(int(max_iter)):
        delta = points - current.unsqueeze(0)
        distance = torch.linalg.vector_norm(delta, dim=1)
        coincident = distance <= eps

        if bool(coincident.any()):
            w0 = weights[coincident].sum()
            nonzero = ~coincident
            if not bool(nonzero.any()):
                return current

            nz_distance = distance[nonzero].clamp(min=eps)
            nz_weights = weights[nonzero]
            nz_delta = delta[nonzero]
            residual = torch.sum(
                nz_weights.unsqueeze(1)
                * nz_delta
                / nz_distance.unsqueeze(1),
                dim=0,
            )
            residual_norm = torch.linalg.vector_norm(residual)
            if float(residual_norm) <= float(w0) + float(tolerance):
                return current

            inv_distance_weight = nz_weights / nz_distance
            weiszfeld = torch.sum(
                inv_distance_weight.unsqueeze(1) * points[nonzero],
                dim=0,
            ) / inv_distance_weight.sum().clamp(min=eps)
            eta = torch.clamp(
                w0 / residual_norm.clamp(min=eps),
                min=0.0,
                max=1.0,
            )
            updated = (1.0 - eta) * weiszfeld + eta * current
        else:
            inv_distance_weight = weights / distance.clamp(min=eps)
            updated = torch.sum(
                inv_distance_weight.unsqueeze(1) * points,
                dim=0,
            ) / inv_distance_weight.sum().clamp(min=eps)

        movement = torch.linalg.vector_norm(updated - current)
        current = updated
        if float(movement) <= float(tolerance):
            break

    return current


def weighted_geometric_median(
    weights: torch.Tensor,
    locations: torch.Tensor,
    *,
    max_iter: int = GEOMETRIC_MEDIAN_MAX_ITER,
    tolerance: float = GEOMETRIC_MEDIAN_TOLERANCE_M,
) -> torch.Tensor:
    weights = _normalize_weights(weights)
    if locations.ndim != 3 or locations.shape[:2] != weights.shape:
        raise ValueError("locations must have shape [batch,support,2]")
    rows = [
        _weighted_geometric_median_row(
            locations[index],
            weights[index],
            max_iter=int(max_iter),
            tolerance=float(tolerance),
        )
        for index in range(len(weights))
    ]
    return torch.stack(rows, dim=0)


def decode_all(
    weights: torch.Tensor,
    locations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "posterior_mean": posterior_mean(weights, locations),
        "map_component": map_component(weights, locations),
        "top8_mean": topk_mean(
            weights,
            locations,
            k=TOPK_DIAGNOSTIC,
        ),
        "geometric_median": weighted_geometric_median(
            weights,
            locations,
        ),
    }


def _group_coordinate_errors(
    row_error: torch.Tensor,
    keys: Sequence[v183.LocationKey],
) -> torch.Tensor:
    if len(row_error) != len(keys):
        raise ValueError("row_error and keys length mismatch")
    grouped: dict[v183.LocationKey, list[int]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(key, []).append(index)
    return torch.stack(
        [
            row_error[
                torch.as_tensor(indices, dtype=torch.long)
            ].mean()
            for indices in grouped.values()
        ]
    ).double()


def _decoder_summary(
    prediction: torch.Tensor,
    target: torch.Tensor,
    keys: Sequence[v183.LocationKey],
) -> dict[str, float | int]:
    error = torch.linalg.vector_norm(prediction - target, dim=1).cpu()
    grouped = _group_coordinate_errors(error, keys)
    coordinate_se = (
        float(grouped.std(unbiased=True) / math.sqrt(len(grouped)))
        if len(grouped) > 1
        else 0.0
    )
    return {
        "coordinate_macro_mae_m": float(grouped.mean()),
        "coordinate_macro_mae_se_m": coordinate_se,
        "row_mae_m": float(error.double().mean()),
        "row_median_error_m": float(torch.quantile(error.double(), 0.50)),
        "row_p75_error_m": float(torch.quantile(error.double(), 0.75)),
        "row_p90_error_m": float(torch.quantile(error.double(), 0.90)),
        "coordinate_groups": int(len(grouped)),
        "rows": int(len(error)),
    }


def _paired_coordinate_difference(
    reference_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    keys: Sequence[v183.LocationKey],
) -> dict[str, float]:
    reference_error = torch.linalg.vector_norm(
        reference_prediction - target,
        dim=1,
    ).cpu()
    candidate_error = torch.linalg.vector_norm(
        candidate_prediction - target,
        dim=1,
    ).cpu()
    reference_group = _group_coordinate_errors(reference_error, keys)
    candidate_group = _group_coordinate_errors(candidate_error, keys)
    improvement = reference_group - candidate_group
    se = (
        float(improvement.std(unbiased=True) / math.sqrt(len(improvement)))
        if len(improvement) > 1
        else 0.0
    )
    return {
        "mean_improvement_m": float(improvement.mean()),
        "improvement_se_m": se,
        "fraction_coordinates_improved": float(
            (improvement > 0.0).double().mean()
        ),
    }


def evaluate_role(
    model: torch.nn.Module,
    checkpoint: dict,
    shard: dict,
    *,
    posterior_temperature: float,
    scale_multiplier: float,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    raw_xy, _building, _floor = v183._shard_arrays(shard)
    origin = checkpoint["coord_origin"].cpu().numpy().astype(np.float32)
    target = torch.from_numpy(
        (raw_xy - origin[None, :]).astype(np.float32)
    )
    keys = v183._decode_shard_keys(shard)

    prediction_parts = {name: [] for name in DECODERS}
    entropy_parts = []
    effective_parts = []
    top1_parts = []

    model.eval()
    for start in range(0, len(raw_xy), int(batch_size)):
        stop = min(start + int(batch_size), len(raw_xy))
        result = v183_predict(
            model,
            shard["rss"][start:stop].to(device),
            posterior_temperature=float(posterior_temperature),
            scale_multiplier=float(scale_multiplier),
            target_coords=None,
            config=v183.V183_CONFIG,
        )
        decoded = decode_all(
            result["candidate_weights"],
            result["component_locations"],
        )

        if not torch.allclose(
            decoded["posterior_mean"],
            result["coords"],
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "decoder probe posterior mean does not reproduce V18.3 coords"
            )

        for name, value in decoded.items():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"non-finite decoder output: {name}")
            prediction_parts[name].append(value.detach().cpu())
        entropy_parts.append(result["anchor_entropy"].detach().cpu())
        effective_parts.append(
            result["effective_anchor_count"].detach().cpu()
        )
        top1_parts.append(result["top1_probability"].detach().cpu())

    predictions = {
        name: torch.cat(parts)
        for name, parts in prediction_parts.items()
    }
    summaries = {
        name: _decoder_summary(prediction, target, keys)
        for name, prediction in predictions.items()
    }
    paired = {
        name: _paired_coordinate_difference(
            predictions["posterior_mean"],
            predictions[name],
            target,
            keys,
        )
        for name in DECODERS
        if name != "posterior_mean"
    }

    mean_mae = summaries["posterior_mean"]["coordinate_macro_mae_m"]
    for name in DECODERS:
        mae = summaries[name]["coordinate_macro_mae_m"]
        summaries[name]["relative_improvement_vs_mean"] = (
            float((mean_mae - mae) / mean_mae)
            if mean_mae > 0.0
            else 0.0
        )

    gm_displacement = torch.linalg.vector_norm(
        predictions["geometric_median"]
        - predictions["posterior_mean"],
        dim=1,
    ).double()

    return {
        "decoder_metrics": summaries,
        "paired_vs_posterior_mean": paired,
        "posterior_diagnostics": {
            "mean_anchor_entropy_nats": float(
                torch.cat(entropy_parts).double().mean()
            ),
            "mean_effective_anchor_count": float(
                torch.cat(effective_parts).double().mean()
            ),
            "mean_top1_probability": float(
                torch.cat(top1_parts).double().mean()
            ),
            "mean_geometric_median_displacement_from_mean_m": float(
                gm_displacement.mean()
            ),
            "p90_geometric_median_displacement_from_mean_m": float(
                torch.quantile(gm_displacement, 0.90)
            ),
        },
    }


def _meets_mechanism_rule(
    role_results: dict[str, dict],
    decoder_name: str,
) -> bool:
    for role in ("early_stop", "calibration"):
        metrics = role_results[role]["decoder_metrics"]
        baseline = float(
            metrics["posterior_mean"]["coordinate_macro_mae_m"]
        )
        candidate = float(
            metrics[decoder_name]["coordinate_macro_mae_m"]
        )
        relative = (
            (baseline - candidate) / baseline
            if baseline > 0.0
            else 0.0
        )
        worsening = candidate - baseline
        if relative < MECHANISM_RELATIVE_IMPROVEMENT:
            return False
        if worsening > MECHANISM_MAX_ROLE_WORSENING_M:
            return False
    return True


def mechanism_decision(role_results: dict[str, dict]) -> dict[str, object]:
    gm_supported = _meets_mechanism_rule(
        role_results,
        "geometric_median",
    )
    top8_supported = _meets_mechanism_rule(
        role_results,
        "top8_mean",
    )

    if gm_supported:
        conclusion = "loss_aligned_geometric_median_supported"
        implication = (
            "Use weighted geometric median as the registered V18.4 point "
            "decoder while keeping the full posterior for NLL/ES/uncertainty."
        )
    elif top8_supported:
        conclusion = "posterior_tail_dilution_supported"
        implication = (
            "Do not promote geometric median on this evidence alone. Make "
            "trust-region posterior contraction/transport the primary V18.4 "
            "fix; retain decoder choice as secondary."
        )
    else:
        conclusion = "posterior_learning_dominates"
        implication = (
            "Neither robust decision rule explains enough error. Prioritize "
            "trust-region posterior learning plus spatial OT; do not claim "
            "decoder mismatch as a primary contribution."
        )

    return {
        "rule": {
            "required_relative_mae_improvement_each_role": (
                MECHANISM_RELATIVE_IMPROVEMENT
            ),
            "maximum_allowed_role_worsening_m": (
                MECHANISM_MAX_ROLE_WORSENING_M
            ),
            "roles": ["early_stop", "calibration"],
        },
        "geometric_median_supported": gm_supported,
        "top8_tail_contraction_supported": top8_supported,
        "conclusion": conclusion,
        "v18_4_implication": implication,
    }


def _write_markdown(path: Path, payload: dict) -> None:
    rows = []
    for role in ("early_stop", "calibration"):
        for decoder in DECODERS:
            metric = payload["roles"][role]["decoder_metrics"][decoder]
            rows.append(
                "| "
                + " | ".join(
                    [
                        role,
                        decoder,
                        f'{metric["coordinate_macro_mae_m"]:.6f}',
                        f'{metric["relative_improvement_vs_mean"]*100:.2f}%',
                        f'{metric["row_mae_m"]:.6f}',
                    ]
                )
                + " |"
            )

    decision = payload["mechanism_decision"]
    report = "\n".join(
        [
            "# V18.4 decoder mechanism probe on frozen V18.3",
            "",
            "This is a read-only diagnostic on already-open V18.3 roles. "
            "It does not consume confirmation or official validation.",
            "",
            "| Role | Decoder | Coordinate-macro MAE (m) | Improvement vs mean | Row MAE (m) |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
            f'Decision: {decision["conclusion"]}',
            "",
            decision["v18_4_implication"],
            "",
            "The predictive posterior is unchanged by this probe; decoder "
            "comparisons do not constitute new NLL/Energy-Score models.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")


def run_probe(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    confirmation_lock: str | Path = DEFAULT_CONFIRMATION_LOCK,
    confirmation_result: str | Path = DEFAULT_CONFIRMATION_RESULT,
    output_path: str | Path = DEFAULT_OUTPUT,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
    device_name: str = "cuda",
) -> dict:
    output = Path(output_path)
    markdown = Path(markdown_path)
    if output.exists() or markdown.exists():
        raise FileExistsError("V18.4 decoder probe output already exists")
    if Path(confirmation_lock).exists() or Path(confirmation_result).exists():
        raise PermissionError(
            "V18.3 confirmation has been consumed; this preregistered "
            "pre-confirmation decoder probe must not be run"
        )

    v183._set_seed(v183.SCORING_SEED)
    registry = v183.load_registry(registry_path)
    freeze = v183.load_freeze(freeze_path, registry=registry)
    checkpoint_file = Path(checkpoint_path)
    if _sha256_file(checkpoint_file) != freeze["selection_checkpoint_sha256"]:
        raise ValueError("V18.3 selection checkpoint SHA does not match freeze")

    device = torch.device(device_name)
    model, checkpoint = v183.load_v183_checkpoint(
        checkpoint_file,
        registry=registry,
        device=device,
        expected_phase="selection",
    )
    if int(checkpoint["selected_epoch"]) != int(
        freeze["frozen_selected_epoch"]
    ):
        raise ValueError("V18.3 selected epoch does not match freeze")

    posterior_temperature = float(
        freeze["frozen_posterior_temperature"]
    )
    scale_multiplier = float(freeze["frozen_scale_multiplier"])

    role_results = {}
    role_hashes = {}
    for role in ("early_stop", "calibration"):
        shard = v183.load_role_shard(registry, role)
        role_results[role] = evaluate_role(
            model,
            checkpoint,
            shard,
            posterior_temperature=posterior_temperature,
            scale_multiplier=scale_multiplier,
            device=device,
        )
        role_hashes[role] = registry["role_shards"][role]["sha256"]

    decision = mechanism_decision(role_results)
    payload = {
        "probe_version": PROBE_VERSION,
        "scientific_status": (
            "read-only mechanism diagnostic on already-open V18.3 roles; "
            "no training, no confirmation, no official benchmark"
        ),
        "v18_3_registry_sha256": registry["registry_sha256"],
        "v18_3_freeze_sha256": freeze["freeze_sha256"],
        "v18_3_selection_checkpoint_sha256": _sha256_file(checkpoint_file),
        "role_shard_sha256": role_hashes,
        "frozen_posterior_temperature": posterior_temperature,
        "frozen_scale_multiplier": scale_multiplier,
        "decoders": list(DECODERS),
        "topk_diagnostic": TOPK_DIAGNOSTIC,
        "roles": role_results,
        "mechanism_decision": decision,
        "confirmation_accessed": False,
        "official_benchmark_accessed": False,
        "model_parameters_modified": False,
        "training_performed": False,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(canonical).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(markdown, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only V18.3 decoder mechanism probe for V18.4"
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument(
        "--confirmation-lock",
        default=str(DEFAULT_CONFIRMATION_LOCK),
    )
    parser.add_argument(
        "--confirmation-result",
        default=str(DEFAULT_CONFIRMATION_RESULT),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = run_probe(
        registry_path=args.registry,
        checkpoint_path=args.checkpoint,
        freeze_path=args.freeze,
        confirmation_lock=args.confirmation_lock,
        confirmation_result=args.confirmation_result,
        output_path=args.output,
        markdown_path=args.markdown,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "result_sha256": payload["result_sha256"],
                "mechanism_decision": payload["mechanism_decision"],
                "roles": {
                    role: payload["roles"][role]["decoder_metrics"]
                    for role in ("early_stop", "calibration")
                },
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
