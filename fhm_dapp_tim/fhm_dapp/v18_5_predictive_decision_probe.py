"""Read-only V18.5 predictive decision probe.

This is a post-formal development diagnostic. It does not retrain V18.5,
change its calibration, or access UJI validation/external test data. The probe
asks one narrow question: how much Euclidean localization error is lost by
using the posterior mean instead of the full predictive spatial median that was
used by the coherent V9 family.

Three distributions are fixed before scoring:
  A. base:             T=1, epsilon=0, scale=1;
  B. calibration-only: selected T, epsilon=0, selected scale;
  C. selected-final:   the frozen V18.5 selected T/epsilon/scale.

All three use the same calibration rows and the same predictive randomness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import load_uji, model_input
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG, v185_distribution
from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    temperature_radio_reference_from_payload,
)
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    DEFAULT_CALIBRATION,
    DEFAULT_FREEZE,
    DEFAULT_REFIT,
    DEFAULT_REGISTRY,
    PREDICTIVE_DRAWS,
    SCORING_SEED,
    _canonical_sha,
    _coordinate_group_ids,
    _coordinate_macro,
    _coordinate_macro_se,
    _load_checkpoint,
    _sha256_file,
    _sha256_indices,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.v19_r0_core import predictive_actions_and_scores


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "evidence" / "v19_r0" / "V18_5_PREDICTIVE_DECISION_PROBE.json"
DEFAULT_MARKDOWN = ROOT / "evidence" / "v19_r0" / "V18_5_PREDICTIVE_DECISION_PROBE.md"


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _validate_v185_inputs(
    registry_path: str | Path,
    calibration_path: str | Path,
    freeze_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[dict, dict, dict, dict]:
    registry = _load_json(registry_path)
    if registry.get("registry_sha256") != _canonical_sha(
        registry, omit="registry_sha256"
    ):
        raise ValueError("V18.5 registry self-hash mismatch")
    calibration = _load_json(calibration_path)
    if calibration.get("calibration_result_sha256") != _canonical_sha(
        calibration, omit="calibration_result_sha256"
    ):
        raise ValueError("V18.5 calibration self-hash mismatch")
    freeze = _load_json(freeze_path)
    if freeze.get("freeze_sha256") != _canonical_sha(
        freeze, omit="freeze_sha256"
    ):
        raise ValueError("V18.5 freeze self-hash mismatch")
    if calibration["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("V18.5 calibration links another registry")
    if freeze["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("V18.5 freeze links another registry")
    if freeze["calibration_result_sha256"] != calibration[
        "calibration_result_sha256"
    ]:
        raise ValueError("V18.5 freeze links another calibration")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if _sha256_file(checkpoint_path) != freeze[
        "selection_refit_checkpoint_sha256"
    ]:
        raise ValueError("V18.5 selection-refit checkpoint hash mismatch")
    if checkpoint.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.5 checkpoint links another registry")
    if checkpoint.get("development_result_sha256") != freeze[
        "development_result_sha256"
    ]:
        raise ValueError("V18.5 checkpoint/development link mismatch")
    return registry, calibration, freeze, checkpoint


def _paired_group_summary(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    keys,
) -> dict[str, float]:
    groups: dict[tuple[int, int, float, float], list[float]] = {}
    for value, ref, key in zip(
        candidate.double().tolist(),
        reference.double().tolist(),
        keys,
    ):
        groups.setdefault(key, []).append(float(value - ref))
    vector = np.asarray(
        [np.mean(groups[key]) for key in sorted(groups)], dtype=np.float64
    )
    mean = float(vector.mean())
    se = 0.0 if len(vector) <= 1 else float(
        vector.std(ddof=1) / math.sqrt(len(vector))
    )
    return {
        "candidate_minus_reference_m": mean,
        "se_m": se,
        "ci95_low_m": mean - 1.96 * se,
        "ci95_high_m": mean + 1.96 * se,
        "coordinate_groups": int(len(vector)),
        "fraction_groups_candidate_better": float(np.mean(vector < 0.0)),
    }


def _score_cell(
    *,
    name: str,
    temperature: float,
    epsilon_kl: float,
    scale_multiplier: float,
    model,
    radio_reference,
    state,
    data,
    rows: np.ndarray,
    randomness,
    device: torch.device,
) -> dict[str, object]:
    mean_errors: list[torch.Tensor] = []
    spatial_errors: list[torch.Tensor] = []
    energy: list[torch.Tensor] = []
    nll: list[torch.Tensor] = []
    batch_size = int(V185_CONFIG["batch_size"])
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        index = rows[start:stop]
        target = torch.from_numpy(
            state["centered"][index].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            distribution = v185_distribution(
                model,
                state["rss"][index].to(device),
                state["observed_mask"][index].to(device),
                radio_memory={
                    "detection_prob": model.anchor_detection_prob,
                    "rss_center": model.anchor_rss_center,
                    "rss_scale": model.anchor_rss_scale,
                },
                radio_reference=(
                    radio_reference if float(epsilon_kl) > 0.0 else None
                ),
                temperature=float(temperature),
                epsilon_kl=float(epsilon_kl),
                scale_multiplier=float(scale_multiplier),
            )
            local_randomness = {
                key: (
                    value[start:stop].to(device)
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                )
                for key, value in randomness.items()
            }
            scored = predictive_actions_and_scores(
                distribution["probability"],
                distribution["component_location"],
                distribution["component_scale"],
                target,
                local_randomness,
                degrees_freedom=float(V185_CONFIG["student_df"]),
            )
        mean_errors.append(scored["posterior_mean_error"].cpu())
        spatial_errors.append(
            scored["predictive_spatial_median_error"].cpu()
        )
        energy.append(scored["energy_score"].cpu())
        nll.append(scored["nll"].cpu())

    mean_error = torch.cat(mean_errors)
    spatial_error = torch.cat(spatial_errors)
    energy_values = torch.cat(energy)
    nll_values = torch.cat(nll)
    keys = _coordinate_group_ids(data, rows)
    return {
        "name": name,
        "temperature": float(temperature),
        "epsilon_kl": float(epsilon_kl),
        "scale_multiplier": float(scale_multiplier),
        "posterior_mean_mae_m": _coordinate_macro(mean_error, keys),
        "posterior_mean_mae_se_m": _coordinate_macro_se(mean_error, keys),
        "predictive_spatial_median_mae_m": _coordinate_macro(
            spatial_error, keys
        ),
        "predictive_spatial_median_mae_se_m": _coordinate_macro_se(
            spatial_error, keys
        ),
        "energy_score_m": _coordinate_macro(energy_values, keys),
        "nll": _coordinate_macro(nll_values, keys),
        "decision_paired": _paired_group_summary(
            spatial_error, mean_error, keys
        ),
        "row_posterior_mean_mae_m": float(mean_error.double().mean()),
        "row_predictive_spatial_median_mae_m": float(
            spatial_error.double().mean()
        ),
    }


def run_probe(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    freeze_path: str | Path = DEFAULT_FREEZE,
    checkpoint_path: str | Path = DEFAULT_REFIT,
    device_name: str = "cuda",
) -> dict[str, object]:
    registry, calibration, freeze, checkpoint = _validate_v185_inputs(
        registry_path, calibration_path, freeze_path, checkpoint_path
    )
    train_path = REPO_ROOT / str(registry["train_path"])
    if train_path.name != "trainingData.csv":
        raise ValueError("probe accepts UJI trainingData registry only")
    if _sha256_file(train_path) != registry["train_sha256"]:
        raise ValueError("UJI trainingData hash mismatch")

    data, _ = load_uji("train", data_root=train_path.parents[1])
    rows = np.asarray(
        registry["roles"]["calibration"]["rows"], dtype=np.int64
    )
    if _sha256_indices(rows) != registry["roles"]["calibration"][
        "row_index_sha256"
    ]:
        raise ValueError("V18.5 calibration row hash mismatch")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, _neighbors, _weights = _load_checkpoint(
        checkpoint_path, device=device
    )
    inputs = model_input(data)
    origin = checkpoint["origin"].cpu().numpy().astype(np.float32)
    state = {
        "rss": inputs.rss,
        "observed_mask": inputs.observed_mask,
        "centered": (data.coords_m.astype(np.float32) - origin[None, :]),
    }
    radio_reference = temperature_radio_reference_from_payload(
        checkpoint["temperature_radio_reference"]
    )
    randomness = common_student_t_randomness(
        len(rows),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V185_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )

    selected_t = float(freeze["posterior_temperature"])
    selected_e = float(freeze["epsilon_kl"])
    selected_s = float(freeze["scale_multiplier"])
    cells = [
        ("base", 1.0, 0.0, 1.0),
        ("calibration_only", selected_t, 0.0, selected_s),
        ("selected_final", selected_t, selected_e, selected_s),
    ]
    results = [
        _score_cell(
            name=name,
            temperature=t,
            epsilon_kl=e,
            scale_multiplier=s,
            model=model,
            radio_reference=radio_reference,
            state=state,
            data=data,
            rows=rows,
            randomness=randomness,
            device=device,
        )
        for name, t, e, s in cells
    ]

    selected = next(
        value for value in results if value["name"] == "selected_final"
    )
    formal_mae = float(calibration["selected"][
        "posterior_mean_mae_m"
    ])
    if abs(float(selected["posterior_mean_mae_m"]) - formal_mae) > 2e-5:
        raise RuntimeError(
            "probe failed to reproduce frozen V18.5 posterior-mean MAE"
        )

    payload = {
        "probe_version": "v18_5_predictive_decision_probe_v1",
        "scientific_status": (
            "post-formal read-only development diagnostic; not independent "
            "confirmation and not a V18.5 method change"
        ),
        "v18_5_registry_sha256": registry["registry_sha256"],
        "v18_5_calibration_sha256": calibration[
            "calibration_result_sha256"
        ],
        "v18_5_freeze_sha256": freeze["freeze_sha256"],
        "v18_5_checkpoint_sha256": _sha256_file(checkpoint_path),
        "rows": int(len(rows)),
        "coordinate_groups": int(
            registry["roles"]["calibration"]["coordinate_count"]
        ),
        "predictive_draws_per_stream": int(PREDICTIVE_DRAWS),
        "scoring_seed": int(SCORING_SEED),
        "cells": results,
        "primary_question": (
            "Does the full Student-t predictive spatial median materially "
            "reduce Euclidean MAE relative to the posterior mean?"
        ),
        "no_training": True,
        "no_model_mutation": True,
        "uji_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    payload["probe_sha256"] = _canonical_sha(payload)
    return payload


def _markdown(payload: Mapping[str, object]) -> str:
    lines = [
        "# V18.5 Predictive Decision Probe",
        "",
        "Read-only post-formal diagnostic. No training, validation, or external test.",
        "",
        "| cell | T | eps | scale | mean MAE | predictive spatial median MAE | delta spatial-mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in payload["cells"]:
        lines.append(
            "| {name} | {temperature:.2f} | {epsilon_kl:.3f} | "
            "{scale_multiplier:.2f} | {posterior_mean_mae_m:.6f} | "
            "{predictive_spatial_median_mae_m:.6f} | {delta:.6f} |".format(
                **cell,
                delta=cell["decision_paired"][
                    "candidate_minus_reference_m"
                ],
            )
        )
    lines.extend([
        "",
        "Negative delta means the predictive spatial median improves Euclidean MAE.",
        "",
        f"Probe SHA256: {payload['probe_sha256']}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    output = Path(args.output)
    markdown = Path(args.markdown)
    if output.exists() or markdown.exists():
        raise FileExistsError("predictive decision probe outputs are write-once")
    payload = run_probe(device_name=args.device)
    _write_json_new(output, payload)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    if markdown.exists():
        raise FileExistsError(markdown)
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
