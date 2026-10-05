"""Post-run artifact validator for the formal V18.5 UJI protocol.

This module is deliberately read-only with respect to formal model artifacts.
It validates the provenance graph and numerical contracts after
prepare -> develop -> calibrate-freeze has completed.

It does not open UJI validation or any external test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import REPO_ROOT
from fhm_dapp_tim.fhm_dapp.v18_5_core import kl_budget_tolerance
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    DEFAULT_CALIBRATION,
    DEFAULT_DEVELOPMENT,
    DEFAULT_FREEZE,
    DEFAULT_REGISTRY,
    PROTOCOL_VERSION,
    _canonical_sha,
    _sha256_file,
    _sha256_indices,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT / "evidence" / "v18_5" / "FORMAL_UJI_ARTIFACT_AUDIT_V18_5.json"
)


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _finite_number(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: nonfinite value")
    return number


def _method_source_compatible(source_sha: str) -> None:
    if len(str(source_sha)) != 40:
        raise ValueError("invalid formal V18.5 source SHA")
    tracked = [
        "fhm_dapp_tim/fhm_dapp",
        "fhm_dapp_tim/evidence/v18_5/DESIGN_V18_5.md",
        "fhm_dapp_tim/evidence/v18_5/RUNBOOK_UJI_V18_5.md",
        "fhm_dapp_tim/evidence/v18_5/RUNBOOK_EXTERNAL_V18_5.md",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", str(source_sha), "HEAD", "--", *tracked],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    if result == 1:
        raise RuntimeError("V18.5 method source changed after formal prepare")
    if result != 0:
        raise RuntimeError("cannot compare V18.5 source provenance")


def _validate_role_registry(
    registry: Mapping[str, object],
) -> dict[str, object]:
    roles = registry["roles"]
    required = {"fit", "development", "selection_refit", "calibration"}
    if set(roles) != required:
        raise ValueError("registry role set mismatch")

    decoded: dict[str, np.ndarray] = {}
    for name, record in roles.items():
        indices = np.asarray(record["rows"], dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"registry role {name} is empty/invalid")
        if _sha256_indices(indices) != record["row_index_sha256"]:
            raise ValueError(f"registry role hash mismatch: {name}")
        if int(record["row_count"]) != len(indices):
            raise ValueError(f"registry role row count mismatch: {name}")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError(f"registry role has duplicate rows: {name}")
        decoded[name] = indices

    fit = set(decoded["fit"].tolist())
    dev = set(decoded["development"].tolist())
    cal = set(decoded["calibration"].tolist())
    refit = set(decoded["selection_refit"].tolist())
    if fit & dev or fit & cal or dev & cal:
        raise ValueError("fit/development/calibration roles overlap")
    if refit != fit | dev:
        raise ValueError("selection_refit is not exactly fit + development")
    expected = set(range(int(registry["training_rows"])))
    if fit | dev | cal != expected:
        raise ValueError("fit/development/calibration do not partition training")
    return {
        name: {
            "rows": int(len(indices)),
            "coordinate_groups": int(roles[name]["coordinate_count"]),
        }
        for name, indices in decoded.items()
    }


def validate_registry(
    path: str | Path = DEFAULT_REGISTRY,
) -> tuple[dict[str, object], dict[str, object]]:
    registry = _load_json(path)
    if registry.get("registry_sha256") != _canonical_sha(
        registry, omit="registry_sha256"
    ):
        raise ValueError("registry self-hash mismatch")
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("registry protocol version mismatch")
    if len(str(registry.get("source_commit_sha", ""))) != 40:
        raise ValueError("registry source SHA invalid")
    if registry.get("tracked_source_dirty") is not False:
        raise ValueError("registry was prepared from dirty tracked source")
    if registry.get("uji_official_validation_used_for_v185_selection") is not False:
        raise ValueError("registry records UJI validation selection use")
    if registry.get("external_test_performance_used_for_v185_selection") is not False:
        raise ValueError("registry records external performance selection use")

    expected_grids = {
        "lambda_spatial_grid": list(map(float, V185_CONFIG["lambda_spatial_grid"])),
        "temperature_grid": list(map(float, V185_CONFIG["temperature_grid"])),
        "epsilon_kl_grid": list(map(float, V185_CONFIG["epsilon_kl_grid"])),
        "scale_grid": list(map(float, V185_CONFIG["scale_grid"])),
    }
    for key, expected in expected_grids.items():
        if list(map(float, registry[key])) != expected:
            raise ValueError(f"registry {key} differs from V18.5 design")

    train_path = REPO_ROOT / str(registry["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if _sha256_file(train_path) != registry["train_sha256"]:
        raise ValueError("registry training source hash mismatch")
    roles = _validate_role_registry(registry)
    _method_source_compatible(str(registry["source_commit_sha"]))
    summary = {
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "training_rows": int(registry["training_rows"]),
        "training_coordinate_groups": int(
            registry["training_coordinate_groups"]
        ),
        "aps": int(registry["aps"]),
        "roles": roles,
    }
    return registry, summary


def _validate_fit_checkpoint(
    path: Path,
    *,
    expected_lambda: float,
    expected_epoch: int,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{path}: checkpoint protocol mismatch")
    if checkpoint.get("phase") != "fit_candidate":
        raise ValueError(f"{path}: checkpoint phase mismatch")
    if float(checkpoint["lambda_spatial"]) != float(expected_lambda):
        raise ValueError(f"{path}: lambda mismatch")
    if int(checkpoint["selected_epoch"]) != int(expected_epoch):
        raise ValueError(f"{path}: selected epoch mismatch")
    scale = _finite_number(
        checkpoint["spatial_scale_m"], name=f"{path}: spatial scale"
    )
    if scale <= 0:
        raise ValueError(f"{path}: nonpositive spatial scale")

    memory = checkpoint["memory"]
    required_memory = {
        "prototype_rss",
        "prototype_observed_mask",
        "detection_prob",
        "rss_center",
        "rss_scale",
        "local_observed_count",
        "ap_pooled_count",
        "ap_pooled_center",
        "ap_pooled_scale",
        "global_rss_scale",
        "coords",
        "building",
        "floor",
        "counts",
        "seen_ap_mask",
    }
    if not required_memory.issubset(memory):
        raise ValueError(f"{path}: incomplete explicit-mask anchor memory")
    if not bool(torch.isfinite(memory["rss_scale"]).all()):
        raise ValueError(f"{path}: nonfinite radio scale")
    if float(memory["rss_scale"].min()) < 0.05 - 1e-8:
        raise ValueError(f"{path}: radio scale below registered floor")
    if memory["prototype_observed_mask"].dtype != torch.bool:
        raise ValueError(f"{path}: prototype observed mask is not bool")
    for name, tensor in checkpoint["model_state"].items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{path}: nonfinite model tensor {name}")
    return {
        "anchors": int(len(memory["coords"])),
        "spatial_scale_m": scale,
        "memory_diagnostics": checkpoint["memory_diagnostics"],
    }


def validate_development(
    development_path: str | Path,
    *,
    registry: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    development = _load_json(development_path)
    if development.get("development_result_sha256") != _canonical_sha(
        development, omit="development_result_sha256"
    ):
        raise ValueError("development self-hash mismatch")
    if development.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("development protocol mismatch")
    if development.get("source_commit_sha") != registry["source_commit_sha"]:
        raise ValueError("development source link mismatch")
    if development.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("development registry link mismatch")
    if development.get("official_validation_accessed") is not False:
        raise ValueError("development accessed UJI validation")
    if development.get("external_test_performance_accessed") is not False:
        raise ValueError("development accessed external performance")

    expected_lambdas = list(map(float, V185_CONFIG["lambda_spatial_grid"]))
    reports = development["lambda_reports"]
    checkpoints = development["checkpoints"]
    if len(reports) != len(expected_lambdas) or len(checkpoints) != len(
        expected_lambdas
    ):
        raise ValueError("development must contain exactly four lambda models")
    report_by_lambda = {float(item["lambda_spatial"]): item for item in reports}
    checkpoint_by_lambda = {
        float(item["lambda_spatial"]): item for item in checkpoints
    }
    if set(report_by_lambda) != set(expected_lambdas):
        raise ValueError("development lambda report grid mismatch")
    if set(checkpoint_by_lambda) != set(expected_lambdas):
        raise ValueError("development checkpoint grid mismatch")

    checkpoint_summary = {}
    for lam in expected_lambdas:
        report = report_by_lambda[lam]
        epoch = int(report["selected_epoch"])
        if not (
            int(V185_CONFIG["minimum_epochs"])
            <= epoch
            <= int(V185_CONFIG["maximum_epochs"])
        ):
            raise ValueError("development selected epoch outside grid")
        if len(report["trajectory"]) != int(V185_CONFIG["maximum_epochs"]):
            raise ValueError("development trajectory length mismatch")
        if [int(x["epoch"]) for x in report["trajectory"]] != list(
            range(1, int(V185_CONFIG["maximum_epochs"]) + 1)
        ):
            raise ValueError("development trajectory epoch sequence mismatch")
        metrics = report["selected_base_metrics"]
        for key in (
            "energy_score_m",
            "energy_score_se_m",
            "nll",
            "nll_se",
            "posterior_mean_mae_m",
            "posterior_mean_mae_se_m",
            "mean_entropy_nats",
            "mean_effective_anchors",
            "mean_top1_probability",
        ):
            _finite_number(metrics[key], name=f"lambda={lam}:{key}")

        record = checkpoint_by_lambda[lam]
        checkpoint_path = REPO_ROOT / str(record["path"])
        if _sha256_file(checkpoint_path) != record["sha256"]:
            raise ValueError(f"lambda={lam}: checkpoint SHA mismatch")
        if report["checkpoint_path"] != record["path"]:
            raise ValueError(f"lambda={lam}: report checkpoint path mismatch")
        if report["checkpoint_sha256"] != record["sha256"]:
            raise ValueError(f"lambda={lam}: report checkpoint SHA mismatch")
        checkpoint_summary[str(lam)] = _validate_fit_checkpoint(
            checkpoint_path,
            expected_lambda=lam,
            expected_epoch=epoch,
        )

    selected = development["selected"]
    selected_lambda = float(selected["lambda_spatial"])
    selected_epoch = int(selected["selected_epoch"])
    if selected_lambda not in report_by_lambda:
        raise ValueError("development selected unregistered lambda")
    if selected_epoch != int(
        report_by_lambda[selected_lambda]["selected_epoch"]
    ):
        raise ValueError("development selected epoch/report mismatch")

    return development, {
        "development_result_sha256": development[
            "development_result_sha256"
        ],
        "selected_lambda_spatial": selected_lambda,
        "selected_epoch": selected_epoch,
        "checkpoint_summary": checkpoint_summary,
    }


def _cell_key(cell: Mapping[str, object]) -> tuple[float, float, float]:
    return (
        float(cell["temperature"]),
        float(cell["epsilon_kl"]),
        float(cell["scale_multiplier"]),
    )


def validate_calibration_surface(
    surface: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected = {
        (float(t), float(e), float(s))
        for t in V185_CONFIG["temperature_grid"]
        for e in V185_CONFIG["epsilon_kl_grid"]
        for s in V185_CONFIG["scale_grid"]
    }
    if len(surface) != len(expected):
        raise ValueError(f"expected 192 calibration cells, got {len(surface)}")
    actual = [_cell_key(cell) for cell in surface]
    if len(set(actual)) != len(actual):
        raise ValueError("calibration surface contains duplicate cells")
    if set(actual) != expected:
        raise ValueError("calibration surface grid mismatch")

    tolerance = kl_budget_tolerance(torch.float32)
    positive_cells = 0
    identity_cells = 0
    max_excess = -float("inf")
    for cell in surface:
        key = _cell_key(cell)
        for metric in (
            "energy_score_m",
            "energy_score_se_m",
            "nll",
            "nll_se",
            "posterior_mean_mae_m",
            "posterior_mean_mae_se_m",
            "mean_entropy_nats",
            "mean_effective_anchors",
            "mean_top1_probability",
            "mean_radio_compatibility",
            "mean_realized_kl",
            "mean_effective_epsilon_kl",
            "max_kl_budget_excess",
            "dual_beta_infinite_fraction",
        ):
            _finite_number(cell[metric], name=f"cell={key}:{metric}")
        beta = cell.get("mean_finite_dual_beta")
        if beta is not None:
            _finite_number(beta, name=f"cell={key}:mean_finite_dual_beta")
            if float(beta) < 0:
                raise ValueError("finite dual beta cannot be negative")
        if not (0.0 <= float(cell["mean_radio_compatibility"]) <= 1.0):
            raise ValueError("radio compatibility outside [0,1]")
        if not (0.0 <= float(cell["mean_top1_probability"]) <= 1.0):
            raise ValueError("top1 probability outside [0,1]")
        if not (0.0 <= float(cell["dual_beta_infinite_fraction"]) <= 1.0):
            raise ValueError("infinite dual fraction outside [0,1]")
        if float(cell["mean_effective_epsilon_kl"]) < -tolerance:
            raise ValueError("negative effective KL budget")
        if float(cell["mean_effective_epsilon_kl"]) > key[1] + tolerance:
            raise ValueError("effective KL budget exceeds registered epsilon")
        excess = float(cell["max_kl_budget_excess"])
        max_excess = max(max_excess, excess)
        if excess > tolerance:
            raise ValueError("calibration cell violates returned-dtype KL budget")

        if key[1] == 0.0:
            identity_cells += 1
            if abs(float(cell["mean_realized_kl"])) > tolerance:
                raise ValueError("epsilon=0 cell has nonzero realized KL")
            if abs(float(cell["mean_effective_epsilon_kl"])) > tolerance:
                raise ValueError("epsilon=0 cell has nonzero effective budget")
            if abs(float(cell["dual_beta_infinite_fraction"])) > tolerance:
                raise ValueError("epsilon=0 cell has infinite dual")
            if beta is None or abs(float(beta)) > tolerance:
                raise ValueError("epsilon=0 cell must have beta=0")
        else:
            positive_cells += 1

    return {
        "cells": int(len(surface)),
        "identity_epsilon_cells": int(identity_cells),
        "positive_epsilon_cells": int(positive_cells),
        "maximum_kl_budget_excess": float(max_excess),
        "allowed_kl_budget_tolerance": float(tolerance),
    }


def _validate_refit_checkpoint(
    path: Path,
    *,
    registry: Mapping[str, object],
    development: Mapping[str, object],
    calibration: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_links = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "selection_refit",
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development[
            "development_result_sha256"
        ],
    }
    for key, expected in expected_links.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"selection-refit checkpoint {key} mismatch")
    if float(checkpoint["lambda_spatial"]) != float(
        development["selected"]["lambda_spatial"]
    ):
        raise ValueError("selection-refit checkpoint lambda mismatch")
    if int(checkpoint["selected_epoch"]) != int(
        development["selected"]["selected_epoch"]
    ):
        raise ValueError("selection-refit checkpoint epoch mismatch")

    references = checkpoint["temperature_radio_reference"]
    expected_temperatures = set(map(float, V185_CONFIG["temperature_grid"]))
    actual_temperatures = {float(key) for key in references}
    if actual_temperatures != expected_temperatures:
        raise ValueError("temperature radio reference grid mismatch")
    expected_rows = int(registry["roles"]["selection_refit"]["row_count"])
    reference_summary = {}
    for key, values in references.items():
        vector = torch.as_tensor(values).double()
        if vector.ndim != 1 or len(vector) != expected_rows:
            raise ValueError("temperature radio reference row count mismatch")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("temperature radio reference is nonfinite")
        if bool((vector[1:] < vector[:-1]).any()):
            raise ValueError("temperature radio reference is not sorted")
        reference_summary[str(float(key))] = {
            "rows": int(len(vector)),
            "minimum": float(vector[0]),
            "median": float(torch.quantile(vector, 0.50)),
            "maximum": float(vector[-1]),
        }

    memory = checkpoint["memory"]
    if memory["prototype_observed_mask"].dtype != torch.bool:
        raise ValueError("selection-refit prototype mask is not bool")
    if float(memory["rss_scale"].min()) < 0.05 - 1e-8:
        raise ValueError("selection-refit radio scale below floor")
    return {
        "anchors": int(len(memory["coords"])),
        "memory_diagnostics": checkpoint["memory_diagnostics"],
        "temperature_references": reference_summary,
    }


def validate_calibration_and_freeze(
    calibration_path: str | Path,
    freeze_path: str | Path,
    *,
    registry: Mapping[str, object],
    development: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    calibration = _load_json(calibration_path)
    if calibration.get("calibration_result_sha256") != _canonical_sha(
        calibration, omit="calibration_result_sha256"
    ):
        raise ValueError("calibration self-hash mismatch")
    for key, expected in (
        ("protocol_version", PROTOCOL_VERSION),
        ("source_commit_sha", registry["source_commit_sha"]),
        ("registry_sha256", registry["registry_sha256"]),
        (
            "development_result_sha256",
            development["development_result_sha256"],
        ),
    ):
        if calibration.get(key) != expected:
            raise ValueError(f"calibration {key} mismatch")
    if calibration.get("official_validation_accessed") is not False:
        raise ValueError("calibration accessed UJI validation")
    if calibration.get("external_test_performance_accessed") is not False:
        raise ValueError("calibration accessed external performance")
    surface_summary = validate_calibration_surface(calibration["surface"])

    selection = calibration["selection"]
    selected_cell = selection["selected_cell"]
    selected_key = _cell_key(selected_cell)
    surface_by_key = {_cell_key(cell): cell for cell in calibration["surface"]}
    if selected_key not in surface_by_key:
        raise ValueError("selected calibration cell absent from surface")
    surface_selected = surface_by_key[selected_key]
    for metric in (
        "nll",
        "energy_score_m",
        "posterior_mean_mae_m",
    ):
        if abs(float(selected_cell[metric]) - float(surface_selected[metric])) > 1e-12:
            raise ValueError(f"selected calibration {metric} mismatch")
    if selection["selected_cell_id"] != surface_selected["cell_id"]:
        raise ValueError("selected calibration cell_id mismatch")
    summary_selected = calibration["selected"]
    if (
        float(summary_selected["temperature"]),
        float(summary_selected["epsilon_kl"]),
        float(summary_selected["scale_multiplier"]),
    ) != selected_key:
        raise ValueError("calibration selected summary parameters mismatch")

    checkpoint_path = REPO_ROOT / str(
        calibration["selection_refit_checkpoint"]
    )
    if _sha256_file(checkpoint_path) != calibration[
        "selection_refit_checkpoint_sha256"
    ]:
        raise ValueError("selection-refit checkpoint hash mismatch")
    checkpoint_summary = _validate_refit_checkpoint(
        checkpoint_path,
        registry=registry,
        development=development,
        calibration=calibration,
    )

    freeze = _load_json(freeze_path)
    if freeze.get("freeze_sha256") != _canonical_sha(
        freeze, omit="freeze_sha256"
    ):
        raise ValueError("method freeze self-hash mismatch")
    links = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development[
            "development_result_sha256"
        ],
        "calibration_result_sha256": calibration[
            "calibration_result_sha256"
        ],
        "selection_refit_checkpoint": calibration[
            "selection_refit_checkpoint"
        ],
        "selection_refit_checkpoint_sha256": calibration[
            "selection_refit_checkpoint_sha256"
        ],
    }
    for key, expected in links.items():
        if freeze.get(key) != expected:
            raise ValueError(f"freeze {key} mismatch")
    if freeze.get("official_validation_accessed") is not False:
        raise ValueError("freeze records UJI validation access")
    if freeze.get("external_test_performance_accessed") is not False:
        raise ValueError("freeze records external performance access")
    if freeze.get("decoder") != "posterior_mean_of_component_locations":
        raise ValueError("freeze decoder mismatch")
    if freeze.get("correction") != "radio_only_exact_KL_projection":
        raise ValueError("freeze correction mismatch")
    if freeze.get("geometry_posterior_correction") is not False:
        raise ValueError("freeze unexpectedly enables geometry correction")

    expected_method = {
        "lambda_spatial": float(development["selected"]["lambda_spatial"]),
        "selected_epoch": int(development["selected"]["selected_epoch"]),
        "posterior_temperature": selected_key[0],
        "epsilon_kl": selected_key[1],
        "scale_multiplier": selected_key[2],
    }
    for key, expected in expected_method.items():
        actual = freeze[key]
        if isinstance(expected, int):
            if int(actual) != expected:
                raise ValueError(f"freeze {key} mismatch")
        elif float(actual) != float(expected):
            raise ValueError(f"freeze {key} mismatch")

    summary = {
        "calibration_result_sha256": calibration[
            "calibration_result_sha256"
        ],
        "freeze_sha256": freeze["freeze_sha256"],
        "selected_method": expected_method,
        "surface": surface_summary,
        "selection_refit": checkpoint_summary,
    }
    return freeze, summary


def run_artifact_audit(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    development_path: str | Path = DEFAULT_DEVELOPMENT,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    freeze_path: str | Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    registry, registry_summary = validate_registry(registry_path)
    development, development_summary = validate_development(
        development_path, registry=registry
    )
    freeze, calibration_summary = validate_calibration_and_freeze(
        calibration_path,
        freeze_path,
        registry=registry,
        development=development,
    )
    return {
        "audit_version": "v18_5_formal_uji_artifact_audit_v1_20261005",
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "registry": registry_summary,
        "development": development_summary,
        "calibration_and_freeze": calibration_summary,
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
        "formal_method_freeze_sha256": freeze["freeze_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--development", default=str(DEFAULT_DEVELOPMENT))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = run_artifact_audit(
        registry_path=args.registry,
        development_path=args.development,
        calibration_path=args.calibration,
        freeze_path=args.freeze,
    )
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                payload, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
