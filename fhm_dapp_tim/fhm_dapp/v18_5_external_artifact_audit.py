"""Read-only artifact audit for V18.5 reserved external evaluation.

Training-freeze audit never loads a provided external test split.  Result audit
is intended only after the irreversible test lock/result already exist; it
validates provenance, hashes and numerical contracts without retuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import REPO_ROOT
from fhm_dapp_tim.fhm_dapp.v18_5_artifact_audit import (
    validate_calibration_surface,
    validate_joint_selection,
    validate_radio_diagnostics,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import kl_budget_tolerance
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG
from fhm_dapp_tim.fhm_dapp import v18_5_external as external


def _load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _finite(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}: nonfinite value")
    return number


def _validate_roles(manifest: Mapping[str, object]) -> dict[str, object]:
    roles = manifest["roles"]
    required = {"fit", "development", "selection_refit", "calibration"}
    if set(roles) != required:
        raise ValueError("external manifest role set mismatch")
    decoded: dict[str, np.ndarray] = {}
    for name, record in roles.items():
        rows = np.asarray(record["rows"], dtype=np.int64)
        if rows.ndim != 1 or len(rows) == 0:
            raise ValueError(f"external role {name} is empty/invalid")
        if len(set(rows.tolist())) != len(rows):
            raise ValueError(f"external role {name} contains duplicate rows")
        if external._sha256_indices(rows) != record["row_index_sha256"]:
            raise ValueError(f"external role {name} row hash mismatch")
        if int(record["row_count"]) != len(rows):
            raise ValueError(f"external role {name} row count mismatch")
        decoded[name] = rows

    fit = set(decoded["fit"].tolist())
    dev = set(decoded["development"].tolist())
    cal = set(decoded["calibration"].tolist())
    refit = set(decoded["selection_refit"].tolist())
    if fit & dev or fit & cal or dev & cal:
        raise ValueError("external fit/development/calibration overlap")
    if refit != fit | dev:
        raise ValueError("external selection_refit != fit + development")
    expected = set(range(int(manifest["training_rows"])))
    if fit | dev | cal != expected:
        raise ValueError("external roles do not partition training rows")
    return {
        name: {
            "rows": int(len(rows)),
            "coordinate_groups": int(roles[name]["coordinate_count"]),
        }
        for name, rows in decoded.items()
    }


def _cell_key(cell: Mapping[str, object]) -> tuple[float, float, float]:
    return (
        float(cell["temperature"]),
        float(cell["epsilon_kl"]),
        float(cell["scale_multiplier"]),
    )


def validate_selection(
    calibration: Mapping[str, object],
) -> dict[str, object]:
    """Compatibility wrapper around the shared UJI/external selector audit."""

    return validate_joint_selection(calibration)


def _validate_checkpoint(
    checkpoint_path: Path,
    *,
    manifest: Mapping[str, object],
    freeze: Mapping[str, object],
) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    expected = {
        "protocol_version": external.PROTOCOL_VERSION,
        "phase": "external_selection_refit",
        "slug": manifest["slug"],
        "dataset": manifest["dataset"],
        "heldout_device": manifest["heldout_device"],
        "execution_source_commit_sha": manifest["execution_source_commit_sha"],
        "uji_method_source_commit_sha": manifest[
            "uji_method_source_commit_sha"
        ],
        "uji_method_freeze_sha256": manifest["uji_method_freeze_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"external checkpoint {key} mismatch")
    for key in ("lambda_spatial", "posterior_temperature", "epsilon_kl", "scale_multiplier"):
        if float(checkpoint[key]) != float(freeze[key]):
            raise ValueError(f"external checkpoint {key} mismatch freeze")
    if int(checkpoint["selected_epoch"]) != int(freeze["selected_epoch"]):
        raise ValueError("external checkpoint epoch mismatch freeze")
    if int(checkpoint["selection_refit_rows"]) != int(
        freeze["selection_refit_rows"]
    ):
        raise ValueError("external checkpoint selection-refit row mismatch")

    memory = checkpoint["memory"]
    required = {
        "prototype_rss",
        "prototype_observed_mask",
        "detection_prob",
        "rss_center",
        "rss_scale",
        "coords",
        "building",
        "floor",
        "counts",
        "seen_ap_mask",
    }
    if not required.issubset(memory):
        raise ValueError("external checkpoint has incomplete anchor memory")
    if memory["prototype_observed_mask"].dtype != torch.bool:
        raise ValueError("external checkpoint prototype mask is not bool")
    if not bool(torch.isfinite(memory["rss_scale"]).all()):
        raise ValueError("external checkpoint radio scale is nonfinite")
    if float(memory["rss_scale"].min()) < 0.05 - 1e-8:
        raise ValueError("external checkpoint radio scale below floor")
    for name, tensor in checkpoint["model_state"].items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"external checkpoint model tensor nonfinite: {name}")

    references = checkpoint["temperature_radio_reference"]
    expected_t = set(map(float, V185_CONFIG["temperature_grid"]))
    if {float(key) for key in references} != expected_t:
        raise ValueError("external checkpoint temperature reference grid mismatch")
    expected_rows = int(freeze["selection_refit_rows"])
    for key, value in references.items():
        vector = torch.as_tensor(value).double()
        if vector.ndim != 1 or len(vector) != expected_rows:
            raise ValueError("external temperature reference row count mismatch")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("external temperature reference is nonfinite")
        if bool((vector[1:] < vector[:-1]).any()):
            raise ValueError("external temperature reference is not sorted")

    reference = checkpoint["dataset_reference"]
    if reference != manifest["dataset_reference"]:
        raise ValueError("external checkpoint dataset reference mismatch")
    if manifest["heldout_device"] is not None:
        if str(manifest["heldout_device"]) in reference["device_mapping"]:
            raise ValueError("held-out device leaked into external checkpoint mapping")

    return {
        "anchors": int(len(memory["coords"])),
        "seen_aps": int(memory["seen_ap_mask"].sum()),
        "temperature_references": int(len(references)),
        "selection_refit_rows": expected_rows,
        "memory_diagnostics": checkpoint["memory_diagnostics"],
    }


def validate_external_training_chain(
    slug: str,
    *,
    uji_freeze_path: str | Path = external.DEFAULT_UJI_FREEZE,
    manifest_path: str | Path | None = None,
    training_freeze_path: str | Path | None = None,
) -> dict[str, object]:
    spec = external._spec(slug)
    defaults = external._default_paths(slug)
    manifest = _load_json(manifest_path or defaults["manifest"])
    freeze = _load_json(training_freeze_path or defaults["freeze"])

    if manifest.get("manifest_sha256") != external._canonical_sha(
        manifest, omit="manifest_sha256"
    ):
        raise ValueError("external manifest self-hash mismatch")
    if freeze.get("training_freeze_sha256") != external._canonical_sha(
        freeze, omit="training_freeze_sha256"
    ):
        raise ValueError("external training freeze self-hash mismatch")
    if manifest.get("protocol_version") != external.PROTOCOL_VERSION:
        raise ValueError("external manifest protocol mismatch")
    if freeze.get("protocol_version") != external.PROTOCOL_VERSION:
        raise ValueError("external training freeze protocol mismatch")
    if manifest.get("slug") != spec.slug or freeze.get("slug") != spec.slug:
        raise ValueError("external slug mismatch")
    if manifest.get("dataset") != spec.dataset or freeze.get("dataset") != spec.dataset:
        raise ValueError("external dataset mismatch")
    if manifest.get("heldout_device") != spec.heldout_device:
        raise ValueError("external manifest heldout device mismatch")
    if freeze.get("heldout_device") != spec.heldout_device:
        raise ValueError("external freeze heldout device mismatch")

    uji = external.load_uji_method_freeze(uji_freeze_path)
    if manifest["uji_method_freeze_sha256"] != uji["freeze_sha256"]:
        raise ValueError("external manifest links wrong UJI freeze")
    if freeze["uji_method_freeze_sha256"] != uji["freeze_sha256"]:
        raise ValueError("external freeze links wrong UJI freeze")
    if freeze["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("external freeze links wrong manifest")
    if float(manifest["uji_frozen_lambda_spatial"]) != float(uji["lambda_spatial"]):
        raise ValueError("external manifest lambda differs from UJI freeze")
    if float(freeze["lambda_spatial"]) != float(uji["lambda_spatial"]):
        raise ValueError("external freeze lambda differs from UJI freeze")
    if freeze["execution_source_commit_sha"] != manifest["execution_source_commit_sha"]:
        raise ValueError("external prepare/fit source commit mismatch")
    external._method_source_compatible(
        str(freeze["execution_source_commit_sha"])
    )

    roles = _validate_roles(manifest)
    for record in manifest["training_files"]:
        source = REPO_ROOT / str(record["path"])
        if not source.exists():
            raise FileNotFoundError(source)
        if external._sha256_file(source) != record["sha256"]:
            raise ValueError("external training source hash mismatch")

    rule = manifest["external_calibration_rule"]
    if list(map(float, rule["temperature_grid"])) != list(
        map(float, V185_CONFIG["temperature_grid"])
    ):
        raise ValueError("external temperature grid mismatch")
    if list(map(float, rule["epsilon_kl_grid"])) != list(
        map(float, V185_CONFIG["epsilon_kl_grid"])
    ):
        raise ValueError("external epsilon grid mismatch")
    if list(map(float, rule["scale_grid"])) != list(
        map(float, V185_CONFIG["scale_grid"])
    ):
        raise ValueError("external scale grid mismatch")
    if int(rule["cells"]) != 192:
        raise ValueError("external calibration rule is not 192 cells")

    if freeze.get("decoder") != "posterior_mean_of_component_locations":
        raise ValueError("external decoder mismatch")
    if freeze.get("correction") != "radio_only_exact_KL_projection":
        raise ValueError("external correction mismatch")
    if freeze.get("geometry_posterior_correction") is not False:
        raise ValueError("external geometry correction must be disabled")
    if freeze.get("formal_test_performance_evaluated") is not False:
        raise ValueError("training freeze already records test evaluation")
    if freeze.get("formal_test_performance_lock_written") is not False:
        raise ValueError("training freeze already records test lock")

    if int(freeze["selected_epoch"]) < int(V185_CONFIG["minimum_epochs"]) or int(
        freeze["selected_epoch"]
    ) > int(V185_CONFIG["maximum_epochs"]):
        raise ValueError("external selected epoch outside registered range")
    if int(freeze["selection_refit_rows"]) != roles["selection_refit"]["rows"]:
        raise ValueError("external selection_refit row count mismatch")

    calibration = freeze["calibration"]
    if int(calibration["rows"]) != roles["calibration"]["rows"]:
        raise ValueError("external calibration row count mismatch")
    surface_summary = validate_calibration_surface(calibration["surface"])
    radio_summary = validate_radio_diagnostics(
        calibration["radio_diagnostics_by_temperature"],
        calibration_rows=int(calibration["rows"]),
        selection_refit_rows=int(freeze["selection_refit_rows"]),
    )
    selection_summary = validate_selection(calibration)
    selected_parameters = selection_summary["selected_parameters"]
    for key, freeze_key in (
        ("temperature", "posterior_temperature"),
        ("epsilon_kl", "epsilon_kl"),
        ("scale_multiplier", "scale_multiplier"),
    ):
        if float(selected_parameters[key]) != float(freeze[freeze_key]):
            raise ValueError(f"external selected {key} mismatch freeze")

    checkpoint = REPO_ROOT / str(freeze["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if external._sha256_file(checkpoint) != freeze["checkpoint_sha256"]:
        raise ValueError("external checkpoint SHA mismatch")
    checkpoint_summary = _validate_checkpoint(
        checkpoint, manifest=manifest, freeze=freeze
    )

    if manifest.get(
        "provided_test_historically_preexposed_for_adapter_validation"
    ) is not True:
        raise ValueError("external manifest lost historical pre-exposure record")
    return {
        "status": "PASS",
        "slug": slug,
        "dataset": spec.dataset,
        "manifest_sha256": manifest["manifest_sha256"],
        "training_freeze_sha256": freeze["training_freeze_sha256"],
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "roles": roles,
        "surface": surface_summary,
        "radio_diagnostics": radio_summary,
        "selection": selection_summary,
        "checkpoint": checkpoint_summary,
        "provided_test_loaded": False,
    }


def _validate_test_metrics(
    metrics: Mapping[str, object],
    *,
    epsilon_kl: float,
) -> dict[str, object]:
    for key in (
        "posterior_mean_coordinate_macro_mae_m",
        "posterior_mean_coordinate_macro_mae_se_m",
        "row_mae_m",
        "row_median_error_m",
        "row_p75_error_m",
        "row_p90_error_m",
        "energy_score_coordinate_macro_m",
        "energy_score_coordinate_macro_se_m",
        "mixture_nll_coordinate_macro",
        "mixture_nll_coordinate_macro_se",
        "mean_entropy_nats",
        "mean_effective_anchors",
        "mean_top1_probability",
        "mean_support_size",
        "natural_truncation_fraction",
    ):
        _finite(metrics[key], name=f"external test metric:{key}")
    if int(metrics["rows"]) <= 0 or int(metrics["coordinate_groups"]) <= 0:
        raise ValueError("external result has empty test rows/groups")
    if not 0.0 <= float(metrics["mean_top1_probability"]) <= 1.0:
        raise ValueError("external result top1 probability outside [0,1]")
    if not 0.0 <= float(metrics["natural_truncation_fraction"]) <= 1.0:
        raise ValueError("external truncation fraction outside [0,1]")

    coverage = metrics["coverage"]
    if set(coverage) != {"50", "80", "90", "95"}:
        raise ValueError("external coverage levels mismatch")
    for level, record in coverage.items():
        for key in (
            "row_coverage",
            "coordinate_macro_coverage",
            "mean_radius_m",
            "median_radius_m",
        ):
            value = _finite(record[key], name=f"coverage {level}:{key}")
            if "coverage" in key and not 0.0 <= value <= 1.0:
                raise ValueError("coverage outside [0,1]")

    projection = metrics["radio_projection"]
    tolerance = kl_budget_tolerance(torch.float32)
    expected_active = float(epsilon_kl) > 0.0
    if bool(projection["active"]) != expected_active:
        raise ValueError("external radio projection active flag mismatch")
    mean_kl = _finite(
        projection["mean_realized_kl"], name="external mean realized KL"
    )
    mean_budget = _finite(
        projection["mean_effective_epsilon_kl"],
        name="external mean effective epsilon",
    )
    excess = _finite(
        projection["max_kl_budget_excess"],
        name="external max KL excess",
    )
    if mean_kl < -tolerance or mean_budget < -tolerance:
        raise ValueError("external KL diagnostic is negative")
    if mean_budget > float(epsilon_kl) + tolerance:
        raise ValueError("external effective budget exceeds frozen epsilon")
    if excess > tolerance:
        raise ValueError("external result violates KL budget")
    if not expected_active:
        if abs(mean_kl) > tolerance or abs(mean_budget) > tolerance:
            raise ValueError("epsilon=0 external result is not identity")
    return {
        "rows": int(metrics["rows"]),
        "coordinate_groups": int(metrics["coordinate_groups"]),
        "mae_m": float(metrics["posterior_mean_coordinate_macro_mae_m"]),
        "energy_score_m": float(metrics["energy_score_coordinate_macro_m"]),
        "nll": float(metrics["mixture_nll_coordinate_macro"]),
        "max_kl_budget_excess": excess,
    }


def validate_external_result_chain(
    slug: str,
    *,
    uji_freeze_path: str | Path = external.DEFAULT_UJI_FREEZE,
    manifest_path: str | Path | None = None,
    training_freeze_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    result_path: str | Path | None = None,
) -> dict[str, object]:
    defaults = external._default_paths(slug)
    training_summary = validate_external_training_chain(
        slug,
        uji_freeze_path=uji_freeze_path,
        manifest_path=manifest_path,
        training_freeze_path=training_freeze_path,
    )
    freeze = _load_json(training_freeze_path or defaults["freeze"])
    lock = _load_json(lock_path or defaults["lock"])
    result = _load_json(result_path or defaults["result"])

    if lock.get("lock_sha256") != external._canonical_sha(
        lock, omit="lock_sha256"
    ):
        raise ValueError("external test lock self-hash mismatch")
    if result.get("result_sha256") != external._canonical_sha(
        result, omit="result_sha256"
    ):
        raise ValueError("external test result self-hash mismatch")
    for key, expected in (
        ("protocol_version", external.PROTOCOL_VERSION),
        ("slug", slug),
        ("dataset", freeze["dataset"]),
        ("heldout_device", freeze["heldout_device"]),
        ("training_freeze_sha256", freeze["training_freeze_sha256"]),
        ("uji_method_freeze_sha256", freeze["uji_method_freeze_sha256"]),
        ("checkpoint_sha256", freeze["checkpoint_sha256"]),
    ):
        if lock.get(key) != expected:
            raise ValueError(f"external test lock {key} mismatch")
    if lock.get("formal_test_performance_evaluation_started") is not True:
        raise ValueError("external lock does not mark evaluation start")
    if lock.get("rerun_after_lock_forbidden") is not True:
        raise ValueError("external lock lost rerun prohibition")

    for key, expected in (
        ("protocol_version", external.PROTOCOL_VERSION),
        ("slug", slug),
        ("dataset", freeze["dataset"]),
        ("heldout_device", freeze["heldout_device"]),
        ("training_freeze_sha256", freeze["training_freeze_sha256"]),
        ("uji_method_freeze_sha256", freeze["uji_method_freeze_sha256"]),
        ("test_consumption_lock_sha256", lock["lock_sha256"]),
        ("checkpoint_sha256", freeze["checkpoint_sha256"]),
    ):
        if result.get(key) != expected:
            raise ValueError(f"external result {key} mismatch")
    if result.get("formal_test_performance_evaluated") is not True:
        raise ValueError("external result does not mark test evaluation")
    if result.get("no_post_test_retuning_permitted") is not True:
        raise ValueError("external result lost no-retuning boundary")
    method = result["method"]
    expected_method = {
        "lambda_spatial": float(freeze["lambda_spatial"]),
        "selected_epoch": int(freeze["selected_epoch"]),
        "posterior_temperature": float(freeze["posterior_temperature"]),
        "epsilon_kl": float(freeze["epsilon_kl"]),
        "scale_multiplier": float(freeze["scale_multiplier"]),
        "decoder": "posterior_mean_of_component_locations",
        "correction": "radio_only_exact_KL_projection",
    }
    for key, expected in expected_method.items():
        actual = method[key]
        if isinstance(expected, str):
            if actual != expected:
                raise ValueError(f"external result method {key} mismatch")
        elif isinstance(expected, int):
            if int(actual) != expected:
                raise ValueError(f"external result method {key} mismatch")
        elif float(actual) != expected:
            raise ValueError(f"external result method {key} mismatch")

    for record in result["test_files"]:
        path = REPO_ROOT / str(record["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if external._sha256_file(path) != record["sha256"]:
            raise ValueError("external test source hash mismatch")

    metric_summary = _validate_test_metrics(
        result["metrics"], epsilon_kl=float(freeze["epsilon_kl"])
    )
    return {
        "status": "PASS",
        "slug": slug,
        "training_chain": training_summary,
        "lock_sha256": lock["lock_sha256"],
        "result_sha256": result["result_sha256"],
        "metrics": metric_summary,
        "post_test_retuning_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", required=True, choices=sorted(external.external_specs())
    )
    parser.add_argument("--uji-freeze", default=str(external.DEFAULT_UJI_FREEZE))
    parser.add_argument("--manifest")
    parser.add_argument("--training-freeze")
    parser.add_argument("--lock")
    parser.add_argument("--result")
    parser.add_argument("--output")
    parser.add_argument(
        "--stage", choices=("training-freeze", "result"), required=True
    )
    args = parser.parse_args()

    if args.stage == "training-freeze":
        payload = validate_external_training_chain(
            args.protocol,
            uji_freeze_path=args.uji_freeze,
            manifest_path=args.manifest,
            training_freeze_path=args.training_freeze,
        )
    else:
        payload = validate_external_result_chain(
            args.protocol,
            uji_freeze_path=args.uji_freeze,
            manifest_path=args.manifest,
            training_freeze_path=args.training_freeze,
            lock_path=args.lock,
            result_path=args.result,
        )
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
