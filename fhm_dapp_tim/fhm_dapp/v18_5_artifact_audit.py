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


def _rederive_lambda_epoch_selection(
    report: Mapping[str, object],
) -> tuple[int, dict[str, object], set[int]]:
    minimum = int(V185_CONFIG["minimum_epochs"])
    maximum = int(V185_CONFIG["maximum_epochs"])
    eligible = []
    for record in report["trajectory"]:
        epoch = int(record["epoch"])
        metrics = record.get("development_base")
        if epoch < minimum:
            if metrics is not None:
                raise ValueError("ineligible epoch contains development metrics")
            continue
        if metrics is None:
            raise ValueError("eligible epoch is missing development metrics")
        if epoch > maximum:
            raise ValueError("trajectory exceeds registered maximum epoch")
        eligible.append((epoch, metrics))
    if [epoch for epoch, _ in eligible] != list(range(minimum, maximum + 1)):
        raise ValueError("eligible development epoch sequence mismatch")

    best_es = min(float(metrics["energy_score_m"]) for _, metrics in eligible)
    margin = float(V185_CONFIG["energy_tie_margin_m"])
    shortlist = [
        (epoch, metrics)
        for epoch, metrics in eligible
        if float(metrics["energy_score_m"]) <= best_es + margin
    ]
    selected_epoch, selected_metrics = min(
        shortlist,
        key=lambda item: (
            float(item[1]["nll"]),
            float(item[1]["posterior_mean_mae_m"]),
            int(item[0]),
        ),
    )
    return (
        int(selected_epoch),
        dict(selected_metrics),
        {int(epoch) for epoch, _ in shortlist},
    )


def _rederive_global_lambda(
    reports: Sequence[Mapping[str, object]],
) -> float:
    best_es = min(
        float(report["selected_base_metrics"]["energy_score_m"])
        for report in reports
    )
    margin = float(V185_CONFIG["energy_tie_margin_m"])
    shortlist = [
        report
        for report in reports
        if float(report["selected_base_metrics"]["energy_score_m"])
        <= best_es + margin
    ]
    selected = min(
        shortlist,
        key=lambda report: (
            float(report["selected_base_metrics"]["nll"]),
            float(report["selected_base_metrics"]["posterior_mean_mae_m"]),
            float(report["lambda_spatial"]),
        ),
    )
    return float(selected["lambda_spatial"])


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
        derived_epoch, derived_metrics, derived_shortlist = (
            _rederive_lambda_epoch_selection(report)
        )
        if epoch != derived_epoch:
            raise ValueError(
                f"lambda={lam}: selected epoch violates registered selector"
            )
        recorded_shortlist = {
            int(item["epoch"]) for item in report["eligible_epoch_shortlist"]
        }
        if recorded_shortlist != derived_shortlist:
            raise ValueError(
                f"lambda={lam}: eligible epoch shortlist mismatch"
            )
        metrics = report["selected_base_metrics"]
        for metric_name in (
            "energy_score_m",
            "energy_score_se_m",
            "nll",
            "nll_se",
            "posterior_mean_mae_m",
            "posterior_mean_mae_se_m",
            "mean_entropy_nats",
            "mean_effective_anchors",
            "mean_top1_probability",
            "rows",
            "coordinate_groups",
        ):
            actual = metrics[metric_name]
            expected = derived_metrics[metric_name]
            if isinstance(expected, int):
                if int(actual) != int(expected):
                    raise ValueError(
                        f"lambda={lam}: selected {metric_name} mismatch trajectory"
                    )
            elif abs(float(actual) - float(expected)) > 1e-12:
                raise ValueError(
                    f"lambda={lam}: selected {metric_name} mismatch trajectory"
                )
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
    derived_lambda = _rederive_global_lambda(reports)
    if selected_lambda != derived_lambda:
        raise ValueError(
            "development selected lambda violates registered global selector"
        )
    if selected_epoch != int(
        report_by_lambda[selected_lambda]["selected_epoch"]
    ):
        raise ValueError("development selected epoch/report mismatch")
    selected_metrics = selected["base_metrics"]
    report_metrics = report_by_lambda[selected_lambda]["selected_base_metrics"]
    for key in (
        "energy_score_m",
        "nll",
        "posterior_mean_mae_m",
    ):
        if abs(float(selected_metrics[key]) - float(report_metrics[key])) > 1e-12:
            raise ValueError(
                f"development selected base metric {key} mismatches lambda report"
            )

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


def _selection_choice_key(cell: Mapping[str, object]) -> tuple:
    return (
        float(cell["posterior_mean_mae_m"]),
        float(cell["epsilon_kl"]),
        abs(math.log(float(cell["temperature"]))),
        abs(math.log(float(cell["scale_multiplier"]))),
        float(cell["temperature"]),
        float(cell["scale_multiplier"]),
        str(cell["cell_id"]),
    )


def _validate_selection_vectors(
    calibration: Mapping[str, object],
    *,
    by_id: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    group_order = list(calibration["selection_group_order"])
    group_count = int(calibration["coordinate_groups"])
    if len(group_order) != group_count or group_count < 2:
        raise ValueError("selection group order/count mismatch")
    normalized_groups = []
    for group in group_order:
        if not isinstance(group, (list, tuple)) or len(group) != 4:
            raise ValueError("selection group identifier must be [b,f,x,y]")
        normalized = (
            int(group[0]),
            int(group[1]),
            float(group[2]),
            float(group[3]),
        )
        if not all(math.isfinite(value) for value in normalized[2:]):
            raise ValueError("selection group coordinate is nonfinite")
        normalized_groups.append(normalized)
    if len(set(normalized_groups)) != len(normalized_groups):
        raise ValueError("selection group order contains duplicates")

    raw_vectors = dict(calibration["selection_vectors"])
    if set(raw_vectors) != set(by_id):
        raise ValueError("selection-vector cell IDs mismatch calibration surface")
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for cell_id, payload in raw_vectors.items():
        record: dict[str, np.ndarray] = {}
        for vector_name, surface_mean, surface_se in (
            ("group_nll", "nll", "nll_se"),
            ("group_energy_score", "energy_score_m", "energy_score_se_m"),
        ):
            array = np.asarray(payload[vector_name], dtype=np.float64)
            if array.shape != (group_count,) or not np.isfinite(array).all():
                raise ValueError(
                    f"{cell_id}:{vector_name} must be finite and aligned"
                )
            mean = float(array.mean())
            se = float(array.std(ddof=1) / math.sqrt(group_count))
            if abs(mean - float(by_id[cell_id][surface_mean])) > 1e-12:
                raise ValueError(
                    f"{cell_id}:{vector_name} mean mismatches surface"
                )
            if abs(se - float(by_id[cell_id][surface_se])) > 1e-12:
                raise ValueError(
                    f"{cell_id}:{vector_name} SE mismatches surface"
                )
            record[vector_name] = array
        vectors[cell_id] = record
    return vectors, group_count


def _validate_one_se_gate(
    gate: Mapping[str, object],
    *,
    allowed_ids: set[str],
    vectors_by_id: Mapping[str, Mapping[str, np.ndarray]],
    vector_name: str,
    name: str,
) -> set[str]:
    records = dict(gate["records"])
    if set(records) != allowed_ids:
        raise ValueError(f"{name}: gate record IDs mismatch")
    if not allowed_ids.issubset(vectors_by_id):
        raise ValueError(f"{name}: missing selection vectors")
    means = {
        key: float(vectors_by_id[key][vector_name].mean())
        for key in allowed_ids
    }
    expected_best = min(
        allowed_ids,
        key=lambda key: (means[key], key),
    )
    best = str(gate["best"])
    if best != expected_best:
        raise ValueError(f"{name}: best cell does not match vector minimum")

    retained = set(map(str, gate["retained"]))
    derived = {
        key for key, record in records.items()
        if bool(record["retained"])
    }
    if retained != derived:
        raise ValueError(f"{name}: retained list/flags disagree")
    if best not in retained:
        raise ValueError(f"{name}: best cell not retained")

    reference = vectors_by_id[best][vector_name]
    for key, record in records.items():
        value = vectors_by_id[key][vector_name]
        difference = value - reference
        expected_loss = means[key]
        expected_difference = float(difference.mean())
        expected_se = float(
            0.0
            if key == best
            else difference.std(ddof=1) / math.sqrt(len(difference))
        )

        mean_loss = _finite_number(
            record["mean_loss"], name=f"{name}:{key}:mean_loss"
        )
        if abs(mean_loss - expected_loss) > 1e-12:
            raise ValueError(f"{name}: recorded mean loss mismatches vectors")
        mean_difference = _finite_number(
            record["mean_difference_vs_best"],
            name=f"{name}:{key}:mean_difference",
        )
        if abs(mean_difference - expected_difference) > 1e-12:
            raise ValueError(
                f"{name}: mean difference mismatches paired vectors"
            )
        paired_se = _finite_number(
            record["paired_se"],
            name=f"{name}:{key}:paired_se",
        )
        if abs(paired_se - expected_se) > 1e-12:
            raise ValueError(f"{name}: paired SE mismatches paired vectors")
        expected_keep = expected_difference <= expected_se + 1e-12
        if bool(record["retained"]) != expected_keep:
            raise ValueError(f"{name}: decision violates one-SE rule")
    return retained


def validate_joint_selection(
    calibration: Mapping[str, object],
) -> dict[str, object]:
    surface = list(calibration["surface"])
    by_id = {str(cell["cell_id"]): cell for cell in surface}
    if len(by_id) != len(surface):
        raise ValueError("calibration cell_id values are not unique")
    vectors, group_count = _validate_selection_vectors(
        calibration, by_id=by_id
    )
    selection = calibration["selection"]

    nll_retained = _validate_one_se_gate(
        selection["nll_one_se"],
        allowed_ids=set(by_id),
        vectors_by_id=vectors,
        vector_name="group_nll",
        name="NLL one-SE",
    )
    if str(selection["raw_nll_best"]) != str(
        selection["nll_one_se"]["best"]
    ):
        raise ValueError("raw NLL best/gate best mismatch")

    es_retained = _validate_one_se_gate(
        selection["es_one_se"],
        allowed_ids=nll_retained,
        vectors_by_id=vectors,
        vector_name="group_energy_score",
        name="ES one-SE",
    )
    if str(selection["raw_es_best_within_nll_set"]) != str(
        selection["es_one_se"]["best"]
    ):
        raise ValueError("raw ES best/gate best mismatch")

    selected_id = str(selection["selected_cell_id"])
    if selected_id not in es_retained:
        raise ValueError("selected cell outside ES one-SE set")
    expected = min(
        es_retained, key=lambda key: _selection_choice_key(by_id[key])
    )
    if selected_id != expected:
        raise ValueError("selected cell violates final parsimony tie-break")
    selected_cell = selection["selected_cell"]
    if str(selected_cell["cell_id"]) != selected_id:
        raise ValueError("selected cell payload ID mismatch")
    surface_cell = by_id[selected_id]
    if _cell_key(selected_cell) != _cell_key(surface_cell):
        raise ValueError("selected cell parameters mismatch surface")
    for metric in ("nll", "energy_score_m", "posterior_mean_mae_m"):
        if abs(float(selected_cell[metric]) - float(surface_cell[metric])) > 1e-12:
            raise ValueError(f"selected cell {metric} mismatch surface")
    for vector_name in ("group_nll", "group_energy_score"):
        selected_vector = np.asarray(
            selected_cell[vector_name], dtype=np.float64
        )
        if selected_vector.shape != (group_count,) or not np.array_equal(
            selected_vector,
            vectors[selected_id][vector_name],
        ):
            raise ValueError(
                f"selected cell {vector_name} mismatches audit vectors"
            )
    return {
        "coordinate_groups": group_count,
        "nll_one_se_size": int(len(nll_retained)),
        "es_one_se_size": int(len(es_retained)),
        "selected_cell_id": selected_id,
        "selected_parameters": {
            "temperature": float(surface_cell["temperature"]),
            "epsilon_kl": float(surface_cell["epsilon_kl"]),
            "scale_multiplier": float(surface_cell["scale_multiplier"]),
        },
    }


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



def _validate_quantiles(
    payload: Mapping[str, object],
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
    strictly_positive: bool = False,
) -> None:
    keys = ("p05", "p25", "p50", "p75", "p95")
    values = [_finite_number(payload[key], name=f"{name}:{key}") for key in keys]
    if values != sorted(values):
        raise ValueError(f"{name}: quantiles are not monotone")
    if lower is not None and values[0] < lower:
        raise ValueError(f"{name}: quantile below lower bound")
    if upper is not None and values[-1] > upper:
        raise ValueError(f"{name}: quantile above upper bound")
    if strictly_positive and values[0] <= 0.0:
        raise ValueError(f"{name}: quantiles must be strictly positive")


def validate_radio_diagnostics(
    records: Sequence[Mapping[str, object]],
    *,
    calibration_rows: int,
    selection_refit_rows: int,
) -> dict[str, object]:
    expected_temperatures = list(map(float, V185_CONFIG["temperature_grid"]))
    if len(records) != len(expected_temperatures):
        raise ValueError("radio diagnostics must contain exactly eight temperatures")
    by_temperature = {float(record["temperature"]): record for record in records}
    if set(by_temperature) != set(expected_temperatures):
        raise ValueError("radio diagnostic temperature grid mismatch")
    if len(by_temperature) != len(records):
        raise ValueError("duplicate radio diagnostic temperature")

    tolerance = kl_budget_tolerance(torch.float32)
    global_max_excess = -float("inf")
    maximum_beta = 0.0
    for temperature in expected_temperatures:
        record = by_temperature[temperature]
        if int(record["rows"]) != int(calibration_rows):
            raise ValueError("radio diagnostic calibration row count mismatch")
        if int(record["training_reference_rows"]) != int(selection_refit_rows):
            raise ValueError("radio diagnostic reference row count mismatch")
        _validate_quantiles(
            record["training_reference_objective_quantiles"],
            name=f"T={temperature}:training_reference_objective",
        )
        _validate_quantiles(
            record["calibration_profiled_objective_quantiles"],
            name=f"T={temperature}:calibration_profiled_objective",
        )
        compatibility_mean = _finite_number(
            record["compatibility_mean"],
            name=f"T={temperature}:compatibility_mean",
        )
        if not 0.0 <= compatibility_mean <= 1.0:
            raise ValueError("radio compatibility mean outside [0,1]")
        _validate_quantiles(
            record["compatibility_quantiles"],
            name=f"T={temperature}:compatibility",
            lower=0.0,
            upper=1.0,
        )
        _finite_number(
            record["mean_abs_style_offset"],
            name=f"T={temperature}:mean_abs_style_offset",
        )
        _validate_quantiles(
            record["abs_style_offset_quantiles"],
            name=f"T={temperature}:abs_style_offset",
            lower=0.0,
        )
        style_mean = _finite_number(
            record["style_scale_mean"],
            name=f"T={temperature}:style_scale_mean",
        )
        if style_mean <= 0.0:
            raise ValueError("style scale mean must be positive")
        _validate_quantiles(
            record["style_scale_quantiles"],
            name=f"T={temperature}:style_scale",
            strictly_positive=True,
        )
        entropy = _finite_number(
            record["q_temperature_entropy_mean"],
            name=f"T={temperature}:q_temperature_entropy_mean",
        )
        if entropy < 0.0:
            raise ValueError("q_T entropy cannot be negative")
        top1 = _finite_number(
            record["q_temperature_top1_mean"],
            name=f"T={temperature}:q_temperature_top1_mean",
        )
        if not 0.0 <= top1 <= 1.0:
            raise ValueError("q_T top1 outside [0,1]")

        epsilon_records = record["epsilon_diagnostics"]
        expected_epsilons = list(map(float, V185_CONFIG["epsilon_kl_grid"]))
        if len(epsilon_records) != len(expected_epsilons):
            raise ValueError("radio diagnostics require four epsilon rows")
        by_epsilon = {
            float(item["epsilon_kl"]): item for item in epsilon_records
        }
        if set(by_epsilon) != set(expected_epsilons):
            raise ValueError("radio diagnostic epsilon grid mismatch")
        if len(by_epsilon) != len(epsilon_records):
            raise ValueError("duplicate radio diagnostic epsilon")
        for epsilon in expected_epsilons:
            item = by_epsilon[epsilon]
            mean_budget = _finite_number(
                item["mean_effective_budget"],
                name=f"T={temperature},eps={epsilon}:mean_budget",
            )
            mean_kl = _finite_number(
                item["mean_realized_kl"],
                name=f"T={temperature},eps={epsilon}:mean_kl",
            )
            if mean_budget < -tolerance or mean_budget > epsilon + tolerance:
                raise ValueError("radio diagnostic effective budget outside epsilon")
            if mean_kl < -tolerance:
                raise ValueError("radio diagnostic realized KL is negative")
            _validate_quantiles(
                item["effective_budget_quantiles"],
                name=f"T={temperature},eps={epsilon}:budget",
                lower=-tolerance,
                upper=epsilon + tolerance,
            )
            _validate_quantiles(
                item["realized_kl_quantiles"],
                name=f"T={temperature},eps={epsilon}:realized_kl",
                lower=-tolerance,
            )
            excess = _finite_number(
                item["max_kl_budget_excess"],
                name=f"T={temperature},eps={epsilon}:max_excess",
            )
            global_max_excess = max(global_max_excess, excess)
            if excess > tolerance:
                raise ValueError("radio diagnostic KL budget excess")
            infinite = _finite_number(
                item["dual_beta_infinite_fraction"],
                name=f"T={temperature},eps={epsilon}:infinite_fraction",
            )
            if not 0.0 <= infinite <= 1.0:
                raise ValueError("radio diagnostic infinite-beta fraction invalid")
            beta_quantiles = item["finite_dual_beta_quantiles"]
            max_beta = item["maximum_finite_dual_beta"]
            if beta_quantiles is None:
                if max_beta is not None:
                    raise ValueError("finite beta maximum exists without quantiles")
            else:
                _validate_quantiles(
                    beta_quantiles,
                    name=f"T={temperature},eps={epsilon}:finite_beta",
                    lower=0.0,
                )
                maximum = _finite_number(
                    max_beta,
                    name=f"T={temperature},eps={epsilon}:maximum_beta",
                )
                if maximum < float(beta_quantiles["p95"]):
                    raise ValueError("maximum beta below p95")
                maximum_beta = max(maximum_beta, maximum)
            if epsilon == 0.0:
                if abs(mean_budget) > tolerance or abs(mean_kl) > tolerance:
                    raise ValueError("epsilon=0 radio diagnostics are not identity")
                if infinite > tolerance:
                    raise ValueError("epsilon=0 radio diagnostics have infinite beta")
                if beta_quantiles is None or abs(float(beta_quantiles["p95"])) > tolerance:
                    raise ValueError("epsilon=0 finite beta must be zero")

    return {
        "temperatures": int(len(records)),
        "epsilon_records": int(
            len(records) * len(V185_CONFIG["epsilon_kl_grid"])
        ),
        "maximum_kl_budget_excess": float(global_max_excess),
        "maximum_finite_dual_beta": float(maximum_beta),
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
    selection_summary = validate_joint_selection(calibration)
    radio_summary = validate_radio_diagnostics(
        calibration["radio_diagnostics_by_temperature"],
        calibration_rows=int(calibration["rows"]),
        selection_refit_rows=int(
            registry["roles"]["selection_refit"]["row_count"]
        ),
    )

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
        "selection": selection_summary,
        "radio_diagnostics": radio_summary,
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
