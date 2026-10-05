"""Reserved external evaluation framework for V18.5 JRT-SP.

V18.5 freezes the global learned method structure and lambda_spatial on UJI.
Each external dataset then uses TRAINING DATA ONLY to:

1. select its epoch under the frozen lambda;
2. refit on selection_refit;
3. fit temperature-matched radio compatibility references;
4. jointly calibrate T x epsilon_KL x Student-t scale with the exact same
   registered 192-cell paired one-SE selector.

Only after the dataset-specific training freeze is complete may the provided
test split be consumed.  evaluate-test writes an irreversible lock before
loading/scoring the formal test split.

The external benchmarks were historically pre-exposed for adapter/schema
validation in earlier versions.  V18.5 therefore does not claim first access;
the guarantee is that their performance outcomes do not enter V18.5 method,
epoch, or calibration selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    REPO_ROOT,
    TUJI_DEVICE_FILES,
)
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import (
    DATA_ROOT,
    CanonicalFingerprintData,
    DatasetReference,
    load_tampere,
    load_tuji1,
    load_uts,
    model_input,
)
from fhm_dapp_tim.fhm_dapp.v18_5_model import (
    V185_CONFIG,
    build_v185_model,
    v185_distribution,
)
from fhm_dapp_tim.fhm_dapp.v18_5_protocol import (
    ROLE_SEEDS,
    TUJI_LODO_ROLE_SEEDS,
    construct_roles,
    coordinate_keys,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    temperature_radio_reference_from_payload,
)
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    DEFAULT_FREEZE as DEFAULT_UJI_FREEZE,
    PREDICTIVE_DRAWS,
    PROTOCOL_VERSION as UJI_PROTOCOL_VERSION,
    SCORING_SEED,
    _calibration_surface,
    _canonical_sha,
    _coordinate_macro,
    _coordinate_macro_se,
    _cpu_memory,
    _cpu_state,
    _fit_one_lambda,
    _fit_temperature_references,
    _reference_payload,
    _repo_relative,
    _row_nll,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _train_fixed_epochs,
    _write_json_new,
)


PROTOCOL_VERSION = "v18_5_external_reserved_v1_20261005"
EXTERNAL_ROOT = (
    Path(__file__).resolve().parents[1] / "evidence" / "v18_5" / "external"
)
MODEL_ROOT = (
    Path(__file__).resolve().parents[1] / "models" / "v18_5" / "external"
)
TUJI_DEVICES = tuple(TUJI_DEVICE_FILES.keys())


@dataclass(frozen=True)
class ExternalSpec:
    slug: str
    dataset: str
    role_seed: int
    training_devices: tuple[str, ...] | None = None
    evaluation_devices: tuple[str, ...] | None = None
    heldout_device: str | None = None


def external_specs() -> dict[str, ExternalSpec]:
    specs = {
        "uts": ExternalSpec(
            slug="uts",
            dataset="UTSIndoorLoc",
            role_seed=int(ROLE_SEEDS["UTSIndoorLoc"]),
        ),
        "tampere": ExternalSpec(
            slug="tampere",
            dataset="Tampere",
            role_seed=int(ROLE_SEEDS["Tampere"]),
        ),
        "tuji-pooled": ExternalSpec(
            slug="tuji-pooled",
            dataset="TUJI1",
            role_seed=int(ROLE_SEEDS["TUJI1"]),
            training_devices=TUJI_DEVICES,
            evaluation_devices=TUJI_DEVICES,
        ),
    }
    for device in TUJI_DEVICES:
        specs[f"tuji-lodo-{device}"] = ExternalSpec(
            slug=f"tuji-lodo-{device}",
            dataset="TUJI1",
            role_seed=int(TUJI_LODO_ROLE_SEEDS[device]),
            training_devices=tuple(
                value for value in TUJI_DEVICES if value != device
            ),
            evaluation_devices=(device,),
            heldout_device=device,
        )
    return specs


def _spec(slug: str) -> ExternalSpec:
    try:
        return external_specs()[slug]
    except KeyError as exc:
        raise ValueError(
            f"unknown V18.5 external protocol {slug!r}; "
            f"choices={sorted(external_specs())}"
        ) from exc


def _current_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if len(value) != 40:
        raise RuntimeError("invalid current Git HEAD")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    if dirty == 1:
        raise RuntimeError(
            "formal V18.5 external phases require no tracked source changes"
        )
    if dirty != 0:
        raise RuntimeError("cannot inspect tracked source state")
    return value


def _method_source_compatible(source_sha: str) -> None:
    """Permit evidence-only commits, reject post-freeze method edits."""

    if len(str(source_sha)) != 40:
        raise ValueError("invalid V18.5 method source SHA")
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
        raise RuntimeError(
            "V18.5 method/protocol source changed after UJI method freeze"
        )
    if result != 0:
        raise RuntimeError("cannot compare current source to UJI method freeze")


def load_uji_method_freeze(
    path: str | Path = DEFAULT_UJI_FREEZE,
) -> dict[str, object]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("freeze_sha256") != _canonical_sha(
        freeze, omit="freeze_sha256"
    ):
        raise ValueError("UJI V18.5 method freeze self-hash mismatch")
    if freeze.get("protocol_version") != UJI_PROTOCOL_VERSION:
        raise ValueError("unexpected UJI V18.5 protocol version")
    if freeze.get("official_validation_accessed") is not False:
        raise ValueError("UJI official validation entered V18.5 selection")
    if freeze.get("external_test_performance_accessed") is not False:
        raise ValueError("external performance entered UJI V18.5 selection")
    if freeze.get("correction") != "radio_only_exact_KL_projection":
        raise ValueError("UJI freeze does not contain V18.5 radio projection")
    if freeze.get("geometry_posterior_correction") is not False:
        raise ValueError("V18.5 external protocol forbids geometry correction")
    if freeze.get("decoder") != "posterior_mean_of_component_locations":
        raise ValueError("unexpected V18.5 registered point decoder")
    if float(freeze["lambda_spatial"]) not in set(
        map(float, V185_CONFIG["lambda_spatial_grid"])
    ):
        raise ValueError("UJI-frozen lambda_spatial outside registered grid")
    if float(freeze["posterior_temperature"]) not in set(
        map(float, V185_CONFIG["temperature_grid"])
    ):
        raise ValueError("UJI calibration temperature outside grid")
    if float(freeze["epsilon_kl"]) not in set(
        map(float, V185_CONFIG["epsilon_kl_grid"])
    ):
        raise ValueError("UJI calibration epsilon outside grid")
    if float(freeze["scale_multiplier"]) not in set(
        map(float, V185_CONFIG["scale_grid"])
    ):
        raise ValueError("UJI calibration scale outside grid")
    _method_source_compatible(str(freeze["source_commit_sha"]))
    checkpoint = REPO_ROOT / str(freeze["selection_refit_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if _sha256_file(checkpoint) != freeze["selection_refit_checkpoint_sha256"]:
        raise ValueError("UJI selection-refit checkpoint SHA mismatch")
    return freeze


def _reference_payload_json(
    reference: DatasetReference,
) -> dict[str, object]:
    return {
        "dataset": reference.dataset,
        "ap_ids": list(reference.ap_ids),
        "building_mapping": dict(reference.building_mapping),
        "floor_mapping": dict(reference.floor_mapping),
        "device_mapping": dict(reference.device_mapping),
        "user_mapping": dict(reference.user_mapping),
        "extra": dict(reference.extra),
    }


def _reference_from_json(
    payload: Mapping[str, object],
) -> DatasetReference:
    return DatasetReference(
        dataset=str(payload["dataset"]),
        ap_ids=tuple(map(str, payload["ap_ids"])),
        building_mapping={
            str(key): int(value)
            for key, value in dict(payload["building_mapping"]).items()
        },
        floor_mapping={
            str(key): int(value)
            for key, value in dict(payload["floor_mapping"]).items()
        },
        device_mapping={
            str(key): int(value)
            for key, value in dict(payload["device_mapping"]).items()
        },
        user_mapping={
            str(key): int(value)
            for key, value in dict(payload["user_mapping"]).items()
        },
        extra=dict(payload.get("extra", {})),
    )


def _training_paths(spec: ExternalSpec) -> list[Path]:
    if spec.slug == "uts":
        return [DATA_ROOT / "UTSIndoorLoc" / "UTS_training.csv"]
    if spec.slug == "tampere":
        return [
            DATA_ROOT / "Tampere" / "Training_rss.csv",
            DATA_ROOT / "Tampere" / "Training_coordinates.csv",
        ]
    assert spec.training_devices is not None
    return [
        DATA_ROOT / "TUJI1" / TUJI_DEVICE_FILES[device]
        for device in spec.training_devices
    ]


def _test_paths(spec: ExternalSpec) -> list[Path]:
    if spec.slug == "uts":
        return [DATA_ROOT / "UTSIndoorLoc" / "UTS_test.csv"]
    if spec.slug == "tampere":
        return [
            DATA_ROOT / "Tampere" / "Test_rss.csv",
            DATA_ROOT / "Tampere" / "Test_coordinates.csv",
        ]
    assert spec.evaluation_devices is not None
    # TUJI train/test records share the same MAT container.  Formal semantic
    # access to results_test is still deferred until after the lock.
    return [
        DATA_ROOT / "TUJI1" / TUJI_DEVICE_FILES[device]
        for device in spec.evaluation_devices
    ]


def _file_manifest(paths: Sequence[Path]) -> list[dict[str, object]]:
    result = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        result.append(
            {
                "path": _repo_relative(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return result


def _load_training(
    spec: ExternalSpec,
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    if spec.slug == "uts":
        return load_uts("train")
    if spec.slug == "tampere":
        return load_tampere("train")
    assert spec.training_devices is not None
    return load_tuji1("train", devices=spec.training_devices)


def _load_test(
    spec: ExternalSpec,
    reference: DatasetReference,
) -> CanonicalFingerprintData:
    if spec.slug == "uts":
        return load_uts("test", reference=reference)[0]
    if spec.slug == "tampere":
        return load_tampere("test", reference=reference)[0]
    assert spec.evaluation_devices is not None
    return load_tuji1(
        "test",
        devices=spec.evaluation_devices,
        reference=reference,
    )[0]


def _role_payload(
    data: CanonicalFingerprintData,
    *,
    seed: int,
) -> dict[str, object]:
    roles = construct_roles(data, seed=int(seed))
    rows = row_indices_for_roles(data, roles)
    keys = coordinate_keys(data)
    result = {}
    for name, indices in rows.items():
        result[name] = {
            "rows": list(map(int, indices.tolist())),
            "row_count": int(len(indices)),
            "coordinate_count": int(
                len({keys[int(index)] for index in indices.tolist()})
            ),
            "row_index_sha256": _sha256_indices(indices),
        }
    return result


def _rows_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, np.ndarray]:
    result = {}
    for name, record in manifest["roles"].items():
        indices = np.asarray(record["rows"], dtype=np.int64)
        if _sha256_indices(indices) != record["row_index_sha256"]:
            raise ValueError(f"external role row hash mismatch: {name}")
        result[name] = indices
    required = {"fit", "development", "selection_refit", "calibration"}
    if set(result) != required:
        raise ValueError("external manifest role set mismatch")
    return result


def _default_paths(slug: str) -> dict[str, Path]:
    return {
        "manifest": EXTERNAL_ROOT / f"{slug}_training_manifest.json",
        "checkpoint": MODEL_ROOT / f"{slug}_selection_refit.pt",
        "freeze": EXTERNAL_ROOT / f"{slug}_training_freeze.json",
        "lock": EXTERNAL_ROOT / f"{slug}_test_consumption_lock.json",
        "result": EXTERNAL_ROOT / f"{slug}_test_result.json",
    }


def prepare_external(
    slug: str,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    spec = _spec(slug)
    uji = load_uji_method_freeze(uji_freeze_path)
    current = _current_head()
    destination = Path(
        manifest_path or _default_paths(spec.slug)["manifest"]
    )
    if destination.exists():
        raise FileExistsError(destination)

    data, reference = _load_training(spec)
    roles = _role_payload(data, seed=spec.role_seed)

    if spec.heldout_device is not None:
        if spec.heldout_device in reference.device_mapping:
            raise RuntimeError("held-out TUJI device leaked into training mapping")
        if spec.training_devices is None:
            raise RuntimeError("LODO protocol lost training-device manifest")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "training_devices": (
            list(spec.training_devices)
            if spec.training_devices is not None
            else None
        ),
        "evaluation_devices": (
            list(spec.evaluation_devices)
            if spec.evaluation_devices is not None
            else None
        ),
        "role_seed": int(spec.role_seed),
        "execution_source_commit_sha": current,
        "uji_method_source_commit_sha": uji["source_commit_sha"],
        "uji_method_freeze_sha256": uji["freeze_sha256"],
        "uji_frozen_lambda_spatial": float(uji["lambda_spatial"]),
        "uji_selected_calibration_for_reference_only": {
            "temperature": float(uji["posterior_temperature"]),
            "epsilon_kl": float(uji["epsilon_kl"]),
            "scale_multiplier": float(uji["scale_multiplier"]),
        },
        "external_calibration_rule": {
            "temperature_grid": list(
                map(float, V185_CONFIG["temperature_grid"])
            ),
            "epsilon_kl_grid": list(
                map(float, V185_CONFIG["epsilon_kl_grid"])
            ),
            "scale_grid": list(map(float, V185_CONFIG["scale_grid"])),
            "cells": (
                len(V185_CONFIG["temperature_grid"])
                * len(V185_CONFIG["epsilon_kl_grid"])
                * len(V185_CONFIG["scale_grid"])
            ),
            "selector": (
                "paired NLL one-SE; paired ES one-SE; minimum posterior-mean "
                "MAE; smaller epsilon; T closest 1; scale closest 1"
            ),
        },
        "dataset_reference": _reference_payload_json(reference),
        "training_files": _file_manifest(_training_paths(spec)),
        "training_rows": int(len(data.coords_m)),
        "training_coordinate_groups": int(len(set(coordinate_keys(data)))),
        "aps": int(len(data.ap_ids)),
        "roles": roles,
        "provided_test_historically_preexposed_for_adapter_validation": True,
        "formal_test_performance_evaluated": False,
    }
    if manifest["external_calibration_rule"]["cells"] != 192:
        raise RuntimeError("V18.5 external calibration grid is not 192 cells")
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    _write_json_new(destination, manifest)
    return manifest


def load_external_manifest(
    path: str | Path,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != _canonical_sha(
        manifest, omit="manifest_sha256"
    ):
        raise ValueError("external manifest self-hash mismatch")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("external manifest protocol mismatch")
    uji = load_uji_method_freeze(uji_freeze_path)
    if manifest.get("uji_method_freeze_sha256") != uji["freeze_sha256"]:
        raise ValueError("external manifest links a different UJI freeze")
    if float(manifest["uji_frozen_lambda_spatial"]) != float(
        uji["lambda_spatial"]
    ):
        raise ValueError("external manifest lambda differs from UJI freeze")
    if _current_head() != manifest["execution_source_commit_sha"]:
        raise RuntimeError(
            "external prepare and fit-freeze must use one unchanged Git commit"
        )
    _method_source_compatible(str(uji["source_commit_sha"]))
    for record in manifest["training_files"]:
        source = REPO_ROOT / str(record["path"])
        if _sha256_file(source) != record["sha256"]:
            raise ValueError("external training source hash changed")
    if (
        manifest.get(
            "provided_test_historically_preexposed_for_adapter_validation"
        )
        is not True
    ):
        raise ValueError("external manifest must record historical pre-exposure")
    if manifest.get("formal_test_performance_evaluated") is not False:
        raise ValueError("external manifest already marks formal test use")
    return manifest


def fit_freeze_external(
    slug: str,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
    manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    freeze_path: str | Path | None = None,
    device_name: str = "cuda",
) -> dict[str, object]:
    spec = _spec(slug)
    defaults = _default_paths(spec.slug)
    manifest = load_external_manifest(
        manifest_path or defaults["manifest"],
        uji_freeze_path=uji_freeze_path,
    )
    checkpoint_destination = Path(
        checkpoint_path or defaults["checkpoint"]
    )
    freeze_destination = Path(freeze_path or defaults["freeze"])
    if checkpoint_destination.exists():
        raise FileExistsError(checkpoint_destination)
    if freeze_destination.exists():
        raise FileExistsError(freeze_destination)

    data, reference = _load_training(spec)
    if _reference_payload_json(reference) != manifest["dataset_reference"]:
        raise ValueError("external training reference changed after prepare")
    rows = _rows_from_manifest(manifest)

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    lambda_spatial = float(manifest["uji_frozen_lambda_spatial"])
    _candidate_checkpoint, epoch_report = _fit_one_lambda(
        lambda_spatial=lambda_spatial,
        data=data,
        rows=rows,
        device=device,
    )
    selected_epoch = int(epoch_report["selected_epoch"])
    if not (
        int(V185_CONFIG["minimum_epochs"])
        <= selected_epoch
        <= int(V185_CONFIG["maximum_epochs"])
    ):
        raise RuntimeError("external selected epoch outside registered range")

    state, refit_trajectory = _train_fixed_epochs(
        lambda_spatial=lambda_spatial,
        epochs=selected_epoch,
        data=data,
        row_indices=rows["selection_refit"],
        device=device,
    )
    radio_reference = _fit_temperature_references(
        state=state,
        row_indices=rows["selection_refit"],
        device=device,
    )
    calibration = _calibration_surface(
        state=state,
        data=data,
        row_indices=rows["calibration"],
        reference=radio_reference,
        device=device,
    )
    if len(calibration["surface"]) != 192:
        raise RuntimeError("external calibration did not produce 192 cells")
    selected_cell = calibration["selection"]["selected_cell"]

    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "model_protocol_version": UJI_PROTOCOL_VERSION,
        "phase": "external_selection_refit",
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "execution_source_commit_sha": manifest[
            "execution_source_commit_sha"
        ],
        "uji_method_source_commit_sha": manifest[
            "uji_method_source_commit_sha"
        ],
        "uji_method_freeze_sha256": manifest[
            "uji_method_freeze_sha256"
        ],
        "manifest_sha256": manifest["manifest_sha256"],
        "lambda_spatial": lambda_spatial,
        "selected_epoch": selected_epoch,
        "posterior_temperature": float(selected_cell["temperature"]),
        "epsilon_kl": float(selected_cell["epsilon_kl"]),
        "scale_multiplier": float(selected_cell["scale_multiplier"]),
        "origin": torch.from_numpy(state["origin"]),
        "spatial_scale_m": float(state["spatial_scale_m"]),
        "n_devices": int(state["n_devices"]),
        "memory": _cpu_memory(state["memory"]),
        "memory_diagnostics": state["memory_diagnostics"],
        "model_state": _cpu_state(state["model"]),
        "temperature_radio_reference": _reference_payload(radio_reference),
        "refit_trajectory": refit_trajectory,
        "model_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in V185_CONFIG.items()
        },
        "dataset_reference": manifest["dataset_reference"],
    }
    _save_torch_new(checkpoint_destination, checkpoint)
    checkpoint_sha = _sha256_file(checkpoint_destination)

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "execution_source_commit_sha": manifest[
            "execution_source_commit_sha"
        ],
        "uji_method_source_commit_sha": manifest[
            "uji_method_source_commit_sha"
        ],
        "uji_method_freeze_sha256": manifest["uji_method_freeze_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_path": _repo_relative(checkpoint_destination),
        "checkpoint_sha256": checkpoint_sha,
        "lambda_spatial": lambda_spatial,
        "selected_epoch": selected_epoch,
        "posterior_temperature": float(selected_cell["temperature"]),
        "epsilon_kl": float(selected_cell["epsilon_kl"]),
        "scale_multiplier": float(selected_cell["scale_multiplier"]),
        "decoder": "posterior_mean_of_component_locations",
        "correction": "radio_only_exact_KL_projection",
        "geometry_posterior_correction": False,
        "epoch_selection": epoch_report,
        "calibration": {
            "rows": int(calibration["rows"]),
            "coordinate_groups": int(calibration["coordinate_groups"]),
            "surface": calibration["surface"],
            "selection": calibration["selection"],
            "selected_metrics": {
                key: value
                for key, value in selected_cell.items()
                if key not in {"group_nll", "group_energy_score"}
            },
        },
        "provided_test_historically_preexposed_for_adapter_validation": True,
        "formal_test_performance_evaluated": False,
        "formal_test_performance_lock_written": False,
    }
    freeze["training_freeze_sha256"] = _canonical_sha(freeze)
    _write_json_new(freeze_destination, freeze)
    return freeze


def _load_external_checkpoint(
    checkpoint_path: Path,
    *,
    freeze: Mapping[str, object],
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    DatasetReference,
    object,
    dict[str, object],
]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("external checkpoint protocol mismatch")
    for key in (
        "manifest_sha256",
        "uji_method_freeze_sha256",
        "execution_source_commit_sha",
    ):
        expected = freeze[
            "manifest_sha256"
            if key == "manifest_sha256"
            else key
        ]
        if checkpoint.get(key) != expected:
            raise ValueError(f"external checkpoint {key} link mismatch")
    if float(checkpoint["lambda_spatial"]) != float(
        freeze["lambda_spatial"]
    ):
        raise ValueError("external checkpoint lambda mismatch")
    if int(checkpoint["selected_epoch"]) != int(freeze["selected_epoch"]):
        raise ValueError("external checkpoint epoch mismatch")
    for key in ("posterior_temperature", "epsilon_kl", "scale_multiplier"):
        if float(checkpoint[key]) != float(freeze[key]):
            raise ValueError(f"external checkpoint {key} mismatch")

    memory = checkpoint["memory"]
    model, _geometry, _neighbors, _weights = build_v185_model(
        memory=memory,
        n_aps=int(memory["prototype_rss"].shape[1]),
        n_buildings=int(memory["building"].max()) + 1,
        n_devices=int(checkpoint["n_devices"]),
        spatial_scale_m=float(checkpoint["spatial_scale_m"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    reference = _reference_from_json(checkpoint["dataset_reference"])
    radio_reference = temperature_radio_reference_from_payload(
        checkpoint["temperature_radio_reference"]
    )
    return model, reference, radio_reference, checkpoint


def load_external_training_freeze(
    path: str | Path,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
) -> dict[str, object]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("training_freeze_sha256") != _canonical_sha(
        freeze, omit="training_freeze_sha256"
    ):
        raise ValueError("external training freeze self-hash mismatch")
    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("external training freeze protocol mismatch")
    uji = load_uji_method_freeze(uji_freeze_path)
    if freeze.get("uji_method_freeze_sha256") != uji["freeze_sha256"]:
        raise ValueError("external training freeze links wrong UJI freeze")
    if float(freeze["lambda_spatial"]) != float(uji["lambda_spatial"]):
        raise ValueError("external training lambda differs from UJI freeze")
    _current_head()
    _method_source_compatible(str(freeze["execution_source_commit_sha"]))
    checkpoint = REPO_ROOT / str(freeze["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if _sha256_file(checkpoint) != freeze["checkpoint_sha256"]:
        raise ValueError("external checkpoint SHA mismatch")
    if (
        freeze.get(
            "provided_test_historically_preexposed_for_adapter_validation"
        )
        is not True
    ):
        raise ValueError("external freeze lost historical pre-exposure record")
    if freeze.get("formal_test_performance_evaluated") is not False:
        raise ValueError("external freeze already marks formal test use")
    if freeze.get("formal_test_performance_lock_written") is not False:
        raise ValueError("external freeze already marks formal test lock")
    if freeze.get("decoder") != "posterior_mean_of_component_locations":
        raise ValueError("external decoder differs from V18.5")
    if freeze.get("correction") != "radio_only_exact_KL_projection":
        raise ValueError("external correction differs from V18.5")
    if freeze.get("geometry_posterior_correction") is not False:
        raise ValueError("external freeze enabled geometry correction")
    if len(freeze["calibration"]["surface"]) != 192:
        raise ValueError("external freeze does not contain 192 calibration cells")
    return freeze


def _aggregate_label_probability(
    probability: torch.Tensor,
    candidate_labels: torch.Tensor,
    *,
    n_labels: int,
) -> torch.Tensor:
    if int(n_labels) <= 0:
        return torch.empty(
            len(probability),
            0,
            dtype=probability.dtype,
            device=probability.device,
        )
    result = torch.zeros(
        len(probability),
        int(n_labels),
        dtype=probability.dtype,
        device=probability.device,
    )
    result.scatter_add_(1, candidate_labels, probability)
    return result


def _coverage_macro(
    covered: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
) -> float:
    return _coordinate_macro(covered.float().cpu(), keys)


def _evaluate_reserved_test(
    *,
    model: torch.nn.Module,
    radio_reference,
    temperature: float,
    epsilon_kl: float,
    scale_multiplier: float,
    test: CanonicalFingerprintData,
    origin: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    inputs = model_input(test)
    centered = (
        test.coords_m.astype(np.float32) - origin[None, :]
    ).astype(np.float32)
    keys = coordinate_keys(test)
    randomness = common_student_t_randomness(
        len(test.coords_m),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V185_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )

    errors = []
    energy_scores = []
    nll_values = []
    entropy_values = []
    top1_values = []
    support_size = []
    natural_pool = []
    truncation = []
    between_component_rms = []
    expected_component_distance = []
    tail_10 = []
    tail_20 = []
    tail_50 = []
    building_correct = []
    building_valid = []
    floor_correct = []
    floor_valid = []
    compatibility = []
    realized_kl = []
    effective_budget = []
    dual_beta = []
    dual_infinite = []
    style_offset = []
    style_scale = []

    coverage_store = {level: [] for level in (0.50, 0.80, 0.90, 0.95)}
    radius_store = {level: [] for level in (0.50, 0.80, 0.90, 0.95)}

    batch_size = int(V185_CONFIG["batch_size"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test.coords_m), batch_size):
            stop = min(start + batch_size, len(test.coords_m))
            rss = inputs.rss[start:stop].to(device)
            observed = inputs.observed_mask[start:stop].to(device)
            target = torch.from_numpy(centered[start:stop]).to(device)
            result = v185_distribution(
                model,
                rss,
                observed,
                radio_memory={
                    "detection_prob": model.anchor_detection_prob,
                    "rss_center": model.anchor_rss_center,
                    "rss_scale": model.anchor_rss_scale,
                },
                radio_reference=radio_reference,
                temperature=float(temperature),
                epsilon_kl=float(epsilon_kl),
                scale_multiplier=float(scale_multiplier),
            )
            probability = result["probability"]
            location = result["component_location"]
            component_scale = result["component_scale"]
            mask = result["candidate_mask"]
            point = result["posterior_mean"]

            error = torch.linalg.vector_norm(point - target, dim=1)
            batch_randomness = {
                key: (
                    value[start:stop].to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in randomness.items()
            }
            es, first, second = continuous_energy_score_mc(
                probability,
                location,
                component_scale,
                target,
                batch_randomness,
            )
            nll = _row_nll(
                probability,
                location,
                component_scale,
                target,
                model,
            )
            ent = -torch.sum(
                torch.where(
                    probability > 0,
                    probability
                    * probability.clamp_min(1e-30).log(),
                    torch.zeros_like(probability),
                ),
                dim=1,
            )

            errors.append(error.cpu())
            energy_scores.append(es.cpu())
            nll_values.append(nll.cpu())
            entropy_values.append(ent.cpu())
            top1_values.append(probability.max(dim=1).values.cpu())
            support_size.append(mask.sum(dim=1).cpu())
            natural_pool.append(
                result["natural_pool_size_before_ceiling"].cpu()
            )
            truncation.append(result["natural_truncated"].cpu())

            delta = location - point.unsqueeze(1)
            squared = torch.sum(delta.square(), dim=2)
            between_component_rms.append(
                torch.sqrt(
                    torch.sum(probability * squared, dim=1).clamp_min(0.0)
                ).cpu()
            )
            target_distance = torch.linalg.vector_norm(
                location - target.unsqueeze(1), dim=2
            )
            expected_component_distance.append(
                torch.sum(probability * target_distance, dim=1).cpu()
            )
            tail_10.append(
                torch.sum(
                    probability * (target_distance > 10.0).to(probability),
                    dim=1,
                ).cpu()
            )
            tail_20.append(
                torch.sum(
                    probability * (target_distance > 20.0).to(probability),
                    dim=1,
                ).cpu()
            )
            tail_50.append(
                torch.sum(
                    probability * (target_distance > 50.0).to(probability),
                    dim=1,
                ).cpu()
            )

            draws = torch.cat([first, second], dim=1)
            radial = torch.linalg.vector_norm(
                draws - point.unsqueeze(1), dim=2
            )
            for level in coverage_store:
                radius = torch.quantile(radial, float(level), dim=1)
                radius_store[level].append(radius.cpu())
                coverage_store[level].append((error <= radius).cpu())

            candidate_indices = result["candidate_indices"]
            candidate_building = model.anchor_buildings[candidate_indices]
            candidate_floor = model.anchor_floors[candidate_indices]
            building_p = _aggregate_label_probability(
                probability,
                candidate_building,
                n_labels=int(model.anchor_buildings.max()) + 1,
            )
            floor_p = _aggregate_label_probability(
                probability,
                candidate_floor,
                n_labels=int(model.anchor_floors.max()) + 1,
            )
            target_building = torch.from_numpy(
                test.building[start:stop]
            ).long().to(device)
            target_floor = torch.from_numpy(
                test.floor[start:stop]
            ).long().to(device)
            valid_b = target_building >= 0
            valid_f = target_floor >= 0
            building_valid.append(valid_b.cpu())
            floor_valid.append(valid_f.cpu())
            building_correct.append(
                (building_p.argmax(dim=1) == target_building).cpu()
            )
            floor_correct.append(
                (floor_p.argmax(dim=1) == target_floor).cpu()
            )

            correction = result["correction"]
            if correction is not None:
                compatibility.append(
                    correction["radio_compatibility"].detach().cpu()
                )
                realized_kl.append(
                    correction["realized_kl"].detach().cpu()
                )
                effective_budget.append(
                    correction["epsilon_kl_effective"].detach().cpu()
                )
                finite = ~correction["dual_beta_infinite"]
                if bool(finite.any()):
                    dual_beta.append(
                        correction["dual_beta"][finite].detach().cpu()
                    )
                dual_infinite.append(
                    correction["dual_beta_infinite"].detach().cpu()
                )
                style_offset.append(
                    correction["style_offset"].detach().cpu()
                )
                style_scale.append(
                    correction["style_scale"].detach().cpu()
                )

    error = torch.cat(errors)
    es = torch.cat(energy_scores)
    nll = torch.cat(nll_values)
    ent = torch.cat(entropy_values)
    support = torch.cat(support_size)
    b_valid = torch.cat(building_valid)
    f_valid = torch.cat(floor_valid)
    b_correct = torch.cat(building_correct)
    f_correct = torch.cat(floor_correct)

    result: dict[str, object] = {
        "rows": int(len(test.coords_m)),
        "coordinate_groups": int(len(set(keys))),
        "posterior_mean_coordinate_macro_mae_m": _coordinate_macro(
            error, keys
        ),
        "posterior_mean_coordinate_macro_mae_se_m": _coordinate_macro_se(
            error, keys
        ),
        "row_mae_m": float(error.double().mean()),
        "row_median_error_m": float(torch.quantile(error.double(), 0.50)),
        "row_p75_error_m": float(torch.quantile(error.double(), 0.75)),
        "row_p90_error_m": float(torch.quantile(error.double(), 0.90)),
        "energy_score_coordinate_macro_m": _coordinate_macro(es, keys),
        "energy_score_coordinate_macro_se_m": _coordinate_macro_se(es, keys),
        "energy_score_row_mean_m": float(es.double().mean()),
        "mixture_nll_coordinate_macro": _coordinate_macro(nll, keys),
        "mixture_nll_coordinate_macro_se": _coordinate_macro_se(nll, keys),
        "mixture_nll_row_mean": float(nll.double().mean()),
        "mean_entropy_nats": float(ent.double().mean()),
        "mean_effective_anchors": float(ent.exp().double().mean()),
        "mean_top1_probability": float(
            torch.cat(top1_values).double().mean()
        ),
        "mean_support_size": float(support.double().mean()),
        "support_size_p90": float(
            torch.quantile(support.double(), 0.90)
        ),
        "mean_natural_pool_size_before_ceiling": float(
            torch.cat(natural_pool).double().mean()
        ),
        "natural_truncation_fraction": float(
            torch.cat(truncation).double().mean()
        ),
        "mean_component_center_rms_about_posterior_mean_m": float(
            torch.cat(between_component_rms).double().mean()
        ),
        "mean_expected_component_distance_to_target_m": float(
            torch.cat(expected_component_distance).double().mean()
        ),
        "mean_mass_beyond_10m_from_target": float(
            torch.cat(tail_10).double().mean()
        ),
        "mean_mass_beyond_20m_from_target": float(
            torch.cat(tail_20).double().mean()
        ),
        "mean_mass_beyond_50m_from_target": float(
            torch.cat(tail_50).double().mean()
        ),
        "building_error": (
            float(1.0 - b_correct[b_valid].double().mean())
            if bool(b_valid.any())
            else None
        ),
        "building_valid_rows": int(b_valid.sum()),
        "floor_error": (
            float(1.0 - f_correct[f_valid].double().mean())
            if bool(f_valid.any())
            else None
        ),
        "floor_valid_rows": int(f_valid.sum()),
        "coverage": {},
        "radio_projection": {
            "active": bool(float(epsilon_kl) > 0.0),
            "mean_radio_compatibility": None,
            "mean_realized_kl": 0.0,
            "mean_effective_epsilon_kl": 0.0,
            "max_kl_budget_excess": 0.0,
            "mean_finite_dual_beta": 0.0,
            "dual_beta_infinite_fraction": 0.0,
            "mean_abs_style_offset": None,
            "mean_style_scale": None,
        },
    }
    for level in coverage_store:
        covered = torch.cat(coverage_store[level])
        radius = torch.cat(radius_store[level])
        label = str(int(level * 100))
        result["coverage"][label] = {
            "row_coverage": float(covered.double().mean()),
            "coordinate_macro_coverage": _coverage_macro(covered, keys),
            "mean_radius_m": float(radius.double().mean()),
            "median_radius_m": float(
                torch.quantile(radius.double(), 0.50)
            ),
        }

    if compatibility:
        joined_kl = torch.cat(realized_kl).double()
        joined_budget = torch.cat(effective_budget).double()
        beta = (
            torch.cat(dual_beta).double()
            if dual_beta
            else torch.empty(0, dtype=torch.float64)
        )
        infinite = torch.cat(dual_infinite).bool()
        result["radio_projection"] = {
            "active": True,
            "mean_radio_compatibility": float(
                torch.cat(compatibility).double().mean()
            ),
            "mean_realized_kl": float(joined_kl.mean()),
            "mean_effective_epsilon_kl": float(joined_budget.mean()),
            "max_kl_budget_excess": float(
                (joined_kl - joined_budget).max()
            ),
            "mean_finite_dual_beta": (
                float(beta.mean()) if beta.numel() else None
            ),
            "dual_beta_infinite_fraction": float(
                infinite.double().mean()
            ),
            "mean_abs_style_offset": float(
                torch.cat(style_offset).abs().double().mean()
            ),
            "mean_style_scale": float(
                torch.cat(style_scale).double().mean()
            ),
        }
    return result


def evaluate_test_once(
    slug: str,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
    training_freeze_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    result_path: str | Path | None = None,
    device_name: str = "cuda",
) -> dict[str, object]:
    spec = _spec(slug)
    defaults = _default_paths(spec.slug)
    freeze = load_external_training_freeze(
        training_freeze_path or defaults["freeze"],
        uji_freeze_path=uji_freeze_path,
    )
    lock_destination = Path(lock_path or defaults["lock"])
    result_destination = Path(result_path or defaults["result"])
    if lock_destination.exists():
        raise RuntimeError(
            "formal external test lock already exists; rerun is forbidden"
        )
    if result_destination.exists():
        raise FileExistsError(result_destination)

    # Preflight everything possible without semantically loading test data.
    for path in _test_paths(spec):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; test lock not written"
        )

    checkpoint_path = REPO_ROOT / str(freeze["checkpoint_path"])
    model, reference, radio_reference, checkpoint = (
        _load_external_checkpoint(
            checkpoint_path,
            freeze=freeze,
            device=device,
        )
    )

    prototype_rss = checkpoint["memory"]["prototype_rss"][:1].to(device)
    prototype_mask = checkpoint["memory"][
        "prototype_observed_mask"
    ][:1].to(device)
    preflight = v185_distribution(
        model,
        prototype_rss,
        prototype_mask,
        radio_memory={
            "detection_prob": model.anchor_detection_prob,
            "rss_center": model.anchor_rss_center,
            "rss_scale": model.anchor_rss_scale,
        },
        radio_reference=radio_reference,
        temperature=float(freeze["posterior_temperature"]),
        epsilon_kl=float(freeze["epsilon_kl"]),
        scale_multiplier=float(freeze["scale_multiplier"]),
    )
    if not bool(torch.isfinite(preflight["probability"]).all()):
        raise RuntimeError(
            "external posterior preflight is nonfinite; lock not written"
        )
    if not bool(
        torch.isfinite(preflight["component_location"]).all()
        and torch.isfinite(preflight["component_scale"]).all()
        and torch.isfinite(preflight["posterior_mean"]).all()
    ):
        raise RuntimeError(
            "external predictive preflight is nonfinite; lock not written"
        )
    correction = preflight["correction"]
    if correction is not None and bool(
        (
            correction["realized_kl"]
            > correction["epsilon_kl_effective"] + 5e-6
        ).any()
    ):
        raise RuntimeError(
            "external preflight violates KL budget; lock not written"
        )

    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "training_freeze_sha256": freeze["training_freeze_sha256"],
        "uji_method_freeze_sha256": freeze["uji_method_freeze_sha256"],
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "execution_source_commit_sha": _current_head(),
        "provided_test_historically_preexposed_for_adapter_validation": True,
        "formal_test_performance_evaluation_started": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rerun_after_lock_forbidden": True,
    }
    lock["lock_sha256"] = _canonical_sha(lock)
    _write_json_new(lock_destination, lock)

    # Formal semantic access and performance scoring start only after the lock.
    test = _load_test(spec, reference)
    test_files = _file_manifest(_test_paths(spec))
    metrics = _evaluate_reserved_test(
        model=model,
        radio_reference=radio_reference,
        temperature=float(freeze["posterior_temperature"]),
        epsilon_kl=float(freeze["epsilon_kl"]),
        scale_multiplier=float(freeze["scale_multiplier"]),
        test=test,
        origin=checkpoint["origin"].cpu().numpy(),
        device=device,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "training_freeze_sha256": freeze["training_freeze_sha256"],
        "uji_method_freeze_sha256": freeze["uji_method_freeze_sha256"],
        "test_consumption_lock_sha256": lock["lock_sha256"],
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "test_files": test_files,
        "evaluation_reference_metadata": test.metadata,
        "metrics": metrics,
        "method": {
            "lambda_spatial": float(freeze["lambda_spatial"]),
            "selected_epoch": int(freeze["selected_epoch"]),
            "posterior_temperature": float(
                freeze["posterior_temperature"]
            ),
            "epsilon_kl": float(freeze["epsilon_kl"]),
            "scale_multiplier": float(freeze["scale_multiplier"]),
            "decoder": "posterior_mean_of_component_locations",
            "correction": "radio_only_exact_KL_projection",
        },
        "provided_test_historically_preexposed_for_adapter_validation": True,
        "formal_test_performance_evaluated": True,
        "no_post_test_retuning_permitted": True,
    }
    result["result_sha256"] = _canonical_sha(result)
    _write_json_new(result_destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        required=True,
        choices=sorted(external_specs()),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--uji-freeze", default=str(DEFAULT_UJI_FREEZE)
    )
    prepare_parser.add_argument("--manifest")

    fit_parser = sub.add_parser("fit-freeze")
    fit_parser.add_argument(
        "--uji-freeze", default=str(DEFAULT_UJI_FREEZE)
    )
    fit_parser.add_argument("--manifest")
    fit_parser.add_argument("--checkpoint")
    fit_parser.add_argument("--freeze")
    fit_parser.add_argument("--device", default="cuda")

    evaluate_parser = sub.add_parser("evaluate-test")
    evaluate_parser.add_argument(
        "--uji-freeze", default=str(DEFAULT_UJI_FREEZE)
    )
    evaluate_parser.add_argument("--training-freeze")
    evaluate_parser.add_argument("--lock")
    evaluate_parser.add_argument("--result")
    evaluate_parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_external(
            args.protocol,
            uji_freeze_path=args.uji_freeze,
            manifest_path=args.manifest,
        )
    elif args.command == "fit-freeze":
        result = fit_freeze_external(
            args.protocol,
            uji_freeze_path=args.uji_freeze,
            manifest_path=args.manifest,
            checkpoint_path=args.checkpoint,
            freeze_path=args.freeze,
            device_name=args.device,
        )
    else:
        result = evaluate_test_once(
            args.protocol,
            uji_freeze_path=args.uji_freeze,
            training_freeze_path=args.training_freeze,
            lock_path=args.lock,
            result_path=args.result,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
