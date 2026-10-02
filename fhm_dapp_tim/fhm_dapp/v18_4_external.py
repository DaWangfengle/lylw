"""Reserved external evaluation framework for V18.4.

The global method law (lambda_OT, epsilon_KL, support/reliability/decoder law)
must first be frozen by UJI training-only development.

For each external protocol this module enforces two gates:

1. prepare / fit-freeze:
   only the dataset training source is accessed. Dataset-specific epoch
   selection and scalar calibration use only training-derived roles.

2. evaluate-test:
   the provided test outcome is accessed once, only after a dataset-specific
   training freeze exists. An irreversible consumption lock is written before
   semantic access to the test split.

TUJI1 stores results_train and results_test in the same MAT containers. Reading
training scans necessarily opens the container bytes, but the results_test
field is not semantically accessed before the evaluate-test lock.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    masked_softmax,
    tail_mass_diagnostics,
    weighted_component_geometric_median,
)
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    DATA_ROOT,
    REPO_ROOT,
    CanonicalFingerprintData,
    DatasetReference,
    TUJI_DEVICE_FILES,
    load_tampere,
    load_tuji1,
    load_uts,
)
from fhm_dapp_tim.fhm_dapp.v18_4_model import (
    CorrectionReference,
    V184_CONFIG,
    v184_distribution,
)
from fhm_dapp_tim.fhm_dapp.v18_4_protocol import (
    ROLE_SEEDS,
    construct_roles,
    coordinate_keys,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.uji_v18_4_development import (
    CALIBRATION_SCALE_GRID,
    CALIBRATION_TEMPERATURE_GRID,
    DEFAULT_FREEZE as DEFAULT_UJI_FREEZE,
    PREDICTIVE_DRAWS,
    SCORING_SEED,
    _cache_corrected_distribution,
    _calibration_nll,
    _canonical_sha,
    _coordinate_group_ids,
    _coordinate_macro,
    _coordinate_macro_se,
    _fit_one_lambda,
    _fit_reference_for_rows,
    _load_model_checkpoint,
    _prediction_entropy,
    _reference_from_payload,
    _reference_payload,
    _repo_relative,
    _row_nll,
    _save_torch_new,
    _selected_calibration_metrics,
    _sha256_file,
    _sha256_indices,
    _train_fixed_epochs,
    _write_json_new,
)


PROTOCOL_VERSION = "v18_4_external_reserved_v1_20261002"
EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "evidence" / "v18_4" / "external"
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "v18_4" / "external"
TUJI_DEVICES = tuple(TUJI_DEVICE_FILES.keys())
TUJI_LODO_ROLE_SEEDS = {
    "A12": 184451,
    "POCO": 184452,
    "S20": 184453,
    "S7": 184454,
    "tabS7": 184455,
}


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
            f"unknown V18.4 external protocol {slug!r}; "
            f"choices={sorted(external_specs())}"
        ) from exc


def _method_source_compatible(source_sha: str) -> None:
    """Allow evidence-only commits but reject any post-UJI method-source edit."""

    if len(str(source_sha)) != 40:
        raise ValueError("invalid UJI method source SHA")
    tracked_paths = [
        "fhm_dapp_tim/fhm_dapp",
        "fhm_dapp_tim/evidence/v18_4/DESIGN_V18_4.md",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", str(source_sha), "HEAD", "--", *tracked_paths],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode == 1:
        raise RuntimeError(
            "V18.4 method-source files changed after the UJI method freeze"
        )
    if result.returncode not in {0, 1}:
        raise RuntimeError("cannot compare current method source to UJI freeze")


def _current_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if len(value) != 40:
        raise RuntimeError("invalid current Git HEAD")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0
    if dirty:
        raise RuntimeError(
            "formal external phases require no tracked working-tree changes"
        )
    return value


def load_uji_method_freeze(
    path: str | Path = DEFAULT_UJI_FREEZE,
) -> dict[str, object]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("freeze_sha256") != _canonical_sha(
        freeze,
        omit="freeze_sha256",
    ):
        raise ValueError("UJI method freeze self-hash mismatch")
    if freeze.get("official_validation_opened") is not False:
        raise ValueError("UJI freeze violated official-validation boundary")
    if freeze.get("reserved_external_test_opened") is not False:
        raise ValueError("external tests were already marked opened in UJI freeze")
    if float(freeze["lambda_ot"]) not in {
        float(value) for value in V184_CONFIG["lambda_ot_grid"]
    }:
        raise ValueError("UJI freeze lambda_OT is outside registered grid")
    if float(freeze["epsilon_kl"]) not in {
        float(value) for value in V184_CONFIG["epsilon_kl_grid"]
    }:
        raise ValueError("UJI freeze epsilon_KL is outside registered grid")
    _method_source_compatible(str(freeze["source_commit_sha"]))
    return freeze


def _reference_payload_json(reference: DatasetReference) -> dict[str, object]:
    return {
        "dataset": reference.dataset,
        "ap_ids": list(reference.ap_ids),
        "building_mapping": dict(reference.building_mapping),
        "floor_mapping": dict(reference.floor_mapping),
        "device_mapping": dict(reference.device_mapping),
        "user_mapping": dict(reference.user_mapping),
        "extra": dict(reference.extra),
    }


def _reference_from_json(payload: Mapping[str, object]) -> DatasetReference:
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


def _load_training(spec: ExternalSpec) -> tuple[
    CanonicalFingerprintData,
    DatasetReference,
]:
    if spec.slug == "uts":
        return load_uts("train")
    if spec.slug == "tampere":
        return load_tampere("train")
    assert spec.training_devices is not None
    return load_tuji1(
        "train",
        devices=spec.training_devices,
    )


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
    payload = {}
    for name, indices in rows.items():
        payload[name] = {
            "rows": list(map(int, indices.tolist())),
            "row_count": int(len(indices)),
            "row_index_sha256": _sha256_indices(indices),
            "coordinate_count": int(
                len({keys[int(index)] for index in indices.tolist()})
            ),
        }
    return payload


def _rows_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, np.ndarray]:
    result = {}
    for name, record in manifest["roles"].items():
        indices = np.asarray(record["rows"], dtype=np.int64)
        if _sha256_indices(indices) != record["row_index_sha256"]:
            raise ValueError(f"external role row hash mismatch: {name}")
        result[name] = indices
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
    freeze = load_uji_method_freeze(uji_freeze_path)
    current = _current_head()
    destination = Path(
        manifest_path or _default_paths(spec.slug)["manifest"]
    )
    if destination.exists():
        raise FileExistsError(destination)

    data, reference = _load_training(spec)
    roles = _role_payload(data, seed=spec.role_seed)
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
        "uji_method_source_commit_sha": freeze["source_commit_sha"],
        "uji_method_freeze_sha256": freeze["freeze_sha256"],
        "lambda_ot": float(freeze["lambda_ot"]),
        "epsilon_kl": float(freeze["epsilon_kl"]),
        "reference": _reference_payload_json(reference),
        "training_files": _file_manifest(_training_paths(spec)),
        "training_rows": int(len(data.coords_m)),
        "training_coordinate_groups": int(len(set(coordinate_keys(data)))),
        "aps": int(len(data.ap_ids)),
        "roles": roles,
        "test_outcome_semantically_accessed": False,
    }
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
        manifest,
        omit="manifest_sha256",
    ):
        raise ValueError("external manifest self-hash mismatch")
    freeze = load_uji_method_freeze(uji_freeze_path)
    if manifest["uji_method_freeze_sha256"] != freeze["freeze_sha256"]:
        raise ValueError("external manifest links a different UJI method freeze")
    if _current_head() != manifest["execution_source_commit_sha"]:
        raise RuntimeError(
            "external prepare/fit-freeze must run on one unchanged Git commit"
        )
    for record in manifest["training_files"]:
        path_value = REPO_ROOT / str(record["path"])
        if _sha256_file(path_value) != record["sha256"]:
            raise ValueError("external training source hash changed")
    if manifest.get("test_outcome_semantically_accessed") is not False:
        raise ValueError("external manifest already marks test outcome access")
    return manifest


def _calibrate_external(
    *,
    model: torch.nn.Module,
    reference: CorrectionReference,
    epsilon_kl: float,
    rss_all: torch.Tensor,
    centered: np.ndarray,
    data: CanonicalFingerprintData,
    calibration_rows: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    cache = _cache_corrected_distribution(
        model=model,
        reference=reference,
        epsilon_kl=float(epsilon_kl),
        rss_all=rss_all,
        row_indices=calibration_rows,
        device=device,
    )
    target = torch.from_numpy(centered[calibration_rows].astype(np.float32))
    keys = _coordinate_group_ids(data, calibration_rows)

    surface = []
    for temperature in CALIBRATION_TEMPERATURE_GRID:
        for scale_multiplier in CALIBRATION_SCALE_GRID:
            surface.append(
                {
                    "temperature": float(temperature),
                    "scale_multiplier": float(scale_multiplier),
                    "coordinate_macro_nll": float(
                        _calibration_nll(
                            cache=cache,
                            target=target,
                            keys=keys,
                            model=model,
                            temperature=float(temperature),
                            scale_multiplier=float(scale_multiplier),
                            device=device,
                        )
                    ),
                }
            )
    selected = min(
        surface,
        key=lambda item: (
            float(item["coordinate_macro_nll"]),
            abs(math.log(float(item["temperature"])))
            + abs(math.log(float(item["scale_multiplier"]))),
            float(item["temperature"]),
            float(item["scale_multiplier"]),
        ),
    )
    metrics = _selected_calibration_metrics(
        cache=cache,
        target=target,
        keys=keys,
        model=model,
        temperature=float(selected["temperature"]),
        scale_multiplier=float(selected["scale_multiplier"]),
        device=device,
    )
    return selected, {
        "temperature_grid": list(CALIBRATION_TEMPERATURE_GRID),
        "scale_grid": list(CALIBRATION_SCALE_GRID),
        "surface": surface,
        "selected_metrics": metrics,
        "rows": int(len(calibration_rows)),
        "coordinate_groups": int(len(set(keys))),
    }


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
    destination_checkpoint = Path(checkpoint_path or defaults["checkpoint"])
    destination_freeze = Path(freeze_path or defaults["freeze"])
    if destination_checkpoint.exists():
        raise FileExistsError(destination_checkpoint)
    if destination_freeze.exists():
        raise FileExistsError(destination_freeze)

    data, fresh_reference = _load_training(spec)
    if _reference_payload_json(fresh_reference) != manifest["reference"]:
        raise ValueError("external training reference changed after prepare")
    rows = _rows_from_manifest(manifest)
    rss_all = torch.from_numpy(data.model_rss()).float()
    device = torch.device(device_name)
    lambda_ot = float(manifest["lambda_ot"])
    epsilon_kl = float(manifest["epsilon_kl"])

    candidate, epoch_report = _fit_one_lambda(
        lambda_ot=lambda_ot,
        data=data,
        rows=rows,
        rss_all=rss_all,
        device=device,
    )
    selected_epoch = int(candidate["selected_epoch"])

    final_checkpoint, model, centered = _train_fixed_epochs(
        data=data,
        training_rows=rows["selection_refit"],
        rss_all=rss_all,
        lambda_ot=lambda_ot,
        epochs=selected_epoch,
        device=device,
    )
    correction_reference = _fit_reference_for_rows(
        model=model,
        rss_all=rss_all,
        row_indices=rows["selection_refit"],
        device=device,
    )
    selected_calibration, calibration_report = _calibrate_external(
        model=model,
        reference=correction_reference,
        epsilon_kl=epsilon_kl,
        rss_all=rss_all,
        centered=centered,
        data=data,
        calibration_rows=rows["calibration"],
        device=device,
    )

    final_checkpoint.update(
        {
            "phase": "external_selection_refit",
            "slug": spec.slug,
            "dataset": spec.dataset,
            "heldout_device": spec.heldout_device,
            "epsilon_kl": epsilon_kl,
            "correction_reference": _reference_payload(correction_reference),
            "posterior_temperature": float(
                selected_calibration["temperature"]
            ),
            "scale_multiplier": float(
                selected_calibration["scale_multiplier"]
            ),
            "dataset_reference": manifest["reference"],
            "manifest_sha256": manifest["manifest_sha256"],
            "uji_method_freeze_sha256": manifest[
                "uji_method_freeze_sha256"
            ],
        }
    )
    _save_torch_new(destination_checkpoint, final_checkpoint)
    checkpoint_sha = _sha256_file(destination_checkpoint)

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
        "checkpoint_path": _repo_relative(destination_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "lambda_ot": lambda_ot,
        "epsilon_kl": epsilon_kl,
        "selected_epoch": selected_epoch,
        "posterior_temperature": float(
            selected_calibration["temperature"]
        ),
        "scale_multiplier": float(
            selected_calibration["scale_multiplier"]
        ),
        "epoch_selection": epoch_report,
        "calibration": {
            "selected": selected_calibration,
            **calibration_report,
        },
        "test_outcome_semantically_accessed": False,
        "test_consumption_lock_written": False,
    }
    freeze["training_freeze_sha256"] = _canonical_sha(freeze)
    _write_json_new(destination_freeze, freeze)
    return freeze


def load_external_training_freeze(
    path: str | Path,
    *,
    uji_freeze_path: str | Path = DEFAULT_UJI_FREEZE,
) -> dict[str, object]:
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    if freeze.get("training_freeze_sha256") != _canonical_sha(
        freeze,
        omit="training_freeze_sha256",
    ):
        raise ValueError("external training freeze self-hash mismatch")
    uji = load_uji_method_freeze(uji_freeze_path)
    if freeze["uji_method_freeze_sha256"] != uji["freeze_sha256"]:
        raise ValueError("external training freeze links wrong UJI method freeze")
    if _current_head() != freeze["execution_source_commit_sha"]:
        raise RuntimeError(
            "external fit-freeze/evaluate-test must use one unchanged Git commit"
        )
    checkpoint = REPO_ROOT / str(freeze["checkpoint_path"])
    if _sha256_file(checkpoint) != freeze["checkpoint_sha256"]:
        raise ValueError("external checkpoint SHA mismatch")
    if freeze.get("test_outcome_semantically_accessed") is not False:
        raise ValueError("external training freeze already marks test access")
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
    correction_reference: CorrectionReference,
    epsilon_kl: float,
    temperature: float,
    scale_multiplier: float,
    test: CanonicalFingerprintData,
    origin: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    rss_all = torch.from_numpy(test.model_rss()).float()
    centered = (
        test.coords_m.astype(np.float32) - origin[None, :]
    ).astype(np.float32)
    keys = coordinate_keys(test)
    randomness = common_student_t_randomness(
        len(test.coords_m),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V184_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )

    row_error = []
    row_mean_error = []
    row_es = []
    row_nll_values = []
    entropy = []
    support_size = []
    top1 = []
    rho = []
    realized_kl = []
    radio_reliability = []
    geometry_reliability = []
    style_offset = []
    style_scale = []
    between_anchor_rms = []
    tail_store: dict[str, list[torch.Tensor]] = {}
    coverage_store = {level: [] for level in (0.50, 0.80, 0.90, 0.95)}
    radius_store = {level: [] for level in (0.50, 0.80, 0.90, 0.95)}
    building_correct = []
    building_valid = []
    floor_correct = []
    floor_valid = []

    batch_size = int(V184_CONFIG["batch_size"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test.coords_m), batch_size):
            stop = min(start + batch_size, len(test.coords_m))
            rss = rss_all[start:stop].to(device)
            target = torch.from_numpy(centered[start:stop]).to(device)
            distribution = v184_distribution(
                model,
                rss,
                correction_reference=correction_reference,
                epsilon_kl=float(epsilon_kl),
                final_temperature=float(temperature),
                scale_multiplier=float(scale_multiplier),
            )
            p = distribution["candidate_weights"]
            location = distribution["component_locations"]
            component_scale = distribution["component_scales"]
            mask = distribution["candidate_mask"]
            gm = weighted_component_geometric_median(
                p,
                location,
                mask=mask,
            )
            mean = torch.sum(p.unsqueeze(2) * location, dim=1)
            error = torch.linalg.vector_norm(gm - target, dim=1)
            mean_error = torch.linalg.vector_norm(mean - target, dim=1)

            batch_randomness = {
                key: (
                    value[start:stop].to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in randomness.items()
            }
            es, first, second = continuous_energy_score_mc(
                p,
                location,
                component_scale,
                target,
                batch_randomness,
            )
            nll = _row_nll(
                p,
                location,
                component_scale,
                target,
                model,
            )

            row_error.append(error.cpu())
            row_mean_error.append(mean_error.cpu())
            row_es.append(es.cpu())
            row_nll_values.append(nll.cpu())
            ent = _prediction_entropy(p)
            entropy.append(ent.cpu())
            support_size.append(mask.sum(dim=1).cpu())
            top1.append(p.max(dim=1).values.cpu())

            rho.append(distribution["correction_rho"].cpu())
            realized_kl.append(
                distribution["correction_realized_kl"].cpu()
            )
            radio_reliability.append(
                distribution["correction_radio_reliability"].cpu()
            )
            geometry_reliability.append(
                distribution["correction_geometry_reliability"].cpu()
            )
            style_offset.append(
                distribution["correction_style_offset"].cpu()
            )
            style_scale.append(
                distribution["correction_style_scale"].cpu()
            )

            delta = location - mean.unsqueeze(1)
            between_anchor_rms.append(
                torch.sqrt(
                    torch.sum(
                        p
                        * torch.sum(delta.square(), dim=2),
                        dim=1,
                    ).clamp(min=0.0)
                ).cpu()
            )

            tail = tail_mass_diagnostics(
                p,
                location,
                target_coords=target,
                mask=mask,
            )
            for name, value in tail.items():
                if name == "geometric_median":
                    continue
                tail_store.setdefault(name, []).append(value.cpu())

            draws = torch.cat([first, second], dim=1)
            radial = torch.linalg.vector_norm(
                draws - gm.unsqueeze(1),
                dim=2,
            )
            for level in coverage_store:
                radius = torch.quantile(radial, float(level), dim=1)
                radius_store[level].append(radius.cpu())
                coverage_store[level].append((error <= radius).cpu())

            candidate_indices = distribution["candidate_indices"]
            candidate_building = model.anchor_buildings[candidate_indices]
            candidate_floor = model.anchor_floors[candidate_indices]
            n_buildings = int(model.anchor_buildings.max().item()) + 1
            n_floors = int(model.anchor_floors.max().item()) + 1
            building_p = _aggregate_label_probability(
                p,
                candidate_building,
                n_labels=n_buildings,
            )
            floor_p = _aggregate_label_probability(
                p,
                candidate_floor,
                n_labels=n_floors,
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
                (
                    building_p.argmax(dim=1) == target_building
                ).cpu()
            )
            floor_correct.append(
                (floor_p.argmax(dim=1) == target_floor).cpu()
            )

    error = torch.cat(row_error)
    mean_error = torch.cat(row_mean_error)
    es = torch.cat(row_es)
    nll = torch.cat(row_nll_values)
    ent = torch.cat(entropy)
    support = torch.cat(support_size)
    b_valid = torch.cat(building_valid)
    f_valid = torch.cat(floor_valid)
    b_correct = torch.cat(building_correct)
    f_correct = torch.cat(floor_correct)

    result: dict[str, object] = {
        "rows": int(len(test.coords_m)),
        "coordinate_groups": int(len(set(keys))),
        "gm_coordinate_macro_mae_m": _coordinate_macro(error, keys),
        "gm_coordinate_macro_mae_se_m": _coordinate_macro_se(error, keys),
        "row_mae_m": float(error.double().mean()),
        "row_median_error_m": float(torch.quantile(error.double(), 0.50)),
        "row_p75_error_m": float(torch.quantile(error.double(), 0.75)),
        "row_p90_error_m": float(torch.quantile(error.double(), 0.90)),
        "posterior_mean_coordinate_macro_mae_m": _coordinate_macro(
            mean_error,
            keys,
        ),
        "energy_score_coordinate_macro_m": _coordinate_macro(es, keys),
        "energy_score_row_mean_m": float(es.double().mean()),
        "mixture_nll_coordinate_macro": _coordinate_macro(nll, keys),
        "mixture_nll_row_mean": float(nll.double().mean()),
        "mean_entropy_nats": float(ent.double().mean()),
        "mean_effective_anchors": float(ent.exp().double().mean()),
        "mean_top1_probability": float(torch.cat(top1).double().mean()),
        "mean_support_size": float(support.double().mean()),
        "support_size_p90": float(torch.quantile(support.double(), 0.90)),
        "mean_between_anchor_rms_m": float(
            torch.cat(between_anchor_rms).double().mean()
        ),
        "mean_rho": float(torch.cat(rho).double().mean()),
        "mean_realized_kl": float(
            torch.cat(realized_kl).double().mean()
        ),
        "mean_radio_reliability": float(
            torch.cat(radio_reliability).double().mean()
        ),
        "mean_geometry_reliability": float(
            torch.cat(geometry_reliability).double().mean()
        ),
        "mean_abs_style_offset": float(
            torch.cat(style_offset).abs().double().mean()
        ),
        "mean_style_scale": float(
            torch.cat(style_scale).double().mean()
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
        "tail_mass": {},
    }
    for level in coverage_store:
        covered = torch.cat(coverage_store[level])
        radius = torch.cat(radius_store[level])
        label = f"{int(level * 100)}"
        result["coverage"][label] = {
            "row_coverage": float(covered.double().mean()),
            "coordinate_macro_coverage": _coverage_macro(covered, keys),
            "mean_radius_m": float(radius.double().mean()),
            "median_radius_m": float(torch.quantile(radius.double(), 0.50)),
        }
    for name, values in tail_store.items():
        joined = torch.cat(values)
        result["tail_mass"][name] = float(joined.double().mean())
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
    train_freeze = load_external_training_freeze(
        training_freeze_path or defaults["freeze"],
        uji_freeze_path=uji_freeze_path,
    )
    destination_lock = Path(lock_path or defaults["lock"])
    destination_result = Path(result_path or defaults["result"])
    if destination_lock.exists():
        raise RuntimeError(
            "external test consumption lock already exists; rerun is forbidden"
        )
    if destination_result.exists():
        raise FileExistsError(destination_result)

    # Complete every check possible before semantic test access.
    for path in _test_paths(spec):
        if not path.exists():
            raise FileNotFoundError(path)
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; test lock not written"
        )

    checkpoint_path = REPO_ROOT / str(train_freeze["checkpoint_path"])
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["manifest_sha256"] != train_freeze["manifest_sha256"]:
        raise ValueError("external checkpoint/manifest link mismatch")

    reference = _reference_from_json(checkpoint["dataset_reference"])
    correction_reference = _reference_from_payload(
        checkpoint["correction_reference"]
    )
    device = torch.device(device_name)
    model = _load_model_checkpoint(checkpoint, device=device)

    # Exercise the complete frozen posterior path on one training-derived
    # prototype scan before the irreversible test lock. This catches device,
    # checkpoint, radio-profiler and calibration-map failures without touching
    # a provided test outcome.
    preflight_rss = model.prototype_rss[:1].detach()
    preflight = v184_distribution(
        model,
        preflight_rss,
        correction_reference=correction_reference,
        epsilon_kl=float(train_freeze["epsilon_kl"]),
        final_temperature=float(train_freeze["posterior_temperature"]),
        scale_multiplier=float(train_freeze["scale_multiplier"]),
    )
    if not bool(torch.isfinite(preflight["candidate_weights"]).all()):
        raise RuntimeError(
            "external frozen posterior preflight is non-finite; "
            "test lock not written"
        )
    if not bool(
        torch.isfinite(preflight["component_locations"]).all()
        and torch.isfinite(preflight["component_scales"]).all()
    ):
        raise RuntimeError(
            "external predictive component preflight is non-finite; "
            "test lock not written"
        )

    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "training_freeze_sha256": train_freeze[
            "training_freeze_sha256"
        ],
        "uji_method_freeze_sha256": train_freeze[
            "uji_method_freeze_sha256"
        ],
        "execution_source_commit_sha": train_freeze[
            "execution_source_commit_sha"
        ],
        "test_outcome_semantic_access_started": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rerun_after_lock_forbidden": True,
    }
    lock["lock_sha256"] = _canonical_sha(lock)
    _write_json_new(destination_lock, lock)

    # First semantic access to the provided test split occurs only after lock.
    test = _load_test(spec, reference)
    test_files = _file_manifest(_test_paths(spec))

    origin = checkpoint["origin"].cpu().numpy()
    metrics = _evaluate_reserved_test(
        model=model,
        correction_reference=correction_reference,
        epsilon_kl=float(train_freeze["epsilon_kl"]),
        temperature=float(train_freeze["posterior_temperature"]),
        scale_multiplier=float(train_freeze["scale_multiplier"]),
        test=test,
        origin=origin,
        device=device,
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "slug": spec.slug,
        "dataset": spec.dataset,
        "heldout_device": spec.heldout_device,
        "training_freeze_sha256": train_freeze[
            "training_freeze_sha256"
        ],
        "uji_method_freeze_sha256": train_freeze[
            "uji_method_freeze_sha256"
        ],
        "test_consumption_lock_sha256": lock["lock_sha256"],
        "test_files": test_files,
        "evaluation_reference_metadata": test.metadata,
        "metrics": metrics,
        "method": {
            "lambda_ot": float(train_freeze["lambda_ot"]),
            "epsilon_kl": float(train_freeze["epsilon_kl"]),
            "selected_epoch": int(train_freeze["selected_epoch"]),
            "posterior_temperature": float(
                train_freeze["posterior_temperature"]
            ),
            "scale_multiplier": float(train_freeze["scale_multiplier"]),
            "decoder": (
                "posterior_weighted_component_center_geometric_median"
            ),
        },
        "no_post_test_retuning_permitted": True,
    }
    result["result_sha256"] = _canonical_sha(result)
    _write_json_new(destination_result, result)
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
        "--uji-freeze",
        default=str(DEFAULT_UJI_FREEZE),
    )
    prepare_parser.add_argument("--manifest")

    fit_parser = sub.add_parser("fit-freeze")
    fit_parser.add_argument("--uji-freeze", default=str(DEFAULT_UJI_FREEZE))
    fit_parser.add_argument("--manifest")
    fit_parser.add_argument("--checkpoint")
    fit_parser.add_argument("--freeze")
    fit_parser.add_argument("--device", default="cuda")

    evaluate_parser = sub.add_parser("evaluate-test")
    evaluate_parser.add_argument(
        "--uji-freeze",
        default=str(DEFAULT_UJI_FREEZE),
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
