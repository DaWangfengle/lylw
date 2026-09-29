"""Final common-core refit and one-shot two-axis confirmation for DR-JAM v4.4."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.device_robust_v4 import (
    estimate_device_effect_covariance,
    load_device_robust_checkpoint,
)
from fhm_dapp_tim.fhm_dapp.evaluate_v4 import (
    _validate_shared_dictionary,
    evaluate_posterior,
)
from fhm_dapp_tim.fhm_dapp.freeze_confirmation_v43 import (
    validate_confirmation_manifest,
)
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    PROTOCOL_VERSION,
    coordinate_axis_partition,
    sha256_indices,
    sha256_location_keys,
    validate_registry,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v4 import (
    FROZEN_MEMBER_V4,
    _configure_cuda_reproducibility,
    _fit_fixed,
    _member_training_spec,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_decisions(
    *,
    registry: dict,
    mechanism_path: Path,
    structure_path: Path,
) -> tuple[dict, dict]:
    mechanism = _load_json(mechanism_path)
    if mechanism.get("artifact_status") != "registered_v44_mechanism_freeze":
        raise ValueError("mechanism artifact has no v4.4 freeze authority")
    if mechanism.get("selection_authority") is not True:
        raise ValueError("mechanism selection authority is false")
    if mechanism.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("mechanism protocol mismatch")
    if mechanism.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("mechanism registry mismatch")

    structure = _load_json(structure_path)
    if structure.get("artifact_status") != "registered_v44_structure_freeze":
        raise ValueError(
            "structure is not frozen; complete any registered boundary extension"
        )
    if structure.get("selection_authority") is not True:
        raise ValueError("structure selection authority is false")
    if structure.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("structure protocol mismatch")
    if structure.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("structure registry mismatch")
    if structure.get("mechanism_freeze_sha256") != _sha256_file(mechanism_path):
        raise ValueError("structure does not bind the supplied mechanism freeze")
    if structure.get("selected_mechanism") != mechanism.get("selected_mechanism"):
        raise ValueError("mechanism and structure selections differ")
    return mechanism, structure


def _registered_member_seeds(registry: dict, structure: dict) -> tuple[int, ...]:
    member_count = int(structure["selected"]["member_count_E"])
    pool = tuple(map(int, registry["structural_member_seeds"]))
    if not 1 <= member_count <= len(pool):
        raise ValueError("selected member count exceeds the registered pool")
    return pool[:member_count]


def _development_checkpoint(
    root: Path, mechanism: str, fold: int, seed: int
) -> Path:
    return root / "coordinate" / mechanism / f"fold{fold}_seed{seed}.pt"


def _validate_epoch_sources(
    *,
    registry: dict,
    training_mechanism: str,
    seed: int,
    checkpoint_root: Path,
    structure_cache_root: Path,
) -> tuple[int, list[dict]]:
    records = []
    epochs = []
    for fold, audit in enumerate(registry["coordinate_development_audit"]):
        path = _development_checkpoint(
            checkpoint_root, training_mechanism, fold, seed
        ).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        if checkpoint.get("artifact_status") != "registered_development_member":
            raise ValueError(f"epoch source has no selection authority: {path}")
        if checkpoint.get("selection_authority") is not True:
            raise ValueError(f"epoch source selection authority is false: {path}")
        if checkpoint.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"epoch source registry mismatch: {path}")
        if config.get("training_protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"epoch source protocol mismatch: {path}")
        if config.get("mechanism") != training_mechanism:
            raise ValueError(f"epoch source mechanism mismatch: {path}")
        if config.get("role_axis") != "coordinate" or int(
            config.get("outer_fold", -1)
        ) != fold:
            raise ValueError(f"epoch source development fold mismatch: {path}")
        if int(checkpoint.get("seed", -1)) != int(seed):
            raise ValueError(f"epoch source seed mismatch: {path}")
        expected_training = np.sort(
            np.concatenate(
                (
                    checkpoint["role_indices"]["member_fit"]
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False),
                    checkpoint["role_indices"]["early_stop"]
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False),
                )
            )
        )
        if sha256_indices(expected_training) != checkpoint.get(
            "training_indices_sha256"
        ):
            raise ValueError(f"epoch source did not refit its full fold: {path}")
        expected_selection = audit["roles"]["structure_development"][
            "row_index_sha256"
        ]
        if checkpoint["role_indices_sha256"][
            "structure_development"
        ] != expected_selection:
            raise ValueError(f"epoch source selection role changed: {path}")

        cache_path = (structure_cache_root / f"fold{fold}.pt").resolve()
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache.get("artifact_status") != "registered_v44_structure_component_cache":
            raise ValueError(f"structure cache has no authority: {cache_path}")
        if cache.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"structure cache registry mismatch: {cache_path}")
        source_hash = _sha256_file(path)
        bound_hashes = cache.get("checkpoint_sha256", {})
        if source_hash not in set(bound_hashes.values()):
            raise ValueError(
                f"selected epoch checkpoint is not bound by structure cache: {path}"
            )
        epoch = int(config.get("selected_epoch", 0))
        if epoch <= 0:
            raise ValueError(f"epoch source has no selected epoch: {path}")
        epochs.append(epoch)
        records.append(
            {
                "development_fold": fold,
                "checkpoint": str(path),
                "checkpoint_sha256": source_hash,
                "structure_cache": str(cache_path),
                "structure_cache_sha256": _sha256_file(cache_path),
                "selected_epoch": epoch,
            }
        )
    if len(epochs) != 5:
        raise ValueError("final epoch rule requires five development folds")
    selected_epoch = int(np.median(np.asarray(epochs, dtype=np.int64)))
    return selected_epoch, records


def _diagnostic_mechanisms(mechanism_freeze: dict) -> dict[str, str]:
    panel = mechanism_freeze.get("coordinate_group_macro_mae_m", {})
    frozen_m1 = sorted(name for name in panel if str(name).startswith("M1-g"))
    if len(frozen_m1) != 1:
        raise ValueError("mechanism freeze does not identify one frozen M1 control")
    return {"M0": "M0", "M1": frozen_m1[0], "M2": "M2"}


def _validate_diagnostic_epoch_sources(
    *,
    registry: dict,
    mechanism_freeze: dict,
    training_mechanism: str,
    checkpoint_root: Path,
    mechanism_cache_root: Path,
) -> tuple[int, list[dict]]:
    """Bind the seed-4101 diagnostic epoch rule to mechanism-stage evidence."""
    seed = 4101
    epochs = []
    records = []
    frozen_cache_hashes = mechanism_freeze.get("cache_sha256", {})
    for fold, audit in enumerate(registry["coordinate_development_audit"]):
        checkpoint_path = _development_checkpoint(
            checkpoint_root, training_mechanism, fold, seed
        ).resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        config = checkpoint.get("config", {})
        checks = {
            "status": checkpoint.get("artifact_status")
            == "registered_development_member",
            "authority": checkpoint.get("selection_authority") is True,
            "registry": checkpoint.get("registry_sha256")
            == registry["registry_sha256"],
            "source": checkpoint.get("source_sha256")
            == registry["training_data_sha256"],
            "protocol": config.get("training_protocol_version")
            == PROTOCOL_VERSION,
            "mechanism": config.get("mechanism") == training_mechanism,
            "role": config.get("role_axis") == "coordinate"
            and int(config.get("outer_fold", -1)) == fold,
            "seed": int(checkpoint.get("seed", -1)) == seed,
            "selection rows": checkpoint.get("role_indices_sha256", {}).get(
                "structure_development"
            )
            == audit["roles"]["structure_development"]["row_index_sha256"],
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(
                "diagnostic epoch source mismatch "
                f"({', '.join(failed)}): {checkpoint_path}"
            )
        cache_path = (
            mechanism_cache_root
            / "coordinate"
            / training_mechanism
            / f"fold{fold}_seed{seed}.pt"
        ).resolve()
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        cache_checks = {
            "status": cache.get("artifact_status")
            == "registered_development_evaluation",
            "protocol": cache.get("protocol_version") == PROTOCOL_VERSION,
            "registry": cache.get("registry_sha256")
            == registry["registry_sha256"],
            "source": cache.get("source_sha256")
            == registry["training_data_sha256"],
            "role": cache.get("evaluation_role") == "structure_development",
            "mechanism": cache.get("summary", {}).get("mechanism")
            == training_mechanism,
            "isolation": cache.get("mechanism_diagnostic_contract")
            == registry["mechanism_diagnostic"],
            "rows": sha256_indices(cache["row_indices"].cpu().numpy())
            == audit["roles"]["structure_development"]["row_index_sha256"],
            "frozen cache": frozen_cache_hashes.get(str(cache_path))
            == _sha256_file(cache_path),
            "checkpoint binding": _sha256_file(checkpoint_path)
            in set(cache.get("checkpoint_sha256", {}).values()),
        }
        cache_failed = [name for name, valid in cache_checks.items() if not valid]
        if cache_failed:
            raise ValueError(
                "diagnostic mechanism cache mismatch "
                f"({', '.join(cache_failed)}): {cache_path}"
            )
        epoch = int(config.get("selected_epoch", 0))
        if epoch <= 0:
            raise ValueError(f"diagnostic epoch is invalid: {checkpoint_path}")
        epochs.append(epoch)
        records.append(
            {
                "development_fold": fold,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "mechanism_cache": str(cache_path),
                "mechanism_cache_sha256": _sha256_file(cache_path),
                "selected_epoch": epoch,
            }
        )
    if len(epochs) != 5:
        raise ValueError("diagnostic epoch rule requires five development folds")
    return int(np.median(np.asarray(epochs, dtype=np.int64))), records


def refit_final_member(args: argparse.Namespace) -> dict:
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    mechanism_path = Path(args.mechanism_freeze).resolve()
    structure_path = Path(args.structure_freeze).resolve()
    mechanism, structure = _validate_decisions(
        registry=registry,
        mechanism_path=mechanism_path,
        structure_path=structure_path,
    )
    confirmation_path = Path(args.confirmation_manifest).resolve()
    confirmation = validate_confirmation_manifest(
        confirmation_path,
        train_data=data_path,
        registry_path=registry_path,
    )
    seeds = _registered_member_seeds(registry, structure)
    seed = int(args.seed)
    if seed not in seeds:
        raise ValueError("seed is not in the canonical first-E member set")
    training_mechanism = str(mechanism["training_member_mechanism"])
    selected_epoch, epoch_sources = _validate_epoch_sources(
        registry=registry,
        training_mechanism=training_mechanism,
        seed=seed,
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        structure_cache_root=Path(args.structure_cache_root).resolve(),
    )

    if torch.cuda.is_available():
        _configure_cuda_reproducibility()
        torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(max(1, min(int(args.threads), torch.get_num_threads())))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        timestamp_t,
        source_stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, source_stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)
    development = tuple(map(int, registry["device_development_phone_ids"]))
    axis = coordinate_axis_partition(keys, phones, development)
    core = set(axis["development_core"])
    train_idx = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in core
        ],
        dtype=np.int64,
    )
    if sha256_indices(train_idx) != confirmation["training_contract"][
        "fit_role"
    ]["row_index_sha256"]:
        raise ValueError("final common-core fit role differs from confirmation freeze")

    started = time.perf_counter()
    (
        model,
        memory,
        centered,
        sample_anchor,
        stats,
        origin,
        training_phones,
        refit_history,
    ) = _fit_fixed(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phones,
        coordinate_keys=keys,
        source_stats=source_stats,
        train_idx=train_idx,
        mechanism=training_mechanism,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=int(args.log_every),
    )
    spec = _member_training_spec(training_mechanism)
    device_effect = None
    if spec["stores_rank2_device_effect"]:
        estimate = estimate_device_effect_covariance(
            rss, sample_anchor, phones, train_idx
        )
        device_effect = {
            "covariance": estimate.covariance.cpu(),
            "phone_ids": list(estimate.phone_ids),
            "effects": estimate.effects.cpu(),
            "observation_counts": list(estimate.observation_counts),
            "shared_anchor_counts": list(estimate.shared_anchor_counts),
            "estimation_indices_sha256": sha256_indices(train_idx),
        }
    config = {
        **FROZEN_MEMBER_V4,
        **spec,
        "training_protocol_version": PROTOCOL_VERSION,
        "role_axis": "final-core",
        "outer_fold": None,
        "held_out_phone": None,
        "n_training_devices": (
            len(training_phones) if spec["member_variant"] == "dann" else 0
        ),
        "selected_epoch": selected_epoch,
        "selected_member_count_E": int(structure["selected"]["member_count_E"]),
        "selected_top_k": int(structure["selected"]["top_k"]),
        "selected_loss_temperature_lambda": float(
            structure["selected"]["loss_temperature_lambda"]
        ),
        "epoch_rule": "integer median across five internal development folds",
    }
    checkpoint = {
        "artifact_status": "registered_v44_final_common_core_member",
        "selection_authority": False,
        "confirmation_evaluation_authority": True,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "radio_memory": {
            key: value.detach().cpu()
            for key, value in memory.items()
            if key
            in {
                "detection_prob",
                "rss_center",
                "rss_scale",
                "counts",
                "device_counts",
            }
        },
        "device_effect": device_effect,
        "config": config,
        "stats": stats,
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "anchor_coordinate_keys": [list(key) for key in memory["coordinate_keys"]],
        "anchor_coordinate_keys_sha256": sha256_location_keys(
            memory["coordinate_keys"]
        ),
        "seed": seed,
        "canonical_first_E_member_seeds": list(seeds),
        "training_phone_ids": list(training_phones),
        "training_indices": torch.as_tensor(train_idx, dtype=torch.long),
        "training_indices_sha256": sha256_indices(train_idx),
        "selected_epoch_sources": epoch_sources,
        "refit_history": refit_history,
        "source_sha256": _sha256_file(data_path),
        "registry_path": str(registry_path),
        "registry_sha256": registry["registry_sha256"],
        "mechanism_freeze_sha256": _sha256_file(mechanism_path),
        "structure_freeze_sha256": _sha256_file(structure_path),
        "confirmation_manifest_sha256": _sha256_file(confirmation_path),
        "timestamp_dtype": str(timestamp_t.numpy().dtype),
        "training_seconds": float(time.perf_counter() - started),
        "centered_coordinate_shape": list(centered.shape),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final member: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    printable = {
        "output": str(output),
        "artifact_status": checkpoint["artifact_status"],
        "seed": seed,
        "selected_epoch": selected_epoch,
        "training_rows": len(train_idx),
        "anchors": model.n_anchors,
        "training_mechanism": training_mechanism,
        "selected_mechanism": mechanism["selected_mechanism"],
        "training_seconds": checkpoint["training_seconds"],
    }
    print(json.dumps(printable, indent=2), flush=True)
    return printable


def refit_isolation_diagnostic(args: argparse.Namespace) -> dict:
    """Refit one frozen E=1/full/lambda=1 attribution member on the core."""
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    mechanism_path = Path(args.mechanism_freeze).resolve()
    structure_path = Path(args.structure_freeze).resolve()
    mechanism, _structure = _validate_decisions(
        registry=registry,
        mechanism_path=mechanism_path,
        structure_path=structure_path,
    )
    baseline_path = Path(args.baseline_freeze).resolve()
    _validate_baseline_freeze(baseline_path, registry)
    confirmation_path = Path(args.confirmation_manifest).resolve()
    confirmation = validate_confirmation_manifest(
        confirmation_path,
        train_data=data_path,
        registry_path=registry_path,
    )
    diagnostic_label = str(args.diagnostic)
    mechanisms = _diagnostic_mechanisms(mechanism)
    if diagnostic_label not in mechanisms:
        raise ValueError("diagnostic must be M0, M1, or M2")
    training_mechanism = mechanisms[diagnostic_label]
    selected_epoch, epoch_sources = _validate_diagnostic_epoch_sources(
        registry=registry,
        mechanism_freeze=mechanism,
        training_mechanism=training_mechanism,
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        mechanism_cache_root=Path(args.mechanism_cache_root).resolve(),
    )

    if torch.cuda.is_available():
        _configure_cuda_reproducibility()
        torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(max(1, min(int(args.threads), torch.get_num_threads())))
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        timestamp_t,
        source_stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, source_stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)
    development = tuple(map(int, registry["device_development_phone_ids"]))
    core = set(
        coordinate_axis_partition(keys, phones, development)["development_core"]
    )
    train_idx = np.asarray(
        [
            row
            for row, phone in enumerate(phones)
            if int(phone) in development and keys[row] in core
        ],
        dtype=np.int64,
    )
    if sha256_indices(train_idx) != confirmation["training_contract"][
        "fit_role"
    ]["row_index_sha256"]:
        raise ValueError("diagnostic common-core role differs from confirmation freeze")

    started = time.perf_counter()
    (
        model,
        memory,
        centered,
        sample_anchor,
        stats,
        origin,
        training_phones,
        refit_history,
    ) = _fit_fixed(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor,
        phone_ids=phones,
        coordinate_keys=keys,
        source_stats=source_stats,
        train_idx=train_idx,
        mechanism=training_mechanism,
        seed=4101,
        epochs=selected_epoch,
        device=device,
        log_every=int(args.log_every),
    )
    spec = _member_training_spec(training_mechanism)
    device_effect = None
    if spec["stores_rank2_device_effect"]:
        estimate = estimate_device_effect_covariance(
            rss, sample_anchor, phones, train_idx
        )
        device_effect = {
            "covariance": estimate.covariance.cpu(),
            "phone_ids": list(estimate.phone_ids),
            "effects": estimate.effects.cpu(),
            "observation_counts": list(estimate.observation_counts),
            "shared_anchor_counts": list(estimate.shared_anchor_counts),
            "estimation_indices_sha256": sha256_indices(train_idx),
        }
    config = {
        **FROZEN_MEMBER_V4,
        **spec,
        "training_protocol_version": PROTOCOL_VERSION,
        "role_axis": "final-core-isolation-diagnostic",
        "outer_fold": None,
        "held_out_phone": None,
        "n_training_devices": (
            len(training_phones) if spec["member_variant"] == "dann" else 0
        ),
        "selected_epoch": selected_epoch,
        "diagnostic_label": diagnostic_label,
        "member_count_E": 1,
        "top_k": "full_dictionary",
        "loss_temperature_lambda": 1.0,
        "epoch_rule": (
            "integer median of seed-4101 epochs across five mechanism-stage folds"
        ),
    }
    checkpoint = {
        "artifact_status": "registered_v44_final_isolation_diagnostic_member",
        "selection_authority": False,
        "confirmation_evaluation_authority": True,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "radio_memory": {
            key: value.detach().cpu()
            for key, value in memory.items()
            if key
            in {
                "detection_prob",
                "rss_center",
                "rss_scale",
                "counts",
                "device_counts",
            }
        },
        "device_effect": device_effect,
        "config": config,
        "stats": stats,
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "anchor_coordinate_keys": [list(key) for key in memory["coordinate_keys"]],
        "anchor_coordinate_keys_sha256": sha256_location_keys(
            memory["coordinate_keys"]
        ),
        "seed": 4101,
        "training_phone_ids": list(training_phones),
        "training_indices": torch.as_tensor(train_idx, dtype=torch.long),
        "training_indices_sha256": sha256_indices(train_idx),
        "selected_epoch_sources": epoch_sources,
        "refit_history": refit_history,
        "source_sha256": _sha256_file(data_path),
        "registry_path": str(registry_path),
        "registry_sha256": registry["registry_sha256"],
        "mechanism_freeze_sha256": _sha256_file(mechanism_path),
        "structure_freeze_sha256": _sha256_file(structure_path),
        "baseline_freeze_sha256": _sha256_file(baseline_path),
        "confirmation_manifest_sha256": _sha256_file(confirmation_path),
        "timestamp_dtype": str(timestamp_t.numpy().dtype),
        "training_seconds": float(time.perf_counter() - started),
        "centered_coordinate_shape": list(centered.shape),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic member: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    printable = {
        "output": str(output),
        "artifact_status": checkpoint["artifact_status"],
        "diagnostic": diagnostic_label,
        "training_mechanism": training_mechanism,
        "selected_epoch": selected_epoch,
        "training_rows": len(train_idx),
        "anchors": model.n_anchors,
        "training_seconds": checkpoint["training_seconds"],
    }
    print(json.dumps(printable, indent=2), flush=True)
    return printable


def _confirmation_roles(
    *,
    keys: list[tuple],
    phones: np.ndarray,
    registry: dict,
    manifest: dict,
) -> dict[str, np.ndarray]:
    development = set(map(int, registry["device_development_phone_ids"]))
    confirmation = set(
        map(int, registry["device_confirmation_selection"]["phone_ids"])
    )
    axis = coordinate_axis_partition(keys, phones, development)
    core = set(axis["development_core"])
    coordinate_confirmation = set(axis["coordinate_confirmation"])
    roles = {
        "coordinate_confirmation": np.asarray(
            [
                row
                for row, phone in enumerate(phones)
                if int(phone) in development and keys[row] in coordinate_confirmation
            ],
            dtype=np.int64,
        ),
        "matched_device_confirmation": np.asarray(
            [
                row
                for row, phone in enumerate(phones)
                if int(phone) in confirmation and keys[row] in core
            ],
            dtype=np.int64,
        ),
        "device_plus_coordinate_confirmation": np.asarray(
            [
                row
                for row, phone in enumerate(phones)
                if int(phone) in confirmation and keys[row] in coordinate_confirmation
            ],
            dtype=np.int64,
        ),
        "device_plus_confirmation_only_coordinate": np.asarray(
            [
                row
                for row, phone in enumerate(phones)
                if (
                    int(phone) in confirmation
                    and keys[row] not in core
                    and keys[row] not in coordinate_confirmation
                )
            ],
            dtype=np.int64,
        ),
    }
    expected_coordinate = manifest["coordinate_confirmation_contract"][
        "development_phone_test_role"
    ]["row_index_sha256"]
    if sha256_indices(roles["coordinate_confirmation"]) != expected_coordinate:
        raise ValueError("coordinate confirmation role differs from its manifest")
    per_phone = manifest["confirmation_contract"]["per_phone"]
    mapping = {
        "matched_device_confirmation": "matched_device_test",
        "device_plus_coordinate_confirmation": (
            "device_plus_registered_coordinate_shift_test"
        ),
        "device_plus_confirmation_only_coordinate": (
            "device_plus_confirmation_only_coordinate_stress"
        ),
    }
    for role_name, manifest_name in mapping.items():
        for phone in confirmation:
            values = roles[role_name][phones[roles[role_name]] == phone]
            expected = per_phone[str(phone)][manifest_name]["row_index_sha256"]
            if sha256_indices(values) != expected:
                raise ValueError(
                    f"confirmation role changed for phone {phone}: {role_name}"
                )
    return roles


def _validate_baseline_freeze(path: Path, registry: dict) -> dict:
    payload = _load_json(path)
    if payload.get("artifact_status") != "registered_v44_baseline_freeze":
        raise ValueError("classical baselines are not frozen")
    if payload.get("selection_authority") is not True:
        raise ValueError("baseline freeze selection authority is false")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("baseline freeze protocol mismatch")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("baseline freeze registry mismatch")
    return payload


@torch.no_grad()
def evaluate_confirmation(args: argparse.Namespace) -> dict:
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    mechanism_path = Path(args.mechanism_freeze).resolve()
    structure_path = Path(args.structure_freeze).resolve()
    mechanism, structure = _validate_decisions(
        registry=registry,
        mechanism_path=mechanism_path,
        structure_path=structure_path,
    )
    baseline_path = Path(args.baseline_freeze).resolve()
    _validate_baseline_freeze(baseline_path, registry)
    manifest_path = Path(args.confirmation_manifest).resolve()
    manifest = validate_confirmation_manifest(
        manifest_path,
        train_data=data_path,
        registry_path=registry_path,
    )
    seeds = _registered_member_seeds(registry, structure)
    paths = [Path(value).resolve() for value in args.checkpoints]
    if len(paths) != len(seeds):
        raise ValueError("final checkpoint count differs from selected E")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    loaded = [load_device_robust_checkpoint(path, device) for path in paths]
    models = [item[0] for item in loaded]
    checkpoints = [item[1] for item in loaded]
    expected_training_mechanism = str(mechanism["training_member_mechanism"])
    for seed, path, checkpoint in zip(seeds, paths, checkpoints):
        config = checkpoint.get("config", {})
        if checkpoint.get("artifact_status") != "registered_v44_final_common_core_member":
            raise ValueError(f"checkpoint is not a final common-core member: {path}")
        if checkpoint.get("confirmation_evaluation_authority") is not True:
            raise ValueError(f"checkpoint has no confirmation authority: {path}")
        if checkpoint.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"final checkpoint registry mismatch: {path}")
        if checkpoint.get("mechanism_freeze_sha256") != _sha256_file(mechanism_path):
            raise ValueError(f"final checkpoint mechanism mismatch: {path}")
        if checkpoint.get("structure_freeze_sha256") != _sha256_file(structure_path):
            raise ValueError(f"final checkpoint structure mismatch: {path}")
        if checkpoint.get("confirmation_manifest_sha256") != _sha256_file(manifest_path):
            raise ValueError(f"final checkpoint confirmation manifest mismatch: {path}")
        if int(checkpoint.get("seed", -1)) != seed:
            raise ValueError(f"final checkpoint seed prefix mismatch: {path}")
        if config.get("role_axis") != "final-core":
            raise ValueError(f"final checkpoint role mismatch: {path}")
        if config.get("mechanism") != expected_training_mechanism:
            raise ValueError(f"final checkpoint mechanism mismatch: {path}")
        if int(config.get("selected_member_count_E", -1)) != len(seeds):
            raise ValueError(f"final checkpoint E mismatch: {path}")
    _validate_shared_dictionary(models, checkpoints)

    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)
    roles = _confirmation_roles(
        keys=keys, phones=phones, registry=registry, manifest=manifest
    )
    selected_mechanism = str(mechanism["selected_mechanism"])
    top_k = int(structure["selected"]["top_k"])
    loss_temperature = float(structure["selected"]["loss_temperature_lambda"])
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation result: {output}")
    if args.dry_run:
        report = {
            "artifact_status": "registered_v44_confirmation_preflight",
            "ready": True,
            "protocol_version": PROTOCOL_VERSION,
            "registry_sha256": registry["registry_sha256"],
            "mechanism_freeze_sha256": _sha256_file(mechanism_path),
            "structure_freeze_sha256": _sha256_file(structure_path),
            "baseline_freeze_sha256": _sha256_file(baseline_path),
            "confirmation_manifest_sha256": _sha256_file(manifest_path),
            "selected_mechanism": selected_mechanism,
            "selected_structure": structure["selected"],
            "member_seeds": list(seeds),
            "checkpoint_sha256": {
                str(path): _sha256_file(path) for path in paths
            },
            "role_rows": {
                name: int(len(indices)) for name, indices in roles.items()
            },
            "role_row_sha256": {
                name: sha256_indices(indices) for name, indices in roles.items()
            },
            "intended_output": str(output),
            "protected_outcomes_evaluated": False,
            "output_written": False,
        }
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return report
    results = {}
    started = time.perf_counter()
    for role_name, indices in roles.items():
        if len(indices) == 0:
            results[role_name] = {"summary": None, "empty_role": True}
            continue
        results[role_name] = evaluate_posterior(
            models=models,
            checkpoints=checkpoints,
            rss=rss,
            raw_xy=raw_xy,
            building=building,
            floor=floor,
            phone_ids=phones,
            coordinate_keys=keys,
            indices=torch.as_tensor(indices, dtype=torch.long),
            mechanism=selected_mechanism,
            top_k=top_k,
            loss_temperature=loss_temperature,
            batch_size=int(args.batch_size),
        )
    payload = {
        "artifact_status": "registered_v44_two_axis_confirmation_evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "mechanism_freeze_sha256": _sha256_file(mechanism_path),
        "structure_freeze_sha256": _sha256_file(structure_path),
        "baseline_freeze_sha256": _sha256_file(baseline_path),
        "confirmation_manifest_sha256": _sha256_file(manifest_path),
        "selected_mechanism": selected_mechanism,
        "selected_structure": structure["selected"],
        "checkpoint_sha256": {
            str(path): _sha256_file(path) for path in paths
        },
        "role_row_sha256": {
            name: sha256_indices(indices) for name, indices in roles.items()
        },
        "results": results,
        "elapsed_seconds": float(time.perf_counter() - started),
        "protected_roles_opened_once": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    printable = {
        "output": str(output),
        "artifact_status": payload["artifact_status"],
        "selected_mechanism": selected_mechanism,
        "selected_structure": structure["selected"],
        "summaries": {
            name: result.get("summary") for name, result in results.items()
        },
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(printable, indent=2, allow_nan=False), flush=True)
    return printable


@torch.no_grad()
def evaluate_isolation_diagnostics(args: argparse.Namespace) -> dict:
    """Evaluate frozen common-core ablations only after primary confirmation locks."""
    data_path = resolve_path(args.train_data).resolve()
    registry_path = Path(args.registry).resolve()
    registry = validate_registry(registry_path, train_data=data_path)
    mechanism_path = Path(args.mechanism_freeze).resolve()
    structure_path = Path(args.structure_freeze).resolve()
    mechanism, _structure = _validate_decisions(
        registry=registry,
        mechanism_path=mechanism_path,
        structure_path=structure_path,
    )
    baseline_path = Path(args.baseline_freeze).resolve()
    _validate_baseline_freeze(baseline_path, registry)
    manifest_path = Path(args.confirmation_manifest).resolve()
    manifest = validate_confirmation_manifest(
        manifest_path, train_data=data_path, registry_path=registry_path
    )

    primary_path = Path(args.primary_confirmation).resolve()
    primary = torch.load(primary_path, map_location="cpu", weights_only=True)
    if primary.get("artifact_status") != "registered_v44_two_axis_confirmation_evaluation":
        raise ValueError("primary DR-JAM confirmation is not locked")
    if primary.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("primary confirmation registry mismatch")
    baseline_confirmation_path = Path(args.baseline_confirmation).resolve()
    baseline_confirmation = _load_json(baseline_confirmation_path)
    if baseline_confirmation.get("artifact_status") != (
        "registered_v44_baseline_two_axis_confirmation"
    ):
        raise ValueError("baseline confirmation is not locked")
    if baseline_confirmation.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("baseline confirmation registry mismatch")
    summary_path = Path(args.confirmation_summary).resolve()
    confirmation_summary = _load_json(summary_path)
    summary_checks = {
        "status": confirmation_summary.get("artifact_status")
        == "registered_v44_confirmation_summary",
        "protocol": confirmation_summary.get("protocol_version")
        == PROTOCOL_VERSION,
        "registry": confirmation_summary.get("registry_sha256")
        == registry["registry_sha256"],
        "primary binding": confirmation_summary.get(
            "dr_jam_confirmation_sha256"
        )
        == _sha256_file(primary_path),
        "baseline binding": confirmation_summary.get(
            "baseline_confirmation_sha256"
        )
        == _sha256_file(baseline_confirmation_path),
    }
    failed_summary = [
        name for name, valid in summary_checks.items() if not valid
    ]
    if failed_summary:
        raise ValueError(
            "confirmation summary is not locked "
            f"({', '.join(failed_summary)})"
        )

    diagnostic_mechanisms = _diagnostic_mechanisms(mechanism)
    checkpoint_paths = [Path(value).resolve() for value in args.checkpoints]
    if len(checkpoint_paths) != 3:
        raise ValueError("diagnostic evaluation requires M0, M1, and M2 checkpoints")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    by_label = {}
    for path in checkpoint_paths:
        model, checkpoint = load_device_robust_checkpoint(path, device)
        config = checkpoint.get("config", {})
        label = str(config.get("diagnostic_label"))
        if label not in diagnostic_mechanisms or label in by_label:
            raise ValueError(f"duplicate or unknown diagnostic checkpoint: {path}")
        expected_mechanism = diagnostic_mechanisms[label]
        checks = {
            "status": checkpoint.get("artifact_status")
            == "registered_v44_final_isolation_diagnostic_member",
            "no selection authority": checkpoint.get("selection_authority")
            is False,
            "confirmation authority": checkpoint.get(
                "confirmation_evaluation_authority"
            )
            is True,
            "registry": checkpoint.get("registry_sha256")
            == registry["registry_sha256"],
            "mechanism freeze": checkpoint.get("mechanism_freeze_sha256")
            == _sha256_file(mechanism_path),
            "structure freeze": checkpoint.get("structure_freeze_sha256")
            == _sha256_file(structure_path),
            "baseline freeze": checkpoint.get("baseline_freeze_sha256")
            == _sha256_file(baseline_path),
            "confirmation manifest": checkpoint.get(
                "confirmation_manifest_sha256"
            )
            == _sha256_file(manifest_path),
            "role": config.get("role_axis")
            == "final-core-isolation-diagnostic",
            "mechanism": config.get("mechanism") == expected_mechanism,
            "seed": int(checkpoint.get("seed", -1)) == 4101,
            "E": int(config.get("member_count_E", -1)) == 1,
            "K": config.get("top_k") == "full_dictionary",
            "lambda": float(config.get("loss_temperature_lambda", -1.0))
            == 1.0,
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise ValueError(
                f"diagnostic checkpoint mismatch ({', '.join(failed)}): {path}"
            )
        by_label[label] = (path, model, checkpoint)
    if set(by_label) != set(diagnostic_mechanisms):
        raise ValueError("diagnostic checkpoint panel is incomplete")
    _validate_shared_dictionary(
        [by_label[label][1] for label in ("M0", "M1", "M2")],
        [by_label[label][2] for label in ("M0", "M1", "M2")],
    )

    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, dataset_type="UJI")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor = floor_t.numpy()
    phones = phone_t.numpy().astype(np.int64, copy=False)
    keys = location_keys(raw_xy, building, floor)
    roles = _confirmation_roles(
        keys=keys, phones=phones, registry=registry, manifest=manifest
    )
    evaluation_specs = {
        "M0": ("M0", "M0"),
        "M1": ("M1", diagnostic_mechanisms["M1"]),
        "M2": ("M2", "M2"),
        "M3": ("M2", "M3"),
    }
    results = {}
    started = time.perf_counter()
    for output_name, (checkpoint_label, evaluation_mechanism) in evaluation_specs.items():
        _path, model, checkpoint = by_label[checkpoint_label]
        role_results = {}
        for role_name, indices in roles.items():
            if len(indices) == 0:
                role_results[role_name] = {"summary": None, "empty_role": True}
                continue
            role_results[role_name] = evaluate_posterior(
                models=[model],
                checkpoints=[checkpoint],
                rss=rss,
                raw_xy=raw_xy,
                building=building,
                floor=floor,
                phone_ids=phones,
                coordinate_keys=keys,
                indices=torch.as_tensor(indices, dtype=torch.long),
                mechanism=evaluation_mechanism,
                top_k=int(model.n_anchors),
                loss_temperature=1.0,
                batch_size=int(args.batch_size),
            )
        results[output_name] = {
            "training_checkpoint": checkpoint_label,
            "evaluation_mechanism": evaluation_mechanism,
            "isolation_contract": {
                "member_count_E": 1,
                "top_k": "full_dictionary",
                "loss_temperature_lambda": 1.0,
            },
            "roles": role_results,
        }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic confirmation: {output}"
        )
    payload = {
        "artifact_status": "registered_v44_isolation_diagnostic_confirmation",
        "selection_authority": False,
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "mechanism_freeze_sha256": _sha256_file(mechanism_path),
        "structure_freeze_sha256": _sha256_file(structure_path),
        "baseline_freeze_sha256": _sha256_file(baseline_path),
        "confirmation_manifest_sha256": _sha256_file(manifest_path),
        "primary_confirmation_sha256": _sha256_file(primary_path),
        "baseline_confirmation_sha256": _sha256_file(
            baseline_confirmation_path
        ),
        "confirmation_summary_sha256": _sha256_file(summary_path),
        "checkpoint_sha256": {
            label: _sha256_file(by_label[label][0]) for label in by_label
        },
        "role_row_sha256": {
            name: sha256_indices(indices) for name, indices in roles.items()
        },
        "results": results,
        "protected_results_cannot_select_or_alter_method": True,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    printable = {
        "output": str(output),
        "artifact_status": payload["artifact_status"],
        "summaries": {
            mechanism_name: {
                role_name: role_result.get("summary")
                for role_name, role_result in mechanism_result["roles"].items()
            }
            for mechanism_name, mechanism_result in results.items()
        },
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    print(json.dumps(printable, indent=2, allow_nan=False), flush=True)
    return printable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DR-JAM v4.4 final workflow")
    children = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    common.add_argument(
        "--registry", default="fhm_dapp_tim/evidence/v44/registry_v4_4.json"
    )
    common.add_argument(
        "--mechanism-freeze",
        default="fhm_dapp_tim/evidence/v44/mechanism_selection_v4_4.json",
    )
    common.add_argument(
        "--structure-freeze",
        default="fhm_dapp_tim/evidence/v44/structure_selection_v4_4.json",
    )
    common.add_argument(
        "--confirmation-manifest",
        default="fhm_dapp_tim/evidence/v44/confirmation_roles_v4_4.json",
    )
    common.add_argument("--device", default="auto")

    refit = children.add_parser("refit", parents=[common])
    refit.add_argument("--seed", type=int, required=True)
    refit.add_argument(
        "--checkpoint-root", default="fhm_dapp_tim/models/v44/mechanism"
    )
    refit.add_argument(
        "--structure-cache-root",
        default="fhm_dapp_tim/evidence/v44/caches/structure",
    )
    refit.add_argument("--threads", type=int, default=4)
    refit.add_argument("--log-every", type=int, default=5)
    refit.add_argument("--output", required=True)
    refit.set_defaults(handler=refit_final_member)

    diagnostic_refit = children.add_parser(
        "diagnostic-refit", parents=[common]
    )
    diagnostic_refit.add_argument(
        "--diagnostic", choices=("M0", "M1", "M2"), required=True
    )
    diagnostic_refit.add_argument(
        "--baseline-freeze",
        default="fhm_dapp_tim/evidence/v44/baseline_selection_v4_4.json",
    )
    diagnostic_refit.add_argument(
        "--checkpoint-root", default="fhm_dapp_tim/models/v44/mechanism"
    )
    diagnostic_refit.add_argument(
        "--mechanism-cache-root",
        default="fhm_dapp_tim/evidence/v44/caches/mechanism",
    )
    diagnostic_refit.add_argument("--threads", type=int, default=4)
    diagnostic_refit.add_argument("--log-every", type=int, default=5)
    diagnostic_refit.add_argument("--output", required=True)
    diagnostic_refit.set_defaults(handler=refit_isolation_diagnostic)

    evaluate = children.add_parser("evaluate", parents=[common])
    evaluate.add_argument(
        "--baseline-freeze",
        default="fhm_dapp_tim/evidence/v44/baseline_selection_v4_4.json",
    )
    evaluate.add_argument("--checkpoints", nargs="+", required=True)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=evaluate_confirmation)

    diagnostic_evaluate = children.add_parser(
        "diagnostic-evaluate", parents=[common]
    )
    diagnostic_evaluate.add_argument(
        "--baseline-freeze",
        default="fhm_dapp_tim/evidence/v44/baseline_selection_v4_4.json",
    )
    diagnostic_evaluate.add_argument(
        "--primary-confirmation",
        default="fhm_dapp_tim/evidence/v44/confirmation_v4_4.pt",
    )
    diagnostic_evaluate.add_argument(
        "--baseline-confirmation",
        default="fhm_dapp_tim/evidence/v44/baseline_confirmation_v4_4.json",
    )
    diagnostic_evaluate.add_argument(
        "--confirmation-summary",
        default="fhm_dapp_tim/evidence/v44/confirmation_summary_v4_4.json",
    )
    diagnostic_evaluate.add_argument("--checkpoints", nargs="+", required=True)
    diagnostic_evaluate.add_argument("--batch-size", type=int, default=16)
    diagnostic_evaluate.add_argument("--output", required=True)
    diagnostic_evaluate.set_defaults(handler=evaluate_isolation_diagnostics)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
