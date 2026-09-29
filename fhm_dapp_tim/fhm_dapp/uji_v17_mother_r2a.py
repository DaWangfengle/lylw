"""Train V17-R2A members 6--10 with the exact registered R0 algorithm."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0_execution
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _save_torch_new,
    _sha256_file,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    load_registry as load_r0_registry,
    registered_role_keys,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2a_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    DEFAULT_REGISTRY,
    PROTOCOL_VERSION,
    checkpoint_path,
    load_registry,
)


def _validate_checkpoint_output(registry: dict, index: int, path: Path) -> None:
    allowed = {
        Path(value).resolve()
        for value in registry["allowed_outputs"]["additional_checkpoints"]
    }
    if path.resolve() not in allowed or path.resolve() != checkpoint_path(index).resolve():
        raise ValueError("R2A checkpoint path is outside the registry")
    if path.exists():
        raise FileExistsError(path)


def train_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    index = int(member_index)
    if index not in ADDITIONAL_MEMBER_INDICES:
        raise ValueError("R2A trains only members 6 through 10")
    destination = checkpoint_path(index)
    _validate_checkpoint_output(registry, index, destination)
    r0_registry = load_r0_registry(registry["bindings"]["r0_registry"]["path"])
    data = load_uji_v16(registry["data_contract"]["train_data"])
    fit_rows = r0_execution._registered_rows(r0_registry, "member_fit", data)
    stop_rows = r0_execution._registered_rows(r0_registry, "early_stop", data)
    refit_rows = r0_execution._registered_rows(r0_registry, "selection_refit", data)
    seed = ADDITIONAL_MEMBER_SEEDS[index - 6]
    device = torch.device(device_name)
    started = time.perf_counter()
    selected_epoch, schedule = r0_execution._fit_schedule(
        data,
        fit_rows,
        stop_rows,
        seed=seed,
        device=device,
        log_every=log_every,
    )
    model, stats, origin, device_mapping, refit_history = r0_execution._fit_refit(
        data,
        refit_rows,
        seed=seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update(
        {
            "training_protocol_version": PROTOCOL_VERSION,
            "model_schema": "v17_r2a_exchangeable_r0_member",
            "n_devices": len(device_mapping),
            "train_all": True,
            "split_strategy": "registered_selection_refit_v17",
            "validation_fraction": 0.0,
            "calibration_fraction": 0.0,
            "epochs": int(selected_epoch),
            "patience": r0_execution.PATIENCE,
            "lr": FROZEN_TRAINING["learning_rate"],
            "target_neighbors": FROZEN_TRAINING["target_neighbors"],
            "early_stop_aggregation": "coordinate_macro_mae",
        }
    )
    role_keys = tuple(sorted(registered_role_keys(r0_registry, "selection_refit")))
    if len(role_keys) != model.n_anchors:
        raise RuntimeError("R2A refit anchors do not equal registered coordinates")
    elapsed = float(time.perf_counter() - started)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": "r2a_selection_refit_panel_extension",
        "member_index": index,
        "member_seed": seed,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": np.asarray(origin, dtype=np.float32),
        "conformal_quantiles": {},
        "epoch": int(selected_epoch),
        "val_mae": float(schedule[selected_epoch - 1]["early_stop"]["row_mae_m"]),
        "seed": seed,
        "development_metrics": None,
        "confirmation_metrics": None,
        "split_contract": {
            "member_fit": registry["data_contract"]["roles"]["member_fit"],
            "early_stop": registry["data_contract"]["roles"]["early_stop"],
            "selection_refit": registry["data_contract"]["roles"]["selection_refit"],
            "development_localization_predictions": 0,
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
        "exchangeability_contract": {
            "same_algorithm_as_v17_r0": True,
            "only_changed_training_factor": "random_seed",
            "base_panel_members_selected_by_outcome": False,
            "new_member_selected_by_outcome": False,
        },
        "device_id_mapping": {
            str(raw): mapped for raw, mapped in device_mapping.items()
        },
        "anchor_coordinate_keys": role_keys,
        "schedule_history": schedule,
        "refit_history": refit_history,
        "training_seconds": elapsed,
    }
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "member_index": index,
        "seed": seed,
        "selected_epoch": int(selected_epoch),
        "anchors": int(model.n_anchors),
        "training_seconds": elapsed,
        "development_localization_predictions": 0,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def check_artifacts(registry_path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = load_registry(registry_path)
    records = []
    for index, value in zip(
        ADDITIONAL_MEMBER_INDICES,
        registry["allowed_outputs"]["additional_checkpoints"],
    ):
        path = Path(value)
        record = {
            "member_index": index,
            "path": str(path.resolve()),
            "exists": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
        if path.exists():
            payload = torch.load(
                path, map_location="cpu", weights_only=True
            )
            record.update(
                {
                    "protocol_version": payload.get("protocol_version"),
                    "registry_sha256": payload.get("registry_sha256"),
                    "seed": payload.get("member_seed"),
                    "selected_epoch": payload.get("epoch"),
                    "anchors": len(payload["anchor_coordinate_keys"]),
                }
            )
        records.append(record)
    return {
        "registry_sha256": registry["registry_sha256"],
        "additional_members": records,
        "development_localization_predictions": 0,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-member")
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)
    check = sub.add_parser("check-artifacts")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "train-member":
        result = train_member(
            member_index=args.member_index,
            registry_path=args.registry,
            device_name=args.device,
            log_every=args.log_every,
        )
    else:
        result = check_artifacts(args.registry)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
