"""Frozen reproduction runner for the historical V9 and V17 UJI systems.

Scope is deliberately narrow: reproduce V9 and V17, verify a sub-6 m
coordinate-macro MAE gate, then stop.  This module does not execute any V18 or
V19 model and does not design a successor model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v9_experiment as v9
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_protocol as v17_protocol
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as v17_r0
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as v17_r2b
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as v17_r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as v17_r3
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as v17_r3_protocol
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _rows_for_keys,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _sha256_keys,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)


PROTOCOL_VERSION = "historical_v9_v17_reproduction_v1_20261007"
SUB_SIX_GATE_M = 6.0

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
EVIDENCE = ROOT / "evidence" / "historical_reproduction"
MODELS = ROOT / "models" / "historical_reproduction"
DESIGN = EVIDENCE / "DESIGN_HISTORICAL_REPRODUCTION.md"
V9_DESIGN = EVIDENCE / "DESIGN_V9_REPRODUCTION.md"
REGISTRY = EVIDENCE / "registry_historical_reproduction.json"
SUMMARY = EVIDENCE / "REPRODUCTION_SUMMARY.json"

V9_EVIDENCE = EVIDENCE / "v9"
V9_SELECTION_MODELS = MODELS / "v9" / "selection"
V9_FINAL_MODELS = MODELS / "v9" / "final"
V9_REGISTRY = V9_EVIDENCE / "registry_v9_uji.json"
V9_FREEZE = V9_EVIDENCE / "selection_freeze_v9_uji.json"
V9_RESULT = V9_EVIDENCE / "outer_test_result_v9_uji.json"
V9_SUMMARY = V9_EVIDENCE / "REPRODUCTION_V9.json"

V17_EVIDENCE = EVIDENCE / "v17"
V17_MODELS = MODELS / "v17"
V17_REGISTRY = V17_EVIDENCE / "registry_v17_reproduction.json"
V17_OUTER_REFIT_CSV = V17_EVIDENCE / "outer_refit_v17_reproduction.csv"
V17_RESULT = V17_EVIDENCE / "outer_result_v17_reproduction.json"
V17_MEMBER_SEEDS = tuple(range(17301, 17326))
V17_PANEL_MEMBERS = 25


class ReproductionError(RuntimeError):
    pass


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_once(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_new(path, dict(payload))


def _gate(mae: float) -> bool:
    return math.isfinite(float(mae)) and 0.0 < float(mae) < SUB_SIX_GATE_M


def _source_manifest() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "v9": Path(v9.__file__).resolve(),
        "v17_protocol": Path(v17_protocol.__file__).resolve(),
        "v17_r0": Path(v17_r0.__file__).resolve(),
        "v17_r2b": Path(v17_r2b.__file__).resolve(),
        "v17_r2f": Path(v17_r2f.__file__).resolve(),
        "v17_r3": Path(v17_r3.__file__).resolve(),
        "v17_r3_protocol": Path(v17_r3_protocol.__file__).resolve(),
    }


def _check_constants() -> None:
    if tuple(v9.MEMBER_SEEDS) != tuple(range(9311, 9323)):
        raise ReproductionError("V9 seed panel changed")
    if v9.INITIAL_MEMBER_COUNT != 8:
        raise ReproductionError("V9 initial member count changed")
    if tuple(v17_r3_protocol.MEMBER_SEEDS) != V17_MEMBER_SEEDS:
        raise ReproductionError("V17 seed panel changed")
    if v17_r3_protocol.PANEL_MEMBER_COUNT != 25:
        raise ReproductionError("V17 panel count changed")
    if v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.SELECTED_NAME] != {
        "E": 16,
        "L": 192,
        "K": 192,
        "alpha": 1.0,
    }:
        raise ReproductionError("V17 selected R3 law changed")
    if v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.HISTORICAL_NAME] != {
        "E": 5,
        "L": 24,
        "K": 24,
        "alpha": 0.25,
    }:
        raise ReproductionError("V17 historical R3 law changed")


def preflight(device_name: str = "cuda") -> dict:
    _check_constants()
    required = [TRAIN_DATA, DESIGN, V9_DESIGN, *_source_manifest().values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing reproduction input: {missing}")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise ReproductionError("CUDA requested but unavailable")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "ready",
        "training_data": str(TRAIN_DATA.resolve()),
        "training_data_sha256": _sha256_file(TRAIN_DATA),
        "source_hashes": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in _source_manifest().items()
        },
        "v9_member_seeds": list(v9.MEMBER_SEEDS),
        "v17_member_seeds": list(V17_MEMBER_SEEDS),
        "sub_six_gate_m": SUB_SIX_GATE_M,
        "official_validation_predictions_authorized": False,
        "external_predictions_authorized": False,
        "successor_model_design_authorized": False,
    }


def prepare(device_name: str = "cuda") -> dict:
    if REGISTRY.exists():
        payload = _read_json(REGISTRY)
        if payload.get("registry_sha256") != _canonical_json_sha256(
            payload, omit="registry_sha256"
        ):
            raise ReproductionError("reproduction registry self-hash failed")
        if payload.get("training_data_sha256") != _sha256_file(TRAIN_DATA):
            raise ReproductionError("training data changed after registration")
        return payload
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    payload = preflight(device_name)
    payload["status"] = "registered_before_reproduction_outcomes"
    payload["v9_contract"] = {
        "execution": "original V9 nested selection/freeze/outer implementation",
        "path_isolation_only": True,
        "initial_E": 8,
        "maximum_E": 12,
    }
    payload["v17_contract"] = {
        "execution": "frozen V17 R3 replay with original R0 training and R3 scoring",
        "selected": copy.deepcopy(
            v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.SELECTED_NAME]
        ),
        "historical": copy.deepcopy(
            v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.HISTORICAL_NAME]
        ),
        "configuration_reselection": False,
    }
    payload["registry_sha256"] = _canonical_json_sha256(payload)
    _write_json_once(REGISTRY, payload)
    return payload


@contextmanager
def _v9_workspace():
    replacements = {
        "DEFAULT_EVIDENCE": V9_EVIDENCE,
        "DEFAULT_DESIGN": V9_DESIGN,
        "DEFAULT_REGISTRY": V9_REGISTRY,
        "DEFAULT_MODELS": V9_SELECTION_MODELS,
        "DEFAULT_FINAL_MODELS": V9_FINAL_MODELS,
        "DEFAULT_FREEZE": V9_FREEZE,
        "DEFAULT_RESULT": V9_RESULT,
    }
    original = {name: getattr(v9, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v9, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(v9, name, value)


def _v9_radio(candidate: str, phase: str) -> Path:
    return V9_EVIDENCE / f"radio_memory_{phase}_{candidate}.pt"


def _v9_cache(candidate: str, members: int) -> Path:
    return V9_EVIDENCE / f"development_cache_{candidate}_E{members:02d}.pt"


def _v9_grid(candidate: str, members: int) -> Path:
    return V9_EVIDENCE / f"development_grid_{candidate}_E{members:02d}.pt"


def _v9_grid_json(candidate: str, members: int) -> Path:
    return V9_EVIDENCE / f"development_grid_{candidate}_E{members:02d}.json"


def _v9_extension(members: int) -> Path:
    return V9_FREEZE.with_name(
        f"selection_extension_required_through_E{members}_v9_uji.json"
    )


def _ensure_v9_registry() -> dict:
    if V9_REGISTRY.exists():
        return v9.load_registry(V9_REGISTRY)
    V9_EVIDENCE.mkdir(parents=True, exist_ok=True)
    return v9.register_protocol(
        train_data=TRAIN_DATA,
        design_path=V9_DESIGN,
        output=V9_REGISTRY,
    )


def _ensure_v9_member(candidate: str, index: int, device_name: str) -> None:
    path = v9.selection_checkpoint_path(candidate, index)
    if path.exists():
        model, payload = v9.load_front_checkpoint(path, torch.device("cpu"))
        del model
        if payload["member_index"] != index or payload["candidate"] != candidate:
            raise ReproductionError("existing V9 selection member identity changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    v9.train_selection_member(
        candidate=candidate,
        member_index=index,
        registry_path=V9_REGISTRY,
        output=path,
        device_name=device_name,
    )


def _ensure_v9_radio(candidate: str, phase: str) -> Path:
    path = _v9_radio(candidate, phase)
    if not path.exists():
        v9.fit_radio_memory(
            candidate=candidate,
            phase=phase,
            registry_path=V9_REGISTRY,
            output=path,
        )
    return path


def _ensure_v9_grid(candidate: str, members: int, device_name: str) -> Path:
    cache = _v9_cache(candidate, members)
    grid = _v9_grid(candidate, members)
    summary = _v9_grid_json(candidate, members)
    radio = _ensure_v9_radio(candidate, "selection")
    if not cache.exists():
        v9.build_cache(
            candidate=candidate,
            role="development",
            registry_path=V9_REGISTRY,
            member_count=members,
            radio_path=radio,
            output=cache,
            device_name=device_name,
        )
    if grid.exists() != summary.exists():
        raise ReproductionError("partial V9 grid output detected")
    if not grid.exists():
        base = None if members == 8 else _v9_grid(candidate, members - 2)
        if base is not None and not base.exists():
            raise FileNotFoundError(base)
        v9.evaluate_development_grid(
            candidate=candidate,
            registry_path=V9_REGISTRY,
            cache_file=cache,
            base_grid_file=base,
            output=grid,
            summary_output=summary,
            device_name=device_name,
        )
    return grid


def run_v9(device_name: str = "cuda") -> dict:
    prepare(device_name)
    with _v9_workspace():
        _ensure_v9_registry()
        members = 8
        while True:
            grids = {}
            for candidate in v9.CANDIDATES:
                for index in range(1, members + 1):
                    _ensure_v9_member(candidate, index, device_name)
                grids[candidate] = _ensure_v9_grid(candidate, members, device_name)
            registry = v9.load_registry(V9_REGISTRY)
            if V9_FREEZE.exists():
                freeze = v9.load_selection_freeze(V9_FREEZE, registry=registry)
                break
            extension = _v9_extension(members)
            if extension.exists():
                decision = _read_json(extension)
            else:
                decision = v9.select_and_freeze(
                    registry_path=V9_REGISTRY,
                    geometry_grid_path=grids["geometry_invariant"],
                    historical_grid_path=grids["historical_affine"],
                    output=V9_FREEZE,
                )
            if decision.get("status") == "selection_frozen":
                freeze = decision
                break
            if decision.get("status") != "extension_required":
                raise ReproductionError(f"unexpected V9 selection status: {decision}")
            requested = [int(x) for x in decision.get("next_member_indices", ())]
            if not requested:
                raise ReproductionError("V9 extension has no member indices")
            next_members = max(requested)
            if next_members != members + 2 or next_members > 12:
                raise ReproductionError("V9 extension left the historical budget")
            members = next_members

        selected = freeze["selected"]
        candidate = selected["candidate"]
        E = int(selected["E"])
        for index in range(1, E + 1):
            path = v9.final_checkpoint_path(candidate, index)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                v9.train_final_member(
                    member_index=index,
                    registry_path=V9_REGISTRY,
                    freeze_path=V9_FREEZE,
                    output=path,
                    device_name=device_name,
                )
        radio = _ensure_v9_radio(candidate, "final")
        outer_cache = V9_EVIDENCE / f"outer_test_cache_{candidate}.pt"
        if not outer_cache.exists():
            v9.build_cache(
                candidate=candidate,
                role="outer_test",
                registry_path=V9_REGISTRY,
                radio_path=radio,
                output=outer_cache,
                device_name=device_name,
            )
        if not V9_RESULT.exists():
            result = v9.evaluate_outer_test(
                registry_path=V9_REGISTRY,
                freeze_path=V9_FREEZE,
                cache_file=outer_cache,
                output=V9_RESULT,
            )
        else:
            result = _read_json(V9_RESULT)
        mae = float(result["metrics"]["coordinate_macro_mae_2d_m"])
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "historical_system": "V9",
            "status": "PASS" if _gate(mae) else "FAIL",
            "selected": copy.deepcopy(selected),
            "outer_coordinate_macro_mae_2d_m": mae,
            "sub_six_gate_passed": _gate(mae),
            "result_sha256": _sha256_file(V9_RESULT),
            "official_validation_predictions": 0,
            "external_predictions": 0,
        }
        if not V9_SUMMARY.exists():
            _write_json_once(V9_SUMMARY, summary)
        elif _read_json(V9_SUMMARY) != summary:
            raise ReproductionError("existing V9 summary differs")
        return summary


def _v17_member_path(index: int) -> Path:
    index = int(index)
    if not 1 <= index <= 25:
        raise ValueError("V17 member index must be in 1..25")
    return V17_MODELS / f"member_{index:02d}_seed{V17_MEMBER_SEEDS[index - 1]}.pt"


def _v17_roles(data: dict) -> tuple[dict, dict, dict]:
    roles, details = v17_protocol.construct_coordinate_roles(set(data["keys"]))
    rows = {
        name: _rows_for_keys(data["keys"], keys)
        for name, keys in roles.items()
    }
    return roles, rows, details


def prepare_v17() -> dict:
    prepare("cuda" if torch.cuda.is_available() else "cpu")
    if V17_REGISTRY.exists():
        payload = _read_json(V17_REGISTRY)
        if payload.get("registry_sha256") != _canonical_json_sha256(
            payload, omit="registry_sha256"
        ):
            raise ReproductionError("V17 replay registry self-hash failed")
        return payload
    data = load_uji_v16(TRAIN_DATA)
    roles, rows, details = _v17_roles(data)
    if len(roles["outer_refit"]) != v17_r3_protocol.OUTER_REFIT_ANCHORS:
        raise ReproductionError("V17 outer-refit coordinate count changed")
    if len(rows["outer_test"]) != v17_r3_protocol.OUTER_TEST_ROWS:
        raise ReproductionError("V17 outer-test row count changed")
    if len(roles["outer_test"]) != v17_r3_protocol.OUTER_TEST_COORDINATES:
        raise ReproductionError("V17 outer-test coordinate count changed")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_frozen_replay_registered",
        "training_data": str(TRAIN_DATA.resolve()),
        "training_data_sha256": _sha256_file(TRAIN_DATA),
        "member_seeds": list(V17_MEMBER_SEEDS),
        "roles": {
            name: {
                "rows": int(len(rows[name])),
                "coordinates": int(len(roles[name])),
                "coordinate_key_sha256": _sha256_keys(roles[name]),
                "row_index_sha256": _sha256_indices(rows[name]),
            }
            for name in roles
        },
        "split_details": details,
        "selected": copy.deepcopy(
            v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.SELECTED_NAME]
        ),
        "historical": copy.deepcopy(
            v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.HISTORICAL_NAME]
        ),
        "configuration_reselection": False,
        "official_validation_predictions_authorized": False,
        "external_predictions_authorized": False,
    }
    payload["registry_sha256"] = _canonical_json_sha256(payload)
    V17_EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_json_once(V17_REGISTRY, payload)
    if not V17_OUTER_REFIT_CSV.exists():
        v17_r0._materialize_csv(
            TRAIN_DATA, V17_OUTER_REFIT_CSV, rows["outer_refit"]
        )
    derived = load_uji_v16(V17_OUTER_REFIT_CSV)
    if set(derived["keys"]) != roles["outer_refit"]:
        raise ReproductionError("V17 outer-refit materialization changed coordinates")
    return payload


def _validated_v17_rows(registry: Mapping[str, object], data: dict):
    roles, rows, _ = _v17_roles(data)
    for name, record in registry["roles"].items():
        if _sha256_keys(roles[name]) != record["coordinate_key_sha256"]:
            raise ReproductionError(f"V17 coordinate role changed: {name}")
        if _sha256_indices(rows[name]) != record["row_index_sha256"]:
            raise ReproductionError(f"V17 row role changed: {name}")
    return roles, rows


def train_v17_member(member_index: int, device_name: str = "cuda") -> dict:
    registry = prepare_v17()
    index = int(member_index)
    path = _v17_member_path(index)
    seed = V17_MEMBER_SEEDS[index - 1]
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checks = (
            payload.get("member_index") == index
            and payload.get("member_seed") == seed
            and payload.get("registry_sha256") == registry["registry_sha256"]
            and payload.get("phase") == "r3_outer_refit"
        )
        if not checks:
            raise ReproductionError(f"existing V17 member {index} is incompatible")
        return {
            "member_index": index,
            "seed": seed,
            "selected_epoch": int(payload["epoch"]),
            "checkpoint_sha256": _sha256_file(path),
            "resumed": True,
        }
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        raise ReproductionError("CUDA requested but unavailable")
    data = load_uji_v16(TRAIN_DATA)
    roles, rows = _validated_v17_rows(registry, data)
    device = torch.device(device_name)
    started = time.perf_counter()
    epoch, schedule = v17_r0._fit_schedule(
        data,
        rows["member_fit"],
        rows["early_stop"],
        seed=seed,
        device=device,
        log_every=5,
    )
    model, stats, origin, device_mapping, history = v17_r0._fit_refit(
        data,
        rows["outer_refit"],
        seed=seed,
        epochs=epoch,
        device=device,
        log_every=5,
    )
    if model.n_anchors != v17_r3_protocol.OUTER_REFIT_ANCHORS:
        raise ReproductionError("V17 replay member anchor count changed")
    config = copy.deepcopy(FROZEN_TRAINING)
    config.update(
        {
            "training_protocol_version": v17_r3_protocol.PROTOCOL_VERSION,
            "model_schema": "v17_r3_outer_refit_historical_acrhap_member",
            "n_devices": len(device_mapping),
            "train_all": True,
            "split_strategy": "registered_outer_refit_v17_r3",
            "validation_fraction": 0.0,
            "calibration_fraction": 0.0,
            "epochs": int(epoch),
            "patience": v17_r0.PATIENCE,
            "lr": FROZEN_TRAINING["learning_rate"],
            "target_neighbors": FROZEN_TRAINING["target_neighbors"],
            "early_stop_aggregation": "coordinate_macro_mae",
            "epoch_selection_source": "frozen_inner_member_fit_to_early_stop",
        }
    )
    payload = {
        "protocol_version": v17_r3_protocol.PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": "r3_outer_refit",
        "member_index": index,
        "member_seed": seed,
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": np.asarray(origin, dtype=np.float32),
        "conformal_quantiles": {},
        "epoch": int(epoch),
        "val_mae": float(schedule[epoch - 1]["early_stop"]["row_mae_m"]),
        "seed": seed,
        "development_metrics": None,
        "confirmation_metrics": None,
        "device_id_mapping": {
            str(raw): mapped for raw, mapped in device_mapping.items()
        },
        "anchor_coordinate_keys": tuple(sorted(roles["outer_refit"])),
        "schedule_history": schedule,
        "refit_history": history,
        "outer_refit_csv_sha256": _sha256_file(V17_OUTER_REFIT_CSV),
        "training_seconds": float(time.perf_counter() - started),
        "split_contract": {
            "outer_test_features_consumed": False,
            "outer_test_labels_consumed": False,
            "official_features_consumed": False,
            "official_labels_consumed": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_torch_new(path, payload)
    return {
        "member_index": index,
        "seed": seed,
        "selected_epoch": int(epoch),
        "early_stop_row_mae_m": payload["val_mae"],
        "checkpoint_sha256": _sha256_file(path),
        "training_seconds": payload["training_seconds"],
        "resumed": False,
    }


def check_v17_members() -> dict:
    registry = prepare_v17()
    records = []
    for index in range(1, 26):
        path = _v17_member_path(index)
        if not path.exists():
            records.append({"member_index": index, "exists": False})
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        seed = V17_MEMBER_SEEDS[index - 1]
        checks = {
            "identity": payload.get("member_index") == index,
            "seed": payload.get("member_seed") == seed,
            "registry": payload.get("registry_sha256") == registry["registry_sha256"],
            "phase": payload.get("phase") == "r3_outer_refit",
            "anchor_count": len(payload.get("anchor_coordinate_keys", ()))
            == v17_r3_protocol.OUTER_REFIT_ANCHORS,
            "outer_unconsumed": payload.get("split_contract", {}).get(
                "outer_test_labels_consumed"
            )
            is False,
            "official_unconsumed": payload.get("split_contract", {}).get(
                "official_labels_consumed"
            )
            is False,
        }
        if not all(checks.values()):
            raise ReproductionError(f"V17 member {index} failed checks: {checks}")
        records.append(
            {
                "member_index": index,
                "exists": True,
                "seed": seed,
                "selected_epoch": int(payload["epoch"]),
                "checkpoint_sha256": _sha256_file(path),
            }
        )
    return {
        "complete_members": sum(record["exists"] for record in records),
        "all_complete": all(record["exists"] for record in records),
        "members": records,
    }


@torch.no_grad()
def evaluate_v17(device_name: str = "cuda") -> dict:
    registry = prepare_v17()
    audit = check_v17_members()
    if not audit["all_complete"]:
        raise ReproductionError("all 25 V17 members are required")
    if V17_RESULT.exists():
        result = _read_json(V17_RESULT)
        if result.get("registry_sha256") != registry["registry_sha256"]:
            raise ReproductionError("existing V17 result registry mismatch")
        return result

    data = load_uji_v16(TRAIN_DATA)
    _, rows = _validated_v17_rows(registry, data)
    paths = [_v17_member_path(i) for i in range(1, 26)]
    with _installed_adapters([_role_record(V17_OUTER_REFIT_CSV)]):
        ensemble = ProjectedEnsemble(
            checkpoints=paths,
            anchor_data=V17_OUTER_REFIT_CSV,
            device=device_name,
            dataset="UJI",
        )
    v17_r3._validate_loaded_panel(ensemble)
    device = torch.device(device_name)
    models = [model for model, _payload in ensemble.checkpoints]
    for model in models:
        model.eval()

    outer_rows = rows["outer_test"]
    rss = data["rss"][outer_rows].float().cpu()
    origin = np.asarray(ensemble.reference["coord_origin"], dtype=np.float64)
    target = torch.from_numpy(
        data["raw_xy"][outer_rows] - origin[None, :]
    ).double()
    group_ids, group_keys = v17_r2b._coordinate_groups(
        [data["keys"][int(index)] for index in outer_rows]
    )
    if len(rss) != v17_r3_protocol.OUTER_TEST_ROWS:
        raise ReproductionError("V17 outer row count changed at evaluation")
    if len(group_keys) != v17_r3_protocol.OUTER_TEST_COORDINATES:
        raise ReproductionError("V17 outer coordinate count changed at evaluation")

    randomness = v17_r3.common_student_t_randomness(
        len(rss),
        draws=v17_r3_protocol.PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=v17_r3_protocol.STUDENT_DF,
        seed=v17_r3_protocol.PREDICTIVE_SEED,
    )
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in randomness.items()
    }
    target_device = target.to(device)
    configurations = v17_r3_protocol.CONFIGURATIONS
    subsets = {
        name: v17_r3_protocol.balanced_member_subsets(config["E"])
        for name, config in configurations.items()
    }
    subset_energy = {
        name: torch.zeros((len(value), len(rss)), dtype=torch.float64)
        for name, value in subsets.items()
    }
    subset_mae = {
        name: torch.zeros((len(value), len(rss)), dtype=torch.float64)
        for name, value in subsets.items()
    }
    retained = {name: [] for name in configurations}

    with v17_r3.r2f_a1._deterministic_algorithms():
        for start in range(0, len(rss), v17_r3_protocol.COMPONENT_BATCH_SIZE):
            stop = min(
                start + v17_r3_protocol.COMPONENT_BATCH_SIZE, len(rss)
            )
            batch = rss[start:stop].to(device)
            prototype_distance = v17_r2b._prototype_distance(models[0], batch)
            neural_members = []
            location_members = []
            scale_members = []
            indices = torch.arange(
                v17_r3_protocol.OUTER_REFIT_ANCHORS,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(len(batch), -1)
            for model in models:
                embedding = model.encoder(batch)
                neural_members.append(
                    model.retrieval_logits_from_embedding(embedding)
                )
                location, scale = model._components(embedding, indices)
                location_members.append(location)
                scale_members.append(scale)
            neural = torch.stack(neural_members)
            location = torch.stack(location_members)
            scale = torch.stack(scale_members)
            probability = v17_r2f._atomic_probability_panel(
                neural,
                prototype_distance,
                v17_r3_protocol.BETA_NEURAL,
                v17_r3_protocol.BETA_PROTOTYPE,
            )
            batch_randomness = {
                key: (
                    value[start:stop]
                    if isinstance(value, torch.Tensor) and value.ndim > 0
                    else value
                )
                for key, value in randomness.items()
            }
            for name, config in configurations.items():
                member_prior, member_indices, member_radial, mass = (
                    v17_r3._member_top_l(
                        probability,
                        location,
                        scale,
                        config["L"],
                        student_df=v17_r3_protocol.STUDENT_DF,
                    )
                )
                retained[name].append(mass.detach().cpu())
                for subset_index, subset in enumerate(subsets[name]):
                    law = v17_r3._dense_joint_law(
                        member_prior,
                        member_indices,
                        member_radial,
                        location,
                        scale,
                        subset,
                        top_k=config["K"],
                        alpha=config["alpha"],
                    )
                    energy, mae = v17_r3._score_joint_law(
                        *law,
                        target_device[start:stop],
                        batch_randomness,
                    )
                    subset_energy[name][subset_index, start:stop] = energy
                    subset_mae[name][subset_index, start:stop] = mae
            print(
                json.dumps(
                    {
                        "v17_outer_rows_completed": stop,
                        "outer_rows_total": len(rss),
                    }
                ),
                flush=True,
            )

    metrics = {}
    vectors = {}
    realization_metrics = {}
    retained_mass = {}
    for name in configurations:
        mean_energy = subset_energy[name].mean(dim=0)
        mean_mae = subset_mae[name].mean(dim=0)
        energy_record, energy_group = v17_r3._metric_record(
            mean_energy, group_ids, len(group_keys)
        )
        mae_record, mae_group = v17_r3._metric_record(
            mean_mae, group_ids, len(group_keys)
        )
        metrics[name] = {
            "energy_score_2d_m": energy_record,
            "exact_mixture_mean_mae_m": mae_record,
        }
        vectors[name] = {
            "energy_score_2d_m": energy_group,
            "exact_mixture_mean_mae_m": mae_group,
        }
        realization_metrics[name] = v17_r3._subset_run_records(
            subset_energy[name],
            subset_mae[name],
            subsets[name],
            group_ids,
            len(group_keys),
        )
        mass = torch.cat(retained[name]).double()
        retained_mass[name] = {
            "mean": float(mass.mean()),
            "q10": float(torch.quantile(mass, 0.10)),
            "median": float(torch.quantile(mass, 0.50)),
            "q90": float(torch.quantile(mass, 0.90)),
        }

    selected_name = v17_r3_protocol.SELECTED_NAME
    historical_name = v17_r3_protocol.HISTORICAL_NAME
    differences = {
        endpoint: vectors[selected_name][endpoint]
        - vectors[historical_name][endpoint]
        for endpoint in (
            "energy_score_2d_m",
            "exact_mixture_mean_mae_m",
        )
    }
    inference = v17_r3.studentized_max_t_family(
        differences,
        repetitions=v17_r3_protocol.BOOTSTRAP_REPETITIONS,
        seed=v17_r3_protocol.BOOTSTRAP_SEED,
        confidence=v17_r3_protocol.CONFIDENCE,
        orientation="selected_E16_minus_historical_E5",
    )
    selected_mae = float(
        metrics[selected_name]["exact_mixture_mean_mae_m"][
            "coordinate_macro_m"
        ]
    )
    historical_mae = float(
        metrics[historical_name]["exact_mixture_mean_mae_m"][
            "coordinate_macro_m"
        ]
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "status": "PASS" if _gate(selected_mae) else "FAIL",
        "selected_name": selected_name,
        "historical_name": historical_name,
        "selected_configuration": copy.deepcopy(configurations[selected_name]),
        "historical_configuration": copy.deepcopy(
            configurations[historical_name]
        ),
        "metrics": metrics,
        "coordinate_group_vectors": {
            name: {
                endpoint: value.tolist()
                for endpoint, value in record.items()
            }
            for name, record in vectors.items()
        },
        "realization_metrics": realization_metrics,
        "paired_familywise_inference": inference,
        "retained_mass_diagnostic": retained_mass,
        "selected_coordinate_macro_mae_m": selected_mae,
        "historical_coordinate_macro_mae_m": historical_mae,
        "sub_six_gate_passed": _gate(selected_mae),
        "member_epoch_records": audit["members"],
        "outer_test_rows": int(len(rss)),
        "outer_test_coordinate_groups": int(len(group_keys)),
        "official_validation_predictions": 0,
        "external_predictions": 0,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_once(V17_RESULT, result)
    return result


def run_v17(device_name: str = "cuda") -> dict:
    prepare_v17()
    for index in range(1, 26):
        train_v17_member(index, device_name)
    return evaluate_v17(device_name)


def summarize() -> dict:
    if not V9_SUMMARY.exists() or not V17_RESULT.exists():
        raise FileNotFoundError(
            "V9 and V17 reproduction results are both required"
        )
    v9_summary = _read_json(V9_SUMMARY)
    v17_result = _read_json(V17_RESULT)
    v9_mae = float(v9_summary["outer_coordinate_macro_mae_2d_m"])
    v17_mae = float(v17_result["selected_coordinate_macro_mae_m"])
    passed = _gate(v9_mae) and _gate(v17_mae)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if passed else "FAIL",
        "v9_coordinate_macro_mae_m": v9_mae,
        "v17_coordinate_macro_mae_m": v17_mae,
        "sub_six_gate_m": SUB_SIX_GATE_M,
        "v9_passed": _gate(v9_mae),
        "v17_passed": _gate(v17_mae),
        "both_historical_baselines_reproduced": passed,
        "next_stage_authorized": passed,
        "next_stage": (
            "design final model from V9/V17 strong skeleton; re-test later "
            "mechanisms one at a time by strict ablation"
            if passed
            else "stop and diagnose reproduction mismatch"
        ),
        "successor_model_implemented_in_this_stage": False,
        "official_validation_predictions": 0,
        "external_predictions": 0,
    }
    payload["summary_sha256"] = _canonical_json_sha256(payload)
    if not SUMMARY.exists():
        _write_json_once(SUMMARY, payload)
    elif _read_json(SUMMARY) != payload:
        raise ReproductionError("existing summary differs")
    return payload


def _parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("prepare")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("run-v9")
    p.add_argument("--device", default="cuda")

    sub.add_parser("prepare-v17")
    p = sub.add_parser("train-v17-member")
    p.add_argument("--member-index", type=int, required=True)
    p.add_argument("--device", default="cuda")
    sub.add_parser("check-v17-members")
    p = sub.add_parser("evaluate-v17")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("run-v17")
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("run-all")
    p.add_argument("--device", default="cuda")
    sub.add_parser("summarize")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.device)
    elif args.command == "prepare":
        result = prepare(args.device)
    elif args.command == "run-v9":
        result = run_v9(args.device)
    elif args.command == "prepare-v17":
        result = prepare_v17()
    elif args.command == "train-v17-member":
        result = train_v17_member(args.member_index, args.device)
    elif args.command == "check-v17-members":
        result = check_v17_members()
    elif args.command == "evaluate-v17":
        result = evaluate_v17(args.device)
    elif args.command == "run-v17":
        result = run_v17(args.device)
    elif args.command == "run-all":
        result = {
            "v9": run_v9(args.device),
            "v17": run_v17(args.device),
            "summary": summarize(),
        }
    elif args.command == "summarize":
        result = summarize()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
