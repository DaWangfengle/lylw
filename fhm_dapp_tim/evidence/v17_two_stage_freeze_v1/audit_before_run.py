"""Read-only environment, training-input, and role audit before formal jobs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pytest
import scipy
import torch
from unittest.mock import patch

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp import msep_v17_two_stage_freeze_v1 as study
from fhm_dapp_tim.fhm_dapp.msep_single_study import _load, _prepare_fit, _seed
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import anchor_site_keys, split_physical_sites
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import same_floor_neighbors, neighbor_contamination
from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING

OUT = REPO / "fhm_dapp_tim/evidence/v17_two_stage_freeze_v1/BEFORE_RUN_AUDIT.json"
assert not OUT.exists(), OUT
torch.set_num_threads(4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


sources = sorted(set(
    list((REPO / "fhm_dapp_tim/fhm_dapp").rglob("*.py"))
    + list((REPO / "tests").rglob("*.py"))
    + [REPO / "V17_TWO_STAGE_FREEZE_CODEX_RUN.txt"]
))
inputs = [REPO / "data/TUJI1" / value for value in ds.TUJI_DEVICE_FILES.values()]
inputs += [REPO / "data/UTSIndoorLoc/UTS_training.csv", REPO / "data/UJIndoorLoc/trainingData.csv"]
assert len(inputs) == 7 and all(p.is_file() for p in inputs)
payload = {
    "python": sys.version, "python_executable": sys.executable,
    "platform": platform.platform(), "torch": torch.__version__,
    "numpy": np.__version__, "scipy": scipy.__version__,
    "pandas": pd.__version__, "pytest": pytest.__version__,
    "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    "cuda_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "threads": 4, "split_seed": 20261017, "cal_fit_split_seed": 20261116,
    "source_sha256": {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in sources},
    "input_sha256": {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in inputs},
    "frozen_training_dictionary": FROZEN_TRAINING,
    "datasets": {}, "tuji_loadmat_calls": [],
    "boundary": "Canonical training files only; TUJI physical MAT containers opened/hashed, results_test not decoded. No old checkpoints or historical score JSON read.",
}
original_loadmat = ds.loadmat


def limited_mat(*args, **kwargs):
    assert tuple(kwargs.get("variable_names", ())) == ("results_train",)
    result = original_loadmat(*args, **kwargs)
    assert "results_test" not in result
    payload["tuji_loadmat_calls"].append({"path": str(args[0]), "variable_names": ["results_train"], "results_test_returned": False})
    return result


for name in study.DATASETS:
    _seed(17301)
    with patch.object(ds, "loadmat", limited_mat):
        data = _load(name, None)
    keys = anchor_site_keys(data.coords_m, data.building, data.floor)
    roles = split_physical_sites(keys, seed=20261017)
    cal_fit, cal_eval = study.split_cal(roles["cal"], keys, 20261116)
    expanded = {k: v for k, v in roles.items() if k != "cal"}
    expanded.update(cal_fit=cal_fit, cal_eval=cal_eval)
    sets = {k: {keys[int(i)] for i in rows} for k, rows in expanded.items()}
    assert all(not sets[a].intersection(sets[b]) for a in sets for b in sets if a != b)
    assert set.union(*sets.values()) == set(keys)
    rss, xy, origin, memory, sample_anchor, obs, device_ids, initial, old_ids, old_w = _prepare_fit(data, roles, torch.device("cpu"))
    centered_keys = anchor_site_keys(xy.numpy(), data.building, data.floor)
    forward, reverse = {}, {}
    for a, b in zip(keys, centered_keys):
        assert a not in forward or forward[a] == b
        assert b not in reverse or reverse[b] == a
        forward[a], reverse[b] = b, a
    assert len(forward) == len(reverse) == len(set(keys))
    floor_ids, floor_w = same_floor_neighbors(memory["coords"], memory["building"], memory["floor"], count=FROZEN_TRAINING["target_neighbors"], temperature_m=FROZEN_TRAINING["spatial_temperature_m"])
    new_contamination = neighbor_contamination(floor_ids, floor_w, memory["building"], memory["floor"])
    assert new_contamination == 0.0
    model_a, model_b = copy.deepcopy(initial), copy.deepcopy(initial)
    assert all(torch.equal(model_a.state_dict()[k], model_b.state_dict()[k]) for k in initial.state_dict())
    for role in expanded:
        if role == "test":
            continue
        bf_fit = {(keys[int(i)][0], keys[int(i)][1]) for i in roles["fit"]}
        assert {(keys[int(i)][0], keys[int(i)][1]) for i in expanded[role]} <= bf_fit
    payload["datasets"][name] = {
        "roles": {k: {"sites": len(sets[k]), "scans": len(rows), "site_keys_sha256": hashlib.sha256(json.dumps(sorted(sets[k]), separators=(",", ":")).encode()).hexdigest()} for k, rows in expanded.items()},
        "role_site_overlap": 0, "raw_centered_mapping_bijective": True,
        "fit_anchor_count": len(memory["coords"]), "fit_seen_aps": int(obs["seen_ap_mask"].sum()),
        "fit_scans_per_site": {"min": int(min(list(keys[int(i)] for i in roles["fit"]).count(key) for key in sets["fit"])), "mean": len(roles["fit"]) / len(sets["fit"])},
        "raw_wrong_neighbor_bf_mass": neighbor_contamination(old_ids, old_w, memory["building"], memory["floor"]),
        "same_floor_wrong_neighbor_bf_mass": new_contamination,
        "paired_initial_states_exact": True,
        "single_bf_noop": bool(torch.equal(old_ids, floor_ids) and torch.allclose(old_w, floor_w, atol=1e-7, rtol=1e-7)),
        "optimizer_updates_per_variant": 80 * ((len(roles["fit"]) + 63) // 64),
        "test_rows_scored": 0,
    }
    del rss, xy, memory, obs, initial, model_a, model_b
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({"audit": "PASS", "source_files": len(payload["source_sha256"]), "input_files": len(inputs), "environment": {k: payload[k] for k in ("python_executable", "torch", "torch_cuda", "gpu", "gpu_total_bytes")}, "datasets": payload["datasets"]}, ensure_ascii=False, indent=2))
