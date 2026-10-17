"""Read permitted training inputs; audit roles and FIT statistics without scoring TEST."""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import platform
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas
import scipy
import torch
from fhm_dapp_tim.fhm_dapp import msep_single_study as old
from fhm_dapp_tim.fhm_dapp import msep_joint_ablation_v1 as study
from fhm_dapp_tim.fhm_dapp import msep_joint_science_v1 as science
from fhm_dapp_tim.fhm_dapp import v18_4_datasets as adapters
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import anchor_site_keys, split_physical_sites


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def main():
    torch.set_num_threads(4)
    result = {"environment": {"python": sys.version, "python_executable": sys.executable,
              "platform": platform.platform(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "gpu": str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None,
              "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
              "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "split_seed": 20261017, "cal_split_seed": 20261116, "datasets": {}}
    mat_calls = []
    original_loadmat = adapters.loadmat

    def inspect_loadmat(*args, **kwargs):
        assert kwargs.get("variable_names") == ("results_train",)
        loaded = original_loadmat(*args, **kwargs)
        assert "results_test" not in loaded and "results_train" in loaded
        mat_calls.append({"path": str(args[0]), "variable_names": list(kwargs["variable_names"]),
                          "returned_variables": sorted(k for k in loaded if not k.startswith("__"))})
        return loaded

    for name in ("tuji", "uts", "uji"):
        old._seed(17301)
        with patch.object(adapters, "loadmat", inspect_loadmat):
            data = old._load(name, None)
        data.validate()
        keys = anchor_site_keys(data.coords_m, data.building, data.floor)
        roles = split_physical_sites(keys, seed=20261017)
        roles["cal_fit"], roles["cal_eval"] = study.split_cal(roles["cal"], keys, 20261116)
        sets = {r: {keys[int(i)] for i in rows} for r, rows in roles.items()}
        assert sets["cal_fit"] | sets["cal_eval"] == sets["cal"]
        assert sum(len(roles[r]) for r in ("fit", "dev", "cal", "test")) == len(keys)
        for a, b in itertools.combinations(("fit", "dev", "cal_fit", "cal_eval", "test"), 2):
            assert not sets[a] & sets[b] and not set(roles[a]) & set(roles[b])
        rss, xy, origin, memory, sample_anchor, obs, devices, model, old_n, old_w = old._prepare_fit(data, roles, torch.device("cpu"))
        new_n, new_w = science.same_floor_neighbors(memory["coords"], memory["building"], memory["floor"])
        assert torch.isfinite(new_w).all() and torch.allclose(new_w.sum(1), torch.ones(len(new_w)))
        assert science.neighbor_contamination(new_n, new_w, memory["building"], memory["floor"]) == 0
        assert len(memory["coords"]) == len(sets["fit"])
        assert all(np.all(sample_anchor[roles[r]] == -1) for r in ("dev", "cal_fit", "cal_eval", "test"))
        centered_keys = anchor_site_keys(xy.numpy(), data.building, data.floor)
        forward, reverse = {}, {}
        for a, b in zip(keys, centered_keys):
            forward.setdefault(a, set()).add(b)
            reverse.setdefault(b, set()).add(a)
        assert all(len(v) == 1 for v in forward.values()) and all(len(v) == 1 for v in reverse.values())
        assert torch.equal(obs["fit_rows_per_anchor"], memory["counts"].float())
        assert all(torch.isfinite(v).all() for v in obs.values())
        sources = ([ROOT / "data/TUJI1" / f for f in adapters.TUJI_DEVICE_FILES.values()]
                   if name == "tuji" else [ROOT / ("data/UTSIndoorLoc/UTS_training.csv" if name == "uts" else "data/UJIndoorLoc/trainingData.csv")])
        role_summary = {}
        for r, rows in roles.items():
            role_summary[r] = {"rows": len(rows), "sites": len(sets[r]),
                               "row_indices_sha256": hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest(),
                               "site_keys_sha256": digest(sorted(sets[r])),
                               "by_building_floor": {str(bf): {"sites": sum(k[:2] == bf for k in sets[r]),
                                                              "rows": sum(keys[int(i)][:2] == bf for i in rows)}
                                                     for bf in sorted({k[:2] for k in keys})}}
            if r != "test":
                role_summary[r]["site_keys"] = sorted(sets[r])
        counts = memory["counts"].numpy()
        _, cls = science.class_structure(memory["building"], memory["floor"])
        class_counts = torch.bincount(cls)
        result["datasets"][name] = {
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
            "roles": role_summary, "physical_roles_disjoint": True,
            "centered_site_mapping_bijective": True, "fit_only_reference_statistics": True,
            "fit_anchor_count": len(memory["coords"]), "input_aps": len(data.ap_ids),
            "fit_seen_aps": int(model.encoder.seen_ap_mask.sum()),
            "ap_ids_sha256": digest(data.ap_ids), "origin_m": origin.tolist(),
            "fit_scan_count_distribution": dict(collections.Counter(map(int, counts))),
            "fit_sites_with_at_least_two_scans": int((counts >= 2).sum()),
            "fit_singleton_bf_strata": int((class_counts == 1).sum()),
            "neighbor_mass_wrong_bf_original": science.neighbor_contamination(old_n, old_w, memory["building"], memory["floor"]),
            "neighbor_mass_wrong_bf_new": 0.0,
            "effective_total_soft_target_wrong_bf_mass_original": .5 * science.neighbor_contamination(old_n, old_w, memory["building"], memory["floor"]),
            "test_role_policy": "only partition metadata counted/hashed; never a model/scoring/calibration input"}
        print(json.dumps({"dataset": name, "roles": {r: {"sites": v["sites"], "rows": v["rows"]} for r, v in role_summary.items()},
                          "old_neighbor_wrong_bf_mass": result["datasets"][name]["neighbor_mass_wrong_bf_original"]}), flush=True)
        del rss, xy, memory, obs, devices, model
    result["tuji_mat_calls"] = mat_calls
    assert len(mat_calls) == 5
    result["data_boundary_note"] = "The mixed MAT containers are opened and hashed; only results_train is decoded. No OS-wide I/O trace was performed."
    with Path(__file__).with_name("DATA_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
