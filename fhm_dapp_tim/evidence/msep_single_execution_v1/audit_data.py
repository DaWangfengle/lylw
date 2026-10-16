"""Read only the three permitted training sources and audit fixed role geometry."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import msep_single_study as study
from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import anchor_site_keys, split_physical_sites, build_fit_observation_memory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    audit = {}
    for name in ("tuji", "uts", "uji"):
        data = study._load(name, None)
        data.validate()
        keys = anchor_site_keys(data.coords_m, data.building, data.floor)
        roles = split_physical_sites(keys, seed=20261016)
        sets = {r: set(keys[int(i)] for i in rows) for r, rows in roles.items()}
        assert sum(map(len, sets.values())) == len(set(keys))
        assert all(not sets[a] & sets[b] for a in sets for b in sets if a != b)
        assert sum(map(len, roles.values())) == len(keys)
        fit = roles["fit"]
        raw = torch.from_numpy(data.model_rss())
        origin = np.asarray(data.coords_m, np.float32)[fit].mean(0).astype(np.float32)
        centered = (np.asarray(data.coords_m, np.float32) - origin).astype(np.float32)
        centered_keys = anchor_site_keys(centered, data.building, data.floor)
        forward, reverse = {}, {}
        for key, centered_key in zip(keys, centered_keys):
            forward.setdefault(key, set()).add(centered_key)
            reverse.setdefault(centered_key, set()).add(key)
        assert all(len(v) == 1 for v in forward.values())
        assert all(len(v) == 1 for v in reverse.values())
        memory, sample_anchor = build_anchor_memory(raw, centered, data.building, data.floor, fit)
        assert len(memory["coords"]) == len(sets["fit"])
        assert all(np.all(sample_anchor[rows] == -1) for role, rows in roles.items() if role != "fit")
        seen = (memory["prototype_rss"] > -.999).any(0)
        obs = build_fit_observation_memory(raw[fit], torch.from_numpy(sample_anchor[fit]), len(memory["coords"]), seen)
        counts = memory["counts"].numpy()
        assert torch.equal(obs["fit_rows_per_anchor"], memory["counts"].float())
        sources = ([ROOT / "data/TUJI1" / f for f in ds.TUJI_DEVICE_FILES.values()]
                   if name == "tuji" else [ROOT / ("data/UTSIndoorLoc/UTS_training.csv" if name == "uts" else "data/UJIndoorLoc/trainingData.csv")])
        audit[name] = {
            "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "ap_ids_sha256": hashlib.sha256(json.dumps(data.ap_ids).encode()).hexdigest(),
            "rows": len(keys), "sites": len(set(keys)), "input_aps": len(data.ap_ids),
            "fit_seen_aps": int(seen.sum()), "fit_anchor_count": len(memory["coords"]),
            "physical_roles_disjoint": True, "centered_site_mapping_bijective": True,
            "non_fit_sample_anchor_ids_all_minus_one": True,
            "g13_statistics_source": "FIT RSS and FIT anchor IDs only; original imported G13 function",
            "fit_scan_count_distribution": {str(n): int((counts == n).sum()) for n in np.unique(counts)},
            "fit_scans_quantiles": np.quantile(counts, [0,.25,.5,.75,.9,1]).tolist(),
            "fit_sites_with_at_least_two_scans": int((counts >= 2).sum()),
            "roles": {r: {"rows": len(rows), "sites": len(sets[r]),
                           "row_indices_sha256": hashlib.sha256(np.asarray(rows, dtype='<i8').tobytes()).hexdigest(),
                           "site_keys_sha256": hashlib.sha256(json.dumps(sorted(sets[r]),separators=(',',':')).encode()).hexdigest(),
                           "by_building_floor": {str(bf): {"sites": sum(k[:2] == bf for k in sets[r]),
                                                           "scans": sum(keys[int(i)][:2] == bf for i in rows)}
                                                 for bf in sorted({k[:2] for k in keys})},
                           "device_rows": {str(int(d)): int((data.device[rows] == d).sum()) for d in np.unique(data.device)}}
                      for r, rows in roles.items()},
            "coordinate_min_m": np.asarray(data.coords_m).min(0).tolist(),
            "coordinate_max_m": np.asarray(data.coords_m).max(0).tolist(),
            "mat_variable_policy": "physical MAT containers opened/hashed; only results_train decoded, results_test absent" if name == 'tuji' else None,
            "official_test_files_opened": False,
        }
        print(json.dumps({"dataset": name, "roles": {r: {k:v for k,v in p.items() if k in ('rows','sites')} for r,p in audit[name]['roles'].items()}, "fit_anchors": len(memory['coords']), "fit_seen_aps": int(seen.sum())}), flush=True)
    target = Path(__file__).with_name("DATA_AUDIT.json")
    with target.open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
