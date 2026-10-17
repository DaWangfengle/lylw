"""Audit saved JSON only; never run a model or select parameters."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            assert key not in ("test_metrics", "test_diagnostics", "test_results")
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(sites, before, after):
    keys = sorted(sites[before])
    pairs = [{"site_key": json.loads(key), "mae_before_m": sites[before][key]["mae_full"],
              "mae_after_m": sites[after][key]["mae_full"],
              "delta_mae_m": sites[after][key]["mae_full"] - sites[before][key]["mae_full"],
              "delta_mae_predicted_bf_m": sites[after][key]["mae_predicted_bf"] - sites[before][key]["mae_predicted_bf"],
              "delta_es_m": sites[after][key]["es"] - sites[before][key]["es"],
              "delta_bf_accuracy": sites[after][key]["bf_accuracy"] - sites[before][key]["bf_accuracy"]}
             for key in keys]
    deltas = np.array([v["delta_mae_m"] for v in pairs])
    return {"before": before, "after": after, "delta_convention": "after minus before; negative error is improvement",
            "n_sites": len(keys), "mean_delta_mae_m": float(deltas.mean()),
            "mean_delta_mae_predicted_bf_m": float(np.mean([v["delta_mae_predicted_bf_m"] for v in pairs])),
            "mean_delta_es_m": float(np.mean([v["delta_es_m"] for v in pairs])),
            "mean_delta_bf_accuracy": float(np.mean([v["delta_bf_accuracy"] for v in pairs])),
            "improved_sites": int((deltas < -1e-6).sum()), "harmed_sites": int((deltas > 1e-6).sum()),
            "ties_within_1e-6m": int((np.abs(deltas) <= 1e-6).sum()),
            "worst_harm": max(pairs, key=lambda p: p["delta_mae_m"]),
            "largest_gain": min(pairs, key=lambda p: p["delta_mae_m"]), "paired_sites": pairs}


def audit_run(domain, seed, *, smoke=False):
    name = f"msep_joint_v1_smoke_{domain}" if smoke else f"msep_joint_v1_{domain}_seed{seed}"
    out = ROOT / "fhm_dapp_tim/evidence" / name
    result, sites = read(out / "RESULT.json"), read(out / "SITE_DIAGNOSTICS.json")
    finite(result)
    finite(sites)
    data = read(HERE / "DATA_AUDIT.json")["datasets"][domain]
    assert result["dataset"] == domain and result["config"]["seed"] == seed
    assert result["config"]["split_seed"] == 20261017
    assert result["cal_fit_sites"] == data["roles"]["cal_fit"]["sites"]
    assert result["cal_eval_sites"] == data["roles"]["cal_eval"]["sites"]
    assert result["site_counts"] == {k: data["roles"][k]["sites"] for k in ("fit", "dev", "cal", "test")}
    expected_keys = {tuple(k) for k in data["roles"]["cal_eval"]["site_keys"]}
    expected_variants = {"raw", "floor_neighbors", "raw_xy", "raw_joint",
                         "floor_neighbors_xy", "floor_neighbors_joint"}
    if not smoke:
        expected_variants |= {"raw_joint_with_evidence", "floor_neighbors_joint_with_evidence"}
    assert set(result["variants"]) == expected_variants
    metrics = {k: v["metrics"] for k, v in result["variants"].items()}
    metrics.update({"cal_"+k: v["metrics"] for k, v in result["calibration"].items()})
    metrics["learned_training_score_same_centers"] = result["score_law_diagnostic"]["learned_training_law_same_centers"]
    assert set(metrics) == set(sites)
    max_difference = 0.
    for policy, met in metrics.items():
        entries = sites[policy]
        assert {tuple(json.loads(k)) for k in entries} == expected_keys
        values = list(entries.values())
        n_scans = sum(v["n_scans"] for v in values)
        assert len(values) == met["n_sites"] == data["roles"]["cal_eval"]["sites"]
        assert n_scans == met["n_scans"] == data["roles"]["cal_eval"]["rows"]
        reconstructed = {
            "macro_mae_full_m": np.mean([v["mae_full"] for v in values]),
            "macro_mae_predicted_bf_m": np.mean([v["mae_predicted_bf"] for v in values]),
            "macro_bf_accuracy": np.mean([v["bf_accuracy"] for v in values]),
            "macro_es_m": np.mean([v["es"] for v in values]),
            "row_bf_accuracy": sum(v["bf_accuracy"]*v["n_scans"] for v in values)/n_scans,
            "site_p95_full_m": np.quantile([v["mae_full"] for v in values], .95)}
        for key, value in reconstructed.items():
            difference = abs(float(value) - met[key])
            assert difference < 2e-5, (domain, seed, policy, key, difference)
            max_difference = max(max_difference, difference)
    for key in metrics["raw"]:
        assert metrics["raw"][key] == metrics["cal_raw"][key]
    assert result["neighbor_mass_wrong_bf_new"] == 0
    assert abs(result["neighbor_mass_wrong_bf_original"] - data["neighbor_mass_wrong_bf_original"]) < 1e-7
    params = {k: v["params"] for k, v in result["calibration"].items()}
    assert set(params) == {"raw", "temp_t06", "temp_fit", "g13_t06", "temp_plus_g13"}
    for key, par in params.items():
        assert .249999 <= par["temperature"] <= 3.000001
        assert all(0 <= v <= 1.5 for v in par["theta"])
    checkpoints = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in out.glob("*.pt")}
    assert len(checkpoints) == (8 if seed == 17301 and not smoke else 0)
    if not smoke:
        assert result["config"]["base_epochs"] == 80 and result["config"]["joint_epochs"] == 10
        assert result["config"]["batch"] == 64 and result["config"]["es_draws"] == 32
        for policy, record in result["variants"].items():
            if policy not in ("raw", "floor_neighbors"):
                assert record["epochs"] == 10
                assert all(0 <= v <= 1.5 for v in record["theta"])
    comparisons = [("raw_xy", "raw_joint"), ("floor_neighbors_xy", "floor_neighbors_joint"),
                   ("raw", "floor_neighbors"), ("raw_xy", "floor_neighbors_xy"),
                   ("raw_joint", "floor_neighbors_joint"), ("cal_temp_t06", "cal_g13_t06"),
                   ("cal_temp_fit", "cal_temp_plus_g13"), ("cal_raw", "cal_temp_fit"),
                   ("cal_raw", "learned_training_score_same_centers")]
    if not smoke:
        comparisons += [("raw_joint", "raw_joint_with_evidence"),
                        ("floor_neighbors_joint", "floor_neighbors_joint_with_evidence"),
                        ("raw_joint_with_evidence", "floor_neighbors_joint_with_evidence")]
    return {"dataset": domain, "seed": seed, "output": out.relative_to(ROOT).as_posix(),
            "config": result["config"], "metrics": metrics, "calibration_parameters": params,
            "base_training": {k: result["variants"][k]["training"] for k in ("raw", "floor_neighbors")},
            "refit_parameters": {k: v["theta"] for k, v in result["variants"].items() if "theta" in v},
            "role_summary": {r: {k: data["roles"][r][k] for k in ("rows", "sites", "row_indices_sha256", "site_keys_sha256")}
                             for r in data["roles"]},
            "max_saved_site_reaggregation_difference": max_difference,
            "saved_site_audit": "PASS; independent read-only macro/P95/BF reconstruction; scan-level errors not saved, no invented scan audit",
            "test_evaluated": False, "checkpoints": checkpoints,
            "output_sha256": {p.name: sha(p) for p in (out/"RESULT.json", out/"SITE_DIAGNOSTICS.json")},
            "comparisons": [compare(sites, a, b) for a, b in comparisons]}


def main():
    mode = sys.argv[1]
    smoke = mode == "smoke"
    seeds = (17301,) if smoke else (17301, 17302, 17303)
    runs = [audit_run(domain, seed, smoke=smoke) for seed in seeds for domain in ("tuji", "uts", "uji")]
    name = "SMOKE_AUDIT.json" if smoke else "SUMMARY.json"
    result = {"status": "PASS", "study": "MSEP joint-location ablation v1",
              "evidence_scope": "split-internal exploratory development; CAL-eval only; no TEST scoring",
              "test_model_inference_or_evaluation": False, "runs": runs}
    if not smoke:
        frozen = read(HERE/"METHOD_FREEZE.json")
        execution = read(HERE/"EXECUTION_RECORD.json")
        result["source_identity"] = {"branch": frozen["initial_branch"],
                                     "commit": frozen["initial_commit"],
                                     "method_freeze_sha256": sha(HERE/"METHOD_FREEZE.json"),
                                     "scientific_source_changes": []}
        result["environment"] = read(HERE/"DATA_AUDIT.json")["environment"]
        result["formal_execution"] = [c for c in execution["commands"] if c["label"].startswith("full_")]
        assert len(result["formal_execution"]) == 9
        result["across_seed_descriptive"] = {
            d: {p: {m: {"mean": float(np.mean(v)), "sample_sd": float(np.std(v, ddof=1)),
                        "min": min(v), "max": max(v)}
                    for m in ("macro_mae_full_m", "macro_mae_predicted_bf_m", "macro_bf_accuracy", "macro_es_m", "macro_joint_nll", "site_p95_full_m")
                    for v in [[r["metrics"][p][m] for r in runs if r["dataset"] == d]]}
                for p in runs[0]["metrics"]} for d in ("tuji", "uts", "uji")}
    with (HERE/name).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": "PASS", "mode": mode, "runs": len(runs),
                      "max_reaggregation_difference": max(r["max_saved_site_reaggregation_difference"] for r in runs)}))


if __name__ == "__main__":
    main()
