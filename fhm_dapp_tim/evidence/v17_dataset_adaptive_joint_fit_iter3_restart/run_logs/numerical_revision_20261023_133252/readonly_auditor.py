"""Read saved continuation results only; do not aggregate, predict, or edit evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(r"D:\liuyanlunwen")
OUT = ROOT / "fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart"
TAG = "continuation_27of27_20261023_125047"
DATASETS = tuple(sys.argv[1:])
assert DATASETS and set(DATASETS) <= {"tuji", "uts", "uji"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def close(a, b, tolerance=2e-5):
    a, b = float(a), float(b)
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b, tolerance)
    return abs(a-b)


context = read(OUT / "run_logs" / f"{TAG}_CONTEXT.json")
frozen = read(OUT / "EXECUTION_FROZEN.json")
revision_path = OUT / "run_logs/numerical_revision_20261023_133252/REVISION_FREEZE.json"
revision = read(revision_path)
assert sha(OUT / "EXECUTION_FROZEN.json") == revision["parent_original_freeze_sha256"]
assert revision["status"] == "AUTHORIZED_NUMERICAL_REVISION_FROZEN"
assert set(revision["source_sha256"]) == set(frozen["source_sha256"])
authorized_changes = [name for name, value in revision["source_sha256"].items()
                      if value != frozen["source_sha256"][name]]
assert set(authorized_changes) == {
    "fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py",
    "tests/test_v17_dataset_adaptive_joint_fit_v1.py"}
for name, expected in revision["source_sha256"].items():
    assert sha(ROOT / name) == expected, name
for name, expected in frozen["training_input_sha256"].items():
    assert sha(ROOT / name) == expected, name
for name, expected in context["all_checkpoint_sha256"].items():
    assert sha(ROOT / name) == expected, name
for name, expected in frozen["preflight_sha256"].items():
    assert sha(OUT / name) == expected, name
for name, expected in context["tuji_raw_result_sha256"].items():
    assert sha(OUT / name) == expected, name
for name, expected in context["untouched_existing_audit_files"].items():
    assert sha(OUT / name) == expected, name
search = read(OUT / "SPATIAL_SCALE_SEARCH_SPACE.json")["outer_geometry_candidates"]
fields = ("macro_mae_full_m", "macro_es_m", "macro_transport_risk_m", "macro_joint_nll",
          "macro_bf_accuracy", "bf_cross_entropy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m")
candidate_fields = ("coordinate_mae_m", "transport_w1_m", "bf_accuracy", "bf_cross_entropy", "joint_nll",
                    "site_p95_mae_m", "site_p99_mae_m", "worst_site_mae_m")
checks, paired, evaluation = [], [], []
maximum_reduction_error = maximum_anchor_drift = maximum_reference_drift = 0.
for dataset in DATASETS:
    for seed in (17301, 17302, 17303):
        for arm in "CDE":
            directory = OUT / "runs" / f"{dataset}_seed{seed}" / arm
            selection = read(directory / "CAL_FIT_SELECTION.json")
            report = read(directory / "CAL_EVAL_REPORT.json")
            identity = (dataset, seed, arm)
            assert (selection["dataset"], selection["seed"], selection["backbone"]) == identity
            assert (report["dataset"], report["seed"], report["backbone"]) == identity
            assert report["status"] == "OK"
            assert report["checkpoint_sha256"] == selection["checkpoint_sha256"]
            checkpoint_matches = [value for name, value in context["all_checkpoint_sha256"].items()
                                  if f"/{dataset}_seed{seed}/" in name and f"/{arm}/" in name]
            assert checkpoint_matches == [selection["checkpoint_sha256"]], identity
            assert report["selection_sha256"] == sha(directory / "CAL_FIT_SELECTION.json")
            assert selection["selection_role"] == "CAL-fit" and report["evaluation_role"] == "CAL-eval"
            assert selection["cal_eval_not_used_for_selection"] is True
            assert report["cal_eval_used_for_selection"] is False
            for item in (selection, report):
                assert item["internal_test_rows_scored"] == item["official_test_rows_scored"] == 0
            assert selection["candidate_count"] == selection["finite_candidate_metric_count"] == 52489
            assert selection["nonfinite_candidate_metric_count"] == 0
            assert selection["geometry_count"] == 12 and selection["inner_grid_count"] == 4374
            with (directory / "CANDIDATE_METRICS.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 52489
            assert len({row["candidate_id"] for row in rows}) == len(rows)
            assert all((row["dataset"], int(row["seed"]), row["backbone"]) == identity for row in rows)
            assert all(row["split_hash"] == selection["fit_split_sha256"] for row in rows)
            refs = [row for row in rows if row["candidate_id"] == "clean_baseline_reference"]
            assert len(refs) == 1
            base = refs[0]
            bm, bw, bc, ba, bn = [float(base[k]) for k in
                                ("coordinate_mae_m", "transport_w1_m", "bf_cross_entropy", "bf_accuracy", "joint_nll")]
            for row in rows:
                assert all(math.isfinite(float(row[k])) for k in candidate_fields)
                m, w, c, a, n = [float(row[k]) for k in
                               ("coordinate_mae_m", "transport_w1_m", "bf_cross_entropy", "bf_accuracy", "joint_nll")]
                guards = {"mae_non_degradation": m <= bm + 1e-6,
                          "bf_accuracy_non_degradation": a >= ba - 1e-12,
                          "bf_cross_entropy_guard": c <= bc * 1.02 + .005}
                for name, value in guards.items():
                    assert (row[name] == "True") == value, (identity, name)
                assert (row["feasible"] == "True") == all(guards.values())
                penalty = max(0., c-bc) / max(bc, .05)
                close(row["normalized_bf_cross_entropy_penalty"], penalty, 1e-12)
                for lam in (.02, .05):
                    objective = m/max(bm, 1e-9) + .10*(w-bw)/max(bw, 1e-9) + .25*penalty + lam*(n-bn)/max(abs(bn), 1.)
                    close(row[f"selection_objective_lambda_{lam:.2f}"], objective, 1e-12)
            geometries = search[f"{dataset}_seed{seed}"]["geometry_candidates"]
            s1_edges, s2_edges, anchors = [], [], []
            for geometry in geometries:
                gid = geometry["geometry_id"]
                subset = [row for row in rows if row["geometry_id"] == gid]
                assert len(subset) == 4374
                support = geometry["support_diagnostics"]
                runtime = selection["geometry_diagnostics"][gid]
                assert runtime["preflight_support_validation"] == support
                assert runtime["retained_edges"] == support["retained_s1_edges"] >= 1
                assert runtime["S2"]["retained_virtual_edges"] == support["retained_virtual_edges"]
                assert support["retained_virtual_edges"] >= math.ceil(.45*support["base_edges"])
                s1_edges.append(runtime["retained_edges"])
                s2_edges.append(runtime["S2"]["retained_virtual_edges"])
                center = selection["joint_calibrated_base"]
                exact = [row for row in subset if all(float(row[k]) == 0. for k in ("alpha_prior", "alpha_obs", "beta"))
                         and float(row["temperature"]) == center["temperature"]
                         and [float(row["theta_v"]), float(row["theta_s"])] == center["theta"]]
                assert len(exact) == 1 and exact[0]["feasible"] == "True"
                maximum_anchor_drift = max(maximum_anchor_drift, abs(float(exact[0]["coordinate_mae_m"])-bm))
                anchors.extend(exact)
            eligible = [row for row in rows if row["candidate_id"] != "clean_baseline_reference" and row["feasible"] == "True"]
            for lam in (.02, .05):
                label = f"lambda_{lam:.2f}"
                def rank(row):
                    return (float(row["selection_objective_" + label]), *(float(row[k]) for k in
                            ("coordinate_mae_m", "transport_w1_m", "joint_nll", "temperature", "theta_v", "theta_s",
                             "alpha_prior", "alpha_obs", "beta")), row["geometry_id"])
                winner = min(eligible, key=rank)
                chosen = selection["selected_joint"][label]
                assert chosen["geometry_id"] == winner["geometry_id"]
                for key in ("temperature", "alpha_prior", "alpha_obs", "beta"):
                    close(chosen[key], winner[key], 1e-12)
                close(chosen["theta"][0], winner["theta_v"], 1e-12)
                close(chosen["theta"][1], winner["theta_s"], 1e-12)
                for key in ("mae_non_degradation", "bf_accuracy_non_degradation", "bf_cross_entropy_guard"):
                    assert chosen[key] is True
            shared_sites = None
            for method, entry in report["metrics"].items():
                metrics, sites = entry["metrics"], entry["site_diagnostics"]
                assert all(math.isfinite(float(metrics[k])) for k in fields)
                if shared_sites is None:
                    shared_sites = set(sites)
                assert set(sites) == shared_sites and len(sites) == metrics["n_sites"]
                assert sum(site["n_scans"] for site in sites.values()) == metrics["n_scans"]
                for aggregate, field in (("macro_mae_full_m", "mae_full_m"), ("macro_bf_accuracy", "bf_accuracy"),
                                         ("macro_es_m", "energy_score_m"), ("macro_joint_nll", "joint_nll"),
                                         ("macro_transport_risk_m", "transport_risk_m")):
                    maximum_reduction_error = max(maximum_reduction_error,
                        close(metrics[aggregate], statistics.mean(site[field] for site in sites.values())))
                for field, q in (("site_p95_full_m", .95), ("site_p99_full_m", .99)):
                    close(metrics[field], np.quantile([site["mae_full_m"] for site in sites.values()], q))
                close(metrics["worst_site_mae_m"], max(site["mae_full_m"] for site in sites.values()))
                evaluation.append({"dataset": dataset, "seed": seed, "backbone": arm, "method": method,
                                   **{k: metrics[k] for k in fields}, "n_sites": metrics["n_sites"], "n_scans": metrics["n_scans"]})
            reference = report["metrics"]["joint_calibrated_base"]
            keys = sorted({tuple(site["site"]) for site in reference["site_diagnostics"].values()})
            assert hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest() == selection["eval_split_sha256"]
            for lam in (.02, .05):
                label = f"lambda_{lam:.2f}"
                entry = report["metrics"]["joint_"+label]
                assert entry["config"] == selection["selected_joint"][label]
                diffs = [entry["site_diagnostics"][key]["mae_full_m"] - reference["site_diagnostics"][key]["mae_full_m"] for key in shared_sites]
                paired.append({"dataset": dataset, "seed": seed, "backbone": arm, "lambda": lam,
                               **{"delta_"+k: entry["metrics"][k]-reference["metrics"][k] for k in fields},
                               "improved_sites_gt_1e6m": sum(d < -1e-6 for d in diffs),
                               "harmed_sites_gt_1e6m": sum(d > 1e-6 for d in diffs),
                               "largest_site_harm_m": max(diffs), "largest_site_improvement_m": min(diffs),
                               "selected_parameters": {k: entry["config"][k] for k in
                                   ("geometry_id", "R_max_S1_m", "R_max_S2_m", "tau_graph_m", "tau_virt_m", "temperature", "theta", "alpha_prior", "alpha_obs", "beta")}})
            previous = read(ROOT / "fhm_dapp_tim/evidence/v17_spatial_composition_v1/runs" / f"{dataset}_seed{seed}" / arm / "RESULT.json")
            for name, old_name in (("joint_calibrated_base", "joint_calibrated_base"), ("old_combo_1D_2A", "combo_1D_2A")):
                if name in report["metrics"]:
                    prior_metrics = previous["methods"][old_name]
                    prior_metrics = prior_metrics.get("metrics", prior_metrics)
                    for field in ("macro_mae_full_m", "macro_es_m", "macro_joint_nll", "macro_bf_accuracy"):
                        maximum_reference_drift = max(maximum_reference_drift,
                            close(report["metrics"][name]["metrics"][field], prior_metrics[field]))
            checks.append({"dataset": dataset, "seed": seed, "backbone": arm, "candidate_rows": len(rows),
                           "feasible_search_candidates": len(eligible), "exact_clean_anchors": len(anchors),
                           "s1_edge_range": [min(s1_edges), max(s1_edges)], "s2_edge_range": [min(s2_edges), max(s2_edges)],
                           "empty_s1_geometries": s1_edges.count(0), "empty_s2_geometries": s2_edges.count(0),
                           "cal_eval_sites": len(shared_sites), "cal_eval_scans": reference["metrics"]["n_scans"]})
            print(json.dumps({"stage": "checkpoint_readonly_verified", **checks[-1]}), flush=True)

assert len(checks) == len(DATASETS)*9
native = read(OUT / "INTEGRITY_REVIEW.json")
expected_native = len(list((OUT / "runs").glob("*/[CDE]/CAL_EVAL_REPORT.json")))
assert native["completed_runs"] == expected_native
assert native["expected_runs"] == 27
assert native["status"] == ("PASS" if expected_native == 27 else "INCOMPLETE")
assert native["internal_test_rows_scored"] == native["official_test_rows_scored"] == 0
summary = []
for dataset in DATASETS:
    for method in ("joint_calibrated_base", "joint_lambda_0.02", "joint_lambda_0.05", "old_combo_1D_2A"):
        group = [row for row in evaluation if row["dataset"] == dataset and row["method"] == method]
        if group:
            summary.append({"dataset": dataset, "method": method, "checkpoint_count": len(group),
                            **{k: statistics.mean(row[k] for row in group) for k in fields}})
for dataset in DATASETS:
    for lam in (.02, .05):
        group = [row for row in paired if row["dataset"] == dataset and row["lambda"] == lam]
        summary.append({"dataset": dataset, "comparison": f"joint_lambda_{lam:.2f} minus clean", "checkpoint_count": len(group),
                        **{"mean_delta_"+k: statistics.mean(row["delta_"+k] for row in group) for k in fields},
                        "mae_improved_checkpoints_strict": sum(row["delta_macro_mae_full_m"] < 0 for row in group),
                        "mae_worsened_checkpoints_strict": sum(row["delta_macro_mae_full_m"] > 0 for row in group),
                        "mae_unchanged_checkpoints": sum(row["delta_macro_mae_full_m"] == 0 for row in group),
                        "alpha_obs_zero_count": sum(row["selected_parameters"]["alpha_obs"] == 0 for row in group),
                        "beta_zero_count": sum(row["selected_parameters"]["beta"] == 0 for row in group)})
result = {"status": "READ_ONLY_SCOPE_PASS", "datasets": DATASETS, "verified_runs": len(checks),
          "native_completed_runs": expected_native, "native_status": native["status"],
          "runs": checks, "evaluation_metrics": evaluation, "paired_checkpoint_deltas": paired, "summary": summary,
          "source_files_matching_authorized_revision": len(revision["source_sha256"]),
          "source_files_unchanged_from_original": len(frozen["source_sha256"])-len(authorized_changes),
          "authorized_revised_files": authorized_changes,
          "source_provenance": {"tuji": revision["old_tuji_source_sha256"],
                                "uts_uji": revision["new_uts_uji_source_sha256"],
                                "limitation": revision["revision_limitation"],
                                "revision_freeze_sha256": sha(revision_path)},
          "checkpoints_unchanged": len(context["all_checkpoint_sha256"]),
          "preflight_files_unchanged": len(frozen["preflight_sha256"]), "tuji_raw_files_unchanged": len(context["tuji_raw_result_sha256"]),
          "maximum_site_reduction_error": maximum_reduction_error, "maximum_exact_clean_anchor_mae_drift": maximum_anchor_drift,
          "maximum_historical_reference_metric_drift": maximum_reference_drift,
          "internal_test_rows_scored": 0, "official_test_rows_scored": 0,
          "audit_limits": ["CAL-eval BF-CE is saved aggregate; per-scan true-class probabilities are not exported for independent reconstruction.",
                           "TEST counters are checked in all outputs and source role guards; no OS-level file-access tracer was installed.",
                           "Earlier TUJI-only audit/publication records remain original stage snapshots; native aggregate reports reflect continuation."]}
print("READ_ONLY_RESULT="+json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
