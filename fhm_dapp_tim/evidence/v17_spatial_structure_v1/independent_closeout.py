"""Independent saved-artifact recomputation, with no model inference or TEST data."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as study

torch.set_num_threads(1)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode() + tensor.numpy().tobytes()).hexdigest()


def main():
    manifest = study.freeze_guard(ROOT)
    historical_root = Path(manifest["input_root"])
    before = study.read_json(historical_root / "BEFORE_RUN_AUDIT.json")
    study.verify_hashes(before["source_sha256"])
    cross_phase = study.read_json(ROOT / "CROSS_PHASE_OA_NUMERICAL_REVIEW.json")
    all_results = sorted(ROOT.glob("*/*/*/RESULT.json"))
    assert len(all_results) == 63
    records, site_records, hashes, checkpoint_records, graph_records = {}, {}, {}, [], {}
    max_reduction, max_composition = 0., 0.
    phase_counts = {"F": 0, "T": 0}
    for path in all_results:
        result = study.read_json(path)
        sites = study.read_json(path.with_name("SITE_DIAGNOSTICS.json"))
        assert finite(result) and finite(sites), f"nonfinite JSON: {path}"
        phase, dataset, seed, backbone = (result[k] for k in ("phase", "dataset", "seed", "backbone"))
        assert result["evaluation_role"] == "CAL-eval"
        assert all(result[k] == 0 for k in study.TEST_COUNTS)
        phase_counts[phase] += 1
        expected = manifest["runs"][f"{dataset}_seed{seed}"]
        for method, entry in result["methods"].items():
            table = sites[method]
            values = list(table.values())
            assert study.key_sha(tuple(s["site"]) for s in values) == expected["cal_eval_key_sha256"]
            assert sum(s["n_scans"] for s in values) == entry["metrics"]["n_scans"] == expected["cal_eval_rows"]
            assert len(values) == entry["metrics"]["n_sites"]
            for key, item in table.items():
                assert json.loads(key) == item["site"] and item["n_scans"] > 0
            for metric, field in (("macro_mae_full_m", "mae_full_m"), ("macro_mae_predicted_bf_m", "mae_predicted_bf_m"),
                                   ("macro_es_m", "energy_score_m"), ("macro_joint_nll", "joint_nll"),
                                   ("macro_bf_accuracy", "bf_accuracy")):
                error = abs(statistics.mean(s[field] for s in values) - entry["metrics"][metric])
                assert error < 2e-5, (path, method, metric, error)
                max_reduction = max(max_reduction, error)
            mae = [s["mae_full_m"] for s in values]
            for metric, actual in (("site_p95_full_m", float(np.quantile(mae, .95))),
                                   ("site_p99_full_m", float(np.quantile(mae, .99))),
                                   ("worst_site_mae_m", max(mae))):
                assert abs(actual - entry["metrics"][metric]) < 1e-10
            row_bf = sum(s["n_scans"] * s["bf_accuracy"] for s in values) / sum(s["n_scans"] for s in values)
            assert abs(row_bf - entry["metrics"]["row_bf_accuracy"]) < 1e-6
            key = (phase, dataset, str(seed), backbone, method)
            assert key not in records
            records[key], site_records[key] = entry["metrics"], table
        if phase == "F":
            selection_path = path.with_name("CAL_FIT_SELECTION.json")
            assert result["selection_sha256"] == study.sha(selection_path)
            selection = study.read_json(selection_path)
            grid = selection["calibration_grid"]
            assert len([r for r in grid if r["stage"] == "coarse"]) == 25 * 7 * 7
            winner = min(grid, key=lambda r: (r["objective"], r["temperature"], *r["theta"]))
            assert selection["joint_calibrated_base"] == {"temperature": winner["temperature"], "theta": winner["theta"]}
            for name, table in selection["structure1_grids"].items():
                best = min(table, key=lambda r: (r["objective"], r["alpha_prior"], r["alpha_obs"]))
                assert all(selection["structure1"][name][k] == v for k, v in best.items())
                assert len(table) == (81 if name.startswith("1D") else 9)
            table = selection["structure2_grid"]
            assert len(table) == 49
            assert {(r["beta"], r["gamma_space"]) for r in table} == set(itertools.product(study.BETAS, study.GAMMAS))
            best = min(table, key=lambda r: (r["objective"], r["beta"], r["gamma_space"]))
            assert best == selection["2B"]
            assert selection["2A"] == {"beta": best["beta"], "gamma_space": 0.}
            assert result["beta_zero_parent_evidence_exact"] and result["beta_zero_joint_nll_ES_allclose"]
            assert result["raw_MAE_reproduction_abs_m"] < 2e-5
            for name, entry in result["methods"].items():
                if name.startswith("1C"):
                    assert max(entry["log_partition_gap_by_scan"]) <= 2e-6
                if "max_raw_bf_mass_error" in entry:
                    assert entry["max_raw_bf_mass_error"] < 2e-6
        else:
            history = read_csv(path.with_name("EPOCH_HISTORY.csv"))
            assert [int(r["epoch"]) for r in history] == list(range(1, 81))
            chosen = min((r for r in history if int(r["epoch"]) >= 21), key=lambda r: float(r["dev_mae_m"]))
            assert int(chosen["epoch"]) == result["selected_epoch"]
            checkpoint = result["checkpoint"]
            cp = Path(checkpoint["path"])
            assert study.sha(cp) == checkpoint["sha256"]
            if backbone in ("O", "A"):
                assert checkpoint["reused"] and checkpoint["new_optimizer_updates"] == 0
                fkey = ("F", dataset, str(seed), study.VARIANTS[backbone], "historical_raw")
                tkey = ("T", dataset, str(seed), backbone, backbone)
                # The protocol's beta=0 identity is checked within the same F
                # pack. Across phases, B=8 vs B=32 forward arithmetic can change
                # CDF decisions and MC ES; keep this as a disclosed diagnostic,
                # not an invented stronger per-site identity gate.
                for site in site_records[tkey]:
                    assert site_records[tkey][site]["bf_accuracy"] == site_records[fkey][site]["bf_accuracy"]
                for metric in ("mae_full_m", "mae_predicted_bf_m"):
                    mean_difference = statistics.mean(site_records[tkey][s][metric] - site_records[fkey][s][metric]
                                                      for s in site_records[tkey])
                    assert abs(mean_difference) < 2e-5
            else:
                assert not checkpoint["reused"]
                fit_rows = expected["roles"]["site_counts"]["fit"]["rows"]
                assert checkpoint["new_optimizer_updates"] == 80 * math.ceil(fit_rows / 64)
                for row in history:
                    composed = float(row["fit_joint_bf_xy_nll"]) + .5 * float(row["fit_retrieval_loss"]) + float(row["fit_hierarchy_loss"]) + .25 * float(row["fit_coordinate_loss"])
                    if backbone in ("D", "E"):
                        composed += .30 * float(row["fit_transport_risk"])
                    error = abs(composed - float(row["fit_loss"]))
                    assert error < 1e-4, (dataset, seed, backbone, row["epoch"], error)
                    max_composition = max(max_composition, error)
            cp_data = torch.load(cp, weights_only=False, map_location="cpu")
            assert cp_data["selected_epoch"] == result["selected_epoch"]
            state = cp_data["state_dict"]
            for name, value in state.items():
                numeric = value.coalesce().values() if value.layout == torch.sparse_coo else value
                assert torch.isfinite(numeric).all(), (cp, name)
            if backbone == "E":
                assert abs(float(state["raw_alpha"].sigmoid()) - result["methods"]["E"]["params"]["alpha"]) < 1e-7
            checkpoint_records.append({"dataset": dataset, "seed": seed, "arm": backbone,
                                       "selected_epoch": result["selected_epoch"], **checkpoint})
        for artifact in path.parent.iterdir():
            if artifact.is_file():
                hashes[artifact.relative_to(ROOT).as_posix()] = study.sha(artifact)
    assert phase_counts == {"F": 18, "T": 45}
    run_rows = read_csv(ROOT / "RUN_LEVEL_METRICS.csv")
    assert len(run_rows) == len(records) == 405
    for row in run_rows:
        key = tuple(row[k] for k in ("phase", "dataset", "seed", "backbone", "method"))
        for name, value in records[key].items():
            assert abs(float(row[name]) - value) < 1e-10
    summaries = read_csv(ROOT / "AGGREGATE_SUMMARY.csv")
    assert len(summaries) == 135
    for row in summaries:
        group = (row["phase"], row["dataset"], row["backbone"], row["method"])
        selected = [m for k, m in records.items() if (k[0], k[1], k[3], k[4]) == group]
        assert len(selected) == 3
        for metric in ("macro_mae_full_m", "macro_mae_predicted_bf_m", "macro_es_m", "macro_joint_nll", "macro_bf_accuracy", "row_bf_accuracy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m"):
            values = [m[metric] for m in selected]
            assert abs(float(row[metric + "_mean"]) - statistics.mean(values)) < 1e-10
            assert abs(float(row[metric + "_sd"]) - statistics.stdev(values)) < 1e-10
    paired = read_csv(ROOT / "PAIRED_DELTAS.csv")
    for row in paired:
        prefix = (row["phase"], row["dataset"], row["seed"], row["backbone"])
        ref = records[(*prefix, row["reference"])] if row["phase"] == "F" else records[("T", row["dataset"], row["seed"], row["reference"], row["reference"])]
        cand = records[(*prefix, row["candidate"])]
        for field, metric in (("mae_delta_m", "macro_mae_full_m"), ("es_delta_m", "macro_es_m"), ("joint_nll_delta", "macro_joint_nll"), ("p95_delta_m", "site_p95_full_m"), ("p99_delta_m", "site_p99_full_m")):
            assert abs(float(row[field]) - (cand[metric] - ref[metric])) < 1e-10
    site_rows = read_csv(ROOT / "SITE_DELTAS_ALL_RUNS.csv")
    for row in site_rows:
        prefix = (row["phase"], row["dataset"], row["seed"], row["backbone"])
        refkey = (*prefix, row["reference"]) if row["phase"] == "F" else ("T", row["dataset"], row["seed"], row["reference"], row["reference"])
        ref, cand = site_records[refkey][row["site_key"]], site_records[(*prefix, row["candidate"])][row["site_key"]]
        for field, metric in (("mae_delta_m", "mae_full_m"), ("energy_score_delta_m", "energy_score_m"), ("bf_accuracy_delta", "bf_accuracy")):
            assert abs(float(row[field]) - (cand[metric] - ref[metric])) < 1e-10
    for dataset in study.DATASETS:
        ref = torch.load(historical_root / f"{dataset}_seed17301/REFERENCE_STATE.pt", weights_only=False, map_location="cpu")
        graph, edges = study.op.build_doubly_stochastic_floor_graph(ref["fit_anchor_coords"], ref["fit_anchor_building"], ref["fit_anchor_floor"])
        graph_records[dataset] = {"canonical_edge_ids_sha256": tensor_sha(edges),
                                 "graph_indices_sha256": tensor_sha(graph.coalesce().indices()),
                                 "graph_values_sha256": tensor_sha(graph.coalesce().values()),
                                 "physical_anchors": graph.shape[0], "unique_edges": len(edges)}
    execution = study.read_json(ROOT / "EXECUTION_RECORD.json")
    for item in execution["commands"]:
        if "stdout_sha256" in item:
            assert study.sha(REPO / item["stdout"]) == item["stdout_sha256"]
            assert study.sha(REPO / item["stderr"]) == item["stderr_sha256"]
    study.write_json(ROOT / "INDEPENDENT_CLOSEOUT.json", {
        "status": "PASS_WITH_DISCLOSED_CROSS_PHASE_DIFFERENCES", "mandatory_protocol_audits_status": "PASS",
        "additional_cross_phase_identity_status": "NOT_IDENTICAL",
        "cross_phase_OA_numerical_review": cross_phase,
        "no_model_inference": True, "phase_counts": phase_counts,
        "run_metric_rows": len(run_rows), "summary_rows": len(summaries), "paired_rows": len(paired),
        "paired_site_rows": len(site_rows), "maximum_site_reduction_difference": max_reduction,
        "maximum_training_composition_rounding_difference": max_composition,
        "training_composition_audit_tolerance": 1e-4, "checkpoint_identity": checkpoint_records,
        "canonical_graph_identity": graph_records, "artifact_sha256": hashes,
        "historical_sources_unchanged": len(before["source_sha256"]),
        "zero_test_scoring_counts": study.TEST_COUNTS,
        "limits": "Recomputed from serialized per-site data; raw per-scan outputs are not emitted, so no independent scan-to-site reconstruction is claimed.",
    })
    print(json.dumps({"status": "PASS_WITH_DISCLOSED_CROSS_PHASE_DIFFERENCES", "mandatory_protocol_audits_status": "PASS",
                      "phase_counts": phase_counts, "run_rows": len(run_rows),
                      "summary_rows": len(summaries), "paired_site_rows": len(site_rows),
                      "maximum_loss_composition_error": max_composition}, ensure_ascii=False))


if __name__ == "__main__":
    main()
