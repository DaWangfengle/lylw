"""G2: read-only, post-outcome support/risk mechanism audit of frozen V17-G1.

No training, no checkpoint loading, no prediction, no candidate selection.
This module MUST NOT be used to make confirmatory claims about G1: both
G1 development partitions were seen when G2 hypotheses were formulated.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fhm_dapp_tim.fhm_dapp import uji_v17_g1_experiment as g1run
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _sha256_keys, _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16


PROTOCOL = "v17_g2_posthoc_support_risk_mechanism_v1_20261008"
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_g2_mechanism"
DESIGN = EVIDENCE / "DESIGN_V17_G2.md"
OUTPUT = EVIDENCE / "G2_MECHANISM_AUDIT.json"
REPORT = EVIDENCE / "RUN_REPORT_V17_G2.md"
G1_INPUTS = {
    "registry": g1run.REGISTRY,
    "calibration": g1run.CALIBRATION,
    "freeze": g1run.FREEZE,
    "confirmation": g1run.CONFIRMATION,
}
# Old frozen hashes are used as evidence identity assertions, not as results to tune.
EXPECTED_HASHES = {
    "registry": "bfd44edf4f5f05f08dea2bc2c9972138b361fff529467c166c3e7ab333ebccf6",
    "calibration": "0729b512f74de0d607afb4f4717919a2f139c1110f62e798cd5c0c443045f86d",
    "freeze": "2bf7e709997f27ad83898e57f06cd63370dab508b622ea2577930b94050e75fe",
    "confirmation": "a54a0a8ff452adf3245bd5145afa52b56f9a2c8eedec650e3e29923b57a9b76c",
}
METRICS = ("exact_mixture_mean_mae_m", "energy_score_2d_m")
SUPPORT_RADIUS_M = 10.0
K_NEIGHBORS = (1, 3, 5)
EXPECTED_ROLES = {"calibration": 72, "confirmation": 79}
DISCLOSURE = (
    "Post-outcome exploratory diagnosis of already opened G1 calibration "
    "and confirmation. NO fresh holdout, confirmatory p-values, new PASS, "
    "model selection, deployment gate, or proof of causality."
)


class G2Error(RuntimeError):
    pass


def read_hashed(path: Path, field: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get(field) != _canonical_json_sha256(result, omit=field):
        raise G2Error(f"self-hash mismatch: {path}")
    return result


def verify_chain(inputs: Mapping[str, dict]) -> None:
    reg, cal, freeze, conf = (inputs[name] for name in
                              ("registry", "calibration", "freeze", "confirmation"))
    if any(inputs[name].get("result_sha256" if name != "registry" else "registry_sha256") != sha
           for name, sha in EXPECTED_HASHES.items()):
        raise G2Error("G1 source results differ from reviewed immutable run")
    if freeze["calibration_result_sha256"] != cal["result_sha256"]:
        raise G2Error("calibration/freezing self-hash chain broken")
    if freeze["calibration_file_sha256"] != _sha256_file(G1_INPUTS["calibration"]):
        raise G2Error("frozen calibration byte hash mismatch")
    if conf["calibration_file_sha256"] != _sha256_file(G1_INPUTS["calibration"]):
        raise G2Error("confirmation calibration byte hash mismatch")
    if conf["freeze_file_sha256"] != _sha256_file(G1_INPUTS["freeze"]):
        raise G2Error("confirmation freeze byte hash mismatch")
    if any(inputs[name]["registry_sha256"] != reg["registry_sha256"]
           for name in ("calibration", "freeze", "confirmation")):
        raise G2Error("registry association mismatch")
    if freeze["chosen_gamma"] != 0.7 or conf["chosen_gamma"] != 0.7:
        raise G2Error("frozen gamma mismatch")
    if conf["decision"] != "STOP_GEOMETRY_DENSITY_MECHANISM" or conf["gate"]["passed"] is not False:
        raise G2Error("historical G1 STOP changed")
    if reg["historical_gate_waiver"] is not False or reg["outer_test_predictions_authorized"] is not False:
        raise G2Error("historical leakage guard changed")
    if any(inputs[x].get("official_validation_predictions", 0) != 0 for x in ("calibration", "freeze", "confirmation")):
        raise G2Error("official validation was consumed")
    if any(inputs[x].get("outer_test_predictions", 0) != 0 for x in ("calibration", "freeze", "confirmation")):
        raise G2Error("outer test was consumed")
    if reg["roles"]["development"]["coordinate_count"] != 151:
        raise G2Error("development role differs from preregistered G1")


def load_inputs() -> dict[str, dict]:
    if not DESIGN.is_file():
        raise FileNotFoundError(DESIGN)
    loaded = {}
    for name, source in G1_INPUTS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        loaded[name] = read_hashed(
            source, "registry_sha256" if name == "registry" else "result_sha256"
        )
    verify_chain(loaded)
    if _sha256_file(g1run.TRAIN_DATA) != g1.EXPECTED_TRAIN_SHA256:
        raise G2Error("G1 source UJI trainingData.csv was changed")
    if loaded["registry"]["input_sha256"] != g1.EXPECTED_TRAIN_SHA256:
        raise G2Error("G1 registry raw dataset hash changed")
    if loaded["registry"]["radius_m"] != SUPPORT_RADIUS_M:
        raise G2Error("G1 support radius differs")
    return loaded


def ordered_groups(data: dict, registry: dict) -> tuple[list[tuple], dict[str, list[tuple]], dict[tuple, int]]:
    """Reconstruct G1 coordinate alignment, independently of vector values."""
    hist = g1run._read_hashed(g1run.HISTORICAL_REGISTRY, "registry_sha256")
    roles, rows = g1run._roles(data, hist)
    if roles["selection_refit"] & roles["development"]:
        raise G2Error("training and development coordinate groups overlap")
    if roles["development"] & roles["outer_test"]:
        raise G2Error("development and outer coordinate groups overlap")
    if _sha256_keys(roles["selection_refit"]) != registry["roles"]["selection_refit"]["coordinate_key_sha256"]:
        raise G2Error("selection_refit geometry hash mismatch")
    if _sha256_keys(roles["development"]) != registry["roles"]["development"]["coordinate_key_sha256"]:
        raise G2Error("development geometry hash mismatch")
    cal, conf = g1.split_development_keys(roles["development"])
    groups = {"calibration": sorted(cal), "confirmation": sorted(conf)}
    for role, coordinates in groups.items():
        reg = registry[role]
        if len(coordinates) != EXPECTED_ROLES[role] or _sha256_keys(set(coordinates)) != reg["coordinate_key_sha256"]:
            raise G2Error(f"{role} coordinate hash/count mismatch")
        if set(coordinates) & roles["selection_refit"]:
            raise G2Error("G1 diagnosis leaked training coordinates")
    count = Counter(data["keys"])
    if sum(count[k] for k in groups["calibration"]) != 1529:
        raise G2Error("G1 calibration row count mismatch")
    if sum(count[k] for k in groups["confirmation"]) != 1625:
        raise G2Error("G1 confirmation row count mismatch")
    return sorted(roles["selection_refit"]), groups, count


def geometry_features(
    training_keys: Sequence[tuple],
    query_keys: Sequence[tuple],
    row_counts: Mapping[tuple, int],
    *, radius_m: float = SUPPORT_RADIUS_M,
) -> list[dict]:
    """Physical *retrospective* diagnosis; query x/y are known true labels.

    NEVER use these query-label-dependent features to gate an online predictor.
    """
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    anchors_by_stratum: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for key in training_keys:
        if len(key) != 4 or not all(math.isfinite(float(x)) for x in key[2:]):
            raise ValueError("malformed training geometry")
        anchors_by_stratum[(int(key[0]), int(key[1]))].append([float(key[2]), float(key[3])])
    answer = []
    for key in query_keys:
        if len(key) != 4 or not all(math.isfinite(float(x)) for x in key[2:]):
            raise ValueError("malformed query coordinate")
        stratum = (int(key[0]), int(key[1]))
        anchors = np.asarray(anchors_by_stratum.get(stratum, []), dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[1:] != (2,):
            raise G2Error(f"no training anchor on stratum {stratum}")
        point = np.asarray(key[2:4], dtype=np.float64)
        distances = np.sort(np.linalg.norm(anchors - point, axis=1))
        row_count = int(row_counts.get(tuple(key), 0))
        if row_count <= 0:
            raise G2Error("diagnostic query has no raw RSS rows")
        record = {
            "coordinate": [int(key[0]), int(key[1]), float(key[2]), float(key[3])],
            "building": int(key[0]), "floor": int(key[1]),
            "stratum": f"B{int(key[0])}_F{int(key[1])}",
            "query_rss_row_count": row_count,
            "training_anchor_count_same_stratum": int(len(distances)),
            "training_anchors_within_10m": int(np.count_nonzero(distances <= radius_m)),
            "nearest_training_anchor_m": float(distances[0]),
            "mean_distance_to_training_anchors_m": float(distances.mean()),
        }
        for k in K_NEIGHBORS:
            record[f"training_anchor_d{k}_m"] = float(distances[min(k, len(distances))-1])
            record[f"training_anchor_k{k}_complete"] = len(distances) >= k
        answer.append(record)
    return answer


def checked_vectors(run: dict, keys: Sequence[tuple], role: str) -> dict[str, dict[str, np.ndarray]]:
    if run["role"] != role or run["groups"] != len(keys):
        raise G2Error("G1 result role/group count mismatch")
    if run["group_key_sha256"] != _sha256_keys(set(keys)):
        raise G2Error("G1 group ordering identity checksum mismatch")
    required = ("gamma_0", "gamma_0p7", "gamma_0p35") if role == "calibration" else (
        "baseline", "selected", "placebo"
    )
    records = {}
    for name in required:
        if name not in run["coordinate_group_vectors"]:
            raise G2Error(f"missing G1 variant: {name}")
        endpoint_values = {}
        for endpoint in METRICS:
            x = np.asarray(run["coordinate_group_vectors"][name][endpoint], dtype=np.float64)
            if x.shape != (len(keys),) or not np.isfinite(x).all() or (x < 0).any():
                raise G2Error(f"invalid {name}/{endpoint} vector")
            macro = run["metrics"][name]["mae_2d_m" if endpoint == METRICS[0]
                                        else "energy_score_2d_m"]
            if not math.isclose(float(x.mean()), float(macro), rel_tol=0, abs_tol=2e-6):
                raise G2Error(f"{name}/{endpoint} macro does not match vectors")
            endpoint_values[endpoint] = x
        records[name] = endpoint_values
    return records


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3:
        raise ValueError("correlation arrays must be same 1-D shape (n>=3)")
    xc, yc = x-x.mean(), y-y.mean()
    norm = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float(np.dot(xc, yc)/norm) if norm > 1e-14 else None


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic mergesort, including tied values."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=np.float64)
    a = 0
    while a < len(x):
        b = a + 1
        while b < len(x) and x[order[b]] == x[order[a]]:
            b += 1
        result[order[a:b]] = (a + b - 1) / 2.0
        a = b
    return result


def _round_or_none(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def correlation_diagnostics(records: list[dict], metric: str,
                            *, baseline: np.ndarray, improvement: np.ndarray) -> dict:
    """Descriptive only; partial rank residuals DO NOT identify causality."""
    x = np.asarray([float(item[metric]) for item in records], dtype=np.float64)
    s = [item["stratum"] for item in records]
    levels = sorted(set(s))
    # Control baseline risk and building/floor strata, not outcome-driven bins.
    nuisance = [np.ones(len(x)), _ranks(baseline)]
    nuisance += [np.array([float(q == level) for q in s]) for level in levels[1:]]
    design = np.stack(nuisance, axis=1)
    xrank, yrank = _ranks(x), _ranks(improvement)
    rx = xrank - design @ np.linalg.lstsq(design, xrank, rcond=None)[0]
    ry = yrank - design @ np.linalg.lstsq(design, yrank, rcond=None)[0]
    return {
        "metric": metric,
        "n": len(x),
        "pearson_with_gain": _round_or_none(_pearson(x, improvement)),
        "spearman_with_gain": _round_or_none(_pearson(xrank, yrank)),
        "partial_rank_after_baseline_and_floor_building": _round_or_none(_pearson(rx, ry)),
        "nuisance_design_rank": int(np.linalg.matrix_rank(design)),
        "meaning": "exploratory observational association, not causal effect",
    }


def _buckets(records: list[dict], feature: str, improvement: np.ndarray,
             baseline: np.ndarray) -> list[dict]:
    """Deterministic index quartiles; tie-breaking uses physical key only."""
    vals = np.asarray([float(r[feature]) for r in records])
    order = sorted(range(len(vals)), key=lambda i:(vals[i], tuple(records[i]["coordinate"])))
    answer = []
    for q in range(4):
        sub = order[q*len(order)//4 : (q+1)*len(order)//4]
        answer.append({
            "quartile": q+1, "size": len(sub),
            "feature_min": float(np.min(vals[sub])),
            "feature_max": float(np.max(vals[sub])),
            "baseline_mae_m": float(baseline[sub].mean()),
            "mean_improvement_m": float(improvement[sub].mean()),
            "median_improvement_m": float(np.median(improvement[sub])),
            "improved_coordinates": int(np.count_nonzero(improvement[sub] > 0)),
            "note": "feature bins are post-outcome exploratory, no inference",
        })
    return answer


def risk_diagnostics(
    features: list[dict], baseline: np.ndarray, corrected: np.ndarray,
    baseline_es: np.ndarray, corrected_es: np.ndarray,
    placebo: np.ndarray | None,
) -> dict:
    """Do not mutate score vectors, select gamma, or conduct a new hypothesis test."""
    n = len(features)
    if not (len(baseline) == len(corrected) == len(baseline_es) == len(corrected_es) == n):
        raise ValueError("score/geometry misalignment")
    gain = baseline - corrected
    es_gain = baseline_es - corrected_es
    freq = np.array([f["query_rss_row_count"] for f in features], dtype=np.int64)
    if freq.min() <= 0:
        raise G2Error("nonpositive row count")
    order = sorted(range(n), key=lambda i:(-gain[i], tuple(features[i]["coordinate"])))
    net = float(gain.sum())
    result = {
        "groups": n, "rows": int(freq.sum()),
        "macro_baseline_mae_m": float(baseline.mean()),
        "macro_g1_mae_m": float(corrected.mean()),
        "macro_gain_m": float(gain.mean()),
        "row_count_weighted_mean_coordinate_error_gain_m": float(np.average(gain, weights=freq)),
        "macro_es_gain_m": float(es_gain.mean()),
        "median_gain_m": float(np.median(gain)),
        "improved_groups": int(np.count_nonzero(gain > 0)),
        "worsened_groups": int(np.count_nonzero(gain < 0)),
        "unchanged_groups": int(np.count_nonzero(gain == 0)),
        "top_gain_influence": [
            {"removed_top_gains": k, "remaining_groups": n-k,
             "remaining_gain_m": float(np.delete(gain, order[:k]).mean())}
            for k in (0, 1, 2, 3, 5, 10) if k < n
        ],
        "top_five_gain_sum_m": float(sum(gain[i] for i in order[:5])),
        "net_group_gain_sum_m": net,
        "net_gain_concentration_top_five": (
            float(sum(gain[i] for i in order[:5]) / net) if abs(net)>1e-14 else None
        ),
        "top_positive_gain_groups": [
            {"coordinate": features[i]["coordinate"],
             "baseline_mae_m": float(baseline[i]),
             "corrected_mae_m": float(corrected[i]),
             "gain_m": float(gain[i]),
             "training_anchors_within_10m": features[i]["training_anchors_within_10m"],
             "nearest_training_anchor_m": features[i]["nearest_training_anchor_m"]}
            for i in order[:5]
        ],
        "associations": {},
        "support_density_quartiles": _buckets(
            features, "training_anchors_within_10m", gain, baseline
        ),
        "baseline_error_quartiles": _buckets(
            features, "baseline_mae_m", gain, baseline
        ) if all("baseline_mae_m" in f for f in features) else None,
        "per_coordinate": [],
    }
    if placebo is not None:
        if len(placebo) != n:
            raise ValueError("placebo length mismatch")
        result["mean_true_vs_placebo_gain_m"] = float((placebo-corrected).mean())
        result["true_beats_placebo_groups"] = int(np.count_nonzero(corrected < placebo))
    for attr in ("training_anchors_within_10m", "nearest_training_anchor_m",
                 "training_anchor_d3_m", "training_anchor_d5_m",
                 "query_rss_row_count", "training_anchor_count_same_stratum"):
        result["associations"][attr] = correlation_diagnostics(
            features, attr, baseline=baseline, improvement=gain
        )
    result["associations"]["baseline_mae_m"] = {
        "pearson_with_gain": _round_or_none(_pearson(baseline, gain)),
        "spearman_with_gain": _round_or_none(_pearson(_ranks(baseline), _ranks(gain))),
    }
    for i, feat in enumerate(features):
        rec = dict(feat)
        rec.update({
            "baseline_mae_m": float(baseline[i]),
            "g1_mae_m": float(corrected[i]),
            "mae_gain_m": float(gain[i]),
            "baseline_es_m": float(baseline_es[i]),
            "g1_es_m": float(corrected_es[i]),
            "es_gain_m": float(es_gain[i]),
        })
        if placebo is not None:
            rec["placebo_mae_m"] = float(placebo[i])
        result["per_coordinate"].append(rec)
    # Build difficulty bin from baseline only; bins calculated after results were viewed.
    result["baseline_error_quartiles"] = _buckets(
        result["per_coordinate"], "baseline_mae_m", gain, baseline
    )
    return result


def compare_partitions(cal: dict, conf: dict) -> dict:
    """Falsification-oriented sign checks, explicitly NON-confirmatory."""
    names = ("training_anchors_within_10m", "nearest_training_anchor_m",
             "training_anchor_d3_m", "query_rss_row_count")
    criteria = {
        "low_density_gain_expected": lambda x: x is not None and x < 0,
        "far_anchor_gain_expected": lambda x: x is not None and x > 0,
    }
    rows = []
    for metric in names:
        a = cal["associations"][metric]["spearman_with_gain"]
        b = conf["associations"][metric]["spearman_with_gain"]
        directional = ("low_density_gain_expected" if metric=="training_anchors_within_10m"
                       else "far_anchor_gain_expected" if metric in (
                           "nearest_training_anchor_m", "training_anchor_d3_m"
                       ) else None)
        flag = None
        if directional is not None:
            flag = bool(criteria[directional](a) and criteria[directional](b))
        rows.append({"proxy": metric, "cal_spearman": a, "conf_spearman": b,
                     "same_signed_direction": bool(a is not None and b is not None and a*b>0),
                     "directional_geometry_story_consistent": flag})
    return {
        "cross_split_descriptive_checks": rows,
        "no_inference_or_model_selection_authority": True,
        "interpretation": (
            "Direction disagreement or disappearance after controlling "
            "baseline difficulty undermines a simple geometry-only story; "
            "agreement is NOT independent confirmation."
        ),
    }


def preflight() -> dict:
    values = load_inputs()
    return {
        "status": "READY_READ_ONLY_POSTHOC",
        "protocol": PROTOCOL, "disclosure": DISCLOSURE,
        "source_hashes": {name: {
            "file_sha256": _sha256_file(path),
            "self_sha256": EXPECTED_HASHES[name],
        } for name, path in G1_INPUTS.items()},
        "train_data_sha256": _sha256_file(g1run.TRAIN_DATA),
        "g1_original_stop_preserved": True,
        "external_and_official_predictions_authorized": False,
        "requires_gpu": False, "requires_checkpoints": False,
    }


def analyze() -> dict:
    audit = preflight()
    if OUTPUT.exists() or REPORT.exists():
        raise G2Error("G2 results already exist; no overwrite or adaptive reruns")
    inputs = load_inputs()
    reg = inputs["registry"]
    data = load_uji_v16(g1run.TRAIN_DATA)
    training, groups, counts = ordered_groups(data, reg)
    result = {
        "protocol": PROTOCOL, "status": "DESCRIPTIVE_MECHANISM_AUDIT",
        "decision": "NO_MODEL_SELECTION_NO_NEW_CONFIRMATION",
        "disclosure": DISCLOSURE,
        "input_evidence": audit,
        "source_code_sha256": _sha256_file(Path(__file__)),
        "design_sha256": _sha256_file(DESIGN),
        "geometry": {
            "radius_m": SUPPORT_RADIUS_M, "same_building_floor_only": True,
            "anchors": len(training),
            "training_anchor_coordinate_key_sha256": _sha256_keys(set(training)),
            "query_coordinates_are_true_labels_diagnostics_only": True,
            "not_available_to_prediction_at_deployment": True,
        },
        "roles": {},
    }
    for role in ("calibration", "confirmation"):
        keys = groups[role]
        g1run_result = inputs[role]
        vect = checked_vectors(g1run_result, keys, role)
        features = geometry_features(training, keys, counts)
        baseline, treatment = (("gamma_0", "gamma_0p7") if role == "calibration"
                               else ("baseline", "selected"))
        maeb = vect[baseline][METRICS[0]]
        maec = vect[treatment][METRICS[0]]
        esb = vect[baseline][METRICS[1]]
        esc = vect[treatment][METRICS[1]]
        placebo = None if role=="calibration" else vect["placebo"][METRICS[0]]
        record = risk_diagnostics(features, maeb, maec, esb, esc, placebo)
        if record["groups"] != EXPECTED_ROLES[role] or record["rows"] != g1run_result["rows"]:
            raise G2Error(f"{role} group/row audit mismatch")
        result["roles"][role] = record
    result["cross_role_analysis"] = compare_partitions(
        result["roles"]["calibration"], result["roles"]["confirmation"]
    )
    result["g1_formal_outcome_unchanged"] = {
        "gamma": 0.7, "status": "STOP_GEOMETRY_DENSITY_MECHANISM",
        "confirmation_mae_familywise_upper_m": inputs["confirmation"][
            "paired_familywise_inference"]["comparisons"][METRICS[0]][
            "simultaneous_upper_bound_m"
        ],
        "new_confirmation_performed": False,
    }
    result["outer_test_predictions"] = 0
    result["official_validation_predictions"] = 0
    result["external_predictions"] = 0
    result["result_sha256"] = _canonical_json_sha256(result)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _write_json_new(OUTPUT, result)
    REPORT.write_text(make_report(result), encoding="utf-8", newline="\n")
    return {
        "output": str(OUTPUT), "report": str(REPORT),
        "result_sha256": result["result_sha256"],
        "status": result["status"],
        "g1_formal_stop_preserved": True,
    }


def make_report(result: dict) -> str:
    c, t = (result["roles"][name] for name in ("calibration", "confirmation"))
    lines = [
        "# G2 mechanism audit | post-outcome, descriptive only",
        "",
        "**This is not a new confirmatory test and cannot authorize a model change.**",
        "",
        result["disclosure"], "",
        "## Frozen G1 provenance", "",
        f"- Source G1 gamma: {result['g1_formal_outcome_unchanged']['gamma']}",
        f"- G1 formal decision: {result['g1_formal_outcome_unchanged']['status']}",
        f"- G1 formal MAE familywise upper: {result['g1_formal_outcome_unchanged']['confirmation_mae_familywise_upper_m']:.6f} m",
        f"- Training anchor groups: {result['geometry']['anchors']}",
        "- No model weight, checkpoint, gradient, hyperparameter, outer-test, official or external data was used to produce new predictions.",
        "",
        "## Reconstructed paired risk and influence", "",
        "| Diagnostic | Calibration | Confirmation |",
        "|---|---:|---:|",
    ]
    for key in ("groups","rows","macro_baseline_mae_m","macro_g1_mae_m",
                "macro_gain_m","macro_es_gain_m","median_gain_m",
                "improved_groups","worsened_groups","top_five_gain_sum_m",
                "net_group_gain_sum_m","net_gain_concentration_top_five"):
        def fmt(x):
            return "N/A" if x is None else f"{x:.6f}" if isinstance(x,float) else str(x)
        lines.append(f"| {key} | {fmt(c[key])} | {fmt(t[key])} |")
    lines += [
        "", "## Support-association diagnostics", "",
        "All associations are exploratory and outcome-dependent. Partial rank correlations control for baseline error and building/floor, but are **not causal effects**.", "",
        "| Geometry/proxy | Calibration Spearman | Confirmation Spearman | Cal partial rank | Conf partial rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in c["associations"]:
        if not isinstance(c["associations"][metric], dict) or "partial_rank_after_baseline_and_floor_building" not in c["associations"][metric]:
            continue
        left, right = c["associations"][metric], t["associations"][metric]
        def fmt(x): return "N/A" if x is None else f"{x:+.4f}"
        lines.append(f"| {metric} | {fmt(left['spearman_with_gain'])} | {fmt(right['spearman_with_gain'])} | {fmt(left['partial_rank_after_baseline_and_floor_building'])} | {fmt(right['partial_rank_after_baseline_and_floor_building'])} |")
    lines += [
        "", "## Cross-split geometry sign checks (descriptive)", "",
        "| Proxy | Direction consistent with simple hypothesis on BOTH opened splits? |",
        "|---|---|",
    ]
    for item in result["cross_role_analysis"]["cross_split_descriptive_checks"]:
        lines.append(f"| {item['proxy']} | {item['directional_geometry_story_consistent']} |")
    lines += [
        "", "## Failure-sensitive concentration", "",
        "| Removed top improving coordinates | Cal remaining gain (m) | Conf remaining gain (m) |",
        "|---|---:|---:|",
    ]
    for cc, tt in zip(c["top_gain_influence"],t["top_gain_influence"]):
        lines.append(f"| {cc['removed_top_gains']} | {cc['remaining_gain_m']:+.5f} | {tt['remaining_gain_m']:+.5f} |")
    lines += [
        "", "## Paper-readiness and next-step boundary", "",
        "- G1 is not a generally beneficial correction: report gains and losses by physical coordinate.",
        "- If support-density association lacks consistent direction or vanishes after risk controls, **do not claim geometric sampling density caused the gain**.",
        "- Even consistent association is observational, potentially confounded and mathematically coupled to baseline error; robust causal evidence is still missing.",
        "- The true query coordinate was used only after prediction, so its local anchor density is NOT an online gating feature.",
        "- New adaptive gates or gamma changes cannot be validated on the already-opened G1 confirmation. Any new hypothesis must be fixed before an independent, compatible evaluation cohort is used.",
        "- Independent benchmark evaluation and detailed prior-art assessment remain necessary before Q1 submission claims.",
        "", f"Evidence SHA256: {result['result_sha256']}", "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight","analyze","check"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight()
    elif args.command == "analyze":
        result = analyze()
    else:
        if not OUTPUT.exists() or not REPORT.exists():
            raise G2Error("G2 output/report missing")
        old = read_hashed(OUTPUT, "result_sha256")
        if REPORT.read_text(encoding="utf-8") != make_report(old):
            raise G2Error("G2 report does not match immutable result")
        if preflight()["status"] != "READY_READ_ONLY_POSTHOC":
            raise G2Error("G1 original input integrity failed")
        result = {"status":"VERIFIED","result_sha256":old["result_sha256"]}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
