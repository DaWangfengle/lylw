"""Regression tests for post-outcome, label-aware G2 diagnostics.

CPU-only, no UJI raw file, no GPU, no historical checkpoint required.
"""
from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pytest

from fhm_dapp_tim.fhm_dapp import uji_v17_g2_mechanism as g2
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _sha256_keys


@pytest.fixture
def test_geometry():
    # The fourth training anchor is deliberately on another floor.
    training = [
        (0, 0, 0.0, 0.0),
        (0, 0, 4.0, 0.0),
        (0, 0, 9.0, 0.0),
        (0, 1, 0.0, 0.0),
        (1, 0, 0.0, 0.0),
    ]
    query = [(0, 0, 1.0, 0.0), (0, 1, 1.0, 0.0)]
    rows = {query[0]: 3, query[1]: 5}
    return training, query, rows


def test_geometry_distance_excludes_other_floor_and_building(test_geometry):
    train, query, counts = test_geometry
    a, b = g2.geometry_features(train, query, counts, radius_m=5.0)
    assert a["training_anchor_count_same_stratum"] == 3
    assert a["training_anchors_within_10m"] == 2
    assert a["nearest_training_anchor_m"] == pytest.approx(1.0)
    assert a["training_anchor_d3_m"] == pytest.approx(8.0)
    assert a["query_rss_row_count"] == 3
    assert b["training_anchor_count_same_stratum"] == 1
    assert b["training_anchors_within_10m"] == 1
    assert b["nearest_training_anchor_m"] == pytest.approx(1.0)
    assert b["training_anchor_k5_complete"] is False


def test_geometry_empty_matching_stratum_is_fatal(test_geometry):
    train, _, _ = test_geometry
    key=(7, 3, 0.0, 0.0)
    with pytest.raises(g2.G2Error, match="no training anchor"):
        g2.geometry_features(train, [key], {key:1})


@pytest.mark.parametrize("radius", [-1.0, 0.0, float("nan"), float("inf")])
def test_bad_radius_is_rejected(test_geometry, radius):
    train, query, counts = test_geometry
    with pytest.raises(ValueError):
        g2.geometry_features(train, query, counts, radius_m=radius)


def test_query_without_rss_is_rejected(test_geometry):
    train, query, _ = test_geometry
    with pytest.raises(g2.G2Error, match="no raw RSS rows"):
        g2.geometry_features(train, query, {})


def test_stable_average_rank_ties():
    assert g2._ranks(np.array([10, 20, 20, 30])).tolist() == [0.0, 1.5, 1.5, 3.0]
    assert g2._ranks(np.array([2, 2, 2])).tolist() == [1.0, 1.0, 1.0]


def test_constant_correlation_is_undefined_not_fabricated():
    assert g2._pearson(np.ones(4), np.arange(4, dtype=float)) is None
    with pytest.raises(ValueError):
        g2._pearson(np.ones(2), np.ones(2))


def test_vectors_reject_wrong_order_count_and_mean():
    keys=[(0, 0, float(x), 0.0) for x in range(4)]
    variants = {}
    scores = {}
    for variant in ("gamma_0", "gamma_0p35", "gamma_0p7"):
        variants[variant] = {
            "exact_mixture_mean_mae_m": [1., 2., 3., 4.],
            "energy_score_2d_m": [2., 3., 4., 5.],
        }
        scores[variant] = {"mae_2d_m": 2.5, "energy_score_2d_m": 3.5}
    raw = {"role":"calibration","groups":4,
           "group_key_sha256":_sha256_keys(set(keys)),
           "coordinate_group_vectors":variants, "metrics":scores}
    got = g2.checked_vectors(raw, keys, "calibration")
    assert np.array_equal(got["gamma_0"][g2.METRICS[0]], [1.,2.,3.,4.])
    bad = dict(raw, group_key_sha256="bad")
    with pytest.raises(g2.G2Error):
        g2.checked_vectors(bad, keys, "calibration")
    bad = dict(raw, groups=3)
    with pytest.raises(g2.G2Error):
        g2.checked_vectors(bad, keys, "calibration")
    bad = json.loads(json.dumps(raw))
    bad["metrics"]["gamma_0"]["mae_2d_m"] = 2.51
    with pytest.raises(g2.G2Error):
        g2.checked_vectors(bad, keys, "calibration")


def synthetic_records(n=12):
    train = [(0, 0, float(x), 0.0) for x in range(4)]
    query = [(0, 0, float(x) + .5, 1.0) for x in range(n)]
    return g2.geometry_features(train, query, {key: x+1 for x,key in enumerate(query)})


def test_risk_diagnostics_concentration_and_row_weighted_mean():
    f = synthetic_records(12)
    baseline=np.array([2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.])
    gain = np.array([2.,1.,.7,.3,0,-.1,-.2,-.3,-.4,-.5,-.6,-.7])
    corrected = baseline-gain
    res=g2.risk_diagnostics(f, baseline, corrected, baseline*.7, corrected*.7, baseline+.2)
    assert res["groups"]==12
    assert res["rows"]==78
    assert res["macro_gain_m"]==pytest.approx(np.mean(gain))
    assert res["improved_groups"] == 4
    assert res["worsened_groups"] == 7
    assert res["unchanged_groups"] == 1
    assert res["top_gain_influence"][0]["remaining_gain_m"] == pytest.approx(np.mean(gain))
    assert res["top_gain_influence"][1]["remaining_gain_m"] == pytest.approx(np.mean(gain[1:]))
    assert res["mean_true_vs_placebo_gain_m"] == pytest.approx(.2 + np.mean(gain))
    assert len(res["per_coordinate"]) == 12
    assert len(res["baseline_error_quartiles"]) == 4
    assert sum(x["size"] for x in res["support_density_quartiles"])==12
    assert res["associations"]["nearest_training_anchor_m"]["spearman_with_gain"] is not None


def test_risk_row_misalignment_and_placebo_shape_fail():
    f=synthetic_records(12)
    base=np.arange(12)+1.
    with pytest.raises(ValueError):
        g2.risk_diagnostics(f, base[:-1], base, base, base, None)
    with pytest.raises(ValueError):
        g2.risk_diagnostics(f, base, base, base, base, np.ones(10))


def test_noncausal_direction_flags_not_conflated_with_formal_gate():
    x = lambda a: {"associations": {
        "training_anchors_within_10m":{"spearman_with_gain":a},
        "nearest_training_anchor_m":{"spearman_with_gain":-a},
        "training_anchor_d3_m":{"spearman_with_gain":-a},
        "query_rss_row_count":{"spearman_with_gain":0.0},
    }}
    out = g2.compare_partitions(x(-.2),x(-.1))
    assert out["cross_split_descriptive_checks"][0]["directional_geometry_story_consistent"] is True
    assert out["cross_split_descriptive_checks"][1]["directional_geometry_story_consistent"] is True
    assert out["no_inference_or_model_selection_authority"] is True
    out = g2.compare_partitions(x(-.2), x(.1))
    assert out["cross_split_descriptive_checks"][0]["directional_geometry_story_consistent"] is False


def test_partial_correlation_not_artificially_declared_causal():
    f=synthetic_records(12)
    a=np.linspace(1.,13.,12)
    b=np.linspace(.01,2.,12)
    corr=g2.correlation_diagnostics(f, "query_rss_row_count",baseline=a,improvement=b)
    assert corr["meaning"].startswith("exploratory")
    assert corr["n"]==12
    assert corr["nuisance_design_rank"]>=2


def test_report_building_does_not_change_scores():
    f=synthetic_records(12)
    base=np.arange(12)+2.
    cal=g2.risk_diagnostics(f,base,base-0.2,base*.7,base*.7-0.1,None)
    conf=g2.risk_diagnostics(f,base,base-0.3,base*.7,base*.7-0.2,base+0.1)
    obj={"disclosure":g2.DISCLOSURE, "geometry":{"anchors":595},
         "g1_formal_outcome_unchanged":{"gamma":.7,
             "status":"STOP_GEOMETRY_DENSITY_MECHANISM",
             "confirmation_mae_familywise_upper_m":.0123528},
         "result_sha256":"fake-for-format-only",
         "roles":{"calibration":cal,"confirmation":conf},
         "cross_role_analysis":g2.compare_partitions(cal,conf)}
    before=(cal["macro_gain_m"],conf["macro_gain_m"])
    markdown=g2.make_report(obj)
    assert "not a new confirmatory test" in markdown
    assert "STOP_GEOMETRY_DENSITY_MECHANISM" in markdown
    assert "same" not in markdown[:5]
    assert before==(cal["macro_gain_m"],conf["macro_gain_m"])


def test_read_only_contract_no_model_loading_or_new_inference():
    source=inspect.getsource(g2)
    assert ".train(" not in source
    assert "torch.load(" not in source
    assert "model.predict(" not in source
    assert "validationData.csv" not in source
    assert "new_confirmation_performed" in source
    assert "POST" not in g2.PROTOCOL.upper() or "posthoc" in g2.PROTOCOL.lower()
    assert g2.OUTPUT.parent != g2.G1_INPUTS["registry"].parent
    assert g2.G1_INPUTS["confirmation"] not in (g2.OUTPUT, g2.REPORT)


def test_expected_hash_contract_frozen():
    assert g2.EXPECTED_HASHES["confirmation"] == "a54a0a8ff452adf3245bd5145afa52b56f9a2c8eedec650e3e29923b57a9b76c"
    assert g2.SUPPORT_RADIUS_M == 10.
    assert g2.EXPECTED_ROLES == {"calibration":72,"confirmation":79}
    assert g2.METRICS == ("exact_mixture_mean_mae_m","energy_score_2d_m")



def test_original_g1_stop_and_chain_cannot_be_rewritten(monkeypatch):
    # Test the protocol chain with tiny synthetic payloads, no raw data files.
    r={"registry_sha256":g2.EXPECTED_HASHES["registry"],
       "historical_gate_waiver":False,"outer_test_predictions_authorized":False,
       "roles":{"development":{"coordinate_count":151}}}
    c={"result_sha256":g2.EXPECTED_HASHES["calibration"],
       "registry_sha256":r["registry_sha256"],
       "outer_test_predictions":0,"official_validation_predictions":0}
    f={"result_sha256":g2.EXPECTED_HASHES["freeze"],
       "registry_sha256":r["registry_sha256"],
       "calibration_result_sha256":c["result_sha256"],
       "calibration_file_sha256":"calfile","chosen_gamma":.7,
       "outer_test_predictions":0,"official_validation_predictions":0}
    q={"result_sha256":g2.EXPECTED_HASHES["confirmation"],
       "registry_sha256":r["registry_sha256"],"chosen_gamma":.7,
       "calibration_file_sha256":"calfile","freeze_file_sha256":"freezefile",
       "decision":"STOP_GEOMETRY_DENSITY_MECHANISM",
       "gate":{"passed":False},
       "outer_test_predictions":0,"official_validation_predictions":0}
    def fake_sha(path):
        if path == g2.G1_INPUTS["calibration"]:
            return "calfile"
        if path == g2.G1_INPUTS["freeze"]:
            return "freezefile"
        raise AssertionError("unexpected file")
    monkeypatch.setattr(g2,"_sha256_file",fake_sha)
    source={"registry":r,"calibration":c,"freeze":f,"confirmation":q}
    g2.verify_chain(source)
    q["decision"]="GO"
    with pytest.raises(g2.G2Error, match="STOP"):
        g2.verify_chain(source)
    q["decision"]="STOP_GEOMETRY_DENSITY_MECHANISM"
    q["gate"]["passed"]=True
    with pytest.raises(g2.G2Error, match="STOP"):
        g2.verify_chain(source)
    q["gate"]["passed"]=False
    f["calibration_file_sha256"]="tampered"
    with pytest.raises(g2.G2Error, match="byte hash"):
        g2.verify_chain(source)


def test_geometry_associations_are_descriptive_and_return_null_when_constant():
    f=synthetic_records(12)
    for rec in f:
        rec["constant_proxy"]=2
    base=np.arange(12, dtype=float)+2
    gain=np.arange(12, dtype=float)*.1
    out=g2.correlation_diagnostics(f,"constant_proxy",baseline=base,improvement=gain)
    assert out["pearson_with_gain"] is None
    assert out["spearman_with_gain"] is None
    assert out["partial_rank_after_baseline_and_floor_building"] is None
    assert "not causal" in out["meaning"]


def test_report_association_order_survives_sorted_json_roundtrip():
    features = synthetic_records(12)
    base = np.arange(12) + 2.0
    cal = g2.risk_diagnostics(features, base, base - 0.2, base * 0.7, base * 0.7 - 0.1, None)
    conf = g2.risk_diagnostics(features, base, base - 0.3, base * 0.7, base * 0.7 - 0.2, base + 0.1)
    payload = {
        "disclosure": g2.DISCLOSURE,
        "geometry": {"anchors": 595},
        "g1_formal_outcome_unchanged": {
            "gamma": 0.7, "status": "STOP_GEOMETRY_DENSITY_MECHANISM",
            "confirmation_mae_familywise_upper_m": 0.012352838,
        },
        "result_sha256": "synthetic-formatting-fixture",
        "roles": {"calibration": cal, "confirmation": conf},
        "cross_role_analysis": g2.compare_partitions(cal, conf),
    }
    expected = g2.make_report(payload)
    reloaded = json.loads(json.dumps(payload, sort_keys=True))
    assert g2.make_report(reloaded) == expected
    for role in reloaded["roles"].values():
        role["associations"] = dict(reversed(list(role["associations"].items())))
    assert g2.make_report(reloaded) == expected
    assert reloaded["roles"]["confirmation"]["macro_gain_m"] == conf["macro_gain_m"]
