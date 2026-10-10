"""G7 CPU scientific controls: pure freeze/gate tests, no target labels opened."""
import inspect
import json

import pytest

from fhm_dapp_tim.fhm_dapp import uji_v17_g7_frozen_transfer as g7


def domain(mae_base, mae_joint, es_base, es_joint, tonly_mae=None):
    # An extremely unfavorable T-only result MUST NOT replace frozen T+s.
    return {"candidate_metrics":{
        g7.BASE:{"mae_2d_m":mae_base,"energy_score_2d_m":es_base},
        g7.JOINT:{"mae_2d_m":mae_joint,"energy_score_2d_m":es_joint},
        g7.TEMP_ONLY:{"mae_2d_m":tonly_mae if tonly_mae is not None else mae_joint,
                       "energy_score_2d_m":es_joint+.01},
        g7.SCALE_ONLY:{"mae_2d_m":mae_base,"energy_score_2d_m":es_base-.01}
    }}


def test_fixed_only_source_policy_and_historic_data():
    assert g7.POLICY=={"policy":"T0.6_S0.5","T":.6,"scale":.5}
    assert g7.DOMAINS==("uts","tampere")
    assert g7.UTS_ROLES==("calibration","confirmation")
    assert g7.TAMPERE_ROLES==("confirmation",)
    assert g7.EXPECTED_CANDIDATES==(
        "T1_S1","T0.6_S0.5","T0.6_S1","T1_S0.5")
    assert g7.RANDOM_STREAM_SEED==g7.g6.SCORE_SEED==19602
    assert g7.MIN_DOMAIN_GAIN_M==.15


def test_positive_two_domain_gate():
    res=g7.transfer_gate({
        "uts":domain(7,6.8,5,4.8),
        "tampere":domain(8,7.6,6,5.7),
    })
    assert res["passed_descriptive_gate"] is True
    assert res["status"]=="TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL"
    assert res["not_independent_blind_evidence"] is True


def test_one_domain_failure_blocks_promotion():
    res=g7.transfer_gate({
        "uts":domain(7,6.7,5,4.8),
        "tampere":domain(8,8.3,6,5.6,tonly_mae=6.0),
    })
    assert not res["passed_descriptive_gate"]
    assert res["status"]=="TRANSFER_NOT_ESTABLISHED"


def test_es_increase_blocks_promotion_despite_better_mae():
    res=g7.transfer_gate({
        "uts":domain(7,6.6,5,5.001),
        "tampere":domain(8,7.5,6,5.7)
    })
    assert res["status"]=="TRANSFER_NOT_ESTABLISHED"
    assert res["checks"]["uts"]["mae_improvement_met"]
    assert not res["checks"]["uts"]["es_nonworsening_met"]


def test_exact_mae_threshold_is_inclusive():
    res=g7.transfer_gate({
        "uts":domain(10,9.85,5,4.9),
        "tampere":domain(10,9.8,5,4.9),
    })
    assert res["passed_descriptive_gate"]


@pytest.mark.parametrize("bad",[float("nan"),float("inf")])
def test_nonfinite_metrics_are_rejected(bad):
    with pytest.raises(g7.G7Error,match="nonfinite"):
        g7.transfer_gate({
            "uts":domain(10,bad,5,4),
            "tampere":domain(10,9,5,4),
        })


def test_no_old_outcome_vectors_are_reused_for_scoring():
    source=inspect.getsource(g7)
    assert "g6._score_role(" in source
    assert 'selected=dict(POLICY)' in source
    assert "_load_historical_models(" in source
    assert "r0._fit_refit" not in source
    assert "train_member(" not in source
    assert "load_uts(\"test\")" not in source
    assert "load_tampere(\"test\")" not in source
    assert "g4._data_roles()" in source
    assert "g5._training()" in source
    assert "g6.CALIBRATION" in source
    assert "g4.CONFIRMATION" not in source
    assert "g5.RESULT" not in source


def test_read_write_never_overwrites_and_detects_mutation(tmp_path):
    path=tmp_path/"thing.json"
    data=g7._save(path,{"research":"pure","role":"historically_seen"})
    assert g7._read(path)==data
    with pytest.raises(FileExistsError):
        g7._save(path,{"research":"altered"})
    data["role"]="new_blind"
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(g7.G7Error,match="integrity"):
        g7._read(path)


def test_one_shot_lock_refusal_occurs_before_preflight(monkeypatch,tmp_path):
    lock=tmp_path/"already.lock"
    lock.write_text("existing")
    monkeypatch.setattr(g7,"LOCK",lock)
    monkeypatch.setattr(g7,"RESULT",tmp_path/"not_existing.json")
    def fail():
        raise AssertionError("prepare cannot run after lock")
    monkeypatch.setattr(g7,"prepare",fail)
    with pytest.raises(g7.G7Error,match="already attempted"):
        g7.run("cpu")


def test_report_has_both_domains_and_retroactive_disclaimer():
    r={"by_domain":{
        "uts":{"scored_roles":{"calibration":domain(6,5,4,3),
                               "confirmation":domain(7,6.8,5,4.8)}},
        "tampere":{"scored_roles":{"confirmation":domain(8,7.6,6,5.7)}}},
       "gate":{"checks":{"uts":{"delta_mae_m":-.2,"delta_es_m":-.2,
                                "mae_improvement_met":True,"es_nonworsening_met":True},
                         "tampere":{"delta_mae_m":-.4,"delta_es_m":-.3,
                                "mae_improvement_met":True,"es_nonworsening_met":True}}},
       "status":"TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL",
       "result_sha256":"fake"}
    report=g7.make_report(r)
    assert "historical" in report.lower()
    assert "not a blind" in report.lower()
    assert "tampere" in report.lower() and "uts" in report.lower()
    assert "T0.6_S1" in report and "T1_S0.5" in report
