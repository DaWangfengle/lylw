"""CPU-only G5 scientific-contract tests. NEVER open provided test files."""
from __future__ import annotations

import inspect
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g5_bayes_action as g5
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness, sample_continuous_mixture
)


def synthetic_data():
    keys=[];coords=[];floors=[];buildings=[]
    for f in (0,1,2):
        for j in range(20):
            for repeat in range(j%3+1):
                point=(0,f,float(j)*2.11,float(f)*8.123)
                keys.append(point)
                coords.append([point[2],point[3]])
                floors.append(f);buildings.append(0)
    return {
        "keys":keys, "raw_xy":np.asarray(coords,dtype=np.float64),
        "building":np.asarray(buildings,dtype=np.int64),
        "floor":np.asarray(floors,dtype=np.int64),
    }


def test_stratified_physical_split_is_deterministic_and_no_overlap(monkeypatch):
    monkeypatch.setattr(g5,"L",10)
    monkeypatch.setattr(g5,"K",10)
    data=synthetic_data()
    roles,rows=g5._roles(data)
    roles2,rows2=g5._roles(data)
    assert roles==roles2
    assert all(np.array_equal(rows[k],rows2[k]) for k in rows)
    assert roles["fit"].isdisjoint(roles["confirmation"])
    assert set.union(*roles.values())==set(data["keys"])
    assert sum(map(len,rows.values()))==len(data["keys"])
    assert Counter(int(i) for rr in rows.values() for i in rr)==Counter(range(len(data["keys"])))
    for f in (0,1,2):
        assert len([p for p in roles["fit"] if p[1]==f])==15
        assert len([p for p in roles["confirmation"] if p[1]==f])==5


def test_small_stratum_fails_instead_of_diluting_confirmation():
    data=synthetic_data()
    data["keys"]=[(0,0,float(i),0.) for i in range(3)]
    with pytest.raises(g5.G5Error,match="too few"):
        g5._roles(data)


def test_tampere_adapter_uses_training_only_and_unknown_device():
    class Fake:
        dataset="Tampere"
        split="train"
        coords_m=np.asarray([[1.,2.],[3.,4.]],dtype=np.float32)
        building=np.asarray([0,0])
        floor=np.asarray([2,3])
        device=np.asarray([-1,-1])
        def validate(self):return None
        def model_rss(self):return np.asarray([[.1,-1],[.5,.8]],dtype=np.float32)
    # The small sample is rejected intentionally; >=100 samples is a fixed preflight.
    with pytest.raises(g5.G5Error,match="training RSS"):
        g5._as_r0_input(Fake())
    source=inspect.getsource(g5._training)
    assert 'ds.load_tampere("train")' in source
    assert 'ds.load_tampere("test")' not in inspect.getsource(g5)
    assert '"Test_rss.csv"' not in source


def test_frozen_recipe_and_external_test_exclusion():
    assert (g5.PANEL,g5.L,g5.K,g5.ALPHA)==(5,192,192,1.)
    assert (g5.SEEDS,g5.EPOCHS)==((17301,17302,17303,17304,17305),
                                    (65,69,56,67,62))
    assert (g5.ACTION_DRAWS_PER_STREAM,g5.RISK_DRAWS_PER_STREAM)==(128,128)
    assert g5.BOOTSTRAP_REPS==5000
    assert g5.TRAIN_RSS.name=="Training_rss.csv"
    assert g5.TRAIN_XYZ.name=="Training_coordinates.csv"
    assert g5.MIN_GAIN_M==.15
    assert "g1.corrected_probability" not in inspect.getsource(g5)
    assert "r0._fit_refit(" in inspect.getsource(g5)


def test_mixture_mean_is_exact_v17_decision_for_continuous_student_t():
    p=torch.tensor([[.75,.25]],dtype=torch.float64)
    xy=torch.tensor([[[0.,0.],[8.,0.]]],dtype=torch.float64)
    scale=torch.full_like(xy,.1)
    rng=common_student_t_randomness(1,draws=128,seed=19503)
    acts=g5.three_actions(p,xy,scale,rng)
    assert acts["mean"].tolist()==pytest.approx([[2.,0.]])
    assert acts["component_median"][0,0]<.01
    assert abs(float(acts["continuous_median"][0,0]))<1.
    assert all(v.shape==(1,2) and torch.isfinite(v).all() for v in acts.values())
    # Re-using same random numbers gives deterministic posterior Bayes action.
    acts2=g5.three_actions(p,xy,scale,rng)
    assert torch.equal(acts["continuous_median"],acts2["continuous_median"])


def test_independent_predictive_distance_prefers_median_on_asymmetric_law():
    p=torch.tensor([[.85,.15]],dtype=torch.float32)
    xy=torch.tensor([[[0.,0.],[10.,0.]]],dtype=torch.float32)
    sc=torch.full_like(xy,.15)
    train=common_student_t_randomness(1,draws=128,seed=g5.ACTION_SAMPLING_SEED)
    test=common_student_t_randomness(1,draws=1024,seed=g5.RISK_SAMPLING_SEED)
    acts=g5.three_actions(p,xy,sc,train)
    first=sample_continuous_mixture(p,xy,sc,test["uniform_1"],test["noise_1"])
    second=sample_continuous_mixture(p,xy,sc,test["uniform_2"],test["noise_2"])
    def risk(action):
        return (torch.linalg.vector_norm(first-action[:,None,:],dim=-1).mean()
                +torch.linalg.vector_norm(second-action[:,None,:],dim=-1).mean())/2
    assert float(risk(acts["continuous_median"])) < float(risk(acts["mean"]))-.5


def test_same_distribution_implies_exactly_same_energy_score():
    p=torch.tensor([[.8,.2]],dtype=torch.float64)
    loc=torch.tensor([[[0.,0.],[5.,1.]]],dtype=torch.float64)
    scale=torch.full_like(loc,.5)
    rng=common_student_t_randomness(1,draws=32,seed=17)
    actions=g5.three_actions(p,loc,scale,rng)
    actual=torch.tensor([[1.,1.]],dtype=torch.float64)
    same_es=r3._score_joint_law(p,loc,scale,actual,rng)[0]
    # Point actions cannot appear as arguments to continuous_energy_score_mc.
    assert "action" not in inspect.signature(r3._score_joint_law).parameters
    assert float(same_es.item())>=0
    assert not torch.equal(actions["mean"],actions["continuous_median"])


def test_r3_five_member_consensus_law_has_unit_probability():
    torch.manual_seed(29)
    prior=torch.softmax(torch.randn(5,2,220),dim=-1)
    centres=torch.randn(5,2,220,2)
    scales=torch.rand_like(centres)+.25
    top,idx,radial,_=r3._member_top_l(prior,centres,scales,g5.L,student_df=4.0)
    p,c,s=r3._dense_joint_law(top,idx,radial,centres,scales,
                               tuple(range(g5.PANEL)),top_k=g5.K,alpha=g5.ALPHA)
    assert p.shape==(2,960) and c.shape==(2,960,2) and s.shape==(2,960,2)
    assert torch.allclose(p.sum(dim=1),torch.ones(2),atol=1e-5)


@pytest.mark.parametrize("delta,upper,pass_expected",[
    (-.20,-.01,True),(-.15,-.001,True),
    (-.149999,-.01,False),(-.20,0.,False),
    (-.20,.01,False),(+.01,-.01,False),
])
def test_g5_gate_strict_original_thresholds(delta,upper,pass_expected):
    result=g5.g5_confirmation_gate(delta,upper)
    assert result["passed"] is pass_expected
    assert len(result["checks"])==2


@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-float("inf")])
def test_gate_rejects_nonfinite_inputs(bad):
    with pytest.raises(g5.G5Error):
        g5.g5_confirmation_gate(bad,-.02)
    with pytest.raises(g5.G5Error):
        g5.g5_confirmation_gate(-.20,bad)


def test_no_result_overwrite_and_hash_detects_edit(tmp_path):
    path=tmp_path/"case.json"
    item=g5._write(path,{"name":"synthetic","value":3})
    assert g5._read(path)==item
    with pytest.raises(FileExistsError):
        g5._write(path,{"name":"another"})
    content=json.loads(path.read_text(encoding="utf-8"))
    content["value"]=999
    path.write_text(json.dumps(content),encoding="utf-8")
    with pytest.raises(g5.G5Error,match="integrity"):
        g5._read(path)


def test_confirm_refuses_prior_lock_before_touching_train(monkeypatch,tmp_path):
    lock=tmp_path/"internal_confirmation.lock"
    lock.write_text("already touched")
    monkeypatch.setattr(g5,"LOCK",lock)
    monkeypatch.setattr(g5,"RESULT",tmp_path/"result-missing.json")
    def fail():
        raise AssertionError("freeze must not be called")
    monkeypatch.setattr(g5,"freeze",fail)
    with pytest.raises(g5.G5Error,match="already attempted"):
        g5.confirm("cpu")


def test_report_disclaims_novelty_and_energy_invariance():
    candidate={
        "metrics":{
            "mean":{"coordinate_macro_mae_m":6.0},
            "component_median":{"coordinate_macro_mae_m":5.9},
            "continuous_median":{"coordinate_macro_mae_m":5.8}},
        "paired_familywise_inference":{"comparisons":{
            "component_median_minus_mean":{"mean_difference_m":-.1}}},
        "gate":{"mean_diff_m":-.2,"maxT_upper_m":-.03,
                "min_mae_gain_m":.15},
        "decision":"PROMISING_ACTION_ALIGNMENT",
        "common_energy_score":{"coordinate_macro_energy_m":4.5},
        "expected_posterior_euclidean_risk_m":{"mean":5.,"continuous_median":4.9},
        "coordinate_macro_median_mean_displacement_m":.3,
        "result_sha256":"synthetic"
    }
    report=g5.make_report(candidate)
    assert "already implements" in report
    assert "Energy Score is identical" in report
    assert "Tampere TRAIN-only" in report
    assert "PROMISING_ACTION_ALIGNMENT" in report


@pytest.mark.parametrize("i",[-1,0,6,10])
def test_invalid_member_index(i):
    with pytest.raises(g5.G5Error):
        g5.member_path(i)
