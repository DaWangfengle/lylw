"""Small G8 CPU scientific tests: exact factorization and two variance laws."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g8_hierarchical_attribution as g8


def _factorized(n=4,e=5,k=192,dtype=torch.float32):
    torch.manual_seed(808)
    w=torch.softmax(torch.randn(n,e,dtype=dtype),dim=-1)
    c=torch.softmax(torch.randn(n,k,dtype=dtype),dim=-1)
    return w,c,(w[:,:,None]*c[:,None,:]).reshape(n,e*k)


def test_four_fixed_variants_only():
    assert g8.DOMAINS==("tuji","uts","tampere")
    assert g8.VARIANTS==("base","member","anchor","both")
    assert g8.T==0.6 and g8.E==5 and g8.K==192
    assert g8.SEED==19602
    assert g8.ROLES["tampere"]==("confirmation",)
    assert g8.ROLES["uts"]==("calibration","confirmation")


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_joint_factorization_matches_original_g6_temperature(dtype):
    w,c,p=_factorized(dtype=dtype)
    variants=g8.hierarchical_laws(p)
    assert variants["base"] is p
    assert set(variants)==set(g8.VARIANTS)
    for label in g8.VARIANTS:
        assert variants[label].shape==p.shape
        assert variants[label].dtype==dtype
        assert torch.allclose(variants[label].sum(dim=-1),
                              torch.ones(len(p),dtype=dtype),atol=1e-6)
    expected=g8.g6.calibrate_joint_law(
        p,torch.zeros(len(p),960,2),torch.ones(len(p),960,2),
        0.6,1.0,
    )[0]
    assert torch.allclose(variants["both"],expected,atol=1e-7,rtol=1e-7)
    only_member=variants["member"].reshape(len(p),5,192)
    only_anchor=variants["anchor"].reshape(len(p),5,192)
    assert torch.allclose(only_member.sum(dim=1),c,atol=1e-6)
    assert torch.allclose(only_anchor.sum(dim=2),w,atol=1e-6)
    assert not torch.allclose(variants["member"],p)
    assert not torch.allclose(variants["anchor"],p)


def test_member_and_anchor_commute_at_fixed_temperature():
    w,c,p=_factorized(dtype=torch.float64)
    candidates=g8.hierarchical_laws(p)
    direct=(g8._tempered(w,.6)[:,:,None]*
            g8._tempered(c,.6)[:,None,:]).reshape(len(p),-1)
    assert torch.allclose(candidates["both"],direct,atol=1e-12,rtol=1e-12)
    # The global temperature is the two-level intervention, not a
    # separately tuned third parameter.
    assert torch.allclose(direct,g8.g6.calibrate_joint_law(
        p,torch.zeros(len(p),960,2,dtype=p.dtype),
        torch.ones(len(p),960,2,dtype=p.dtype),.6,1.
    )[0],atol=1e-12,rtol=1e-12)


def test_independent_product_violation_is_a_hard_stop():
    w,c,p=_factorized(n=2,e=5,k=192)
    v=p.clone().reshape(2,5,192)
    delta=float(min(v[0,0,0],v[0,0,1]))*0.5
    v[0,0,0]+=delta
    v[0,0,1]-=delta
    assert (v>=0).all()
    with pytest.raises(g8.G8Error,match="factorization"):
        g8.hierarchical_laws(v.reshape(2,-1))


def test_shape_and_negative_probability_stop():
    _,_,p=_factorized(n=1)
    with pytest.raises(g8.G8Error,match="E\\*K"):
        g8.hierarchical_laws(p[:,:-1])
    bad=p.clone();bad[0,0]=-0.1
    with pytest.raises(g8.G8Error,match="invalid"):
        g8.hierarchical_laws(bad)


def test_two_exact_nested_variance_decompositions():
    torch.manual_seed(21)
    w,c,q=_factorized(n=3,e=5,k=192,dtype=torch.float64)
    loc=torch.randn(3,960,2,dtype=torch.float64)*8
    scale=0.2+torch.rand_like(loc)
    item=g8.decompose_trace(q,loc,scale)
    assert all(v.shape==(3,) for v in item.values())
    inner=item["within_student_t_trace_m2"]
    outer=item["between_centres_trace_m2"]
    assert torch.allclose(inner+outer,item["total_predictive_trace_m2"])
    assert torch.allclose(
        outer,item["anchor_between_given_member_average_m2"]
              +item["member_within_anchor_m2"],atol=1e-9)
    assert torch.allclose(
        outer,item["member_between_given_anchor_average_m2"]
              +item["anchor_within_member_m2"],atol=1e-9)
    assert (inner>0).all() and (outer>0).all()


def test_actual_r3_dense_joint_factorization_cpu():
    # Use the real frozen R3 assembly function rather than relying only
    # on synthetic q=w*c construction.
    torch.manual_seed(190)
    member_prior=torch.softmax(torch.randn(5,2,210),dim=-1)
    full_location=torch.randn(5,2,210,2)
    full_scale=0.2+torch.rand_like(full_location)
    top,ix,radial,_=g8.r3._member_top_l(
        member_prior,full_location,full_scale,192,student_df=4.)
    joint,xy,sig=g8.r3._dense_joint_law(
        top,ix,radial,full_location,full_scale,
        tuple(range(5)),top_k=192,alpha=1.)
    assert joint.shape==(2,960)
    assert xy.shape==sig.shape==(2,960,2)
    methods=g8.hierarchical_laws(joint)
    assert methods["base"] is joint
    assert torch.isfinite(methods["both"]).all()
    # Disentangling the probability transform must never move component
    # locations or change conditional Student-t scales.
    expected=g8.g6.calibrate_joint_law(joint,xy,sig,.6,1.)[0]
    assert torch.allclose(methods["both"],expected,atol=1e-6)


def test_analytic_two_mode_variance_and_scale():
    # Only anchor variation, no member variation. Different nesting
    # identities must each recover precisely Var(anchor centroid).
    w=torch.ones(1,2,dtype=torch.float64)/2
    c=torch.tensor([[.25,.75]],dtype=torch.float64)
    p=(w[:,:,None]*c[:,None,:]).reshape(1,4)
    loc=torch.tensor([[[0.,0.],[4.,0.],[0.,0.],[4.,0.]]])
    scale=torch.ones_like(loc)*.5
    diag=g8.decompose_trace(p,loc,scale,members=2,anchors=2)
    assert diag["between_centres_trace_m2"].item()==pytest.approx(3.0)
    assert diag["within_student_t_trace_m2"].item()==pytest.approx(1.)
    assert diag["total_predictive_trace_m2"].item()==pytest.approx(4.)
    assert diag["anchor_between_given_member_average_m2"].item()==pytest.approx(3.)
    assert diag["member_between_given_anchor_average_m2"].item()==pytest.approx(0.)


def test_factorial_interaction_is_not_assumed_zero():
    m={
        "base":{"mae_2d_m":10.,"energy_score_2d_m":7.},
        "member":{"mae_2d_m":9.,"energy_score_2d_m":6.5},
        "anchor":{"mae_2d_m":8.,"energy_score_2d_m":5.},
        "both":{"mae_2d_m":6.,"energy_score_2d_m":4.},
    }
    r=g8.factorial_effects(m)
    assert r["mae_2d_m"]["member_delta"]==-1.
    assert r["mae_2d_m"]["anchor_delta"]==-2.
    assert r["mae_2d_m"]["both_delta"]==-4.
    assert r["mae_2d_m"]["factorial_interaction"]==-1.
    assert r["mae_2d_m"]["anchor_fraction_of_both"]==.5


def test_trace_differs_with_member_dependent_centres():
    w=torch.tensor([[.4,.6]],dtype=torch.float64)
    c=torch.tensor([[.3,.7]],dtype=torch.float64)
    p=(w[:,:,None]*c[:,None,:]).reshape(1,4)
    loc=torch.tensor([[[0.,0.],[4.,0.],[1.,0.],[9.,0.]]])
    scale=torch.ones_like(loc)
    d=g8.decompose_trace(p,loc,scale,members=2,anchors=2)
    assert not torch.allclose(
        d["anchor_between_given_member_average_m2"],
        d["member_between_given_anchor_average_m2"])
    assert torch.allclose(
        d["between_centres_trace_m2"],
        d["anchor_between_given_member_average_m2"]+
        d["member_within_anchor_m2"])


def test_immutable_records_and_lock(tmp_path,monkeypatch):
    p=tmp_path/"evidence.json"
    x=g8._write(p,{"fixed_T":.6,"role":"old_confirm"})
    assert g8._read(p)==x
    with pytest.raises(FileExistsError):g8._write(p,x)
    x["fixed_T"]=.75;p.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(g8.G8Error,match="hash changed"):g8._read(p)
    lock=tmp_path/"LOCK"
    lock.write_text("already created",encoding="utf-8")
    monkeypatch.setattr(g8,"LOCK",lock)
    monkeypatch.setattr(g8,"RESULT",tmp_path/"missing")
    with pytest.raises(g8.G8Error,match="already attempted"):
        g8.run("cpu")


def test_no_retraining_no_provided_tests_no_old_result_read():
    source=inspect.getsource(g8)
    assert "g7._data_and_old_protocol(" in source
    assert "g6._training()" in source
    assert "g6._score_role" not in source
    assert "r3._score_joint_law" in source
    assert "r3._dense_joint_law" in source
    assert "g5._load_models" not in source
    assert "_fit_refit(" not in source
    assert "train_member(" not in source
    assert 'load_uts("test")' not in source
    assert 'load_tampere("test")' not in source
    assert "g7.RESULT" not in source
    assert "g5.RESULT" not in source
    assert "g4.CONFIRMATION" not in source


def test_report_is_deterministic_through_json_round_trip():
    entry={
        "candidate_metrics":{v:{"mae_2d_m":i+3.5,
                                 "energy_score_2d_m":i+2.2}
                             for i,v in enumerate(g8.VARIANTS)},
        "factorial_effects":{
            label:{"member_delta":-.1,"anchor_delta":-.2,
                   "both_delta":-.3,"factorial_interaction":0.}
            for label in ("mae_2d_m","energy_score_2d_m")},
    }
    j={
        "domains":{d:{"roles":{role:entry for role in g8.ROLES[d]}}
                   for d in g8.DOMAINS},
        "result_sha256":"synthetic"
    }
    assert g8._overview(j)==g8._overview(json.loads(json.dumps(j,sort_keys=True)))
    report=g8._overview(j)
    assert "No training or new blind confirmation." in report
    assert "already-viewed" in report
    assert "No p-values or model promotion." in report
