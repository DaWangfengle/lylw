"""G12 fixed-likelihood science tests: geometry, posterior invariance and rank attribution.

All tests are synthetic CPU-only; none loads historical CAL/CONFIRM outcomes.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g12_action_ranking as g12
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
)


def test_fixed_configuration_and_scientific_scope():
    assert (g12.E,g12.L,g12.K)==(5,192,192)
    assert g12.TEMPERATURE==.6
    assert g12.ACTION_DRAWS==g12.RISK_DRAWS==128
    assert g12.ACTION_SEED!=g12.RISK_SEED
    assert g12.RADIUS==10.
    assert g12.ACTIONS==("mean","component_median","continuous_median")
    assert g12.DOMAINS==("tuji","uts","tampere")


def _simple_law():
    torch.manual_seed(33)
    b=2
    p=torch.softmax(torch.randn(b,960),dim=1)
    xy=torch.randn(b,960,2)*4
    scale=torch.ones_like(xy)*.4
    return p,xy,scale


def test_all_decisions_label_free_and_preserve_baseline_mean():
    p,xy,scale=_simple_law()
    rng=common_student_t_randomness(2,draws=128,degrees_freedom=4,seed=8)
    actions=g12.actions_from_frozen_law(p,xy,scale,rng)
    assert set(actions)==set(g12.ACTIONS)
    assert all(a.shape==(2,2) for a in actions.values())
    assert torch.equal(actions["mean"],(p[:,:,None]*xy).sum(1))
    assert "target" not in inspect.signature(g12.actions_from_frozen_law).parameters
    assert "true_floor" not in inspect.signature(g12.actions_from_frozen_law).parameters


def test_symmetric_frozen_mixture_medians_near_mean():
    p=torch.full((1,960),1/960)
    points=torch.zeros(1,960,2)
    points[:,:480,0]=-2
    points[:,480:,0]=2
    scale=torch.ones_like(points)*.01
    rng=common_student_t_randomness(1,draws=128,degrees_freedom=4,seed=19)
    a=g12.actions_from_frozen_law(p,points,scale,rng)
    assert abs(a["mean"][0,0])<1e-4
    # A symmetric two-point median is not unique; zero is a valid minimizer.
    assert abs(a["component_median"][0,0])<.1
    assert abs(a["continuous_median"][0,1])<.2


def test_component_median_and_continuous_median_distinct_when_scale_matters():
    p=torch.full((1,960),1/960)
    point=torch.zeros((1,960,2))
    point[:,500:,0]=10.
    scale=torch.ones_like(point)
    scale[:,:500]*=.01
    scale[:,500:]*=1.
    rng=common_student_t_randomness(1,draws=128,degrees_freedom=4,seed=14)
    a=g12.actions_from_frozen_law(p,point,scale,rng)
    assert all(torch.isfinite(z).all() for z in a.values())
    # No artificially imposed equality between Dirac centre and continuous actions.
    assert not torch.equal(a["component_median"],a["continuous_median"])


def test_independent_model_risk_evaluates_only_posterior_not_true_location():
    p,xy,scale=_simple_law()
    action_rng=common_student_t_randomness(2,draws=128,degrees_freedom=4,seed=23)
    risk_rng=common_student_t_randomness(2,draws=128,degrees_freedom=4,seed=24)
    actions=g12.actions_from_frozen_law(p,xy,scale,action_rng)
    risk=g12.predicted_point_risk(actions,p,xy,scale,risk_rng)
    assert set(risk)==set(g12.ACTIONS)
    assert all(v.shape==(2,) and torch.isfinite(v).all() for v in risk.values())
    assert "target" not in inspect.signature(g12.predicted_point_risk).parameters


def _synthetic_stage():
    torch.manual_seed(12)
    b=2;a=195
    distance=torch.rand(b,a)
    logits=torch.randn(5,b,a)
    full=torch.softmax(logits-2*distance[None],dim=2)
    ids=torch.arange(192)[None,None,:].expand(5,b,192).clone()
    top=full.gather(2,ids)
    top=top/top.sum(2,keepdim=True)
    rad=torch.ones(5,b)
    # Final stage can have any CONSENSUS top K; identity in this test.
    consensus_ids=torch.arange(192)[None,:].expand(b,192).clone()
    c=torch.softmax(torch.randn(b,192),dim=1)
    return distance,logits,full,top,ids,rad,consensus_ids,c


def test_six_stage_probabilities_shapes_and_mass():
    args=_synthetic_stage()
    maps=g12.ranking_probability_panels(*args)
    assert tuple(maps)==g12.STAGES
    for name,p in maps.items():
        assert p.shape==(2,195),name
        assert torch.isfinite(p).all(),name
        assert torch.all(p>=0)
        assert torch.allclose(p.sum(1),torch.ones(2),atol=1e-6)


def test_fused_equal_and_radial_identical_if_member_radials_equal():
    args=list(_synthetic_stage())
    a=g12.ranking_probability_panels(*args)
    assert torch.allclose(a["fused_equal_members"],a["fused_radial_members"],
                          atol=1e-7)


def test_concentration_does_not_change_top_one():
    p=torch.softmax(torch.randn(4,192),dim=1)
    assert torch.equal(p.argmax(1),p.pow(1/.6).argmax(1))


def test_truth_only_in_retrospective_rank_diagnostic():
    torch.manual_seed(2)
    args=_synthetic_stage()
    maps=g12.ranking_probability_panels(*args)
    xy=torch.stack((torch.arange(195)*25.,torch.zeros(195)),dim=1).float()
    building=torch.zeros(195,dtype=torch.long)
    floor=torch.zeros(195,dtype=torch.long)
    floor[0]=1
    true_xy=torch.tensor([[0.,0.],[50.,0.]])
    tb=torch.zeros(2,dtype=torch.long)
    tf=torch.zeros(2,dtype=torch.long)
    x=g12.ranking_label_diagnostics(maps,xy,building,floor,true_xy,tb,tf)
    assert x["near_reference_exists"].shape==(2,)
    assert x["near_reference_exists"].tolist()==[0.,1.]
    for stage in g12.STAGES:
        assert x[stage+"_top1_in10m"].shape==(2,)
        assert x[stage+"_near10_mass"].shape==(2,)
        assert x[stage+"_best_near_rank"].shape==(2,)
        assert torch.isnan(x[stage+"_best_near_rank"][0])
    assert "true_xy" not in inspect.signature(g12.ranking_probability_panels).parameters


def test_rank_near_mass_respects_correct_floor_even_with_same_coordinate():
    a=192
    p=torch.full((1,a),1/a)
    maps={name:p for name in g12.STAGES}
    xy=torch.zeros(a,2)
    floor=torch.ones(a,dtype=torch.long)
    floor[0]=0
    valid=g12.ranking_label_diagnostics(
        maps,xy,torch.zeros(a,dtype=torch.long),floor,
        torch.zeros(1,2),torch.zeros(1,dtype=torch.long),
        torch.zeros(1,dtype=torch.long))
    assert valid["R3_consensus_K192_near10_mass"].item()==pytest.approx(1/192)
    assert valid["near_reference_exists"].item()==1.


def test_site_macro_and_null_rank_aggregation():
    g=torch.tensor([0,0,1,1])
    v=torch.tensor([1.,float("nan"),3.,5.])
    assert g12._site_vector(v,g,2)==[1.,4.]
    assert g12._site_vector(torch.full((4,),float("nan")),g,2)==[None,None]
    metrics=g12._summary({"mean_mae_m":[1.,3.,8.]})
    assert metrics["mean_mae_m"]["site_mean"]==4
    assert metrics["mean_mae_m"]["max_m"]==8


def test_input_shape_fail_closed():
    args=list(_synthetic_stage())
    args[3]=args[3][:,:-1,:]
    with pytest.raises(g12.G12Error,match="mismatched"):
        g12.ranking_probability_panels(*args)
    p,xy,scale=_simple_law()
    with pytest.raises(g12.G12Error,match="invalid"):
        g12.actions_from_frozen_law(p,xy,scale*0,
            common_student_t_randomness(2,draws=128))


def test_action_risk_not_mislabeled_energy_score():
    text=inspect.getsource(g12)
    assert "shared_student_t_es_m" in text
    assert "mean_model_risk_m" in text or 'name+"_model_risk_m"' in text
    assert "g8.hierarchical_laws(joint,E,K)" in text
    assert '["anchor"]' in text
    assert "g5.weighted_component_geometric_median(" in text
    assert "spatial_median_from_draws(" in text


def test_old_source_untouched_and_no_model_learning():
    source=inspect.getsource(g12)
    assert "r2b._prototype_distance(" in source
    assert "r3._member_top_l(" in source
    assert "r3._dense_joint_law(" in source
    assert "r3.dense_alpha_consensus_prior(" in source
    assert "g8._load_models(" in source
    assert "optimizer.step()" not in source
    assert "train_member(" not in source
    assert "official_test_opened" in source


def test_frozen_result_cannot_be_overwritten(tmp_path):
    path=tmp_path/"frozen.json"
    x=g12._write(path,{"data":"immutable"})
    assert g12._read(path)==x
    with pytest.raises(FileExistsError):
        g12._write(path,{"data":"changed"})
    x["data"]="tampered"
    path.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(g12.G12Error,match="hash changed"):
        g12._read(path)


def test_lock_prohibits_second_run(monkeypatch,tmp_path):
    lock=tmp_path/"run.lock"
    lock.write_text("yes")
    monkeypatch.setattr(g12,"LOCK",lock)
    monkeypatch.setattr(g12,"RESULT",tmp_path/"none")
    with pytest.raises(g12.G12Error,match="already attempted"):
        g12.run("cpu")


def test_rng_seed_and_sampling_are_frozen_for_three_domains():
    assert (g12.ACTION_SEED,g12.RISK_SEED)==(19801,19802)
    assert g12.PREDICTIVE_ES_SEED==g12.g8.SEED
    assert g12.MEDIAN_ITERATIONS==64
    assert g12.MEDIAN_TOL==1e-5
