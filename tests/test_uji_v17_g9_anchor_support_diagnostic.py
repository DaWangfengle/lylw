"""G9 focused CPU tests: geometry != retrieval != posterior mass."""
from __future__ import annotations

import inspect
import json

import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g9_anchor_support_diagnostic as g9


def toy():
    # Anchor 0 is the ONLY same-floor dictionary anchor inside 5m.
    # Anchor 193 is at identical XY but on the WRONG FLOOR.
    a=194
    xy=torch.full((a,2),100.,dtype=torch.float32)
    xy[0]=torch.tensor([0.,0.])
    xy[1]=torch.tensor([8.,0.])
    xy[3]=torch.tensor([100.,0.])
    xy[193]=torch.tensor([0.,0.])
    building=torch.zeros(a,dtype=torch.long)
    floor=torch.zeros(a,dtype=torch.long)
    floor[193]=1
    target=torch.tensor([[0.,0.]],dtype=torch.float32)
    b=torch.zeros(1,dtype=torch.long)
    f=torch.zeros(1,dtype=torch.long)
    # Every member Top-L has 0..191. Final Top-K instead selects
    # 1..192: correct neighbour 0 dies in the CONSENSUS stage.
    top=torch.arange(192).reshape(1,1,192).expand(5,1,192).clone()
    selected=torch.arange(1,193).reshape(1,192)
    prob=torch.ones(1,192)/192.
    return xy,building,floor,target,b,f,top,selected,prob


def test_three_distinct_support_stages_and_floor_awareness():
    x=g9.stage_metrics(*toy())
    assert x["nearest_fit_m"].item()==0.
    assert x["nearest_member_union_m"].item()==0.
    assert x["nearest_consensus_m"].item()==pytest.approx(8.)
    assert x["r5_fit_exists"].item()==1.
    assert x["r5_union_exists"].item()==1.
    assert x["r5_consensus_exists"].item()==0.
    assert x["r5_consensus_and_union_exists"].item()==0.
    assert x["r5_base_mass"].item()==0.
    assert x["r5_anchor_t_mass"].item()==0.
    assert x["r10_consensus_exists"].item()==1.
    assert x["r10_consensus_and_union_exists"].item()==1.
    assert x["r10_base_mass"].item()==pytest.approx(1/192.,rel=1e-5)
    assert x["consensus_selected_outside_union_count"].item()==1.
    assert x["consensus_top1_matches_floor"].item()==1.
    assert x["consensus_top1_anchor_distance_m"].item()==8.


def test_same_xy_wrong_floor_must_not_count_as_support():
    xy,build,floor,target,tb,tf,top,selected,p=toy()
    selected=selected.clone()
    selected[0,0]=193
    # Site anchor 0 and final consensus wrong-floor anchor 193 overlap in 2D.
    # No same-floor candidate <5m in the final selected list.
    ans=g9.stage_metrics(xy,build,floor,target,tb,tf,top,selected,p)
    assert ans["r5_fit_exists"].item()==1.
    assert ans["r5_consensus_exists"].item()==0.
    assert ans["consensus_top1_matches_floor"].item()==0.
    assert torch.isinf(ans["consensus_top1_anchor_distance_m"]).item()


def test_learned_residual_centres_are_separate_from_anchor_dictionary():
    inp=toy()
    loc=torch.full((1,5*192,2),80.)
    # Selected anchor index 3 is 100m away, but first member's learned
    # continuous residual centre is at the true coordinate.
    loc[0,1,:]=0.
    x=g9.stage_metrics(*inp,component_location=loc)
    assert x["r5_consensus_exists"].item()==0.
    assert x["r5_residual_centre_exists"].item()==1.
    assert x["nearest_residual_centre_m"].item()==0.


def test_probability_mass_is_not_a_binary_coverage_flag():
    xy,building,floor,target,tb,tf,top,sel,p=toy()
    sel=sel.clone();sel[0,0]=0
    p=torch.full((1,192),.5/191)
    p[0,0]=.5
    x=g9.stage_metrics(xy,building,floor,target,tb,tf,top,sel,p)
    assert x["r5_consensus_exists"].item()==1.
    assert x["r5_base_mass"].item()==pytest.approx(.5,rel=1e-6)
    assert x["r5_anchor_t_mass"].item()>.5
    assert x["r5_top1_inside"].item()==1.


def test_no_correct_floor_in_fit_is_not_an_infinite_json_number():
    xy,building,floor,target,tb,tf,top,sel,p=toy()
    tf[0]=3
    ans=g9.stage_metrics(xy,building,floor,target,tb,tf,top,sel,p)
    assert ans["same_floor_fit_exists"].item()==0.
    assert torch.isinf(ans["nearest_fit_m"]).item()
    v=g9._mean_vector(ans["nearest_fit_m"],torch.tensor([0]),1)
    assert v==[None]
    assert g9._mean_vector(ans["r5_fit_exists"],torch.tensor([0]),1)==[0.]


def test_partial_missing_row_keeps_finite_site_distances():
    x=torch.tensor([float("inf"),4.,8.,float("inf")],dtype=torch.float64)
    site=g9._mean_vector(x,torch.tensor([0,0,1,1]),2)
    assert site==[4.,8.]


def test_conditional_recall_uses_joint_hits_not_site_minima():
    site={
        "r5_fit_exists":[1.,1.],
        "r5_union_exists":[.5,1.],
        "r5_consensus_exists":[.5,.5],
        "r5_consensus_and_union_exists":[0.,.5],
        "r10_fit_exists":[1.,1.],
        "r10_union_exists":[.5,1.],
        "r10_consensus_exists":[.5,.5],
        "r10_consensus_and_union_exists":[0.,.5]
    }
    stats=g9._summary(site,2)
    assert stats["r5_retrieval_recall_given_fit"]==pytest.approx(.75)
    assert stats["r5_consensus_recall_given_fit"]==pytest.approx(.5)
    assert stats["r5_consensus_recall_given_union"]==pytest.approx(1/3.)
    assert stats["r10_consensus_recall_given_union"]==pytest.approx(1/3.)


def test_predeclared_fixed_domains_and_radii():
    assert g9.DOMAINS==("tuji","uts","tampere")
    assert g9.RADII_M==(5.,10.)
    assert (g9.T,g9.E,g9.L,g9.K)==(.6,5,192,192)
    assert g9.BATCH==16
    assert g9.ROLES["tuji"]==("calibration","confirmation")


def test_stage_hard_fails_on_posterior_normalization():
    a=list(toy());p=a[-1].clone();p*=.9;a[-1]=p
    with pytest.raises(g9.G9Error,match="invalid consensus"):
        g9.stage_metrics(*a)


def test_conservation_of_original_r3_consensus_marginal_on_cpu():
    # Original R3 dense joint has exact member x shared-consensus structure.
    torch.manual_seed(902)
    probs=torch.softmax(torch.randn(5,1,194),dim=2)
    centres=torch.randn(5,1,194,2)
    sigma=.2+torch.rand_like(centres)
    top,ix,rad,_=g9.r3._member_top_l(probs,centres,sigma,192,student_df=4.)
    p,loc,scale=g9.r3._dense_joint_law(
        top,ix,rad,centres,sigma,tuple(range(5)),top_k=192,alpha=1.)
    mw=rad.square().clamp_min(1e-8).reciprocal()
    mw=mw/mw.sum(dim=0,keepdim=True)
    idx,c=g9.r3.dense_alpha_consensus_prior(
        top,ix,mw,194,192,epsilon=g9.r3.CONSENSUS_EPSILON,alpha=1.)
    assert p.shape==(1,960) and loc.shape==(1,960,2)
    assert torch.allclose(p.reshape(1,5,192).sum(1),c,atol=1e-6)
    gather=idx[:,None,:,None].expand(1,5,192,2)
    expected=torch.gather(centres.permute(1,0,2,3),2,gather).reshape(1,960,2)
    assert torch.allclose(expected,loc)


def test_immutable_plan_and_run_guard(tmp_path,monkeypatch):
    file=tmp_path/"snapshot.json"
    item=g9._write(file,{"reused":True})
    assert g9._read(file)==item
    with pytest.raises(FileExistsError):g9._write(file,{"reused":False})
    item["reused"]=False;file.write_text(json.dumps(item))
    with pytest.raises(g9.G9Error,match="self hash"):g9._read(file)
    l=tmp_path/"lock";l.write_text("yes")
    monkeypatch.setattr(g9,"LOCK",l)
    monkeypatch.setattr(g9,"RESULT",tmp_path/"none")
    with pytest.raises(g9.G9Error,match="already locked"):
        g9.run("cpu")


def test_train_only_inference_no_old_confirm_outcomes():
    code=inspect.getsource(g9)
    assert "g8._pack(domain)" in code
    assert "g8._load_models(" in code
    assert "r3._dense_joint_law(" in code
    assert "r3.dense_alpha_consensus_prior(" in code
    assert "train_member(" not in code
    assert "g8.RESULT" not in code
    assert "g7.RESULT" not in code
    assert "confirmation_v17" not in code
    assert "load_uts(\"test\")" not in code
    assert "load_tampere(\"test\")" not in code
