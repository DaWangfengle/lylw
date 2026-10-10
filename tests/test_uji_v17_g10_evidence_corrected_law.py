"""G10 CPU mechanistic tests: label-free evidence, proper surrogate, strict CAL fit."""
from __future__ import annotations

import inspect
import json

import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g10_evidence_corrected_law as g10


def synthetic_evidence(n=2,a=194):
    torch.manual_seed(31)
    proto=torch.rand(n,a)
    member_prior=torch.softmax(torch.randn(g10.E,n,a),dim=-1)
    top=torch.arange(g10.L).view(1,1,-1).expand(g10.E,n,-1).clone()
    ids=torch.arange(1,1+g10.K).view(1,-1).expand(n,-1).clone()
    c=torch.softmax(torch.randn(n,g10.K),dim=-1)
    xy=torch.stack((torch.arange(a,dtype=torch.float32)*2.,
                    torch.zeros(a)),dim=-1)
    floors=torch.zeros(a,dtype=torch.long)
    floors[100:]=1
    buildings=torch.zeros(a,dtype=torch.long)
    return proto,member_prior,top,ids,c,xy,buildings,floors


def test_fixed_four_features_and_protocol():
    assert g10.NFEATURES==4
    assert g10.E==5 and g10.L==192 and g10.K==192
    assert g10.LEARN_DOMAINS==("tuji","uts")
    assert g10.DOMAINS==("tuji","uts","tampere")
    assert g10.EPOCHS==24 and g10.T==.6
    assert g10.VARIANTS==("base","anchor_t06","evidence_correction")


def test_features_are_computed_without_labels_and_bounded():
    args=synthetic_evidence()
    feat=g10.label_free_features(*args)
    assert feat.shape==(2,192,4)
    assert torch.isfinite(feat).all()
    assert feat.abs().max()<=3.00001
    # Constant shift in prototype distances per query has no effect.
    shifted=list(args)
    shifted[0]=args[0]+torch.tensor([[3.],[100.]])
    feat2=g10.label_free_features(*shifted)
    assert torch.allclose(feat,feat2,atol=1e-5)
    # Anchor-only features never access query target or true floor.
    assert "true_xy" not in inspect.signature(g10.label_free_features).parameters
    assert "true_floor" not in inspect.signature(g10.label_free_features).parameters


def test_zero_parameters_exactly_recover_fixed_anchor_baseline():
    features=g10.label_free_features(*synthetic_evidence())
    c=synthetic_evidence()[4]
    theta=torch.zeros(g10.NFEATURES)
    q=g10.anchor_posterior(c,features,theta)
    assert torch.allclose(q,torch.softmax(c.log()/g10.T,dim=1),atol=1e-7)
    wm=torch.softmax(torch.randn(2,5),dim=1)
    q_joint=g10.original_joint(wm,q)
    p=g10.original_joint(wm,c)
    assert torch.allclose(q_joint,g10.g8.hierarchical_laws(p,5,192)["anchor"],
                          atol=2e-6,rtol=2e-6)


def test_four_parameters_cannot_exceed_fixed_coefficient_cap():
    t=g10.bounded_parameters(torch.tensor([100.,-100.,0.,2.]))
    assert t.abs().max()<=g10.CAP
    assert t[0].item()==pytest.approx(1.5)
    assert t[1].item()==pytest.approx(-1.5)
    assert t[2].item()==0.


def test_exact_discrete_energy_score_two_point_example():
    q=torch.tensor([[.5,.5]],dtype=torch.float32)
    loc=torch.tensor([[[0.,0.],[2.,0.]]])
    truth=torch.tensor([[0.,0.]])
    # E||X-y||=1; .5 E||X-X'||=.5.
    assert g10.discrete_energy_loss(q,loc,truth).item()==pytest.approx(.5,abs=1e-5)


def test_training_loss_equals_zero_parameter_baseline_surrogate_plus_point():
    n=2;k=192
    torch.manual_seed(16)
    c=torch.softmax(torch.randn(n,k),dim=1)
    loc=torch.randn(n,k,2)*6
    target=torch.randn(n,2)
    q=g10.anchor_posterior(c,torch.randn(n,k,4),torch.zeros(4))
    obj=g10.calibration_objective(q,c,loc,target)
    es=g10.discrete_energy_loss(q,loc,target)
    mae=torch.linalg.vector_norm((q[:,:,None]*loc).sum(1)-target,dim=1)
    assert torch.allclose(obj,(es+g10.POINT_WEIGHT*mae)/g10.SCALE_M,
                          atol=1e-5)


def test_equal_physical_site_and_equal_domain_calibration_weights():
    group=torch.tensor([0,0,0,1,1,2])
    weight=g10._row_weights(group,3)
    assert weight.shape==(6,)
    assert sum(weight[group==0]).item()==pytest.approx(1/6)
    assert sum(weight[group==1]).item()==pytest.approx(1/6)
    assert sum(weight[group==2]).item()==pytest.approx(1/6)
    assert weight.sum().item()==pytest.approx(.5)


def test_one_small_synthetic_calibration_fit_cpu():
    torch.manual_seed(14)
    n=4
    cal={}
    for domain in g10.LEARN_DOMAINS:
        data={
            "features":torch.randn(n,192,4)*.25,
            "c":torch.softmax(torch.randn(n,192),dim=-1),
            "centres":torch.randn(n,192,2)*2,
            "target":torch.randn(n,2).double(),
            "gid":torch.tensor([0,0,1,1]),
            "sites":2,"rows":n,
        }
        cal[domain]=data
    result=g10.fit_once(cal,"cpu")
    assert result["epochs"]==24
    assert len(result["train_loss_by_epoch"])==24
    assert len(result["bounded_coefficients"])==4
    assert all(abs(x)<=1.5 for x in result["bounded_coefficients"])
    assert all(torch.isfinite(torch.tensor(result["train_loss_by_epoch"])))
    assert result["no_confirmation_gradients"] is True


def test_gradient_reaches_coefficient_through_energy_surrogate():
    torch.manual_seed(21)
    theta=torch.zeros(4,requires_grad=True)
    c=torch.softmax(torch.randn(2,192),dim=1)
    feat=torch.randn(2,192,4)
    centres=torch.randn(2,192,2)
    truth=torch.ones(2,2)*4
    q=g10.anchor_posterior(c,feat,g10.bounded_parameters(theta))
    l=g10.calibration_objective(q,c,centres,truth).mean()
    l.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum()>0


def test_original_five_member_joint_and_score_protocol():
    torch.manual_seed(2)
    w=torch.softmax(torch.randn(2,5),dim=1)
    q=torch.softmax(torch.randn(2,192),dim=1)
    joint=g10.original_joint(w,q)
    assert joint.shape==(2,960)
    assert torch.allclose(joint.sum(dim=1),torch.ones(2),atol=1e-6)
    with pytest.raises(g10.G10Error,match="shapes"):
        g10.original_joint(w[:,:-1],q)


def test_full_continuous_es_scoring_and_coordinate_keys_cpu():
    torch.manual_seed(1)
    n=2
    wm=torch.softmax(torch.randn(n,5),dim=1)
    c=torch.softmax(torch.randn(n,192),dim=1)
    features=torch.randn(n,192,4)*.25
    locations=torch.randn(n,960,2)
    scales=torch.ones_like(locations)*.2
    target=torch.zeros(n,2,dtype=torch.float64)
    pack={
        "rows":n,"sites":2,"keys":[[0,0,0,0],[0,0,1,1]],
        "gid":torch.tensor([0,1]),
        "c":c,"wm":wm,"features":features,
        "component_location":locations,"component_scale":scales,
        "target":target,
    }
    out=g10.score_confirmation(pack,[0.]*4,"cpu")
    assert set(out["methods"])==set(g10.VARIANTS)
    for name,info in out["methods"].items():
        assert info["coordinate_macro_mae_m"]>=0
        assert info["coordinate_macro_es_m"]>=0
        assert len(info["coordinate_vector_mae_m"])==2
        assert len(info["coordinate_vector_es_m"])==2
    assert out["methods"]["anchor_t06"]["coordinate_macro_mae_m"]==pytest.approx(
        out["methods"]["evidence_correction"]["coordinate_macro_mae_m"],abs=1e-6)
    assert out["methods"]["anchor_t06"]["coordinate_macro_es_m"]==pytest.approx(
        out["methods"]["evidence_correction"]["coordinate_macro_es_m"],abs=1e-6)


def test_immutable_fit_and_lock_refuse_rerun(tmp_path,monkeypatch):
    p=tmp_path/"fitted.json"
    x=g10._write(p,{"coefficient":[0,0,0,0]})
    assert g10._read(p)==x
    with pytest.raises(FileExistsError):g10._write(p,x)
    x["coefficient"]=[0,0,0,1]
    p.write_text(json.dumps(x),encoding="utf-8")
    with pytest.raises(g10.G10Error,match="hash mismatch"):g10._read(p)
    lock=tmp_path/"lock";lock.write_text("yes")
    monkeypatch.setattr(g10,"LOCK",lock)
    monkeypatch.setattr(g10,"FIT",tmp_path/"nonefit")
    monkeypatch.setattr(g10,"RESULT",tmp_path/"none")
    with pytest.raises(g10.G10Error,match="already attempted"):
        g10.run("cpu")


def test_retrospective_calibration_vs_confirmation_is_code_enforced():
    source=inspect.getsource(g10.run)
    a=source.index('data["rows"]["calibration"]')
    b=source.index("_write(FIT,")
    c=source.index('data["rows"]["confirmation"]')
    assert a<b<c
    assert "fit_once(cal" in source
    assert "train_member(" not in inspect.getsource(g10)
    assert "original_joint(" in inspect.getsource(g10.score_confirmation)
    assert "common_student_t_randomness(" in inspect.getsource(g10.score_confirmation)
    assert "official_test_used" in source
