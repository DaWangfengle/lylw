"""G13 scientific CPU invariants; no private data, no GPUs, no historical labels."""
from __future__ import annotations

import inspect
import json

import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g13_multiscan_observation as g13


def toy_fit():
    # Four repeated RSS observations at each of two FIT reference anchors.
    fit=torch.tensor([
        [.20,.11],[.22,.12],[.18,.10],[.21,.13],
        [-1.,-.45],[-1.,-.40],[-1.,-.43],[-1.,-.42]
    ])
    ids=torch.tensor([0,0,0,0,1,1,1,1],dtype=torch.long)
    seen=torch.tensor([True,True])
    return fit,ids,seen


def test_study_excludes_tampere_and_reserves_uji():
    assert g13.DOMAINS==("tuji","uts")
    assert "tampere" not in g13.DOMAINS
    assert g13.ROLES==("calibration","confirmation")
    assert g13.VARIANTS==("base","anchor_t06","visibility","strength","joint")
    assert (g13.E,g13.L,g13.K)==(5,192,192)


def test_fit_only_detection_and_intensity_are_different_channels():
    fit,ids,seen=toy_fit()
    mem=g13.build_fit_observation_memory(fit,ids,2,seen)
    assert mem["fit_rows_per_anchor"].tolist()==[4.,4.]
    assert mem["model_ap_count"]==2
    # First AP always detected at A, absent at B, Beta-shrunk from global.
    p=mem["detection_probability"]
    assert 0.<p[1,0]<.5<p[0,0]<1.
    assert mem["conditional_mean"][0,1] > mem["conditional_mean"][1,1]
    assert torch.isfinite(mem["conditional_variance"]).all()
    assert (mem["conditional_variance"]>=g13.SIGMA_FLOOR**2).all()


def test_query_evidence_favors_matching_fit_anchor():
    fit,ids,seen=toy_fit()
    mem=g13.build_fit_observation_memory(fit,ids,2,seen)
    q=torch.tensor([[.21,.115],[-1.,-.425]])
    anchor=torch.tensor([[0,1],[0,1]])
    feat=g13.observation_features(q,anchor,seen,mem)
    assert feat.shape==(2,2,2)
    assert feat[0,0,0]>feat[0,1,0] # AP detection likelihood
    assert feat[0,0,1]>feat[0,1,1] # conditional RSS likelihood
    assert feat[1,1,0]>feat[1,0,0] # AP absence likelihood
    assert torch.isfinite(feat).all()


def test_observed_negative_one_is_missing_not_strength_measurement():
    fit,ids,seen=toy_fit()
    mem=g13.build_fit_observation_memory(fit,ids,2,seen)
    ap_ids=torch.tensor([[0,1]])
    q1=torch.tensor([[-1.,-.425]])
    feature=g13.observation_features(q1,ap_ids,seen,mem)
    assert torch.isfinite(feature).all()
    assert feature[0,1,0]>feature[0,0,0]


def test_single_scan_never_collapses_rss_variance_to_zero():
    fit,ids,seen=toy_fit()
    one=g13.build_fit_observation_memory(fit[[0,4]],torch.tensor([0,1]),2,seen)
    assert (one["conditional_variance"]>=g13.SIGMA_FLOOR**2).all()
    assert torch.isfinite(one["conditional_mean"]).all()


def test_never_overwrite_original_r3_anchors_at_zero_coefficients():
    torch.manual_seed(14)
    w=torch.softmax(torch.randn(3,5),dim=1)
    c=torch.softmax(torch.randn(3,192),dim=1)
    p=(w[:,:,None]*c[:,None,:]).reshape(3,960)
    features=torch.randn(3,192,2)
    pack={"component_p":p,"features":features}
    zeros=torch.zeros(2)
    policies=g13._variant_probabilities(pack,0,3,zeros,torch.device("cpu"))
    assert torch.equal(policies["base"],p)
    assert torch.allclose(policies["joint"],policies["anchor_t06"],atol=3e-6)
    assert torch.allclose(policies["visibility"],policies["anchor_t06"],atol=3e-6)
    assert torch.allclose(policies["strength"],policies["anchor_t06"],atol=3e-6)
    for v in policies.values():
        assert torch.isfinite(v).all()
        assert torch.allclose(v.sum(1),torch.ones(3),atol=1e-6)


def test_joint_reweighted_law_preserves_original_member_marginal():
    torch.manual_seed(18)
    w=torch.softmax(torch.randn(2,5),dim=1)
    c=torch.softmax(torch.randn(2,192),dim=1)
    p=(w[:,:,None]*c[:,None,:]).reshape(2,960)
    pack={"component_p":p,"features":torch.randn(2,192,2)}
    q=g13._variant_probabilities(
        pack,0,2,torch.tensor([.7,.5]),torch.device("cpu"))
    for variant in g13.VARIANTS:
        margin=q[variant].reshape(2,5,192).sum(2)
        assert torch.allclose(margin,w,atol=1e-6)


def test_no_observed_APs_produce_neutral_strength_evidence():
    fit,ids,seen=toy_fit()
    mem=g13.build_fit_observation_memory(fit,ids,2,seen)
    f=g13.observation_features(
        torch.tensor([[-1.,-1.]]),torch.tensor([[0,1]]),seen,mem)
    assert torch.allclose(f[0,:,1],torch.zeros(2))


def test_invalid_AP_mapping_and_probability_fail_explicitly():
    fit,ids,seen=toy_fit()
    with pytest.raises(g13.G13Error,match="invalid FIT"):
        g13.build_fit_observation_memory(fit,ids,2,torch.tensor([False,False]))
    mem=g13.build_fit_observation_memory(fit,ids,2,seen)
    with pytest.raises(g13.G13Error,match="candidate IDs"):
        g13.observation_features(fit[[0]],torch.tensor([[0,2]]),seen,mem)
    with pytest.raises(g13.G13Error,match="coefficients"):
        g13.probability(torch.ones(1,2)/2,torch.zeros(1,2,2),
                        torch.tensor([3.,0.]))


def test_training_only_two_coefficients_and_zero_data_leak():
    torch.manual_seed(9)
    # Synthetic two-site evidence, original anchor posterior deliberately flat.
    p={
        "c":torch.tensor([[.5,.5],[.5,.5]]),
        "features":torch.tensor([[[1.,1.],[-1.,-1.]],
                                 [[-1.,-1.],[1.,1.]]]),
        "centres":torch.tensor([[[0.,0.],[10.,0.]],
                                [[0.,0.],[10.,0.]]]),
        "target":torch.tensor([[0.,0.],[10.,0.]]),
        "gid":torch.tensor([0,1]),
        "sites":2,"rows":2,
    }
    fitted=g13.fit({"tuji":p,"uts":p},"cpu")
    q=fitted["coefficients"]
    assert 0<=q["visibility"]<=g13.COEFF_CAP
    assert 0<=q["strength"]<=g13.COEFF_CAP
    assert len(fitted["train_loss_by_epoch"])==g13.EPOCHS
    assert fitted["no_confirmation_fitting"] is True


def test_immutable_artifacts_and_one_shot_run_guard(tmp_path,monkeypatch):
    f=tmp_path/"a.json"
    record=g13._write(f,{"scientific_status":"FIT-only"})
    assert g13._read(f)==record
    with pytest.raises(FileExistsError):g13._write(f,{"retry":True})
    record["scientific_status"]="bad"
    f.write_text(json.dumps(record),encoding="utf-8")
    with pytest.raises(g13.G13Error,match="hash mismatch"):g13._read(f)
    lock=tmp_path/"already.lock";lock.write_text("yes")
    monkeypatch.setattr(g13,"LOCK",lock)
    monkeypatch.setattr(g13,"RESULT",tmp_path/"notresult")
    monkeypatch.setattr(g13,"COEFFICIENTS",tmp_path/"notcoeff")
    with pytest.raises(g13.G13Error,match="already locked"):
        g13.run("cpu")


def test_source_does_not_fit_or_evaluate_tampere():
    s=inspect.getsource(g13)
    assert 'DOMAINS=("tuji","uts")' in s
    assert "g11.map_fit_rows_to_model_anchors" in s
    assert 'p["rows"]["fit"]' in s
    assert 'p["rows"]["calibration"]' in s
    assert 'p["rows"]["confirmation"]' in s
    assert "r3._score_joint_law" in s
    assert "_fit_refit(" not in s
    assert "train_member(" not in s
    assert "load_tampere(" not in s
    assert "load_uts(\"test\")" not in s
    assert "g8.RESULT" not in s
    assert "g10.RESULT" not in s
