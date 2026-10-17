"""Minimal numerical invariants for the MSEP causal pilot.

Run: python -m pytest tests/test_msep_joint_ablation_v1.py -q
"""
import math

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import (
    class_structure, full_joint_loss, mask_self_when_supported,
    neighbor_contamination, probability, same_floor_neighbors,
    spatial_metrics, student_t_logpdf, calibrate_probability,
)


def test_same_floor_neighbors_exclude_cross_floor_even_when_coincident():
    xy = torch.tensor([[0.,0.], [2.,0.], [0.,0.], [3.,0.], [99.,1.]])
    buildings = torch.tensor([0,0,0,0,1])
    floors = torch.tensor([0,0,1,1,0])
    ids, w = same_floor_neighbors(xy, buildings, floors, count=4)
    assert ids.shape == w.shape == (5,4)
    assert torch.allclose(w.sum(1), torch.ones(5))
    assert neighbor_contamination(ids,w,buildings,floors) == 0.0
    assert (ids[4] == 4).all()  # singleton fallback remains in its own class
    assert w[4,0] == 1.0
    assert w[0, 0] > 0


def test_old_cross_floor_label_mass_audited():
    ids = torch.tensor([[1,2],[0,2],[0,1]])
    w = torch.tensor([[.25,.75],[.4,.6],[.5,.5]])
    b = torch.zeros(3,dtype=torch.long)
    f = torch.tensor([0,0,1])
    assert neighbor_contamination(ids,w,b,f) == pytest.approx((.75+.6+1.)/3)


def test_joint_location_nll_penalizes_wrong_floor_even_at_identical_xy():
    anchor_build = torch.tensor([0,0])
    anchor_floor = torch.tensor([0,1])
    pairs, cls = class_structure(anchor_build,anchor_floor)
    centers = torch.zeros(1,2,2)
    scales = torch.ones_like(centers)
    y = torch.zeros(1,2)
    ids = torch.tensor([[0,1]])
    p_good = torch.tensor([[.9,.1]])
    p_bad = torch.tensor([[.1,.9]])
    nll_good = full_joint_loss(p_good,ids,centers,scales,y,
                               torch.tensor([0]),cls,point_weight=0.)
    nll_bad = full_joint_loss(p_bad,ids,centers,scales,y,
                              torch.tensor([0]),cls,point_weight=0.)
    assert (nll_bad-nll_good).item() == pytest.approx(math.log(9.), rel=1e-5)


def test_refit_mask_excludes_own_only_when_class_has_alternative():
    cls = torch.tensor([0,0,1])
    ids = torch.tensor([[0,1,2],[0,1,2]])
    scores = torch.zeros((2,3))
    updated, singleton = mask_self_when_supported(
        scores, ids, torch.tensor([0,2]), cls)
    assert updated[0,0] < -1e8
    assert updated[1,2] == 0
    assert singleton == 1


def test_joint_nll_has_gradient_for_both_logits_and_centers():
    _, cls = class_structure(torch.tensor([0,0,0]), torch.tensor([0,1,0]))
    raw = torch.tensor([[0.4,-0.5,0.1]],requires_grad=True)
    center = torch.tensor([[[1.,0.],[0.,0.],[2.,0.]]],requires_grad=True)
    sigma = torch.ones_like(center)
    q = torch.softmax(raw,dim=1)
    loss = full_joint_loss(q, torch.tensor([[0,1,2]]), center, sigma,
                           torch.tensor([[1.2,0.]]), torch.tensor([0]),
                           cls,point_weight=.3)
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.isfinite(center.grad).all()
    assert raw.grad.abs().sum() > 0
    assert center.grad.abs().sum() > 0


def test_joint_model_metrics_never_use_truth_to_select_class():
    building = torch.tensor([0,0])
    floor = torch.tensor([0,1])
    pairs, cls = class_structure(building,floor)
    q = torch.tensor([[.05,.95]])
    centers = torch.tensor([[[0.,0.],[2.,0.]]])
    y = torch.tensor([[0.,0.]])
    m,_ = spatial_metrics(q,torch.tensor([[0,1]]),centers,y,
                          torch.tensor([0]),torch.tensor([0]),cls,pairs,
                          keys=[(0,0,0.,0.)])
    assert m["macro_bf_accuracy"] == 0.
    assert m["macro_mae_full_m"] == pytest.approx(1.9)
    assert m["macro_mae_predicted_bf_m"] == pytest.approx(2.)


def test_temperature_evidence_optimization_numerically_valid():
    p = {
        "scores": torch.tensor([[1.,0.],[.4,.2],[-.3,.6]],dtype=torch.float32),
        "z": torch.tensor([[[0.,1.],[.2,0.]],[[0.,.5],[.3,0.]],[[.4,0.],[0.,.2]]]),
        "xy": torch.tensor([[[0.,0.],[3.,0.]]]*3),
        "sigma": torch.ones(3,2,2),
        "target": torch.tensor([[0.,0.],[0.,0.],[3.,0.]]),
        "weight": torch.tensor([1/3]*3),
    }
    for mode in ("temperature","g13_fixed","joint"):
        params = calibrate_probability(p, mode=mode, steps=8)
        assert .25 <= params["temperature"] <= 3.0
        assert all(0 <= x <= 1.5 for x in params["theta"])
        q = probability(p["scores"],p["z"],params)
        assert torch.isfinite(q).all()
        assert torch.allclose(q.sum(1),torch.ones(3),atol=1e-6)


def test_invalid_joint_class_support_raises():
    with pytest.raises(ValueError,match="absent"):
        full_joint_loss(torch.tensor([[.5,.5]]),torch.tensor([[0,1]]),
                        torch.zeros(1,2,2),torch.ones(1,2,2),torch.zeros(1,2),
                        torch.tensor([9]),torch.tensor([0,1]))
