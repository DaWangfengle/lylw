"""Safety and mathematical identity tests; no data or GPU required."""
import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_single_posterior import (
    UnifiedAnchorPosterior, anchor_site_keys, energy_score_mc,
    mixture_loss, posterior, reliability_features, site_metrics,
    split_physical_sites,
)


def test_physical_split_keeps_all_scans_together():
    coords = np.asarray([[float(i), i * .5] for i in range(80) for _ in range(3)], dtype=np.float32)
    building = np.zeros(len(coords), dtype=np.int64)
    floor = np.zeros(len(coords), dtype=np.int64)
    keys = anchor_site_keys(coords, building, floor)
    roles = split_physical_sites(keys)
    assert len({key for key in keys}) == 80
    assert sum(map(len, roles.values())) == len(coords)
    assert all(len(v) for v in roles.values())
    site_roles = {}
    for name, indices in roles.items():
        for index in indices:
            assert keys[index] not in site_roles or site_roles[keys[index]] == name
            site_roles[keys[index]] = name


def test_split_reproducible_and_floor_stratified():
    coords = np.asarray([[float(j), 0.] for f in range(3) for j in range(50)], np.float32)
    floors = np.repeat(np.arange(3), 50)
    buildings = np.zeros(len(coords), dtype=np.int64)
    keys = anchor_site_keys(coords, buildings, floors)
    p = split_physical_sites(keys, seed=5)
    q = split_physical_sites(keys, seed=5)
    assert all(np.array_equal(p[r],q[r]) for r in p)
    assert all(len(set(floors[ids])) == 3 for ids in p.values())


def test_g13_and_unified_zero_gate_exact():
    torch.manual_seed(2)
    logits = torch.randn(9, 15)
    z = torch.randn(9, 15, 2)
    rel = torch.rand(9, 15, 4)
    h = UnifiedAnchorPosterior(theta=(.85,.75))
    p = posterior(logits, z, mode="g13", theta=(.85,.75))
    q = posterior(logits, z, mode="unified", reliability=rel, unified=h)
    assert torch.allclose(p,q,atol=1e-7,rtol=1e-7)
    assert torch.allclose(p.sum(1), torch.ones(9))
    with torch.no_grad():
        h.theta.zero_()
    zero = posterior(logits,z,mode="unified",reliability=rel,unified=h)
    t = posterior(logits,z,mode="temperature")
    assert torch.allclose(zero,t,atol=1e-7,rtol=1e-7)


def test_g13_support_zero_not_same_as_uniform_random():
    scores = torch.tensor([[10.,1.,-6.]])
    z = torch.zeros(1,3,2)
    expect = torch.softmax(scores/.6,dim=1)
    assert torch.allclose(posterior(scores,z,mode="g13",theta=(0,0)),expect)
    assert not torch.allclose(expect,torch.full_like(expect,1/3))


def test_reliability_fit_only_and_query():
    ids = torch.tensor([[1,0],[0,1]])
    m = {"fit_rows_per_anchor":torch.tensor([1.,10.]),
         "detection_probability":torch.tensor([[.5,.9],[.95,.6]]),
         "conditional_variance":torch.tensor([[.04,.01],[.02,.08]]),
         "seen_ap_mask":torch.tensor([True,True,False])}
    q = torch.tensor([[.1,-1.,-1.],[-1.,.2,-1.]])
    r = reliability_features(q,ids,m)
    assert r.shape == (2,2,4)
    assert torch.isfinite(r).all() and (r>=0).all() and (r<=1).all()
    assert r[0,0,0] > r[0,1,0]


def test_student_t_proper_shapes_gradient():
    logits = torch.tensor([[.1,.4,.8]],requires_grad=True)
    p = torch.softmax(logits,dim=1)
    loc = torch.tensor([[[0.,0.],[3.,3.],[5.,5.]]],requires_grad=True)
    scale = torch.ones_like(loc)
    y = torch.tensor([[2.,2.]])
    loss = mixture_loss(p,loc,scale,y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(loc.grad).all() and torch.isfinite(logits.grad).all()
    es1 = energy_score_mc(p.detach(),loc.detach(),scale,y,draws=32,seed=13)
    es2 = energy_score_mc(p.detach(),loc.detach(),scale,y,draws=32,seed=13)
    assert torch.allclose(es1,es2)


def test_site_macro_not_scan_macro():
    keys=[(0,0,0.,0.)]*10+[(0,0,4.,4.)]
    errors=np.asarray([1.]*10+[11.]); es=errors.copy()
    m=site_metrics(errors,es,keys)
    assert m["sites"] == 2 and m["rows"] == 11
    assert m["site_macro_mae_m"] == 6.
    assert m["row_mae_m"] < 2.


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        split_physical_sites([(0,0,1.,2.)],seed=1)
    with pytest.raises(ValueError):
        posterior(torch.zeros(1,2),torch.zeros(1,2,2),mode="g13",theta=(-1.,0.))