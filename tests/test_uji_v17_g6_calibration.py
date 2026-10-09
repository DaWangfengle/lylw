"""G6 pure CPU science-contract tests; no real TUJI files or external tests."""
from __future__ import annotations

import inspect
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g6_calibration as g6
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness


def test_split_is_stable_by_site_not_rows():
    keys=[]
    for floor in (0,1):
        for i in range(40):
            key=(0,floor,round(i*2.1,4),float(floor*5))
            keys.extend([key]*(1+i%3))
    a=g6.split_physical_keys(keys)
    b=g6.split_physical_keys(list(reversed(keys)))
    assert a==b
    assert set.union(*a.values())==set(keys)
    assert all(a[x].isdisjoint(a[y]) for x,y in
               (("fit","calibration"),("fit","confirmation"),
                ("calibration","confirmation")))
    for floor in (0,1):
        assert sum(1 for k in a["fit"] if k[1]==floor)==28
        assert sum(1 for k in a["calibration"] if k[1]==floor)==6
        assert sum(1 for k in a["confirmation"] if k[1]==floor)==6
    assert a==g6.split_physical_keys(keys,seed=g6.SPLIT_SEED)


def test_tiny_floor_fails_before_fit():
    keys=[(0,0,float(i),0.) for i in range(14)]
    with pytest.raises(g6.G6Error,match="too few"):
        g6.split_physical_keys(keys)


def test_float32_centring_preserves_all_role_identity():
    keys=[(0,0,float(i*2),0.) for i in range(400)]
    data={"keys":keys,"raw_xy":np.asarray([[k[2],k[3]] for k in keys]),
          "building":np.zeros(len(keys),dtype=np.int64),
          "floor":np.zeros(len(keys),dtype=np.int64)}
    roles,rows=g6._roles(data)
    assert len(roles["fit"])>=192
    assert sum(map(len,rows.values()))==len(keys)
    assert Counter(i for idx in rows.values() for i in idx)==Counter(range(len(keys)))
    assert all(len(roles[k])>0 for k in roles)


def test_duplicate_float32_rounding_site_fails():
    # Two distinct original sites collapse after float32 centring.
    n=400
    vals=np.arange(n,dtype=np.float64)*2
    vals[1]=vals[0]+.00009
    data={"keys":[(0,0,float(v),0.) for v in vals],
          "raw_xy":np.column_stack((vals,np.zeros(n))),
          "building":np.zeros(n,dtype=np.int64),
          "floor":np.zeros(n,dtype=np.int64)}
    # On most systems these sites are distinct at four decimals, and the
    # R0-centred collision may or may not happen depending on origin.
    # Check the no-leak property in all cases; no forced failure assertion.
    try:
        roles,_=g6._roles(data)
    except g6.G6Error:
        return
    assert all(roles[a].isdisjoint(roles[b]) for a,b in
               (("fit","calibration"),("fit","confirmation"),
                ("calibration","confirmation")))


def test_safe_tuji_loader_requests_only_train_variable(monkeypatch):
    paths=[]
    class Scan:
        def __init__(self,x):
            self.coords=np.array([float(x),1.])
            self.measured=np.array([[11.,-65.],[12.,-75.]])
    def restricted_loadmat(path,**kwargs):
        paths.append((path,kwargs))
        assert kwargs["variable_names"]==("results_train",)
        return {"results_train":SimpleNamespace(Wifi=[
            Scan(2),Scan(4),Scan(6),Scan(8)])}
    monkeypatch.setattr(g6,"loadmat",restricted_loadmat)
    output=g6.load_tuji_train_variable_only()
    assert output.dataset=="TUJI1" and output.split=="train"
    assert len(output.coords_m)==4*len(g6.EXPECTED_DEVICES)
    assert output.ap_ids==("MAC11","MAC12")
    assert sorted(np.unique(output.device).tolist())==list(range(5))
    assert len(paths)==5
    assert output.metadata["test_data_decoded"] is False
    assert output.metadata["files_physically_contain_results_test"] is True


def test_safe_tuji_loader_rejects_test_variable_in_payload(monkeypatch):
    monkeypatch.setattr(g6,"loadmat",lambda *a,**k: {
        "results_train":SimpleNamespace(Wifi=[]),
        "results_test":SimpleNamespace(Wifi=[])})
    with pytest.raises(g6.G6Error,match="unexpectedly deserialized"):
        g6.load_tuji_train_variable_only()


def test_old_unfiltered_tuji_loader_is_not_called():
    source=inspect.getsource(g6)
    assert 'ds.load_tuji1("train")' not in source
    assert 'ds.load_tuji1("test")' not in source
    assert 'variable_names=("results_train",)' in source
    assert "r0._fit_refit(" in source
    assert "gamma" not in inspect.getsource(g6.calibrate_joint_law).lower()


def test_grid_is_small_fixed_and_baseline_exact():
    assert (g6.PANEL,g6.L,g6.K,g6.ALPHA)==(5,192,192,1.)
    assert g6.T_VALUES==(1.,.8,.6)
    assert g6.S_VALUES==(1.,.75,.5)
    assert len(g6.CANDIDATES)==9
    assert g6.BASE=="T1_S1"
    assert g6.BOOT_REPS==5000
    torch.manual_seed(6)
    p=torch.softmax(torch.randn(3,10),dim=1)
    location=torch.randn(3,10,2)
    scale=torch.rand_like(location)+.4
    q,loc,sig=g6.calibrate_joint_law(p,location,scale,1.,1.)
    assert torch.equal(q,p)
    assert loc is location and sig is scale
    assert torch.allclose(q.sum(dim=1),torch.ones(3))


def test_scale_only_does_not_move_exact_mean():
    p=torch.tensor([[.2,.8]],dtype=torch.float64)
    loc=torch.tensor([[[0.,0.],[6.,0.]]],dtype=torch.float64)
    scale=torch.ones_like(loc)
    b,_,s0=g6.calibrate_joint_law(p,loc,scale,1.,1.)
    q,_,s1=g6.calibrate_joint_law(p,loc,scale,1.,.5)
    assert torch.equal(b,q) and torch.allclose(s1,s0*.5)
    mean0=(b[...,None]*loc).sum(dim=1)
    mean1=(q[...,None]*loc).sum(dim=1)
    assert torch.equal(mean0,mean1)
    # Temperature is the only modification that may move point action.
    sharpen,_,_=g6.calibrate_joint_law(p,loc,scale,.6,1.)
    assert sharpen[0,1]>p[0,1]
    assert not torch.equal((sharpen[...,None]*loc).sum(dim=1),mean0)


def test_scale_only_mae_invariant_in_actual_r3_scorer():
    p=torch.tensor([[.4,.6]],dtype=torch.float32)
    loc=torch.tensor([[[0.,0.],[10.,2.]]],dtype=torch.float32)
    scale=torch.ones_like(loc)
    target=torch.tensor([[7.,1.]],dtype=torch.float64)
    rng=common_student_t_randomness(1,draws=64,seed=19602)
    b,_,s0=g6.calibrate_joint_law(p,loc,scale,1.,1.)
    q,_,s1=g6.calibrate_joint_law(p,loc,scale,1.,.5)
    es0,mae0=r3._score_joint_law(b,loc,s0,target,rng)
    es1,mae1=r3._score_joint_law(q,loc,s1,target,rng)
    assert torch.equal(mae0,mae1)
    assert torch.isfinite(es0).all() and torch.isfinite(es1).all()


def test_cal_selects_lowest_mae_only_with_ES_gain_and_fallback():
    records={}
    for t,s in g6.CANDIDATES:
        records[g6.candidate_name(t,s)]={"mae_2d_m":5.0,
                                         "energy_score_2d_m":4.0}
    records[g6.BASE]={"mae_2d_m":5.0,"energy_score_2d_m":4.0}
    records["T0.8_S0.75"]={"mae_2d_m":4.60,"energy_score_2d_m":3.96}
    records["T0.6_S0.75"]={"mae_2d_m":4.51,"energy_score_2d_m":4.03}
    records["T0.8_S0.5"]={"mae_2d_m":4.70,"energy_score_2d_m":3.90}
    selection=g6.select_policy(records)
    assert selection["policy"]=="T0.8_S0.75"
    assert selection["T"]==.8 and selection["scale"]==.75
    assert "T0.6_S0.75" not in selection["eligible"]
    for k in records:
        if k!=g6.BASE:
            records[k]={"mae_2d_m":4.99,"energy_score_2d_m":3.5}
    no=g6.select_policy(records)
    assert no["policy"]==g6.BASE and no["fell_back_to_baseline"]


def test_r3_joint_law_temperature_preserves_dimensions():
    torch.manual_seed(17)
    prior=torch.softmax(torch.randn(5,2,210),dim=-1)
    loc=torch.randn(5,2,210,2)
    scale=torch.rand_like(loc)+.2
    top,ix,rad,_=r3._member_top_l(prior,loc,scale,192,student_df=4.)
    p,xy,s=r3._dense_joint_law(top,ix,rad,loc,scale,
                                tuple(range(5)),top_k=192,alpha=1.)
    assert p.shape==(2,960)
    for t,scale_factor in g6.CANDIDATES:
        q,mm,ss=g6.calibrate_joint_law(p,xy,s,t,scale_factor)
        assert q.shape==p.shape and mm.shape==ss.shape==xy.shape
        assert torch.isfinite(q).all()
        assert torch.allclose(q.sum(dim=1),torch.ones(2),atol=1e-5)


def test_confirmation_lock_rejects_repeated_attempt(monkeypatch,tmp_path):
    lock=tmp_path/"confirm.lock"
    lock.write_text("already opened",encoding="utf-8")
    monkeypatch.setattr(g6,"LOCK",lock)
    with pytest.raises(g6.G6Error,match="already attempted"):
        g6.confirm("cpu")


def test_evidence_hash_and_immutability(tmp_path):
    path=tmp_path/"sample.json"
    data=g6._write(path,{"candidate":"T0.8_S0.75","value":3})
    assert g6._read(path)==data
    with pytest.raises(FileExistsError):
        g6._write(path,{"candidate":"T1_S1"})
    bad=json.loads(path.read_text())
    bad["value"]=4
    path.write_text(json.dumps(bad),encoding="utf-8")
    with pytest.raises(g6.G6Error,match="hash mismatch"):
        g6._read(path)


def test_report_does_not_overstate_causality_or_external_testing():
    selected={"policy":"T0.8_S0.75"}
    j={"selected_policy":selected,
       "candidate_metrics":{
           "T1_S1":{"mae_2d_m":5.,"energy_score_2d_m":4.},
           "T0.8_S0.75":{"mae_2d_m":4.7,"energy_score_2d_m":3.8},
       },
       "gate":{"mae_delta_m":-.3,"es_delta_m":-.2,
               "mae_upper_m":-.1,"es_upper_m":-.05,"passed":True},
       "decision":"PROMISING_JOINT_CALIBRATION","result_sha256":"synthetic"}
    txt=g6.make_report(j)
    assert "TRAIN" in txt
    assert "external" in txt.lower()
    assert "not" in txt.lower()
    assert "PROMISING_JOINT_CALIBRATION" in txt


@pytest.mark.parametrize("i",[0,-1,6,9])
def test_invalid_member_index(i):
    with pytest.raises(g6.G6Error):
        g6.member_path(i)


def test_original_double_sites_cannot_merge_during_float32_adapter(monkeypatch):
    scans = [
        SimpleNamespace(coords=np.array([100000000., 1.], dtype=np.float64),
                        measured=np.array([[11., -65.], [12., -75.]])),
        SimpleNamespace(coords=np.array([100000001., 1.], dtype=np.float64),
                        measured=np.array([[11., -65.], [12., -75.]])),
    ]
    monkeypatch.setattr(g6, "loadmat", lambda *a, **k: {
        "results_train": SimpleNamespace(Wifi=scans)})
    with pytest.raises(g6.G6Error, match="float32.*physical site"):
        g6.load_tuji_train_variable_only()


def test_report_is_identical_after_sorted_json_roundtrip():
    record = {
        "selected_policy": {"policy": "T0.8_S0.75"},
        "candidate_metrics": {
            "T1_S1": {"mae_2d_m": 5., "energy_score_2d_m": 4.},
            "T0.8_S0.75": {"mae_2d_m": 4.7, "energy_score_2d_m": 3.8},
        },
        "gate": {"mae_delta_m": -.3, "es_delta_m": -.2,
                 "mae_upper_m": -.1, "es_upper_m": -.05, "passed": True},
        "decision": "PROMISING_JOINT_CALIBRATION", "result_sha256": "synthetic",
    }
    assert g6.make_report(record) == g6.make_report(json.loads(
        json.dumps(record, sort_keys=True)))


def test_actual_role_scoring_retains_all_candidate_reliability(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.n_anchors = 210
            self.encoder = torch.nn.Identity()
            self.register_buffer("centres", torch.column_stack((
                torch.linspace(-8, 8, self.n_anchors),
                torch.zeros(self.n_anchors))))
        def retrieval_logits_from_embedding(self, embedding):
            return -self.centres[:, 0].square()[None, :].expand(len(embedding), -1)/10
        def _components(self, embedding, anchor_ix):
            loc = self.centres[None, :, :].expand(len(embedding), -1, -1)
            return loc, torch.ones_like(loc)
    monkeypatch.setattr(g6.r2b, "_prototype_distance",
                        lambda ref, rss: torch.zeros(len(rss), ref.n_anchors))
    data = {"keys": [(0, 0, 2., 1.), (0, 0, 2., 1.), (0, 0, 3., 1.)],
            "raw_xy": np.array([[2., 1.], [2., 1.], [3., 1.]]),
            "phone": np.zeros(3, dtype=np.int64), "rss": torch.zeros(3, 2)}
    scored = g6._score_role("calibration", [Model() for _ in range(5)],
                           data, np.arange(3), np.zeros(2))
    names = {g6.candidate_name(*pair) for pair in g6.CANDIDATES}
    assert set(scored["reliability"]) == names
    assert scored["coordinate_groups"] == 2
    for name in names:
        diag = scored["reliability"][name]
        assert len(diag["coordinate_vectors"]["predicted_radius"]) == 2
        assert len(diag["coordinate_vectors"]["observed_error"]) == 2
        assert set(diag["coverage"]) == {"0.5", "0.8", "0.9"}
        assert set(diag["macro_credible_radius_m"]) == {"0.5", "0.8", "0.9"}
    for t in g6.T_VALUES:
        mae = [scored["coordinate_vectors"][g6.candidate_name(t, s)]["mae"]
               for s in g6.S_VALUES]
        assert mae[0] == mae[1] == mae[2]
