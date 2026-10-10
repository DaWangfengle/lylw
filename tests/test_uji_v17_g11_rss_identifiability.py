"""G11 low-cost CPU scientific tests, no real CAL/CONFIRM consumption."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g11_rss_identifiability as g11


def test_prespecified_protocol_and_no_learning():
    assert g11.DOMAINS==("tuji","uts","tampere")
    assert (g11.E,g11.L,g11.K)==(5,192,192)
    assert (g11.NEAR,g11.FAR)==(10.,20.)
    assert (g11.KNN,g11.KNN_EPS)==(5,.001)
    assert g11.BETA_PRIOR==.5
    assert g11.METRICS==("raw_l2_1nn","raw_l2_5nn",
                        "ap_visibility_1nn","v17_anchor_t06")
    assert "tampere" in g11.ROLES
    assert "confirmation" in g11.ROLES["tampere"]


def test_presence_fit_only_beta_prior_and_grouping():
    rss=torch.tensor([
        [-1.,.2,-1.],[-1.,.8,-1.],
        [.4,-1.,.2]],dtype=torch.float32)
    groups=torch.tensor([0,0,1])
    seen=torch.tensor([True,True,False])
    p=g11.fit_site_ap_frequency(rss,groups,2,seen)
    assert p.shape==(2,2)
    assert p[0,0].item()==pytest.approx(.5/3)
    assert p[0,1].item()==pytest.approx(2.5/3)
    assert p[1,0].item()==pytest.approx(1.5/2)
    assert p[1,1].item()==pytest.approx(.5/2)


def test_ap_presence_clipping_limitation_and_brier_exact():
    reference=torch.tensor([[.75,.25],[.25,.75]])
    query=torch.tensor([[.4,-1.],[-1.,.1]],dtype=torch.float32)
    dist=g11.visibility_brier_distance(query,reference)
    assert dist[0,0].item()==pytest.approx(
        ((1-.75)**2+(0-.25)**2)/2)
    assert dist[0,1].item()==pytest.approx(
        ((1-.25)**2+(0-.75)**2)/2)
    assert torch.equal(torch.argmin(dist,1),torch.tensor([0,1]))
    # Exact -1 means missing OR min-clipped, either way treated unseen.
    assert (query>g11.AP_MISSING+g11.AP_EPS).int().tolist()==[[1,0],[0,1]]


def test_original_fit_anchor_permutation_verified_with_real_geometry():
    class MockRef:
        n_anchors=3
        anchor_coords=torch.tensor([[30.,1.],[10.,1.],[20.,1.]])
        anchor_buildings=torch.tensor([0,0,0])
        anchor_floors=torch.tensor([1,1,1])
    keys=[(0,1,20.,1.),(0,1,10.,1.),(0,1,30.,1.),
          (0,1,20.,1.)]
    data={"keys":keys}
    groups,ordered=g11.map_fit_rows_to_model_anchors(
        np.arange(4),data,MockRef(),np.zeros(2))
    assert groups.tolist()==[2,1,0,2]
    assert len(ordered)==3
    # No assumption that sorted source FIT positions are model-memory ordered.


def test_wrong_fit_to_anchor_bijection_fails_closed():
    class BadRef:
        n_anchors=2
        anchor_coords=torch.tensor([[10.,0.],[10.,0.]])
        anchor_buildings=torch.tensor([0,0])
        anchor_floors=torch.tensor([0,0])
    with pytest.raises(g11.G11Error,match="bijection"):
        g11.map_fit_rows_to_model_anchors(
            np.arange(2),{"keys":[(0,0,10.,0.),(0,0,20.,0.)]},
            BadRef(),np.zeros(2))


def test_fit_anchor_mapping_rejects_wrong_floor():
    class BadRef:
        n_anchors=2
        anchor_coords=torch.tensor([[10.,0.],[20.,0.]])
        anchor_buildings=torch.tensor([0,0])
        anchor_floors=torch.tensor([0,0])
    with pytest.raises(g11.G11Error,match="bijection"):
        g11.map_fit_rows_to_model_anchors(
            np.arange(2),{"keys":[(0,0,10.,0.),(0,1,20.,0.)]},
            BadRef(),np.zeros(2))


def test_fit_anchor_mapping_avoids_float32_cdist_cancellation():
    # More than 25 anchors selects cdist's GEMM path by default. Its
    # float32 norm expansion can invent >0.03m distances for identical XY.
    index=torch.arange(64,dtype=torch.float32)
    xy=torch.stack((137.437+index*.71,121.193+torch.sin(index)),dim=1)
    order=torch.arange(63,-1,-1)
    ref=SimpleNamespace(
        n_anchors=64,anchor_coords=xy[order],
        anchor_buildings=torch.zeros(64,dtype=torch.long),
        anchor_floors=torch.ones(64,dtype=torch.long))
    keys=[(0,1,float(x),float(y)) for x,y in xy.tolist()]
    groups,_=g11.map_fit_rows_to_model_anchors(
        np.arange(64),{"keys":keys},ref,np.zeros(2))
    assert groups.tolist()==order.tolist()
    # Keep the scientific 0.03m gate: a real displacement still fails.
    ref.anchor_coords=ref.anchor_coords+torch.tensor([.031,0.])
    with pytest.raises(g11.G11Error,match="bijection"):
        g11.map_fit_rows_to_model_anchors(
            np.arange(64),{"keys":keys},ref,np.zeros(2))


def test_1nn_and_5nn_point_predictions_no_true_labels():
    xy=torch.stack((torch.arange(6,dtype=torch.float32)*10,
                    torch.zeros(6)),dim=1)
    dist=torch.tensor([[0.,1.,2.,3.,4.,5.],
                       [5.,4.,3.,2.,1.,0.]])
    x=g11.rss_knn_predictions(dist,xy)
    assert torch.equal(x["raw_l2_top1_id"],torch.tensor([0,5]))
    assert x["raw_l2_1nn"][:,0].tolist()==[0.,50.]
    assert 0.<=x["raw_l2_5nn"][0,0].item()<=20.
    assert 30.<=x["raw_l2_5nn"][1,0].item()<=50.
    assert "true_building" not in inspect.signature(
        g11.rss_knn_predictions).parameters


def _toy_geometry():
    a=192
    xy=torch.stack((torch.arange(a,dtype=torch.float32)*30,
                    torch.zeros(a)),dim=1)
    floor=torch.zeros(a,dtype=torch.long)
    floor[1]=1
    building=torch.zeros(a,dtype=torch.long)
    target=torch.tensor([[0.,0.]])
    b=torch.zeros(1,dtype=torch.long)
    f=torch.zeros(1,dtype=torch.long)
    raw=torch.arange(a,dtype=torch.float32)[None,:]
    ap=raw.flip(1).clone()
    selected=torch.arange(a).reshape(1,-1)
    mass=torch.full((1,a),1/a)
    return target,b,f,xy,building,floor,raw,ap,selected,mass


def test_distinct_raw_top1_vs_v17_ranking():
    y,b,f,xy,ab,af,raw,ap,selected,mass=_toy_geometry()
    diag=g11.distinguishability(
        y,b,f,xy,ab,af,raw,ap,
        torch.tensor([0]),torch.tensor([191]),torch.tensor([191]),
        selected,mass)
    assert diag["raw_l2_top1_in_10m"].item()==1.
    assert diag["ap_visibility_top1_in_10m"].item()==0.
    assert diag["v17_top1_in_10m"].item()==0.
    assert diag["raw_near_far_positive"].item()==1.
    assert diag["ap_near_far_positive"].item()==0.
    assert diag["v17_best_anchor_rank"].item()==1.
    assert diag["v17_near_10m_mass"].item()==pytest.approx(1/192.)


def test_other_floor_at_same_xy_is_clearly_wrong_not_correct():
    y,b,f,xy,ab,af,raw,ap,selected,mass=_toy_geometry()
    xy[1]=torch.tensor([0.,0.])
    raw[0,1]=-4.
    diag=g11.distinguishability(
        y,b,f,xy,ab,af,raw,ap,
        torch.tensor([1]),torch.tensor([1]),torch.tensor([1]),
        selected,mass)
    assert diag["raw_l2_top1_in_10m"].item()==0.
    assert diag["v17_top1_in_10m"].item()==0.
    assert diag["raw_near_far_margin"].item()<0
    assert diag["near_fit_anchor_exists"].item()==1.


def test_no_nearby_fit_site_causes_null_margin_not_oracle_hit():
    y,b,f,xy,ab,af,raw,ap,selected,mass=_toy_geometry()
    y=torch.tensor([[-100.,0.]])
    diag=g11.distinguishability(
        y,b,f,xy,ab,af,raw,ap,
        torch.tensor([0]),torch.tensor([191]),torch.tensor([191]),
        selected,mass)
    assert diag["near_fit_anchor_exists"].item()==0.
    assert torch.isnan(diag["raw_near_far_margin"]).item()
    assert torch.isnan(diag["v17_best_anchor_rank"]).item()


def test_site_macro_and_null_handling():
    group=torch.tensor([0,0,1,1])
    v=torch.tensor([2.,float("nan"),5.,9.])
    assert g11._site_vector(v,group,2)==[2.,7.]
    none=torch.full((4,),float("nan"))
    assert g11._site_vector(none,group,2)==[None,None]
    stats=g11._metric_summaries({"raw_l2_1nn_point_mae_m":[2.,3.,9.]})
    assert stats["raw_l2_1nn_point_mae_m"]["macro_mean"]==pytest.approx(14/3)
    assert stats["raw_l2_1nn_point_mae_m"]["max_mae_m"]==9.


def test_prototype_and_detection_metrics_have_different_rankings():
    y,b,f,xy,ab,af,raw,ap,selected,mass=_toy_geometry()
    proto_nn=torch.argmin(raw,dim=1)
    presence_nn=torch.argmin(ap,dim=1)
    assert proto_nn.item()==0 and presence_nn.item()==191
    assert proto_nn.item()!=presence_nn.item()


def test_immutability_of_g11_evidence(tmp_path):
    path=tmp_path/"result.json"
    record=g11._write(path,{"frozen":True})
    assert g11._read(path)==record
    with pytest.raises(FileExistsError):
        g11._write(path,{"frozen":False})
    record["frozen"]=False
    path.write_text(json.dumps(record))
    with pytest.raises(g11.G11Error,match="mismatch"):
        g11._read(path)


def test_science_scope_and_r3_original_operators():
    source=inspect.getsource(g11)
    assert "r2b._prototype_distance(ref,rss)" in source
    assert "r3._member_top_l(" in source
    assert "r3._dense_joint_law(" in source
    assert "r3.dense_alpha_consensus_prior(" in source
    assert 'g8.hierarchical_laws(joint,E,K)["anchor"]' in source
    assert "g8._load_models(" in source
    assert 'data["rows"]["fit"]' in source
    assert "train_member(" not in source
    assert "optimizer" not in source.lower()
    assert "load_tampere_test" not in source
    assert "official_test_accessed" in source


def test_lock_exists_refuses_formal_rerun(monkeypatch,tmp_path):
    lock=tmp_path/"g11.lock"
    lock.write_text("attempted")
    monkeypatch.setattr(g11,"LOCK",lock)
    monkeypatch.setattr(g11,"RESULT",tmp_path/"none")
    with pytest.raises(g11.G11Error,match="existing lock"):
        g11.run("cpu")
