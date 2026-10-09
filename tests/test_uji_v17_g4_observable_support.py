"""Fast CPU fixtures for G4 training-only split, observable gate and R3 law.

No real dataset opens, model fitting, GPU usage, or external-test access.
"""
from __future__ import annotations

import inspect
import json
from collections import Counter

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g4_observable_support as g4
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3


def synthetic_keys():
    keys=[]
    for floor in (0,1,2):
        for anchor in range(16):
            key=(0,floor,float(anchor*1.1),float(floor*2.2))
            keys.extend([key]*(anchor%3+1))
    return keys


def test_coordinate_strata_splits_are_stable_disjoint_and_exhaustive():
    keys=synthetic_keys()
    a=g4.split_keys(keys)
    b=g4.split_keys(list(reversed(keys)))
    assert a==b
    assert set.union(*a.values())==set(keys)
    assert a["fit"].isdisjoint(a["calibration"])
    assert a["fit"].isdisjoint(a["confirmation"])
    assert a["calibration"].isdisjoint(a["confirmation"])
    for role in a:
        assert {x[1] for x in a[role]}=={0,1,2}
    data={"keys":keys}
    assigned=[g4.rows_for_role(data,a,r) for r in a]
    assert sum(map(len,assigned))==len(keys)
    counts=Counter(int(k) for rows in assigned for k in rows)
    assert counts==Counter(range(len(keys)))
    assert all(set(keys[i] for i in assigned[j]).isdisjoint(
        set(keys[i] for i in assigned[k]))
        for j,k in ((0,1),(0,2),(1,2)))


def test_tiny_floor_cannot_produce_fake_independence():
    keys=[(0,0,float(i),0.) for i in range(4)]
    with pytest.raises(g4.G4Error,match="too few"):
        g4.split_keys(keys)


def test_fixed_science_is_independent_of_g3_external_metrics():
    assert g4.SEEDS==g1.SEEDS[:5]
    assert g4.EPOCHS==g1.EPOCHS[:5]
    assert g4.GAMMA==.70
    assert g4.RADIUS_M==10.
    assert (g4.PANEL,g4.L,g4.K,g4.ALPHA)==(5,192,192,1.0)
    assert g4.CAL_QUANTILES==(.25,.50,.75)
    assert g4.CONF_MIN_GAIN_M==.15
    assert g4.BOOTSTRAP_REPS==5000
    src=inspect.getsource(g4)
    assert 'load_uts("test"' not in src
    assert "G3_RESULT" not in src
    assert "UTS_test.csv" in src  # documented prohibited asset, not read
    assert "r0._fit_refit(" in src
    assert "model.training_loss" not in src  # frozen R0 handles the training loss


def test_query_support_score_uses_prior_not_true_position():
    logd=torch.tensor([0.,np.log(2),np.log(9)],dtype=torch.float64)
    p=torch.tensor([
        [[1.,0.,0.],[0.,0.,1.]],
        [[0.,1.,0.],[0.,0.,1.]]
    ],dtype=torch.float64)
    obs=g4.posterior_observable(p,logd)
    assert obs.shape==(2,)
    assert obs.tolist()==pytest.approx([.5*np.log(2),np.log(9)])
    assert "target" not in inspect.getsource(g4.posterior_observable)
    assert "raw_xy" not in inspect.getsource(g4.posterior_observable)


def test_selective_posterior_uses_identity_on_high_support():
    p=torch.tensor([[[.5,.3,.2],[.4,.4,.2]],
                    [[.2,.7,.1],[.1,.7,.2]]],dtype=torch.float64)
    logd=torch.tensor([0.,.8,1.6],dtype=torch.float64)
    score=torch.tensor([.3,2.0],dtype=torch.float64)
    out=g4.mixture_prior(p,logd,mode="selective",
                         observable=score,threshold=.5)
    g=g1.corrected_probability(p,logd,g4.GAMMA)
    assert torch.allclose(out[:,0],g[:,0])
    assert torch.equal(out[:,1],p[:,1])
    assert torch.equal(g4.mixture_prior(p,logd,mode="baseline"),p)
    assert not torch.allclose(g,p)
    with pytest.raises(g4.G4Error):
        g4.mixture_prior(p,logd,mode="selective",observable=score)


def test_calibration_only_selection_and_fallback():
    metrics={
        "baseline":{"mae_2d_m":5.0,"energy_score_2d_m":4.0},
        "g1":{"mae_2d_m":4.5,"energy_score_2d_m":4.4},
        "gate_q25":{"mae_2d_m":4.80,"energy_score_2d_m":4.015},
        "gate_q50":{"mae_2d_m":4.70,"energy_score_2d_m":4.01},
        "gate_q75":{"mae_2d_m":4.60,"energy_score_2d_m":4.09},
    }
    feature_by_group={(0,0,float(i),0.):float(i) for i in range(12)}
    selected=g4.select_calibration_policy(metrics,feature_by_group)
    assert selected["policy"]=="gate_q50"
    assert selected["threshold"]==pytest.approx(5.5)
    assert selected["feasible_policies"]==["gate_q25","gate_q50"]
    worse={k:dict(v) for k,v in metrics.items()}
    worse["gate_q25"]["mae_2d_m"]=5.01
    worse["gate_q50"]["mae_2d_m"]=5.01
    no=g4.select_calibration_policy(worse,feature_by_group)
    assert no["policy"]=="baseline"
    assert no["threshold"] is None


def test_calibration_threshold_measured_over_physical_sites_not_scans():
    keys=[(0,0,0.,0.)]*3+[(0,0,1.,0.)]*1+[(0,0,2.,0.)]*5
    values=np.array([.3,.3,.3,.6,.9,.9,.9,.9,.9])
    data={"keys":keys}
    grouped=g4._group_means(np.arange(len(keys)),data,values)
    assert len(grouped)==3
    assert np.quantile(list(grouped.values()),.5)==pytest.approx(.6)
    assert np.quantile(values,.5)==pytest.approx(.9)


def test_five_member_joint_inference_shape_matches_v17_r3():
    torch.manual_seed(18)
    prior=torch.softmax(torch.randn(5,2,201),dim=-1)
    loc=torch.randn(5,2,201,2)
    scale=torch.rand(5,2,201,2)+.5
    top,ids,radial,retained=r3._member_top_l(
        prior,loc,scale,g4.L,student_df=4.
    )
    joint,location,scales=r3._dense_joint_law(
        top,ids,radial,loc,scale,tuple(range(g4.PANEL)),
        top_k=g4.K,alpha=g4.ALPHA,
    )
    assert joint.shape==(2,5*192)
    assert location.shape==(2,5*192,2)
    assert scales.shape==(2,5*192,2)
    assert torch.allclose(joint.sum(dim=1),torch.ones(2),atol=1e-5)


def test_json_chain_rejects_modified_result(tmp_path):
    target=tmp_path/"synthetic_v17_g4.json"
    obj=g4._write(target,{"experiment":"dummy","gain":[1,2,3]})
    assert g4._read(target,"result_sha256")==obj
    with pytest.raises(FileExistsError):
        g4._write(target,{"experiment":"dummy"})
    raw=json.loads(target.read_text(encoding="utf-8"))
    raw["gain"][0]=999
    target.write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(g4.G4Error,match="self hash"):
        g4._read(target,"result_sha256")


def test_confirmation_lock_refuses_duplicate_or_prior_attempt(monkeypatch,tmp_path):
    # Confirm checks irreversible lock before making any calls to prepare.
    lock=tmp_path/"exists"
    lock.write_text("already attempted")
    monkeypatch.setattr(g4,"CONFIRM_LOCK",lock)
    with pytest.raises(g4.G4Error,match="already"):
        g4.confirm("cpu")


def test_g4_report_keeps_all_scientific_caveats():
    selected={"policy":"gate_q25","threshold":0.4}
    cal={"selected_policy":selected,
         "observable_support_vs_fixed_g1_gain_spearman":-.3}
    conf={"observable_support_vs_fixed_g1_gain_spearman":-.2,
          "competing_observable_controls":{
              "entropy":{"spearman_vs_g1_gain":+.1},
              "visible_aps":{"spearman_vs_g1_gain":-.05}},
          "metrics":{"baseline":{"mae_2d_m":5.,"energy_score_2d_m":4.},
                     "g1":{"mae_2d_m":4.9,"energy_score_2d_m":4.1},
                     "gate_q25":{"mae_2d_m":4.7,"energy_score_2d_m":3.9},
                     "placebo":{"mae_2d_m":5.1,"energy_score_2d_m":4.1}},
          "confirmation_gate":{"passed":True,"delta_mae_m":-.3,"delta_es_m":-.1},
          "decision":"PROMISING_INTERNAL_PILOT","result_sha256":"synthetic"}
    txt=g4.make_report(cal,conf)
    assert "UTS_training.csv" in txt
    assert "UTS_test.csv" in txt
    assert "not causal" in txt.lower()
    assert "PROMISING_INTERNAL_PILOT" in txt


@pytest.mark.parametrize("invalid",[0,-1,6,8])
def test_member_index_validity(invalid):
    with pytest.raises(g4.G4Error):
        g4.member_path(invalid)
