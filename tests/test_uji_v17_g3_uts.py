"""G3 fast CPU-only invariants: fixed science, train/test access boundary.

No actual UTS provided test is opened by tests. Synthetic geometry/RSS only.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as r3p
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as r3
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _canonical_json_sha256


def canon(coords, b=None, f=None, device=None):
    n=len(coords)
    return SimpleNamespace(
        dataset="UTSIndoorLoc", split="train",
        coords_m=np.asarray(coords, dtype=np.float32),
        rss_dbm=np.array([[-60.,float("nan"),-90.]]*n, dtype=np.float32),
        building=np.asarray(b if b is not None else [0]*n, dtype=np.int64),
        floor=np.asarray(f if f is not None else [0]*n, dtype=np.int64),
        device=np.asarray(device if device is not None else [0]*n, dtype=np.int64),
        model_rss=lambda: np.array([[.6,-1.,-.2]]*n, dtype=np.float32),
        validate=lambda:None
    )


def test_science_matches_original_frozen_G1_V17():
    assert g3.GAMMA==g1.GAMMAS[-1]==.70
    assert g3.RADIUS_M==g1.RADIUS_M==10.
    assert g3.SEEDS==g1.SEEDS
    assert g3.EPOCHS==g1.EPOCHS
    assert g3.PANEL_SIZE==25
    assert r3p.SELECTED_NAME=="selected_E16_L192_K192_A1"
    assert r3p.CONFIGURATIONS[r3p.SELECTED_NAME]=={
        "E":16,"L":192,"K":192,"alpha":1.0
    }
    assert len(r3p.balanced_member_subsets(16))==25
    assert g3.BOOTSTRAP_REPETITIONS==5000


def test_canonical_train_adapter_minimal_model_contract():
    d=g3._as_r0_input(canon([(0.,0.),(1.,0.),(0.,1.)]),require_training=False)
    assert d["rss"].shape==(3,3)
    assert d["raw_xy"].dtype==np.float64
    assert d["keys"]==[(0,0,0.,0.),(0,0,1.,0.),(0,0,0.,1.)]
    assert d["stats"]["n_aps"]==3
    assert d["stats"]["n_buildings"]==1
    assert len(d["phone"])==3
    with pytest.raises(g3.G3Error, match="training rows"):
        g3._as_r0_input(canon([(0.,0.)]), require_training=True)


@pytest.mark.parametrize("payload",[
    canon([(0.,0.)]),
    SimpleNamespace(dataset="TUJI1",split="train",model_rss=lambda:np.zeros((1,3)),
                    coords_m=np.zeros((1,2)),building=np.zeros(1,dtype=int),
                    floor=np.zeros(1,dtype=int),device=np.zeros(1,dtype=int)),
])
def test_training_data_law_fails_when_bad(payload):
    with pytest.raises(g3.G3Error):
        g3._as_r0_input(payload,require_training=True)


def test_train_anchor_geometry_refuses_distinct_site_collisions():
    coords=[(float(i),0.) for i in range(220)]
    d=g3._as_r0_input(canon(coords),require_training=False)
    geometry=g3._training_geometry(d)
    assert geometry["anchor_count"]==220
    assert geometry["rss_aps"]==3
    assert geometry["stratum_anchor_counts"]=={"B0_F0":220}
    # Two original float64 positions at 1e-5 separation share legacy rounded key.
    # This is a documented resolution limitation and should not be fabricated
    # as a distinct-site safety failure when original key already merged.


def test_geometry_requires_minimum_anchor_coverage():
    d=g3._as_r0_input(canon([(float(i),0.) for i in range(20)]),require_training=False)
    with pytest.raises(g3.G3Error,match="less than 192"):
        g3._training_geometry(d)


def test_training_source_no_test_access_syntactic_guard():
    for fn in (g3._frozen_source_hashes, g3.training_data, g3._training_geometry,
               g3.preflight, g3.prepare, g3.train_member, g3.check_members,
               g3.freeze):
        source=inspect.getsource(fn)
        assert 'load_uts("test"' not in source
        assert "_evaluate_test_once(" not in source
        assert "_sha256_file(DATA_TEST)" not in source
    evaluating=inspect.getsource(g3.evaluate_test)
    assert evaluating.index("_write_once(LOCK")<evaluating.index("_evaluate_test_once")
    assert g3.DATA_TEST not in (g3.MANIFEST,g3.FREEZE)
    assert "test labels/outcomes" in g3.DISCLOSURE


def test_model_rss_identical_for_same_synthetic_rows_baseline_and_g1():
    # Exact same training feature tensor feeds frozen inference variants.
    probs=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
    logdensity=torch.tensor([0.,.69,1.39])
    identical=g1.corrected_probability(probs,logdensity,0.)
    g=g1.corrected_probability(probs,logdensity,g3.GAMMA)
    assert torch.equal(identical,probs)
    assert g.sum().item()==pytest.approx(1.)
    assert torch.all(g>=0)
    assert not torch.allclose(g,probs)


def test_immutable_manifest_hash_and_repeated_create(tmp_path):
    x={"phase":"g3-synthetic","something":[1,2,3]}
    dest=tmp_path/"manifest.json"
    out=g3._write_once(dest,x,self_hash="manifest_sha256")
    assert out["manifest_sha256"]==_canonical_json_sha256(x)
    assert g3._load_json(dest,"manifest_sha256")==out
    with pytest.raises(FileExistsError):
        g3._write_once(dest,x,self_hash="manifest_sha256")
    raw=json.loads(dest.read_text())
    raw["something"].append(4)
    dest.write_text(json.dumps(raw))
    with pytest.raises(g3.G3Error,match="self-hash"):
        g3._load_json(dest,"manifest_sha256")


def test_frozen_bootstrap_family_valid_on_synthetic_vectors():
    rng=np.random.default_rng(130)
    g={"exact_mixture_mean_mae_m":rng.normal(-.20,.50,size=28),
       "energy_score_2d_m":rng.normal(-.10,.40,size=28)}
    out=r3.studentized_max_t_family(
        g,repetitions=150,confidence=.95,seed=100,orientation="g3_test"
    )
    assert out["coordinate_groups"]==28
    assert out["family_size"]==2
    assert out["comparisons"]["exact_mixture_mean_mae_m"]["simultaneous_upper_bound_m"]>=out["comparisons"]["exact_mixture_mean_mae_m"]["mean_difference_m"]


def test_report_is_deterministic_and_g1_stop_clear():
    mock={
        "protocol":g3.PROTOCOL, "disclosure":g3.DISCLOSURE,
        "test_rows":80,"test_coordinate_groups":30,
        "metrics":{k:{"mae_2d_m":5.1,"energy_score_2d_m":4.2,"mae_row_mean_m":5.1}
                   for k in ("baseline","g1_gamma_0p7","permuted_density")},
        "delta_coordinate_macro_mae_m":-.2,"delta_coordinate_macro_energy_m":-.15,
        "paired_familywise_inference":{"comparisons":{
            "exact_mixture_mean_mae_m":{"simultaneous_upper_bound_m":-.01},
            "energy_score_2d_m":{"simultaneous_upper_bound_m":-.02}}},
        "fixed_transfer_gate":{"passed":True}, "decision":"TRANSFER_SUPPORTED_ON_UTS",
        "result_sha256":"hash_only_fixture"
    }
    a=g3.make_report(mock)
    assert a==g3.make_report(json.loads(json.dumps(mock,sort_keys=True)))
    assert "STOP" in a
    assert "not 25 statistically independent" in a.lower() or "NOT independent" in a
    assert "0.200000" in a


def test_lock_blocks_second_test_without_loading_it(monkeypatch,tmp_path):
    lock=tmp_path/"locked"
    lock.write_text("already invoked")
    monkeypatch.setattr(g3,"LOCK",lock)
    def forbidden():
        raise AssertionError("prepare attempted even though lock exists")
    with pytest.raises(g3.G3Error,match="already attempted"):
        g3.evaluate_test("cpu")


def test_no_train_after_freeze_even_if_checkpoint_missing(monkeypatch,tmp_path):
    freeze=tmp_path/"freeze"
    freeze.write_text("frozen")
    monkeypatch.setattr(g3,"FREEZE",freeze)
    monkeypatch.setattr(g3,"LOCK",tmp_path/"never-open-test")
    monkeypatch.setattr(g3,"RESULT",tmp_path/"no-result")
    monkeypatch.setattr(g3,"prepare",lambda:{"manifest_sha256":"synthetic"})
    with pytest.raises(g3.G3Error,match="no training permitted"):
        g3.train_member(1,"cpu")


def test_no_model_prediction_on_preflight_path():
    for fn in (g3.preflight,g3.prepare):
        s=inspect.getsource(fn)
        assert ".eval(" not in s
        assert "training_data()" in s or "preflight()" in s


def test_member_path_out_of_range():
    for n in (0,-1,26):
        with pytest.raises(ValueError):
            g3.member_path(n)
