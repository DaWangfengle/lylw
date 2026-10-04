from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import v18_4_coupling_stratum_probe as p
from fhm_dapp_tim.fhm_dapp import v18_4_mechanism_probe as old
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.v18_4_core import tail_mass_diagnostics
from fhm_dapp_tim.fhm_dapp.v18_4_model import CorrectionReference, apply_correction, _masked_geometry_energy


def production_fixture(device="cpu", dtype=torch.float64):
    q = torch.tensor([[.5, .25, .25, 0.], [.125, .375, .5, 0.]], dtype=dtype, device=device)
    mask = q > 0
    raw = {"radio_energy_raw": q.new_tensor([[0., 2., 6., 0.], [4., 1., 0., 0.]]),
           "geometry_energy_raw": q.new_tensor([[3., 1., 0., 0.], [0., 2., 5., 0.]]),
           "style_offset": q.new_zeros(2), "style_scale": q.new_ones(2)}
    reference = CorrectionReference(q.new_tensor(2.), q.new_tensor(3.),
        torch.linspace(0., 1., 64, dtype=dtype, device=device),
        torch.linspace(0., 3., 64, dtype=dtype, device=device))
    full = apply_correction(base_probability=q, candidate_mask=mask, raw=raw, reference=reference, epsilon_kl=.025)
    return q, mask, full


def fake_surface():
    scenarios = {}
    for name in ("fit_development", "fit_calibration", "refit_calibration"):
        cells = {}
        for c in p.cell_definitions():
            effects = {m: {"mean_delta": 1.} for m in p.METRICS}
            cells[p.cell_key(c["temperature"], c["mode"], c["epsilon_kl"])] = {**c, "paired_correction_effect": effects}
        scenarios[name] = {"cells": cells}
    return scenarios


def set_benefit(scenarios, name, temperatures, eps=.025, mode="radio_only", mae=0.):
    for t in temperatures:
        effects = scenarios[name]["cells"][p.cell_key(t, mode, eps)]["paired_correction_effect"]
        effects["energy_score_m"]["mean_delta"] = -.2
        effects["nll"]["mean_delta"] = -.1
        effects["posterior_mean_mae_m"]["mean_delta"] = mae


def geometry_fixture(device="cpu"):
    q = torch.tensor([[.7, .29, .01, 0.]], dtype=torch.float64, device=device)
    mask = q > 0
    building = torch.tensor([[0, 0, 1, 9]], device=device)
    floor = torch.tensor([[0, 0, 0, 9]], device=device)
    anchors = q.new_tensor([[[0., 0.], [20., 0.], [200., 0.], [9999., 9999.]]])
    components = q.new_tensor([[[0., 0.], [2., 0.], [201., 0.], [-9999., -9999.]]])
    kwargs = {"spatial_scale_m": 10., "sigma_scale": 1.5, "margin_scale": .5}
    prior = q * mask
    prior = prior / prior.sum(1, keepdim=True)
    production = _masked_geometry_energy(prior, anchors, mask, building, floor, **kwargs)
    return q, mask, building, floor, anchors, components, kwargs, production


def hypothesis_fixture():
    return {name: {"temperature": p.SEMANTIC[name], "variants": {
        g: {"mean_argmin_component_error_m": 100., "mean_row_spearman": .2,
            "low_mass_subsets": {"0.05": {"fraction": .5}}} for g in p.GEOMETRIES}} for name in p.PRIMARY}


def test_exact_registered_surface_and_scenario_packages():
    assert p.TEMPERATURES == (.5,.75,1.,1.25,1.5,2.,2.5,3.) == p.CALIBRATION_TEMPERATURE_GRID
    assert p.EPSILONS == (0.,.025,.05,.1)
    assert p.MODES == ("none","radio_only","geometry_only","full")
    cells = p.cell_definitions()
    assert len(cells) == 104
    assert len({p.cell_key(c["temperature"],c["mode"],c["epsilon_kl"]) for c in cells}) == 104
    assert sum(c["mode"] == "none" for c in cells) == 8
    assert all(c["epsilon_kl"] == 0 for c in cells if c["mode"] == "none")
    assert all(c["scale_multiplier"] == 1 for c in cells)
    assert p.SCENARIOS[0]["checkpoint"] == p.SCENARIOS[1]["checkpoint"]
    assert p.SCENARIOS[0]["reference_role"] == p.SCENARIOS[1]["reference_role"] == "fit"
    assert p.SCENARIOS[1]["role"] == p.SCENARIOS[2]["role"] == "calibration"
    assert not any(c["role"] == "development" and c["reference_role"] == "selection_refit" for c in p.SCENARIOS)


@pytest.mark.parametrize("defect", ["missing", "extra", "wrong_temperature", "duplicate_definition"])
def test_surface_missing_extra_or_duplicate_fails_closed(defect):
    cells = fake_surface()[p.PRIMARY[0]]["cells"]
    p.assert_surface_complete(cells)
    first = next(iter(cells))
    if defect == "missing":
        cells.pop(first)
    elif defect == "extra":
        cells["extra"] = cells[first]
    elif defect == "wrong_temperature":
        cells["T9|none|eps0"] = cells.pop(first)
    else:
        cells.pop(first)
        cells["duplicate"] = next(iter(cells.values()))
    with pytest.raises(RuntimeError, match="surface cell"):
        p.assert_surface_complete(cells)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_surface_identity_production_equality_and_budgets(device, dtype):
    q, mask, production = production_fixture(device, dtype)
    surface = p.surface_posteriors(q, mask, production)
    assert len(surface) == 13
    for (mode, eps), correction in surface.items():
        assert bool(torch.isfinite(correction["posterior"]).all())
        assert bool((correction["posterior"][~mask] == 0).all())
        assert bool((correction["realized_kl"] <= correction["epsilon_kl_effective"] + p.KL_TOLERANCE).all())
        if eps == 0:
            assert torch.equal(correction["posterior"], q)
            assert bool((correction["realized_kl"] == 0).all())
            assert bool((correction["rho"] == 0).all())
        if mode in ("radio_only","geometry_only"):
            inactive = "geometry" if mode == "radio_only" else "radio"
            active = "radio" if mode == "radio_only" else "geometry"
            assert bool((correction[inactive+"_energy"] == 0).all())
            assert bool((correction[inactive+"_reliability"] == 0).all())
            assert torch.equal(correction["epsilon_kl_effective"], production[active+"_reliability"].double()*eps)
    assert torch.equal(surface[("full",.025)]["posterior"], production["posterior"])
    assert surface[("none",0.)]["posterior"] is q


@pytest.mark.parametrize("defect", ["nonfinite_energy", "padding", "nonfinite_q", "KL"])
def test_surface_numerical_failures_stop(defect):
    q, mask, production = production_fixture()
    if defect == "nonfinite_energy":
        production["radio_energy"][0,0] = float("nan")
    elif defect == "padding":
        production["posterior"][0,-1] = .001
    elif defect == "nonfinite_q":
        q[0,0] = float("nan")
    else:
        production["epsilon_kl_effective"] = torch.zeros(2,dtype=q.dtype)
    with pytest.raises((RuntimeError, ValueError)):
        p.surface_posteriors(q,mask,production)


@pytest.mark.parametrize("values,classification", [
    ([-1.]*8,"SIGN_STABLE_BENEFICIAL"), ([1.]*8,"SIGN_STABLE_HARMFUL"),
    ([-1.]*7+[1.],"TEMPERATURE_DEPENDENT"), ([-1.]*7+[0.],"TEMPERATURE_DEPENDENT"),
    ([0.]*8,"TEMPERATURE_DEPENDENT"),
])
def test_temperature_sign_map_all_eight_and_zero_boundary(values, classification):
    record = p.sign_classification(values)
    assert record["classification"] == classification
    assert record["sign_vector"] == list(np.sign(values).astype(int))


def test_temperature_map_requires_full_grid_and_finite_values():
    with pytest.raises(RuntimeError,match="eight"):
        p.sign_classification([-1.]*7)
    with pytest.raises(RuntimeError,match="non-finite"):
        p.sign_classification([-1.]*7+[float("nan")])


@pytest.mark.parametrize("name,mode,triggers", [
    ("fit_development","radio_only",True), ("refit_calibration","full",True),
    ("fit_calibration","radio_only",False), ("fit_development","geometry_only",False),
])
def test_joint_selection_only_registered_primary_radio_full(name,mode,triggers):
    scenarios = fake_surface()
    scenarios[name]["cells"][p.cell_key(1.,mode,.05)]["paired_correction_effect"]["nll"]["mean_delta"] = -.1
    report = p.coupling_maps(scenarios)
    assert report["temperature_and_correction_require_joint_selection"] is triggers
    assert len(report["scenarios"][name]) == 9


def test_mixed_stable_metric_directions_are_not_jointly_separable():
    scenarios = fake_surface()
    set_benefit(scenarios,p.PRIMARY[0],p.TEMPERATURES,mae=.01)
    report = p.coupling_maps(scenarios)
    record = next(r for r in report["scenarios"][p.PRIMARY[0]] if r["mode"]=="radio_only" and r["epsilon_kl"]==.025)
    assert record["metrics"]["posterior_mean_mae_m"]["classification"] == "SIGN_STABLE_HARMFUL"
    assert record["classification"] == "NONSEPARABLE_FROM_TEMPERATURE"


@pytest.mark.parametrize("mode", ["radio_only","full"])
@pytest.mark.parametrize("temperatures,passes", [
    ([.75,1.],True), ([2.5,3.],True), ([1.],False), ([.5,1.],False), ([],False),
])
def test_robust_region_adjacency_requires_two_common_registered_temperatures(mode, temperatures, passes):
    scenarios = fake_surface()
    for name in p.PRIMARY:
        set_benefit(scenarios,name,temperatures,mode=mode)
    gate = p.robust_common_region(scenarios,mode)
    assert gate["any_robust_common_region"] is passes
    assert gate["epsilon_reports"][0]["ROBUST_COMMON_REGION"] is passes


def test_robust_region_requires_same_epsilon_and_both_primary_scenarios():
    scenarios = fake_surface()
    set_benefit(scenarios,p.PRIMARY[0],[1.,1.25],eps=.025)
    set_benefit(scenarios,p.PRIMARY[1],[1.,1.25],eps=.05)
    set_benefit(scenarios,"fit_calibration",p.TEMPERATURES,eps=.025)
    assert not p.robust_common_region(scenarios,"radio_only")["any_robust_common_region"]


@pytest.mark.parametrize("mae,passes", [(.05,True),(.05000001,False),(-.1,True)])
def test_radio_mean_error_gate_inclusive_fixed_0p05(mae,passes):
    scenarios = fake_surface()
    for name in p.PRIMARY:
        set_benefit(scenarios,name,[1.,1.25],mae=mae)
    assert p.robust_common_region(scenarios,"radio_only")["any_robust_common_region"] is passes


@pytest.mark.parametrize("metric", ["energy_score_m","nll"])
def test_radio_probabilistic_benefit_is_strictly_negative(metric):
    scenarios = fake_surface()
    for name in p.PRIMARY:
        set_benefit(scenarios,name,[1.,1.25])
        scenarios[name]["cells"][p.cell_key(1.25,"radio_only",.025)]["paired_correction_effect"][metric]["mean_delta"] = 0.
    assert not p.robust_common_region(scenarios,"radio_only")["any_robust_common_region"]


@pytest.mark.parametrize("rho,fraction,passes", [(.31,.8,True),(.30,.8,False),(.31,.75,False)])
def test_radio_intrinsic_median_boundaries_both_roles(rho,fraction,passes):
    scenarios = {name: {"source_informativeness": {f"{t:g}": {"radio": {
        "mean_row_spearman": rho, "fraction_oracle_gain_positive": fraction}} for t in p.TEMPERATURES}} for name in p.PRIMARY}
    scenarios[p.PRIMARY[0]]["source_informativeness"]["0.5"]["radio"]["mean_row_spearman"] = 0.
    assert p.intrinsically_informative(scenarios)["RADIO_INTRINSICALLY_INFORMATIVE"] is passes
    scenarios[p.PRIMARY[1]]["source_informativeness"] = {f"{t:g}": {"radio": {
        "mean_row_spearman": .1, "fraction_oracle_gain_positive": .5}} for t in p.TEMPERATURES}
    assert not p.intrinsically_informative(scenarios)["RADIO_INTRINSICALLY_INFORMATIVE"]


def decomposition_fixture():
    scenarios, cache = {}, {}
    for name in ("fit_development","fit_calibration","refit_calibration"):
        scenarios[name] = {"definition": {"checkpoint_sha256": "fit" if name!="refit_calibration" else "refit",
            "reference_tensor_sha256":"fitref" if name!="refit_calibration" else "refitref", "memory_anchors": 2,
            "lambda_spatial":.1,"selected_epoch":16,"row_index_sha256":"cal" if name!="fit_development" else "dev",
            "coordinate_group_order_sha256":"groups","rows":2,"coordinate_groups":2}, "cells":{}}
        cache[name] = {}
        for mode in ("none","full"):
            key = p.cell_key(1.,mode,0. if mode=="none" else .025)
            baseline, corrected = {"fit_development": (8.,10.), "fit_calibration":(7.,12.), "refit_calibration":(6.,11.)}[name]
            value = baseline if mode=="none" else corrected
            scenarios[name]["cells"][key] = {"temperature":1., "metrics": {m:value for m in
                (*p.METRICS,"entropy_nats","effective_anchor_count","top1_probability")},
                "paired_correction_effect":{m:{"mean_delta":0. if mode=="none" else corrected-baseline} for m in p.METRICS}}
            vector = ({"none": [1.,2.], "full":[2.,6.]} if name=="fit_calibration" else
                      {"none":[1.,1.],"full":[3.,4.]})[mode]
            cache[name][key] = {m:np.asarray(vector) for m in p.METRICS}
    return scenarios,cache


def test_role_shift_metric_and_correction_effect_formulas_exact():
    scenarios,_ = decomposition_fixture()
    result = p.role_shift_decomposition(scenarios)
    assert result["paired_statistical_comparison"] is False
    record = result["cells"][p.cell_key(1.,"full",.025)]
    assert all(v==2. for v in record["role_shift_metric_delta"].values())
    assert all(v==3. for v in record["role_shift_in_correction_effect"].values())


def test_role_shift_rejects_changed_checkpoint_reference_package():
    scenarios,_ = decomposition_fixture()
    scenarios["fit_calibration"]["definition"]["reference_tensor_sha256"] = "changed"
    with pytest.raises(RuntimeError,match="FIT package"):
        p.role_shift_decomposition(scenarios)


def test_paired_refit_group_delta_se_ci_and_difference_in_differences():
    scenarios,cache = decomposition_fixture()
    result = p.refit_decomposition(scenarios,cache)
    record = result["cells"][p.cell_key(1.,"full",.025)]
    for metric in p.METRICS:
        direct = record["paired_refit_metric_delta"][metric]
        assert direct["mean_delta"] == -.5
        assert direct["paired_se"] == pytest.approx(1.5)
        assert direct["normal_95_ci"] == pytest.approx([-3.44,2.44])
        gains = record["paired_refit_effect_on_correction"][metric]
        assert gains["mean_delta"] == 0.
        assert gains["paired_se"] == pytest.approx(1.)
        assert gains["normal_95_ci"] == pytest.approx([-1.96,1.96])


@pytest.mark.parametrize("key", ["row_index_sha256","coordinate_group_order_sha256","rows"])
def test_refit_pairing_fails_on_different_rows_or_group_order(key):
    scenarios,cache = decomposition_fixture()
    scenarios["refit_calibration"]["definition"][key] = "wrong"
    with pytest.raises(RuntimeError,match="rows/groups differ"):
        p.refit_decomposition(scenarios,cache)


@pytest.mark.parametrize("device", ["cpu","cuda"])
def test_exact_building_floor_stratum_mass_ignores_padding(device):
    q = torch.tensor([[.2,.3,.1,.4,99.]],dtype=torch.float64,device=device)
    mask = torch.tensor([[True,True,True,True,False]],device=device)
    building = torch.tensor([[0,0,0,1,0]],device=device)
    floor = torch.tensor([[0,0,1,0,0]],device=device)
    assert torch.allclose(p.stratum_mass(q,mask,building,floor),q.new_tensor([[.5,.5,.1,.4,0.]]),atol=0,rtol=0)


def test_stratum_argmin_indices_mass_errors_and_near_zero_strata():
    q,mask,building,floor,_,_,_,_ = geometry_fixture()
    energy = np.array([[3.,2.,0.,0.]])
    anchor = np.array([[5.,6.,100.,999.]])
    component = np.array([[4.,7.,110.,999.]])
    residual = np.array([[1.,2.,10.,999.]])
    report = p.geometry_stratum_audit(energy,q.numpy(),mask.numpy(),building.numpy(),floor.numpy(),
        anchor,component,residual,np.array([[7,8,9,10]]))
    row = report["row_records"][0]
    assert row["argmin_support_index"] == 2 and row["argmin_anchor_index"] == 9
    assert row["argmin_stratum_mass"] == .01
    assert not row["argmin_stratum_is_q_T_MAP_stratum"]
    assert row["argmin_component_error_m"] == 110.
    assert row["argmin_anchor_error_m"] == 100.
    assert row["argmin_anchor_component_residual_m"] == 10.
    assert row["unique_building_floor_strata"] == 2
    assert row["strata_min_geometry_energy_le_1e_8"] == 1
    assert report["low_mass_subsets"]["0.01"] == {"fraction":0.,"row_count":0,"mean_argmin_component_error_m":None}
    assert report["low_mass_subsets"]["0.05"]["fraction"] == 1.
    assert report["mean_q_T_MAP_component_error_m"] == 4.


def test_low_mass_thresholds_strict_exact_and_subset_error_mean():
    q = np.array([[.009,.991],[.01,.99],[.05,.95]])
    mask = np.ones_like(q,dtype=bool)
    building,floor = np.tile([0,1],(3,1)),np.zeros_like(q)
    distance = np.array([[10.,1.],[20.,1.],[30.,1.]])
    result = p.geometry_stratum_audit(np.tile([0.,1.],(3,1)),q,mask,building,floor,distance,distance,
        np.zeros_like(q),np.tile([0,1],(3,1)))
    assert result["low_mass_subsets"]["0.01"]["row_count"] == 1
    assert result["low_mass_subsets"]["0.05"]["row_count"] == 2
    assert result["low_mass_subsets"]["0.1"]["row_count"] == 3
    assert result["low_mass_subsets"]["0.05"]["mean_argmin_component_error_m"] == 15.


@pytest.mark.parametrize("device", ["cpu","cuda"])
def test_G1_exact_production_and_G2_only_coordinate_replacement(device):
    q,mask,b,f,a,c,kwargs,g1 = geometry_fixture(device)
    result = p.geometry_variants(q,a,c,mask,b,f,spatial_scale_m=10.,production_geometry=g1)
    assert torch.equal(result[p.GEOMETRIES[0]],g1)
    prior = q / q.sum(1,keepdim=True)
    expected_g2 = _masked_geometry_energy(prior,c,mask,b,f,**kwargs)
    assert torch.equal(result[p.GEOMETRIES[1]],expected_g2)
    assert not torch.equal(result[p.GEOMETRIES[1]],g1)
    for conditional,coords,name in ((True,a,p.GEOMETRIES[0]),(True,c,p.GEOMETRIES[1]),
                                    (False,a,p.GEOMETRIES[2]),(False,c,p.GEOMETRIES[3])):
        energy = p.geometry_energy(q,coords,mask,b,f,conditional=conditional,**kwargs)
        assert torch.equal(energy,result[name])
        assert bool(torch.isfinite(energy).all()) and bool((energy>=0).all())
        assert float(energy[0,-1]) == 0.


@pytest.mark.parametrize("device", ["cpu","cuda"])
def test_G3_G4_global_mass_penalty_exact_singleton_formula(device):
    q,mask,b,f,a,c,kwargs,g1 = geometry_fixture(device)
    variants = p.geometry_variants(q,a,c,mask,b,f,spatial_scale_m=10.,production_geometry=g1)
    expected = (max(np.sqrt(-2*15.**2*np.log(.01))-5.,0.)/10.)**2
    assert float(g1[0,2]) == 0.
    for name in p.GEOMETRIES[2:]:
        assert float(variants[name][0,2]) == pytest.approx(expected,rel=1e-12,abs=1e-12)
        assert float(variants[name][0,2]) > float(g1[0,2])
    assert not torch.equal(variants[p.GEOMETRIES[2]],variants[p.GEOMETRIES[3]])


def test_geometry_padding_coordinates_and_strata_do_not_change_valid_energy():
    q,mask,b,f,a,c,kwargs,_ = geometry_fixture()
    for conditional in (True,False):
        baseline = p.geometry_energy(q,a,mask,b,f,conditional=conditional,**kwargs)
        changed = a.clone(); changed[0,-1] = -1e6
        bb = b.clone(); bb[0,-1] = 0
        ff = f.clone(); ff[0,-1] = 0
        assert torch.equal(baseline,p.geometry_energy(q,changed,mask,bb,ff,conditional=conditional,**kwargs))


@pytest.mark.parametrize("error,passes", [(75.,True),(75.000001,False),(74.,True)])
def test_H1_exact_25_percent_boundary_and_both_semantic_scenarios(error,passes):
    semantic = hypothesis_fixture()
    for name in p.PRIMARY:
        semantic[name]["variants"][p.GEOMETRIES[2]]["mean_argmin_component_error_m"] = error
        semantic[name]["variants"][p.GEOMETRIES[2]]["low_mass_subsets"]["0.05"]["fraction"] = .4
    assert p.geometry_hypothesis_gates(semantic)["H1"]["supported"] is passes
    semantic[p.PRIMARY[1]] = hypothesis_fixture()[p.PRIMARY[1]]
    assert not p.geometry_hypothesis_gates(semantic)["H1"]["supported"]


def test_H1_requires_error_and_low_mass_improvement_in_same_comparison():
    semantic = hypothesis_fixture()
    for name in p.PRIMARY:
        semantic[name]["variants"][p.GEOMETRIES[2]]["mean_argmin_component_error_m"] = 70.
        semantic[name]["variants"][p.GEOMETRIES[3]]["low_mass_subsets"]["0.05"]["fraction"] = .4
    assert not p.geometry_hypothesis_gates(semantic)["H1"]["supported"]


@pytest.mark.parametrize("rho,error,passes", [(.25,100.,True),(.249999,100.,False),(.2,90.,True),(.2,90.000001,False)])
def test_H2_exact_spearman_and_error_boundaries_both_semantic_scenarios(rho,error,passes):
    semantic = hypothesis_fixture()
    for name in p.PRIMARY:
        semantic[name]["variants"][p.GEOMETRIES[1]]["mean_row_spearman"] = rho
        semantic[name]["variants"][p.GEOMETRIES[1]]["mean_argmin_component_error_m"] = error
    assert p.geometry_hypothesis_gates(semantic)["H2"]["supported"] is passes
    semantic[p.PRIMARY[1]] = hypothesis_fixture()[p.PRIMARY[1]]
    assert not p.geometry_hypothesis_gates(semantic)["H2"]["supported"]


def test_H1_H2_may_use_different_valid_comparisons_in_each_semantic_scenario():
    semantic = hypothesis_fixture()
    semantic[p.PRIMARY[0]]["variants"][p.GEOMETRIES[2]]["mean_argmin_component_error_m"] = 70.
    semantic[p.PRIMARY[0]]["variants"][p.GEOMETRIES[2]]["low_mass_subsets"]["0.05"]["fraction"] = .4
    semantic[p.PRIMARY[1]]["variants"][p.GEOMETRIES[3]]["mean_argmin_component_error_m"] = 70.
    semantic[p.PRIMARY[1]]["variants"][p.GEOMETRIES[3]]["low_mass_subsets"]["0.05"]["fraction"] = .4
    semantic[p.PRIMARY[0]]["variants"][p.GEOMETRIES[1]]["mean_row_spearman"] = .3
    semantic[p.PRIMARY[1]]["variants"][p.GEOMETRIES[3]]["mean_row_spearman"] = .3
    gate = p.geometry_hypothesis_gates(semantic)
    assert gate["H1"]["supported"] and gate["H2"]["supported"]


def test_shared_prior_cell_fixed_reproduction_tolerance_and_missing_metric():
    checks = p.compare_shared_cell({"energy_score_m":5.+1e-7},{"energy_score_m":5.},"synthetic")
    assert checks[0]["passes"]
    with pytest.raises(RuntimeError,match="reproduction failed"):
        p.compare_shared_cell({"energy_score_m":5.01},{"energy_score_m":5.},"synthetic")
    with pytest.raises(RuntimeError,match="metric missing"):
        p.compare_shared_cell({}, {"nll":5.},"synthetic")


def test_prior_shared_cell_mapping_all_four_modes_and_semantic_temperatures(monkeypatch):
    prior = {"scenarios": {"calibration_frozen": {"modes": {m:{
        "probabilistic_metrics":{"coordinate_macro_energy_score_m":1.,"coordinate_macro_energy_score_m_se":.1,
            "coordinate_macro_nll":2.,"coordinate_macro_nll_se":.2,"mean_entropy_nats":.3},
        "decoders":{"posterior_mean":{"coordinate_macro_mae_m":3.,"coordinate_macro_mae_se_m":.4,"row_mae_m":3.}},
        "tail_diagnostics":{"truth_far_mass_10m":.1,"mean_gm_displacement_from_mean_m":.2}}
        for m in p.MODES}}}}
    cells = {p.cell_key(1.5,m,0. if m=="none" else .025):{"metrics":{
        "energy_score_m":1.,"energy_score_m_se":.1,"nll":2.,"nll_se":.2,"entropy_nats":.3,
        "posterior_mean_mae_m":3.,"posterior_mean_mae_m_se":.4,"row_mae_m":3.,
        "truth_far_mass_10m":.1,"mean_gm_displacement_from_mean_m":.2}} for m in p.MODES}
    assert len(p.prior_reproduction_checks("refit_calibration",1.5,cells,prior)) == 40
    assert p.prior_reproduction_checks("refit_calibration",1.,{},prior) == []
    assert p.prior_reproduction_checks("fit_calibration",1.5,{},prior) == []


def prior_fixture(monkeypatch):
    prior = {"frozen_source_commit_sha":p.FROZEN_SOURCE_SHA,**old.BOUNDARIES}
    old.seal(prior,"result_sha256")
    monkeypatch.setattr(p,"PRIOR_CANONICAL",prior["result_sha256"])
    execution = {"status":"PASS","probe_exit_code":0,"result_self_hash":prior["result_sha256"],
        "frozen_source_commit_sha":p.FROZEN_SOURCE_SHA,"output_sha256":dict(p.PRIOR_EXPECTED)}
    old.seal(execution,"execution_record_sha256")
    return prior,execution


def test_prior_integrity_canonical_and_cross_links_pass(monkeypatch):
    prior,execution = prior_fixture(monkeypatch)
    assert p.validate_prior_payloads(prior,execution)["status"] == "PASS"


@pytest.mark.parametrize("defect", ["prior_hash","execution_hash","execution_fail","result_link","output_link","source_link"])
def test_prior_integrity_failure_paths(monkeypatch,defect):
    prior,execution = prior_fixture(monkeypatch)
    if defect == "prior_hash":
        prior["training_performed"] = True
    elif defect == "execution_hash":
        execution["status"] = "FAIL_CLOSED"
    else:
        if defect == "execution_fail": execution["status"] = "FAIL_CLOSED"
        if defect == "result_link": execution["result_self_hash"] = "wrong"
        if defect == "output_link": execution["output_sha256"][next(iter(p.PRIOR_EXPECTED))] = "wrong"
        if defect == "source_link": execution["frozen_source_commit_sha"] = "wrong"
        old.seal(execution,"execution_record_sha256")
    with pytest.raises(RuntimeError):
        p.validate_prior_payloads(prior,execution)


def test_prior_byte_hash_mismatch_stops_before_payload_io(monkeypatch):
    monkeypatch.setattr(old,"verify_integrity",lambda: ({},{}) )
    monkeypatch.setattr(p,"sha256_file",lambda path: "wrong")
    with pytest.raises(RuntimeError,match="prior file-byte"):
        p.verify_integrity()


@pytest.mark.parametrize("device", ["cpu","cuda"])
def test_surface_scoring_common_draws_frozen_tails_and_mean_only(device):
    q,mask,production = production_fixture(device,dtype=torch.float32)
    locations = q.new_tensor([[[0.,0.],[15.,0.],[40.,0.],[999.,999.]],[[0.,0.],[15.,0.],[40.,0.],[999.,999.]]])
    scale, target = torch.ones_like(locations),q.new_zeros(2,2)
    draws = common_student_t_randomness(2,draws=8,seed=p.SCORING_SEED)
    draws = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in draws.items()}
    digest = p.randomness_hash(draws)
    class StudentModel:
        def student_log_prob(self,target,location,scale):
            return -((location-target[:,None,:])/scale).square().sum(2)
    model = StudentModel()
    for correction in p.surface_posteriors(q,mask,production).values():
        weights = correction["posterior"]
        values = p.row_surface_metrics(weights,locations,scale,target,mask,model,draws)
        expected = tail_mass_diagnostics(weights,locations,target_coords=target,mask=mask)
        for name,value in expected.items():
            if name.startswith(("truth_far_mass_","mode_far_mass_")):
                assert torch.equal(values[name],value)
        assert torch.equal(values["posterior_mean_mae_m"],torch.linalg.vector_norm((weights[:,:,None]*locations).sum(1)-target,dim=1))
        assert p.randomness_hash(draws) == digest
        assert not any("map_error" in key or "top8" in key for key in values)


def test_macro_cell_summary_equal_groups_and_tail_percentile():
    values = {m:np.array([0.,2.,10.]) for m in p.METRICS}
    for name in ("entropy_nats","effective_anchor_count","top1_probability","realized_kl","epsilon_kl_effective","rho","radio_reliability","geometry_reliability","gm_displacement_m"):
        values[name] = np.array([0.,2.,10.])
    values["truth_far_mass_10m"] = np.array([0.,0.,1.])
    metrics = p.summarize_cell(values,["a","a","b"])
    for name in p.METRICS:
        assert metrics[name] == 5.5
        assert metrics[name+"_se"] == pytest.approx(4.5)
    assert metrics["row_mae_m"] == 4.
    assert metrics["truth_far_mass_10m"] == pytest.approx(1/3)
    assert metrics["p90_gm_displacement_from_mean_m"] == pytest.approx(8.4)


def test_residual_distribution_quantiles_and_strict_thresholds():
    report = p.distribution_summary([5.,10.,20.,30.],displacement=True)
    assert report["fraction_gt_5m"] == .75
    assert report["fraction_gt_10m"] == .5
    assert report["fraction_gt_20m"] == .25
    assert report["p95"] == pytest.approx(28.5)


def test_write_once_never_overwrites_existing_output(tmp_path):
    path = tmp_path/"result.json"
    p.write_once(path,b"first\n")
    with pytest.raises(FileExistsError): p.write_once(path,b"second\n")
    with pytest.raises(RuntimeError,match="already exists"): p.assert_unused([path])
    assert path.read_bytes() == b"first\n"


@pytest.mark.parametrize("relative", ["data/UJIndoorLoc/validationData.csv","data/UTS/Test_rss.csv",
    "data/Tampere/Test_rss.csv","data/TUJI1/test.json","fhm_dapp_tim/evidence/v18_4/external/results.json"])
def test_forbidden_file_open_audit_fails_before_io(tmp_path,relative):
    with p.FileOpenAudit(tmp_path) as audit:
        with pytest.raises(RuntimeError,match="file-open audit denied"):
            (tmp_path/relative).open("rb")
    assert len(audit.denied) == 1


def test_only_training_reads_recorded_and_writes_denied(tmp_path):
    train = tmp_path/"data/UJIndoorLoc/trainingData.csv"
    train.parent.mkdir(parents=True); train.write_bytes(b"synthetic\n")
    with p.FileOpenAudit(tmp_path) as audit:
        assert train.read_bytes() == b"synthetic\n"
        with pytest.raises(RuntimeError): train.open("wb")
    assert [o["allowed"] for o in audit.opens] == [True,False]
    assert train.read_bytes() == b"synthetic\n"


@pytest.mark.parametrize("relative", ["fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py",p.MODULE,
    "tests/test_v18_4_mechanism_probe.py",p.TEST_MODULE,
    "fhm_dapp_tim/models/v18_4/model.pt","fhm_dapp_tim/evidence/v18_4/MECHANISM_PROBE_V18_4.json"])
def test_all_frozen_and_new_source_tests_and_prior_artifact_writes_forbidden(tmp_path,relative):
    path = tmp_path/relative
    path.parent.mkdir(parents=True); path.write_bytes(b"preserve")
    with p.FileOpenAudit(tmp_path):
        with pytest.raises(RuntimeError,match="write forbidden"): path.open("wb")
    assert path.read_bytes() == b"preserve"


def test_new_outputs_allowed_but_old_probe_outputs_protected(tmp_path):
    audit = p.FileOpenAudit(tmp_path)
    for path in p.OUTPUTS:
        audit.check(tmp_path/path.relative_to(p.REPO_ROOT),"xb")
    for path in old.OUTPUTS:
        with pytest.raises(RuntimeError,match="write forbidden"):
            audit.check(tmp_path/path.relative_to(p.REPO_ROOT),"wb")


AUTH = {"radio_temporary_adam_allowed":True,"scope":"existing two-tensor six-step radio authorization"}


def test_no_training_optimizer_network_backward_and_state_grad_integrity():
    model = torch.nn.Linear(2,2).eval()
    before = p.tensor_state_hash(model.state_dict())
    with p.ReadOnlyModelGuard(model,AUTH) as guard:
        with pytest.raises(RuntimeError,match="training mode"): model.train()
        with pytest.raises(RuntimeError,match="optimizer forbidden"): torch.optim.Adam(model.parameters())
        with pytest.raises(RuntimeError,match="backward forbidden"): torch.tensor(1.,requires_grad=True).backward()
    assert guard.verification["before_sha256"] == guard.verification["after_sha256"] == before
    assert all(param.grad is None for param in model.parameters())


def test_model_buffer_mutation_fails_closed():
    model = torch.nn.Linear(2,2).eval()
    model.register_buffer("frozen_buffer",torch.tensor([1.]))
    with pytest.raises(RuntimeError,match="changed bitwise"):
        with p.ReadOnlyModelGuard(model,AUTH): model.frozen_buffer.add_(1.)


def test_model_gradient_appearance_fails_closed():
    model = torch.nn.Linear(2,2).eval()
    with pytest.raises(RuntimeError,match="gradient created"):
        with p.ReadOnlyModelGuard(model,AUTH): model.weight.grad = torch.ones_like(model.weight)


def test_gate_requires_pass_count_complete_suite_and_tested_hashes(monkeypatch):
    paths = {p.MODULE,p.TEST_MODULE,*p.TEST_FILES,*old.FORMAL_EXPECTED,"fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py"}
    receipt = {"compile":{"exit_code":0},"pytest":{"exit_code":0,"test_files":list(p.TEST_FILES),"new_tests":2,
        "passed":130,"failed":0,"errors":0,"skipped":0,"xfailed":0,"xpassed":0},
        "human_authorization":AUTH,"tested_file_sha256":{path:"digest" for path in paths}}
    monkeypatch.setattr(p,"sha256_file",lambda path:"digest")
    monkeypatch.setenv("V184_COUPLING_GATE_RECEIPT",json.dumps(receipt))
    assert p.gate_receipt()["pytest"]["passed"] == 130
    receipt["pytest"]["skipped"] = 1
    monkeypatch.setenv("V184_COUPLING_GATE_RECEIPT",json.dumps(receipt))
    with pytest.raises(RuntimeError,match="count/status"): p.gate_receipt()
    receipt["pytest"]["skipped"] = 0
    receipt["tested_file_sha256"][p.MODULE] = "changed"
    monkeypatch.setenv("V184_COUPLING_GATE_RECEIPT",json.dumps(receipt))
    with pytest.raises(RuntimeError,match="changed after gate"): p.gate_receipt()
