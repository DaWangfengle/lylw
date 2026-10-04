from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import v18_4_mechanism_probe as probe
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness, continuous_energy_score_mc,
    profile_shared_affine_marked_energy,
)
from fhm_dapp_tim.fhm_dapp.v18_4_core import kl_p_to_q, weighted_component_geometric_median
from fhm_dapp_tim.fhm_dapp.v18_4_model import CorrectionReference, apply_correction


def production_fixture(device: str = "cpu"):
    q = torch.tensor([[.5, .25, .25, 0.], [.125, .375, .5, 0.]], dtype=torch.float64, device=device)
    mask = q > 0
    raw = {
        "radio_energy_raw": torch.tensor([[0., 2., 6., 0.], [4., 1., 0., 0.]], dtype=q.dtype, device=device),
        "geometry_energy_raw": torch.tensor([[3., 1., 0., 0.], [0., 2., 5., 0.]], dtype=q.dtype, device=device),
        "style_offset": torch.zeros(2, dtype=q.dtype, device=device),
        "style_scale": torch.ones(2, dtype=q.dtype, device=device),
    }
    reference = CorrectionReference(
        radio_energy_scale=q.new_tensor(2.), geometry_energy_scale=q.new_tensor(3.),
        radio_mismatch_reference=torch.linspace(0., 1., 64, dtype=q.dtype, device=device),
        geometry_mismatch_reference=torch.linspace(0., 3., 64, dtype=q.dtype, device=device))
    full = apply_correction(base_probability=q, candidate_mask=mask, raw=raw,
                            reference=reference, epsilon_kl=.025)
    return q, mask, full


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_none_bitwise_and_full_exact_production(device):
    q, mask, production = production_fixture(device)
    modes = probe.decompose_correction(q, mask, production)
    assert tuple(modes) == probe.MODES
    assert torch.equal(modes["none"]["posterior"], q)
    assert torch.equal(modes["full"]["posterior"], production["posterior"])
    for key in ("rho", "realized_kl", "epsilon_kl_effective"):
        assert bool((modes["none"][key] == 0).all())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_source_only_budget_and_inactive_source_exact(device):
    q, mask, production = production_fixture(device)
    modes = probe.decompose_correction(q, mask, production)
    for active, inactive in (("radio", "geometry"), ("geometry", "radio")):
        mode = modes[f"{active}_only"]
        assert torch.equal(mode["epsilon_kl_effective"], production[f"{active}_reliability"].double() * .025)
        assert bool((mode[f"{inactive}_energy"] == 0).all())
        assert bool((mode[f"{inactive}_reliability"] == 0).all())
        assert torch.equal(mode[f"{active}_energy"], production[f"{active}_energy"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decomposition_normalized_padding_and_actual_kl(device):
    q, mask, production = production_fixture(device)
    modes = probe.decompose_correction(q, mask, production)
    for mode in modes.values():
        p = mode["posterior"]
        assert torch.allclose(p.sum(1), torch.ones(2, dtype=q.dtype, device=device), atol=1e-14, rtol=0)
        assert bool((p[~mask] == 0).all())
        actual = kl_p_to_q(p.double(), q.double(), mask).clamp(min=0)
        assert bool((actual <= mode["epsilon_kl_effective"] + 1e-10).all())
        assert torch.allclose(actual, mode["realized_kl"], atol=1e-14, rtol=0)


def test_zero_source_reliability_preserves_base_exactly():
    q, mask, production = production_fixture()
    production["radio_reliability"] = torch.zeros_like(production["radio_reliability"])
    mode = probe.decompose_correction(q, mask, production)["radio_only"]
    assert torch.equal(mode["posterior"], q)
    assert bool((mode["epsilon_kl_effective"] == 0).all())
    assert bool((mode["rho"] == 0).all())


def test_posterior_padding_violation_fails_closed():
    q, mask, production = production_fixture()
    production["posterior"] = production["posterior"].clone()
    production["posterior"][0, -1] = .01
    with pytest.raises(RuntimeError, match="padding received"):
        probe.decompose_correction(q, mask, production)


def test_actual_kl_budget_violation_fails_closed():
    q, mask, production = production_fixture()
    production["epsilon_kl_effective"] = torch.zeros_like(production["epsilon_kl_effective"])
    with pytest.raises(RuntimeError, match="KL exceeds"):
        probe.decompose_correction(q, mask, production)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decoders_exact_mean_map_and_core_gm(device):
    p = torch.tensor([[.25, .5, .25, 0.]], dtype=torch.float64, device=device)
    points = torch.tensor([[[0., 0.], [4., 0.], [0., 4.], [999., 999.]]], dtype=p.dtype, device=device)
    mask = p > 0
    result = probe.decode_points(p, points, mask)
    assert torch.equal(result["posterior_mean"], p.new_tensor([[2., 1.]]))
    assert torch.equal(result["map_component"], p.new_tensor([[4., 0.]]))
    assert torch.equal(result["top8_mean"], result["posterior_mean"])
    assert torch.equal(result["geometric_median"], weighted_component_geometric_median(p, points, mask=mask))


def test_top8_valid_only_and_renormalized():
    p = torch.arange(1., 11., dtype=torch.float64).unsqueeze(0)
    p = torch.cat((p / p.sum(), torch.zeros(1, 2, dtype=p.dtype)), 1)
    mask = p > 0
    coords = torch.stack((torch.arange(12., dtype=p.dtype), torch.zeros(12, dtype=p.dtype)), -1).unsqueeze(0)
    coords[0, -2:] = 1e9
    result = probe.decode_points(p, coords, mask)
    selected = torch.arange(2, 10)
    expected = (p[:, selected, None] * coords[:, selected]).sum(1) / p[:, selected].sum(1, keepdim=True)
    assert torch.allclose(result["top8_mean"], expected, atol=1e-14, rtol=0)
    assert float(result["top8_mean"][0, 0]) < 10


def test_top8_ties_use_stable_valid_candidate_order():
    p = torch.full((1, 10), .1, dtype=torch.float64)
    coords = torch.stack((torch.arange(10., dtype=p.dtype), torch.zeros(10, dtype=p.dtype)), -1).unsqueeze(0)
    result = probe.decode_points(p, coords, torch.ones_like(p, dtype=torch.bool))
    assert float(result["top8_mean"][0, 0]) == pytest.approx(3.5)


def test_single_valid_component_all_decoders_equal():
    p = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    coords = torch.tensor([[[999., 999.], [7., 9.], [-999., -999.]]], dtype=p.dtype)
    result = probe.decode_points(p, coords, p > 0)
    assert all(torch.equal(value, p.new_tensor([[7., 9.]])) for value in result.values())


def test_coordinate_macro_equal_groups_not_rows():
    values, groups = [0., 2., 10.], ["a", "a", "b"]
    metrics = probe.decoder_metrics(values, groups)
    assert metrics["coordinate_macro_mae_m"] == 5.5
    assert metrics["coordinate_macro_mae_se_m"] == pytest.approx(4.5)
    assert metrics["row_mae_m"] == 4.
    assert metrics["row_median_error_m"] == 2.
    assert metrics["row_p75_error_m"] == 6.
    assert metrics["row_p90_error_m"] == pytest.approx(8.4)


def test_paired_delta_sign_se_ci_and_group_improved_fraction():
    # Row deltas [-1,-3,+2]; group deltas [-2,+2] have mean 0, sample SE 2.
    stats = probe.paired_statistics([9., 7., 12.], [10., 10., 10.], ["a", "a", "b"])
    assert stats["mean_delta"] == 0.
    assert stats["paired_se"] == pytest.approx(2.)
    assert stats["normal_95_ci"] == pytest.approx([-3.92, 3.92])
    assert stats["fraction_groups_improved"] == .5
    positive = probe.paired_statistics([10., 10., 10.], [9., 7., 12.], ["a", "a", "b"], positive_is_better=True)
    assert positive["mean_delta"] == -stats["mean_delta"]
    assert positive["fraction_groups_improved"] == .5


def test_paired_improvement_fraction_counts_groups_and_ties():
    groups = ["a", "a", "a", "b", "c"]
    stats = probe.paired_statistics([1., 1., 1., 3., 2.], [2.] * 5, groups)
    assert stats["fraction_groups_improved"] == pytest.approx(1 / 3)
    assert stats["coordinate_groups"] == 3
    single = probe.paired_statistics([1., 3.], [2., 4.], ["a", "a"])
    assert single["mean_delta"] == -1.
    assert single["paired_se"] == 0.
    assert single["normal_95_ci"] == [-1., -1.]


@pytest.mark.parametrize("direction,expected", [(1, 1.), (-1, -1.)])
def test_spearman_perfect_or_reversed(direction, expected):
    assert probe.spearman([1., 2., 3., 4.], np.arange(4.) * direction) == pytest.approx(expected)


def test_spearman_ties_constant_and_single_finite_deterministic():
    assert np.array_equal(probe.average_ranks([3., 1., 1., 4.]), [2., .5, .5, 3.])
    assert probe.spearman([1., 1., 2., 3.], [1., 1., 2., 3.]) == pytest.approx(1.)
    assert probe.spearman([1., 1., 1.], [1., 2., 3.]) == 0.
    assert probe.pearson([3.], [7.]) == 0.
    assert probe.spearman([2., 2., 3.], [1., 2., 3.]) == probe.spearman([2., 2., 3.], [1., 2., 3.])


def test_source_energy_truth_alignment_excludes_padding_and_beats_map():
    distance = np.array([[1., 2., 4., 999.], [4., 2., 1., 999.]])
    energy = np.array([[0., 1., 2., -999.], [2., 1., 0., -999.]])
    q = np.array([[.1, .2, .7, 0.], [.7, .2, .1, 0.]])
    report = probe.source_energy_alignment(energy, distance, q, q > 0)
    assert report["mean_row_spearman"] == pytest.approx(1.)
    assert report["fraction_spearman_positive"] == 1.
    assert report["source_argmin_candidate_mean_error_m"] == 1.
    assert report["q_T_map_candidate_mean_error_m"] == 4.
    assert report["fraction_source_argmin_beats_q_T_map"] == 1.
    reversed_report = probe.source_energy_alignment(-energy, distance, q, q > 0)
    assert reversed_report["mean_row_spearman"] == pytest.approx(-1.)


@pytest.mark.parametrize("size", [4, 9, 17])
def test_reliability_quartiles_cover_every_row_once_and_stable_ties(size):
    r = np.full(size, .5)
    quartiles = probe.reliability_quartiles(r)
    assert np.array_equal(np.concatenate(quartiles), np.arange(size))
    assert max(map(len, quartiles)) - min(map(len, quartiles)) <= 1
    report = probe.reliability_utility(r, np.arange(size, dtype=float) - 2.)
    assert sum(q["row_count"] for q in report["quartiles"]) == size
    assert set(report["quartile_assignment_role_row_order"]) == {1, 2, 3, 4}
    assert report["spearman_reliability_oracle_gain"] == 0.
    assert report["pearson_reliability_oracle_gain"] == 0.


def test_reliability_utility_gain_sign_correlations_and_quartile_summary():
    r = np.array([.1, .2, .3, .4, .5, .6, .7, .8])
    gain = np.arange(8.) - 3.
    report = probe.reliability_utility(r, gain)
    assert report["mean_oracle_gain_m"] == .5
    assert report["median_oracle_gain_m"] == .5
    assert report["fraction_oracle_gain_positive"] == .5
    assert report["fraction_oracle_gain_negative"] == .375
    assert report["spearman_reliability_oracle_gain"] == pytest.approx(1.)
    assert report["pearson_reliability_oracle_gain"] == pytest.approx(1.)
    assert report["quartiles"][0]["mean_oracle_gain_m"] == -2.5
    assert report["quartiles"][3]["fraction_oracle_gain_positive"] == 1.


def test_radio_geometry_agreement_uses_valid_support_and_source_reliabilities():
    radio = np.array([[0., 1., 2., -9.], [0., 1., 2., -9.]])
    geometry = np.array([[0., 1., 2., -9.], [2., 1., 0., -9.]])
    points = np.array([[[0., 0.], [1., 0.], [2., 0.], [999., 999.]]] * 2)
    mask = np.array([[True, True, True, False]] * 2)
    report = probe.source_agreement(radio, geometry, points, mask, [.2, .8], [.8, .2])
    assert report["mean_row_energy_spearman"] == pytest.approx(0.)
    assert report["fraction_row_energy_spearman_positive"] == .5
    assert report["fraction_same_energy_argmin_candidate"] == .5
    assert report["mean_argmin_spatial_distance_m"] == 1.
    assert report["pearson_source_reliabilities"] == pytest.approx(-1.)
    assert report["spearman_source_reliabilities"] == pytest.approx(-1.)


def test_variable_width_support_concatenation_preserves_mask_and_locations():
    first = np.array([[True, False]])
    second = np.array([[True, True, False]])
    mask = probe.concat_support_arrays([first, second])
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, [[True, False, False], [True, True, False]])
    points = probe.concat_support_arrays([np.ones((1, 2, 2)), np.full((1, 3, 2), 2.)])
    assert points.shape == (2, 3, 2)
    assert np.array_equal(points[0, 2], [0., 0.])
    assert np.array_equal(probe.concat_support_arrays([np.array([1.]), np.array([2., 3.])]), [1., 2., 3.])


def decisions_fixture():
    result = {}
    for name in (*probe.PRIMARY, "calibration_T1_control"):
        result[name] = {
            "paired_correction_effects": {mode: {metric: {"mean_delta": -.1} for metric in probe.DELTA_METRICS} for mode in probe.MODES[1:]},
            "modes": {"none": {"decoders": {d: {"coordinate_macro_mae_m": 10. if d == "posterior_mean" else 11.} for d in probe.DECODERS}}}}
    return result


def set_deltas(scenarios, mode, es, nll, mae, names=probe.PRIMARY):
    for name in names:
        for key, value in zip(probe.DELTA_METRICS, (es, nll, mae)):
            scenarios[name]["paired_correction_effects"][mode][key]["mean_delta"] = value


def test_correction_beneficial_harmful_mixed_and_overall_full():
    scenarios = decisions_fixture()
    set_deltas(scenarios, "radio_only", -.1, -.2, .05)
    set_deltas(scenarios, "geometry_only", .1, .2, 0.)
    set_deltas(scenarios, "full", -.1, .1, 0.)
    report = probe.primary_correction_decision(scenarios)
    assert report["classifications"] == {"radio_only": "CONSISTENTLY_BENEFICIAL", "geometry_only": "CONSISTENTLY_HARMFUL", "full": "MIXED_OR_NEUTRAL"}
    assert report["conclusion"] == "single_source_correction_supported:radio"
    set_deltas(scenarios, "full", -.1, -.2, 0.)
    assert probe.primary_correction_decision(scenarios)["conclusion"] == "full_correction_supported"


@pytest.mark.parametrize("source", ["radio_only", "geometry_only"])
def test_exactly_one_beneficial_source_takes_registered_branch(source):
    scenarios = decisions_fixture()
    for mode in probe.MODES[1:]:
        set_deltas(scenarios, mode, .1, .2, .1)
    set_deltas(scenarios, source, -.1, -.2, 0.)
    assert probe.primary_correction_decision(scenarios)["conclusion"] == f"single_source_correction_supported:{source.removesuffix('_only')}"


def test_no_beneficial_source_and_harmful_full_not_supported():
    scenarios = decisions_fixture()
    for mode in probe.MODES[1:]:
        set_deltas(scenarios, mode, .1, .2, .1)
    assert probe.primary_correction_decision(scenarios)["conclusion"] == "structured_correction_not_supported"


def test_both_beneficial_sources_without_beneficial_full_are_mixed():
    scenarios = decisions_fixture()
    set_deltas(scenarios, "full", 0., 0., 0.)
    assert probe.primary_correction_decision(scenarios)["conclusion"] == "correction_evidence_mixed"
    set_deltas(scenarios, "radio_only", -.1, -.1, .0500001)
    assert probe.primary_correction_decision(scenarios)["classifications"]["radio_only"] == "MIXED_OR_NEUTRAL"
    set_deltas(scenarios, "geometry_only", -.1, -.1, 0., names=(probe.PRIMARY[0],))
    set_deltas(scenarios, "geometry_only", .1, .1, 0., names=(probe.PRIMARY[1],))
    assert probe.primary_correction_decision(scenarios)["classifications"]["geometry_only"] == "MIXED_OR_NEUTRAL"


def test_exact_decoder_ten_percent_and_quarter_meter_boundaries():
    assert probe.decoder_promotion_gate(10., 9.)["passes"] is True
    assert probe.decoder_promotion_gate(10., 9.000001)["passes"] is False
    assert probe.decoder_promotion_gate(7.1, 6.39)["passes"] is True
    assert probe.decoder_promotion_gate(10., 10.25)["worsening_at_most_0p25_m"] is True
    assert probe.decoder_promotion_gate(10., 10.250001)["worsening_at_most_0p25_m"] is False


def test_primary_decoder_promotion_requires_both_roles_and_reports_all_passers():
    scenarios = decisions_fixture()
    default = probe.primary_decoder_decision(scenarios)
    assert default["conclusion"] == "posterior_mean_supported_as_default"
    assert default["posterior_mean_directly_best_both_roles"] is True
    for name in probe.PRIMARY:
        scenarios[name]["modes"]["none"]["decoders"]["top8_mean"]["coordinate_macro_mae_m"] = 9.
        scenarios[name]["modes"]["none"]["decoders"]["geometric_median"]["coordinate_macro_mae_m"] = 8.9
    promoted = probe.primary_decoder_decision(scenarios)
    assert promoted["supported_alternatives"] == ["top8_mean", "geometric_median"]
    assert promoted["posterior_mean_directly_best_both_roles"] is False
    scenarios[probe.PRIMARY[1]]["modes"]["none"]["decoders"]["top8_mean"]["coordinate_macro_mae_m"] = 9.1
    assert probe.primary_decoder_decision(scenarios)["supported_alternatives"] == ["geometric_median"]


def test_decoder_direct_best_ties_count_as_best_and_subthreshold_improvement_fails():
    scenarios = decisions_fixture()
    for name in probe.PRIMARY:
        for decoder in probe.DECODERS:
            scenarios[name]["modes"]["none"]["decoders"][decoder]["coordinate_macro_mae_m"] = 10.
    assert probe.primary_decoder_decision(scenarios)["posterior_mean_directly_best_both_roles"] is True
    scenarios[probe.PRIMARY[0]]["modes"]["none"]["decoders"]["map_component"]["coordinate_macro_mae_m"] = 9.5
    report = probe.primary_decoder_decision(scenarios)
    assert report["supported_alternatives"] == []
    assert report["posterior_mean_directly_best_both_roles"] is False


def test_auxiliary_scenario_c_cannot_change_either_primary_decision():
    scenarios = decisions_fixture()
    before_correction = probe.primary_correction_decision(scenarios)
    before_decoder = probe.primary_decoder_decision(scenarios)
    scenarios["calibration_T1_control"] = {"unusable_for_primary_decisions": float("nan")}
    assert probe.primary_correction_decision(scenarios) == before_correction
    assert probe.primary_decoder_decision(scenarios) == before_decoder


def test_temperature_sign_stability_requires_all_three_metrics_including_zero():
    scenarios = decisions_fixture()
    set_deltas(scenarios, "radio_only", -.1, -.2, 0., names=("calibration_frozen", "calibration_T1_control"))
    set_deltas(scenarios, "geometry_only", .1, -.2, 0., names=("calibration_T1_control",))
    set_deltas(scenarios, "full", -.1, -.2, 0., names=("calibration_T1_control",))
    result = probe.temperature_interaction(scenarios)
    assert result["radio_only"]["classification"] == "sign_stable"
    assert result["geometry_only"]["classification"] == "sign_changed"
    assert result["full"]["classification"] == "sign_changed"


def test_write_once_exclusive_and_preflight_all_outputs(tmp_path):
    path = tmp_path / "result.json"
    probe.write_once(path, b"first\n")
    with pytest.raises(FileExistsError):
        probe.write_once(path, b"overwrite\n")
    assert path.read_bytes() == b"first\n"
    with pytest.raises(RuntimeError, match="already exists"):
        probe.assert_unused([tmp_path / "unused.json", path])


@pytest.mark.parametrize("relative", [
    "data/UJIndoorLoc/validationData.csv", "data/UTS/Test_rss.csv",
    "data/Tampere/Test_rss.csv", "data/TUJI1/test.json",
    "elsewhere/UJIndoorLoc/validationData.csv", "fhm_dapp_tim/evidence/v18_4/external/results.json",
])
def test_forbidden_data_and_external_result_opens_denied_before_io(tmp_path, relative):
    audit = probe.FileOpenAudit(tmp_path)
    with audit:
        with pytest.raises(RuntimeError, match="file-open audit denied"):
            (tmp_path / relative).open("rb")
    assert len(audit.denied) == 1


def test_allowed_train_reads_recorded_and_train_write_denied(tmp_path):
    path = tmp_path / "data/UJIndoorLoc/trainingData.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture\n")
    with probe.FileOpenAudit(tmp_path) as audit:
        assert path.read_bytes() == b"fixture\n"
        assert path.read_text(encoding="utf-8") == "fixture\n"
        with pytest.raises(RuntimeError, match="read-only"):
            path.open("wb")
    assert len(audit.opens) == 3
    assert [item["allowed"] for item in audit.opens] == [True, True, False]
    assert path.read_bytes() == b"fixture\n"


@pytest.mark.parametrize("relative", [
    "fhm_dapp_tim/models/v18_4/model.pt", "fhm_dapp_tim/evidence/v18_4/uji_method_freeze_v18_4.json",
    "fhm_dapp_tim/fhm_dapp/v18_4_core.py",
])
def test_frozen_checkpoint_evidence_source_writes_denied(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"frozen")
    with probe.FileOpenAudit(tmp_path):
        with pytest.raises(RuntimeError, match="write forbidden"):
            path.open("wb")
    assert path.read_bytes() == b"frozen"


def test_audit_fd_writes_and_context_deactivation(tmp_path):
    audit = probe.FileOpenAudit(tmp_path)
    with pytest.raises(RuntimeError, match="file-open audit denied"):
        audit.check(tmp_path / "data/UJIndoorLoc/trainingData.csv", None, os.O_WRONLY | os.O_CREAT)
    path = tmp_path / "data/UJIndoorLoc/validationData.csv"
    path.parent.mkdir(parents=True)
    with audit:
        pass
    path.write_bytes(b"synthetic fixture only after guard inactive")
    assert audit.active is False


def test_json_self_hash_nonfinite_rejection_and_state_hash_mutation():
    payload = {"a": 1, "nested": {"b": [2., 3.]}}
    sealed = probe.seal(copy.deepcopy(payload), "result_sha256")
    assert probe.canonical_hash(sealed, "result_sha256") == sealed["result_sha256"]
    sealed["a"] = 2
    assert probe.canonical_hash(sealed, "result_sha256") != sealed["result_sha256"]
    with pytest.raises(ValueError):
        probe.json_bytes({"a": float("nan")})
    with pytest.raises(RuntimeError, match="non-finite"):
        probe.assert_finite({"a": [float("inf")]})
    state = {"scalar": torch.tensor(1., dtype=torch.float32), "buffer": torch.tensor([True, False])}
    digest = probe.tensor_state_hash(state)
    assert digest == probe.tensor_state_hash({k: v.clone() for k, v in state.items()})
    state["scalar"].add_(1.)
    assert probe.tensor_state_hash(state) != digest


def test_config_comparison_preserves_frozen_law_across_json_tuple_arrays():
    assert probe.canonical_config({"grid": (.1, .25, .5), "nested": {"x": (1., 2.)}}) == {
        "grid": [.1, .25, .5], "nested": {"x": [1., 2.]}}
    assert probe.canonical_config({"grid": (.1, .25)}) != probe.canonical_config({"grid": [.1, .5]})


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_common_student_t_draws_shared_and_unchanged_across_modes(device):
    draws = common_student_t_randomness(2, draws=8, seed=probe.SCORING_SEED)
    draws = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in draws.items()}
    digest = probe.randomness_hash(draws)
    q, mask, production = production_fixture(device)
    modes = probe.decompose_correction(q, mask, production)
    locations = torch.zeros(2, 4, 2, dtype=q.dtype, device=device)
    locations[:, :, 0] = torch.arange(4., dtype=q.dtype, device=device)
    scale = torch.ones_like(locations)
    target = torch.zeros(2, 2, dtype=q.dtype, device=device)
    for mode in modes.values():
        es, _, _ = continuous_energy_score_mc(mode["posterior"], locations, scale, target, draws)
        assert bool(torch.isfinite(es).all())
        assert probe.randomness_hash(draws) == digest
    draws["noise_1"][0, 0, 0].add_(1.)
    assert probe.randomness_hash(draws) != digest


AUTHORIZATION = {"radio_temporary_adam_allowed": True, "human_reply": "authorized nuisance Adam only"}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_guard_real_six_step_nuisance_profiler_and_bitwise_state(device):
    model = torch.nn.Linear(2, 2).to(device).eval()
    before = probe.tensor_state_hash(model.state_dict())
    query = torch.full((2, 3), -.7, device=device)
    indices = torch.tensor([[0, 1, 2, 3]] * 2, dtype=torch.long, device=device)
    fields = {"detection_probability": torch.full((4, 3), .8, device=device),
              "rss_center": torch.full((4, 3), -.8, device=device),
              "rss_scale": torch.full((4, 3), .08, device=device)}
    prior = torch.full((2, 4), .25, device=device)
    with probe.ReadOnlyModelGuard(model, AUTHORIZATION) as guard:
        with guard.radio_batch(2):
            energy, _ = profile_shared_affine_marked_energy(query, indices, fields, prior, steps=6)
            assert bool(torch.isfinite(energy).all())
    assert guard.verification["temporary_adam_steps"] == 6
    assert guard.verification["temporary_nuisance_backward_calls"] == 6
    assert guard.verification["temporary_adam_initializations"] == 1
    assert guard.verification["radio_batches"] == 1
    assert probe.tensor_state_hash(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())


def test_model_guard_rejects_training_optimizer_and_unscoped_backward():
    model = torch.nn.Linear(2, 2).eval()
    with probe.ReadOnlyModelGuard(model, AUTHORIZATION):
        with pytest.raises(RuntimeError, match="training mode forbidden"):
            model.train()
        with pytest.raises(RuntimeError, match="optimizer forbidden"):
            torch.optim.Adam(model.parameters(), lr=.1)
        with pytest.raises(RuntimeError, match="backward forbidden"):
            torch.tensor(1., requires_grad=True).backward()
    assert all(p.grad is None for p in model.parameters())


def test_model_guard_rejects_extra_autograd_leaf_before_backward():
    model = torch.nn.Linear(2, 2).eval()
    with probe.ReadOnlyModelGuard(model, AUTHORIZATION) as guard:
        with pytest.raises(RuntimeError, match="leaves other than"):
            with guard.radio_batch(2):
                a = torch.zeros(2, requires_grad=True)
                b = torch.zeros(2, requires_grad=True)
                torch.optim.Adam([a, b], lr=.2)
                extra = torch.tensor(1., requires_grad=True)
                (a.sum() + b.sum() + extra).backward()
        assert guard.counts["temporary_nuisance_backward_calls"] == 0
    assert all(p.grad is None for p in model.parameters())


def test_model_guard_detects_bitwise_state_change():
    model = torch.nn.Linear(2, 2).eval()
    with pytest.raises(RuntimeError, match="changed bitwise"):
        with probe.ReadOnlyModelGuard(model, AUTHORIZATION):
            with torch.no_grad():
                model.weight[0, 0].add_(1.)


def test_model_guard_requires_human_exception_and_restores_global_methods():
    model = torch.nn.Linear(2, 2).eval()
    originals = (torch.nn.Module.train, torch.optim.Optimizer.__init__, torch.optim.Adam.step, torch.autograd.backward)
    with pytest.raises(RuntimeError, match="human radio nuisance"):
        with probe.ReadOnlyModelGuard(model, {"radio_temporary_adam_allowed": False}):
            pass
    with probe.ReadOnlyModelGuard(model, AUTHORIZATION):
        pass
    assert originals == (torch.nn.Module.train, torch.optim.Optimizer.__init__, torch.optim.Adam.step, torch.autograd.backward)
