import json
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2b as execution
from fhm_dapp_tim.fhm_dapp.anchor import load_anchor_checkpoint
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import (
    ANCHOR_COUNT,
    BETA_GRID,
    HISTORICAL_BETA_NEURAL,
    HISTORICAL_BETA_PROTOTYPE,
    HISTORICAL_VARIANT,
    LOG_MULTIPLIERS,
    MEMBER_COUNT,
    VARIANTS,
    beta_name,
)


def _metrics(best: str):
    return {
        name: {
            "energy_score_2d_m": {
                "coordinate_macro_m": 4.0 if name == best else 5.0
            },
            "exact_mixture_mean_mae_m": {"coordinate_macro_m": 6.0},
        }
        for name in VARIANTS
    }


def _family(value_by_name):
    return {
        "comparisons": {
            name: {"simultaneous_upper_bound_m": value_by_name.get(name, 0.0)}
            for name in VARIANTS
            if name != HISTORICAL_VARIANT
        }
    }


def test_r2b_grid_is_identifiable_and_contains_one_historical_point():
    assert MEMBER_COUNT == 10
    assert ANCHOR_COUNT == 595
    assert len(BETA_GRID) == 25
    assert len(set(VARIANTS)) == 25
    assert VARIANTS.count(HISTORICAL_VARIANT) == 1
    assert beta_name(
        HISTORICAL_BETA_NEURAL, HISTORICAL_BETA_PROTOTYPE
    ) == HISTORICAL_VARIANT
    assert LOG_MULTIPLIERS == (0.25, 0.5, 1.0, 2.0, 4.0)


def test_selection_retains_history_without_familywise_energy_evidence():
    candidate = VARIANTS[0]
    selected = execution.select_registered_variant(
        _metrics(candidate),
        _family({candidate: 0.01}),
        _family({candidate: 0.05}),
    )
    assert selected["pointwise_best_eligible"] == candidate
    assert selected["variant"] == HISTORICAL_VARIANT
    assert selected["historical_retained_due_to_insufficient_evidence"] is True


def test_selection_changes_only_after_mae_and_energy_gates_pass():
    candidate = VARIANTS[0]
    selected = execution.select_registered_variant(
        _metrics(candidate),
        _family({candidate: -0.01}),
        _family({candidate: 0.05}),
    )
    assert selected["variant"] == candidate
    assert selected["replacement_evidence_passed"] is True
    rejected = execution.select_registered_variant(
        _metrics(candidate),
        _family({candidate: -0.01}),
        _family({candidate: 0.11}),
    )
    assert rejected["variant"] == HISTORICAL_VARIANT


def test_r2b_has_no_outer_or_official_entry_point():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert 'sub.add_parser("build-cache")' in source
    assert 'sub.add_parser("evaluate")' in source
    assert 'sub.add_parser("check-result")' in source
    assert "outer_test_predictions\": 0" in source
    assert "official_validation_predictions\": 0" in source
    assert "validationData.csv" not in source


def test_atomic_score_precedes_all_support_and_consensus_operations():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert "torch.softmax(" in source
    assert "member_consensus_applied\": False" in source
    assert "full_untruncated_support\": True" in source
    assert "topk(" not in source.lower()
    assert "alpha_consensus_prior" not in source


def test_identifiable_coefficients_exactly_reproduce_historical_probability_law():
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "fhm_dapp_tim"
        / "models"
        / "v17_acrhap_mother_r0"
        / "member_01_seed17301.pt"
    )
    model, _payload = load_anchor_checkpoint(checkpoint, torch.device("cpu"))
    model.eval()
    rss = model.prototype_rss[:3].clone()
    with torch.no_grad():
        historical_logits, embedding = model.retrieval_logits(
            rss,
            neural_scale=0.75,
            prototype_distance_weight=8.0,
        )
        historical = torch.softmax(historical_logits / 0.75, dim=1)
        neural = model.retrieval_logits_from_embedding(embedding)
        distance = execution._prototype_distance(model, rss)
        identifiable = torch.softmax(
            HISTORICAL_BETA_NEURAL * neural
            - HISTORICAL_BETA_PROTOTYPE * distance,
            dim=1,
        )
    assert torch.allclose(identifiable, historical, atol=1e-7, rtol=1e-6)


def test_ten_member_panel_loads_as_one_aligned_atomic_family():
    root = Path(__file__).resolve().parents[1] / "fhm_dapp_tim"
    r2a = json.loads(
        (root / "evidence" / "v17_acrhap_mother" / "registry_v17_r2a_panel.json")
        .read_text(encoding="utf-8")
    )
    r1 = json.loads(
        (root / "evidence" / "v17_acrhap_mother" / "registry_v17_r1_backend.json")
        .read_text(encoding="utf-8")
    )
    checkpoints = [item["path"] for item in r2a["base_checkpoints"]]
    checkpoints.extend(r2a["allowed_outputs"]["additional_checkpoints"])
    selection_csv = Path(r1["data_contract"]["selection_refit_csv"])
    with _installed_adapters([_role_record(selection_csv)]):
        ensemble = ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device="cpu",
            dataset="UJI",
        )
    assert len(ensemble.checkpoints) == MEMBER_COUNT
    assert all(model.n_anchors == ANCHOR_COUNT for model, _ in ensemble.checkpoints)
    assert all(not model.use_candidate_reranker for model, _ in ensemble.checkpoints)
    assert len(ensemble.training_indices) > 0
