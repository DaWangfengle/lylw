from __future__ import annotations

import json
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as registered_r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f_runtime_a1 as runtime
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _canonical_json_sha256, _sha256_file


def _fixture() -> tuple[torch.Tensor, ...]:
    prior = torch.tensor(
        [
            [[0.60, 0.30, 0.10], [0.55, 0.35, 0.10]],
            [[0.50, 0.30, 0.20], [0.50, 0.25, 0.25]],
        ],
        dtype=torch.float64,
    )
    indices = torch.tensor(
        [
            [[0, 2, 4], [0, 1, 3]],
            [[1, 2, 3], [0, 2, 4]],
        ],
        dtype=torch.long,
    )
    radial = torch.tensor(
        [[1.0, 1.5], [2.0, 1.0]], dtype=torch.float64
    )
    locations = torch.arange(2 * 2 * 5 * 2, dtype=torch.float64).reshape(
        2, 2, 5, 2
    )
    scales = torch.full_like(locations, 0.75)
    return prior, indices, radial, locations, scales


def test_dense_projected_joint_law_is_exactly_repeatable_and_normalized() -> None:
    inputs = _fixture()
    first = runtime._dense_projected_joint_law(
        *inputs,
        (0, 1),
        construction_k=5,
        output_k=2,
    )
    second = runtime._dense_projected_joint_law(
        *inputs,
        (0, 1),
        construction_k=5,
        output_k=2,
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    probability, location, scale = first
    assert probability.shape == (2, 4)
    assert location.shape == (2, 4, 2)
    assert scale.shape == (2, 4, 2)
    assert torch.allclose(
        probability.sum(dim=1), torch.ones(2, dtype=torch.float64)
    )


def test_construct_then_project_matches_direct_dense_top_k() -> None:
    inputs = _fixture()
    projected = runtime._dense_projected_joint_law(
        *inputs,
        (0, 1),
        construction_k=5,
        output_k=2,
    )
    direct = runtime._dense_projected_joint_law(
        *inputs,
        (0, 1),
        construction_k=2,
        output_k=2,
    )
    assert torch.allclose(projected[0], direct[0], atol=1e-15, rtol=0.0)
    assert torch.equal(projected[1], direct[1])
    assert torch.equal(projected[2], direct[2])


def test_deterministic_context_restores_process_setting() -> None:
    before = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with runtime._deterministic_algorithms():
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.are_deterministic_algorithms_enabled() is before
    assert torch.is_deterministic_algorithms_warn_only_enabled() is before_warn_only


def test_amendment_is_self_hashed_and_scientifically_null() -> None:
    payload = json.loads(runtime.DEFAULT_AMENDMENT.read_text(encoding="utf-8"))
    assert payload["amendment_sha256"] == _canonical_json_sha256(
        payload, omit="amendment_sha256"
    )
    assert payload["scientific_change"] is False
    assert payload["selection_authority_for_v17_r2f"] is False
    assert payload["may_modify_v17_r2f_model"] is False
    assert payload["created_after_failed_r2f_evaluation_attempt"] is True
    assert payload["incident"]["development_result_existed_after_failure"] is False
    assert payload["incident"]["outer_or_official_data_accessed"] is False


def test_amendment_binds_all_runtime_inputs() -> None:
    payload = runtime._load_amendment()
    for record in payload["bindings"].values():
        assert Path(record["path"]).is_file()
        assert _sha256_file(record["path"]) == record["file_sha256"]
    registry = registered_r2f.load_registry(
        payload["bindings"]["registered_registry"]["path"]
    )
    assert (
        registry["registry_sha256"]
        == payload["bindings"]["registered_registry"]["registry_sha256"]
    )


def test_amendment_preserves_registered_grids_and_decision_thresholds() -> None:
    payload = runtime._load_amendment()
    unchanged = payload["unchanged_scientific_contract"]
    assert unchanged["member_grid"] == list(registered_r2f.MEMBER_GRID)
    assert unchanged["support_l_grid"] == list(registered_r2f.SUPPORT_L_GRID)
    assert unchanged["support_k"] == registered_r2f.SUPPORT_K
    assert unchanged["alpha"] == registered_r2f.CONSENSUS_ALPHA
    assert (
        unchanged["mae_noninferiority_margin_m"]
        == registered_r2f.MAE_NONINFERIORITY_MARGIN_M
    )
    assert (
        unchanged["energy_equivalence_margin_m"]
        == registered_r2f.ENERGY_EQUIVALENCE_MARGIN_M
    )
    assert unchanged["bootstrap_repetitions"] == registered_r2f.BOOTSTRAP_REPETITIONS
    assert unchanged["bootstrap_seed"] == registered_r2f.BOOTSTRAP_SEED
    assert unchanged["outer_test_access"] is False
    assert unchanged["official_validation_access"] is False
