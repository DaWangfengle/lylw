import json
import math
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2d as r2d_execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2e as execution
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2c import _joint_consensus_law
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2e_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    ANCHOR_COUNT,
    CONFIGURATIONS,
    DEFAULT_R2D_RESULT,
    DENSE_CONSENSUS_FROM_L,
    MEMBER_GRID,
    PANEL_MEMBER_COUNT,
    SUBSET_OFFSET_ORDER,
    SUPPORT_K_GRID,
    SUPPORT_L_GRID,
    VARIANTS,
    _stagewise_r2d_evidence,
    balanced_member_subsets,
    checkpoint_path,
    variant_name,
)


def test_r2e_grid_jointly_closes_members_and_finite_support():
    assert MEMBER_GRID == tuple(range(10, 21))
    assert SUPPORT_L_GRID == (48, 96, 192, 384, 595)
    assert SUPPORT_K_GRID == (24, 48, 96, 192, 384, 595)
    assert max(SUPPORT_L_GRID) == max(SUPPORT_K_GRID) == ANCHOR_COUNT == 595
    assert len(CONFIGURATIONS) == len(VARIANTS) == len(set(VARIANTS)) == 220
    assert all(K <= L for _E, L, K in CONFIGURATIONS)
    assert all(name.endswith("_A1") for name in VARIANTS)
    assert variant_name(18, 384, 192) == "E18_L384_K192_A1"


def test_twenty_member_cyclic_design_is_nested_and_member_balanced():
    assert len(SUBSET_OFFSET_ORDER) == len(set(SUBSET_OFFSET_ORDER)) == 20
    assert math.gcd(7, PANEL_MEMBER_COUNT) == 1
    previous = None
    for E in MEMBER_GRID[:-1]:
        subsets = balanced_member_subsets(E)
        assert len(subsets) == PANEL_MEMBER_COUNT
        counts = np.bincount(np.asarray(subsets).reshape(-1), minlength=20)
        assert np.array_equal(counts, np.full(20, E))
        assert all(len(set(value)) == E for value in subsets)
        if previous is not None:
            assert all(set(left).issubset(right) for left, right in zip(previous, subsets))
        previous = subsets
    assert balanced_member_subsets(20) == (tuple(range(20)),)


def test_r2d_evidence_licenses_only_registered_stagewise_pruning():
    result = json.loads(DEFAULT_R2D_RESULT.read_text(encoding="utf-8"))
    evidence = _stagewise_r2d_evidence(result)
    assert evidence["r2d_global_boundary_variant"] == "E15_L192_K048_A1"
    assert evidence["r2d_platform_was_boundary_only"] is True
    assert set(evidence["removed_low_member_profiles"]) == {
        str(value) for value in range(5, 10)
    }
    for record in evidence["removed_low_member_profiles"].values():
        assert record["energy_minus_r2d_global_m"] > 0.05
    for record in evidence["removed_low_support_profiles"].values():
        assert record["best_K16_minus_within_E_best_energy_m"] > 0.05
        assert record["best_L24_minus_within_E_best_energy_m"] > 0.05


def test_new_member_paths_are_seed_identity_locked():
    assert ADDITIONAL_MEMBER_INDICES == tuple(range(16, 21))
    assert ADDITIONAL_MEMBER_SEEDS == tuple(range(17316, 17321))
    for index, seed in zip(ADDITIONAL_MEMBER_INDICES, ADDITIONAL_MEMBER_SEEDS):
        assert checkpoint_path(index).name == f"member_{index:02d}_seed{seed}.pt"


def test_atomic_probability_extension_preserves_the_first_fifteen_blocks():
    generator = torch.Generator().manual_seed(79)
    neural = torch.randn(20, 4, 9, generator=generator)
    distance = torch.rand(4, 9, generator=generator)
    extended = execution._atomic_probability_panel(neural, distance, 1.0, 32.0 / 3.0)
    frozen = r2d_execution._atomic_probability_panel(
        neural[:15], distance, 1.0, 32.0 / 3.0
    )
    assert torch.equal(extended[:15], frozen)
    assert torch.allclose(extended.sum(dim=2), torch.ones(20, 4))


def test_dense_memory_safe_path_is_the_same_consensus_operator():
    generator = torch.Generator().manual_seed(83)
    members, rows, anchors = 3, 2, 400
    support = DENSE_CONSENSUS_FROM_L
    dense_probability = torch.rand(members, rows, anchors, generator=generator)
    dense_probability = dense_probability / dense_probability.sum(dim=2, keepdim=True)
    prior, indices = torch.topk(dense_probability, support, dim=2, sorted=True)
    prior = prior / prior.sum(dim=2, keepdim=True)
    radial = torch.rand(members, rows, generator=generator) + 0.5
    location = torch.randn(members, rows, anchors, 2, generator=generator)
    scale = torch.rand(members, rows, anchors, 2, generator=generator) + 0.1
    subset = tuple(range(members))
    sparse = _joint_consensus_law(
        prior,
        indices,
        radial,
        location,
        scale,
        subset,
        K=24,
        alpha=1.0,
    )
    memory_safe = execution._joint_consensus_law_memory_safe(
        prior,
        indices,
        radial,
        location,
        scale,
        subset,
        K=24,
        alpha=1.0,
    )
    for observed, expected in zip(memory_safe, sparse):
        assert torch.allclose(observed, expected, atol=2e-6, rtol=2e-6)


def test_r2e_has_no_outer_or_official_execution_entry_point():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert 'sub.add_parser("train-member")' in source
    assert 'sub.add_parser("build-cache")' in source
    assert 'sub.add_parser("evaluate")' in source
    assert 'sub.add_parser("check-result")' in source
    assert "validationData.csv" not in source
    assert '"may_access_outer_test": False' in source
    assert '"outer_test_predictions": 0' in source
    assert '"official_validation_predictions": 0' in source

