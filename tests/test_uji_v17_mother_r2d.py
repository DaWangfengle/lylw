import json
import math
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2d as execution
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2d_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    CONFIGURATIONS,
    DEFAULT_R2C_RESULT,
    MEMBER_GRID,
    PANEL_MEMBER_COUNT,
    SUBSET_OFFSET_ORDER,
    SUPPORT_K_GRID,
    SUPPORT_L_GRID,
    VARIANTS,
    _stagewise_r2c_evidence,
    balanced_member_subsets,
    checkpoint_path,
    variant_name,
)


def test_r2d_grid_is_the_registered_boundary_extension_only():
    assert MEMBER_GRID == tuple(range(5, 16))
    assert SUPPORT_L_GRID == (24, 48, 96, 192)
    assert SUPPORT_K_GRID == (16, 24, 48)
    assert len(CONFIGURATIONS) == len(VARIANTS) == len(set(VARIANTS)) == 121
    assert all(K <= L for _E, L, K in CONFIGURATIONS)
    assert all(name.endswith("_A1") for name in VARIANTS)
    assert variant_name(13, 96, 24) == "E13_L096_K024_A1"


def test_fifteen_member_cyclic_design_is_nested_and_member_balanced():
    assert len(SUBSET_OFFSET_ORDER) == len(set(SUBSET_OFFSET_ORDER)) == 15
    assert math.gcd(7, PANEL_MEMBER_COUNT) == 1
    previous = None
    for E in MEMBER_GRID[:-1]:
        subsets = balanced_member_subsets(E)
        assert len(subsets) == PANEL_MEMBER_COUNT
        counts = np.bincount(np.asarray(subsets).reshape(-1), minlength=15)
        assert np.array_equal(counts, np.full(15, E))
        assert all(len(set(value)) == E for value in subsets)
        if previous is not None:
            assert all(set(left).issubset(right) for left, right in zip(previous, subsets))
        previous = subsets
    assert balanced_member_subsets(15) == (tuple(range(15)),)


def test_r2c_evidence_supports_the_frozen_alpha_and_pruned_low_support():
    result = json.loads(DEFAULT_R2C_RESULT.read_text(encoding="utf-8"))
    evidence = _stagewise_r2c_evidence(result)
    assert evidence["eligible_profile_alpha1_for_every_E_5_through_10"] is True
    for E in range(5, 11):
        profile = evidence["alpha_profiles"][str(E)]
        best = min(
            profile,
            key=lambda key: profile[key]["energy_score_coordinate_macro_m"],
        )
        assert best == "1.0"
        removed = evidence["removed_support_profiles"][str(E)]
        assert removed["best_L12_minus_within_E_best_energy_m"] > 0.15
        assert removed["best_K8_minus_within_E_best_energy_m"] > 0.15


def test_new_member_paths_are_seed_identity_locked():
    assert ADDITIONAL_MEMBER_INDICES == tuple(range(11, 16))
    assert ADDITIONAL_MEMBER_SEEDS == tuple(range(17311, 17316))
    for index, seed in zip(ADDITIONAL_MEMBER_INDICES, ADDITIONAL_MEMBER_SEEDS):
        assert checkpoint_path(index).name == f"member_{index:02d}_seed{seed}.pt"


def test_atomic_probability_extension_preserves_the_first_ten_numerical_block():
    generator = torch.Generator().manual_seed(73)
    neural = torch.randn(15, 4, 9, generator=generator)
    distance = torch.rand(4, 9, generator=generator)
    extended = execution._atomic_probability_panel(neural, distance, 1.0, 32.0 / 3.0)
    frozen = torch.softmax(
        neural[:10] - (32.0 / 3.0) * distance[None, :, :], dim=2
    )
    assert torch.equal(extended[:10], frozen)
    assert torch.allclose(extended.sum(dim=2), torch.ones(15, 4))


def test_r2d_has_no_outer_or_official_execution_entry_point():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert 'sub.add_parser("train-member")' in source
    assert 'sub.add_parser("build-cache")' in source
    assert 'sub.add_parser("evaluate")' in source
    assert 'sub.add_parser("check-result")' in source
    assert "validationData.csv" not in source
    assert '"may_access_outer_test": False' in source
    assert '"outer_test_predictions": 0' in source
    assert '"official_validation_predictions": 0' in source
