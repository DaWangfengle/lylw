import json
import math
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2e as r2e_execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as execution
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2f_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    ANCHOR_COUNT,
    CONFIGURATIONS,
    DEFAULT_R2E_RESULT,
    MEMBER_GRID,
    PANEL_MEMBER_COUNT,
    R2E_COMPACT_REFERENCE,
    R2E_GLOBAL_REFERENCE,
    SUBSET_OFFSET_ORDER,
    SUPPORT_K,
    SUPPORT_L_GRID,
    VARIANTS,
    _stagewise_r2e_evidence,
    balanced_member_subsets,
    checkpoint_path,
    variant_name,
)


def test_r2f_grid_changes_only_member_information_and_retains_support_sentinel():
    assert MEMBER_GRID == tuple(range(16, 26))
    assert SUPPORT_L_GRID == (192, 595)
    assert SUPPORT_K == 192
    assert max(SUPPORT_L_GRID) == ANCHOR_COUNT == 595
    assert len(CONFIGURATIONS) == len(VARIANTS) == len(set(VARIANTS)) == 20
    assert all(name.endswith("_K192_A1") for name in VARIANTS)
    assert variant_name(23, 192) == "E23_L192_K192_A1"


def test_twenty_five_member_cyclic_design_is_nested_and_balanced():
    assert len(SUBSET_OFFSET_ORDER) == len(set(SUBSET_OFFSET_ORDER)) == 25
    assert math.gcd(7, PANEL_MEMBER_COUNT) == 1
    previous = None
    for E in MEMBER_GRID[:-1]:
        subsets = balanced_member_subsets(E)
        assert len(subsets) == PANEL_MEMBER_COUNT
        counts = np.bincount(np.asarray(subsets).reshape(-1), minlength=25)
        assert np.array_equal(counts, np.full(25, E))
        assert all(len(set(value)) == E for value in subsets)
        if previous is not None:
            assert all(set(left).issubset(right) for left, right in zip(previous, subsets))
        previous = subsets
    assert balanced_member_subsets(25) == (tuple(range(25)),)


def test_r2e_evidence_licenses_the_registered_dimension_reduction():
    result = json.loads(DEFAULT_R2E_RESULT.read_text(encoding="utf-8"))
    evidence = _stagewise_r2e_evidence(result)
    assert evidence["r2e_global_reference"] == R2E_GLOBAL_REFERENCE
    assert set(evidence["r2e_energy_platform"]) == {
        R2E_GLOBAL_REFERENCE,
        R2E_COMPACT_REFERENCE,
    }
    assert evidence["fixed_K"] == 192
    assert evidence["retained_L"] == [192, 595]
    assert set(evidence["removed_E_10_through_15"]) == {
        str(value) for value in range(10, 16)
    }
    for record in evidence["removed_E_10_through_15"].values():
        assert record["simultaneous_upper_bound_m"] > 0.05


def test_new_member_paths_are_seed_identity_locked():
    assert ADDITIONAL_MEMBER_INDICES == tuple(range(21, 26))
    assert ADDITIONAL_MEMBER_SEEDS == tuple(range(17321, 17326))
    for index, seed in zip(ADDITIONAL_MEMBER_INDICES, ADDITIONAL_MEMBER_SEEDS):
        assert checkpoint_path(index).name == f"member_{index:02d}_seed{seed}.pt"


def test_atomic_probability_extension_preserves_the_first_twenty_exactly():
    generator = torch.Generator().manual_seed(89)
    neural = torch.randn(25, 4, 9, generator=generator)
    distance = torch.rand(4, 9, generator=generator)
    extended = execution._atomic_probability_panel(neural, distance, 1.0, 32.0 / 3.0)
    frozen = r2e_execution._atomic_probability_panel(
        neural[:20], distance, 1.0, 32.0 / 3.0
    )
    assert torch.equal(extended[:20], frozen)
    assert torch.allclose(extended.sum(dim=2), torch.ones(25, 4))


def test_r2f_has_no_outer_or_official_execution_entry_point():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert 'sub.add_parser("train-member")' in source
    assert 'sub.add_parser("build-cache")' in source
    assert 'sub.add_parser("evaluate")' in source
    assert 'sub.add_parser("check-result")' in source
    assert "validationData.csv" not in source
    assert '"may_access_outer_test": False' in source
    assert '"outer_test_predictions": 0' in source
    assert '"official_validation_predictions": 0' in source

