from __future__ import annotations

import pytest
import numpy as np

from fhm_dapp_tim.fhm_dapp.uji_v9_wknn_protocol import (
    K_GRID,
    _validate_status,
    select_k_by_coordinate_macro,
)
from fhm_dapp_tim.fhm_dapp.v9_wknn_common import coordinate_prototypes


def test_select_k_uses_development_minimum_and_smallest_numeric_tie():
    scores = {1: 2.0, 3: 1.0, 5: 1.0, 8: 1.2, 12: 1.3, 20: 1.4}
    assert select_k_by_coordinate_macro(scores) == 3


def test_select_k_rejects_incomplete_grid():
    with pytest.raises(ValueError, match="k grid differs"):
        select_k_by_coordinate_macro({k: float(k) for k in K_GRID[:-1]})


def test_status_contract_rejects_preregistration_rebranding():
    valid = {
        "created_after_v9_outer_test": True,
        "selection_authority_for_v9": False,
        "may_modify_v9": False,
        "comparison_status": "post_lock_comparator",
        "not_a_preregistration": True,
        "internal_v9_variant_expansion_allowed": False,
    }
    _validate_status(valid)
    invalid = dict(valid, created_after_v9_outer_test=False)
    with pytest.raises(ValueError, match="created_after_v9_outer_test"):
        _validate_status(invalid)


def test_coordinate_prototype_accumulates_metre_scale_targets_in_float64():
    rows = 49
    rss = np.full((rows, 2), -0.5, dtype=np.float32)
    xy = np.tile(np.asarray([[-7312.662109375, 4864807.5]], dtype=np.float32), (rows, 1))
    phones = np.asarray([1] * 24 + [2] * 25, dtype=np.int64)
    keys = [(2, 3, -7312.6621, 4864807.5)] * rows
    _prototype, targets, _keys = coordinate_prototypes(
        rss, xy, phones, keys, np.arange(rows, dtype=np.int64)
    )
    assert np.allclose(targets, [[-7312.662109375, 4864807.5]], atol=1e-9)
