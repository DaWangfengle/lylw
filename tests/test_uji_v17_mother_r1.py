from __future__ import annotations

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.dapp import robust_hurdle_anchor_energy
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1 import (
    _group_vector,
    _paired_simultaneous_family,
    balanced_affine_energy_grid,
    marginal_effective_loss,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    BOOTSTRAP_REPETITIONS,
    MAE_NONINFERIORITY_MARGIN_M,
    MEMBER_COUNT,
    POSTERIOR_VARIANTS,
    QUADRATURE_ORDER,
    TOP_K,
    TOP_L,
    VARIANTS,
)


def _synthetic_hurdle_inputs():
    detection = torch.tensor(
        [[[0.8, 0.3, 0.6], [0.4, 0.9, 0.2]]], dtype=torch.float64
    )
    center = torch.tensor(
        [[[-0.3, 0.1, -0.6], [-0.1, 0.4, -0.4]]], dtype=torch.float64
    )
    scale = torch.full_like(center, 0.15)
    query = torch.tensor([[-0.25, 0.0, 0.0]], dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])
    reliability = torch.ones(3, dtype=torch.float64)
    return detection, center, scale, query, mask, reliability


def test_registered_factorial_contract_is_single_and_closed():
    assert MEMBER_COUNT == 5
    assert TOP_L == TOP_K == 24
    assert QUADRATURE_ORDER == 5
    assert POSTERIOR_VARIANTS == ("P0", "N0", "N1")
    assert VARIANTS == (
        "P0D0",
        "P0D1",
        "N0D0",
        "N0D1",
        "N1D0",
        "N1D1",
    )
    assert MAE_NONINFERIORITY_MARGIN_M == 0.10
    assert BOOTSTRAP_REPETITIONS == 5000


def test_balanced_affine_grid_zero_node_matches_direct_historical_loss():
    detection, center, scale, query, mask, reliability = (
        _synthetic_hurdle_inputs()
    )
    nodes = torch.zeros(1, 2, dtype=torch.float64)
    grid = balanced_affine_energy_grid(
        detection,
        center,
        scale,
        query,
        mask,
        reliability,
        nodes,
    )
    direct = robust_hurdle_anchor_energy(
        detection,
        center,
        scale,
        query,
        mask,
        reliability,
        style_offset=0.0,
        style_scale=1.0,
        visible_factor=1.0,
        invisible_factor=1.0,
        distribution="student_t",
        degrees_freedom=4.0,
        score="balanced_log",
    )
    assert grid.shape == (1, 2, 1)
    torch.testing.assert_close(grid[:, :, 0], direct, atol=1e-12, rtol=0.0)


def test_marginal_effective_loss_preserves_one_or_repeated_node():
    energy = torch.tensor([[[1.25], [0.50]]], dtype=torch.float64)
    one = marginal_effective_loss(
        energy, torch.ones(1, dtype=torch.float64), lambda_value=0.5
    )
    torch.testing.assert_close(one, energy[:, :, 0], atol=1e-12, rtol=0.0)
    repeated = energy.expand(-1, -1, 3).clone()
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    many = marginal_effective_loss(repeated, weights, lambda_value=0.5)
    torch.testing.assert_close(many, energy[:, :, 0], atol=1e-12, rtol=0.0)


def test_group_vector_uses_coordinate_macro_aggregation():
    row = torch.tensor([1.0, 3.0, 9.0], dtype=torch.float32)
    group = torch.tensor([0, 0, 1], dtype=torch.long)
    result = _group_vector(row, group, 2)
    torch.testing.assert_close(
        result, torch.tensor([2.0, 9.0], dtype=torch.float64)
    )


def test_simultaneous_family_is_deterministic_and_preserves_orientation():
    differences = {
        "A-B": np.asarray([-2.0, -1.0, -3.0, -2.0]),
        "C-B": np.asarray([1.0, 2.0, 0.0, 1.0]),
    }
    first = _paired_simultaneous_family(
        differences, repetitions=500, seed=17001
    )
    second = _paired_simultaneous_family(
        differences, repetitions=500, seed=17001
    )
    assert first == second
    assert first["comparisons"]["A-B"]["mean_difference_m"] == -2.0
    assert first["comparisons"]["C-B"]["mean_difference_m"] == 1.0
    assert first["family_size"] == 2
    assert first["common_coordinate_group_resamples"] is True
