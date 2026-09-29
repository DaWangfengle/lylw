import json
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2c as execution
from fhm_dapp_tim.fhm_dapp.consensus import alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2c_protocol import (
    CONFIGURATIONS,
    DEFAULT_R1_RESULT,
    DEFAULT_R2B_CACHE,
    DEFAULT_R2B_REGISTRY,
    HISTORICAL_CONFIGURATION,
    HISTORICAL_VARIANT,
    MEMBER_GRID,
    SUPPORT_K_GRID,
    SUPPORT_L_GRID,
    VARIANTS,
    balanced_member_subsets,
    variant_name,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b import load_cache
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import load_registry


def test_joint_grid_is_complete_without_unidentified_E1_alpha_duplicates():
    assert len(CONFIGURATIONS) == 444
    assert len(VARIANTS) == len(set(VARIANTS)) == 444
    assert HISTORICAL_CONFIGURATION in CONFIGURATIONS
    assert HISTORICAL_VARIANT == "E05_L024_K024_A0p25"
    assert all(K <= L for _E, L, K, _alpha in CONFIGURATIONS)
    assert all(
        alpha == 0.25 for E, _L, _K, alpha in CONFIGURATIONS if E == 1
    )
    assert SUPPORT_L_GRID == (12, 24, 48, 96)
    assert SUPPORT_K_GRID == (8, 16, 24, 48)


def test_cyclic_subsets_are_nested_and_exactly_member_balanced():
    previous = None
    for E in MEMBER_GRID[:-1]:
        subsets = balanced_member_subsets(E)
        assert len(subsets) == 10
        counts = np.bincount(np.asarray(subsets).reshape(-1), minlength=10)
        assert np.array_equal(counts, np.full(10, E))
        assert all(len(set(value)) == E for value in subsets)
        if previous is not None:
            assert all(set(left).issubset(right) for left, right in zip(previous, subsets))
        previous = subsets
    assert balanced_member_subsets(10) == (tuple(range(10)),)


def test_topK_prefix_of_maximum_consensus_is_the_exact_smaller_projection():
    generator = torch.Generator().manual_seed(42)
    members, rows, width, anchors = 3, 4, 20, 100
    raw = torch.rand(members, rows, width, generator=generator)
    priors = raw / raw.sum(dim=2, keepdim=True)
    indices = torch.stack(
        [
            torch.stack(
                [torch.randperm(anchors, generator=generator)[:width] for _ in range(rows)]
            )
            for _ in range(members)
        ]
    )
    member_weight = torch.rand(members, rows, generator=generator)
    large_indices, large_probability = alpha_consensus_prior(
        priors, indices, member_weight, anchors, 48, epsilon=1e-6, alpha=0.25
    )
    small_indices, small_probability = alpha_consensus_prior(
        priors, indices, member_weight, anchors, 24, epsilon=1e-6, alpha=0.25
    )
    prefix = large_probability[:, :24]
    prefix = prefix / prefix.sum(dim=1, keepdim=True)
    assert torch.equal(large_indices[:, :24], small_indices)
    assert torch.allclose(prefix, small_probability, atol=1e-7, rtol=1e-6)


def test_studentized_max_t_does_not_use_one_raw_meter_critical_for_all_scales():
    x = np.linspace(-1.0, 1.0, 31)
    family = execution.studentized_max_t_family(
        {"small": 0.02 + 0.01 * x, "large": 2.0 + 5.0 * x},
        repetitions=500,
        seed=17,
        orientation="fixture",
    )
    small = family["comparisons"]["small"]
    large = family["comparisons"]["large"]
    assert family["studentized"] is True
    assert small["simultaneous_upper_bound_m"] < 0.04
    assert large["simultaneous_upper_bound_m"] > 2.0


def test_r2c_has_no_outer_or_official_entry_point():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert 'sub.add_parser("evaluate")' in source
    assert 'sub.add_parser("check-result")' in source
    assert "validationData.csv" not in source
    assert "outer_test_predictions\": 0" in source
    assert "official_validation_predictions\": 0" in source


def test_member_topL_is_an_exact_reverse_KL_projection():
    probability = torch.tensor([[[0.5, 0.3, 0.15, 0.05]]])
    location = torch.zeros(1, 1, 4, 2)
    scale = torch.ones_like(location)
    weights, indices, radial, mass = execution._member_top_l(
        probability, location, scale, 2
    )
    assert torch.equal(indices, torch.tensor([[[0, 1]]]))
    assert torch.allclose(mass, torch.tensor([[0.8]]))
    assert torch.allclose(weights, torch.tensor([[[0.625, 0.375]]]))
    assert bool((radial > 0.0).all())


def test_frozen_first_five_historical_law_reproduces_r1_p0d0():
    registry = load_registry(DEFAULT_R2B_REGISTRY)
    cache = load_cache(DEFAULT_R2B_CACHE, registry=registry)
    probability = torch.softmax(
        cache["member_neural_logits"][:5]
        - (32.0 / 3.0) * cache["prototype_distance"][None, :, :],
        dim=2,
    )
    location = cache["member_locations"][:5]
    scale = cache["member_scales"][:5]
    prior, indices, radial, _mass = execution._member_top_l(
        probability, location, scale, 24
    )
    joint, selected_location, _selected_scale = execution._joint_consensus_law(
        prior,
        indices,
        radial,
        location,
        scale,
        tuple(range(5)),
        K=24,
        alpha=0.25,
    )
    action = torch.sum(joint[:, :, None] * selected_location, dim=1)
    observed = torch.linalg.vector_norm(
        action.double() - cache["target_xy"].double(), dim=1
    )
    reference = json.loads(DEFAULT_R1_RESULT.read_text(encoding="utf-8"))
    expected = torch.tensor(
        reference["row_vectors"]["euclidean_error"]["P0D0"],
        dtype=torch.float64,
    )
    assert observed.shape == expected.shape
    observed_macro = execution._group_vector(
        observed,
        cache["coordinate_group_ids"],
        len(cache["coordinate_group_keys"]),
    ).mean()
    expected_macro = execution._group_vector(
        expected,
        cache["coordinate_group_ids"],
        len(cache["coordinate_group_keys"]),
    ).mean()
    assert float((observed_macro - expected_macro).abs()) <= 1e-4
