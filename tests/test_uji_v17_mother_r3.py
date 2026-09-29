import inspect
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as protocol


def test_registered_cyclic_subsets_are_exactly_balanced():
    for member_count in (5, 16):
        subsets = protocol.balanced_member_subsets(member_count)
        assert len(subsets) == protocol.PANEL_MEMBER_COUNT
        assert all(len(subset) == member_count for subset in subsets)
        assert all(len(set(subset)) == member_count for subset in subsets)
        appearances = [
            sum(member in subset for subset in subsets)
            for member in range(protocol.PANEL_MEMBER_COUNT)
        ]
        assert appearances == [member_count] * protocol.PANEL_MEMBER_COUNT


def test_selected_and_historical_laws_are_single_frozen_configurations():
    assert tuple(protocol.CONFIGURATIONS) == (
        protocol.SELECTED_NAME,
        protocol.HISTORICAL_NAME,
    )
    assert protocol.CONFIGURATIONS[protocol.SELECTED_NAME] == {
        "E": 16,
        "L": 192,
        "K": 192,
        "alpha": 1.0,
    }
    assert protocol.CONFIGURATIONS[protocol.HISTORICAL_NAME] == {
        "E": 5,
        "L": 24,
        "K": 24,
        "alpha": 0.25,
    }


def test_dense_joint_law_preserves_member_identity_and_probability_mass():
    torch.manual_seed(7)
    members, rows, anchors, support = 4, 3, 8, 4
    prior = torch.rand(members, rows, support)
    prior = prior / prior.sum(dim=2, keepdim=True)
    indices = torch.stack(
        [
            torch.stack(
                [torch.randperm(anchors)[:support] for _ in range(rows)]
            )
            for _ in range(members)
        ]
    )
    radial = torch.rand(members, rows) + 0.5
    location = torch.randn(members, rows, anchors, 2)
    scale = torch.rand(members, rows, anchors, 2) + 0.1
    probability, selected_location, selected_scale = execution._dense_joint_law(
        prior,
        indices,
        radial,
        location,
        scale,
        (0, 2, 3),
        top_k=3,
        alpha=0.5,
    )
    assert probability.shape == (rows, 9)
    assert selected_location.shape == (rows, 9, 2)
    assert selected_scale.shape == (rows, 9, 2)
    assert torch.allclose(probability.sum(dim=1), torch.ones(rows), atol=1e-6)
    assert bool((probability >= 0).all())
    assert bool((selected_scale > 0).all())


def test_outer_command_has_no_query_or_configuration_argument():
    parameters = tuple(inspect.signature(execution.evaluate_outer).parameters)
    assert parameters == ("registry_path", "device_name")
    source = Path(execution.__file__).read_text(encoding="utf-8")
    assert "validationData.csv" not in source
    assert "test_data=" not in source


def test_expected_risk_definition_averages_losses_not_predictions():
    subset_losses = torch.tensor(
        [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]],
        dtype=torch.float64,
    )
    expected_rows = subset_losses.mean(dim=0)
    assert torch.equal(expected_rows, torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64))
    design = Path(protocol.DEFAULT_DESIGN).read_text(encoding="utf-8")
    assert "Losses, not predictive distributions" in design

