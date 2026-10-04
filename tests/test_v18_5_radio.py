import inspect

import pytest
import torch

from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    fit_temperature_radio_reference,
    marked_radio_energy,
    profile_radio_nuisance,
    radio_compatibility,
    temperature_radio_reference_from_payload,
)


def _memory():
    return {
        "detection_prob": torch.tensor(
            [[0.9, 0.2], [0.4, 0.8]], dtype=torch.float32
        ),
        "rss_center": torch.tensor(
            [[-0.8, 0.2], [-0.1, 0.7]], dtype=torch.float32
        ),
        "rss_scale": torch.tensor(
            [[0.2, 0.3], [0.25, 0.2]], dtype=torch.float32
        ),
    }


def test_radio_energy_ignores_numeric_fill_when_mask_is_false():
    indices = torch.tensor([[0, 1]], dtype=torch.long)
    observed = torch.tensor([[True, False]])
    first = marked_radio_energy(
        torch.tensor([[-1.0, -1.0]]),
        observed,
        indices,
        _memory(),
    )
    second = marked_radio_energy(
        torch.tensor([[-1.0, 0.91]]),
        observed,
        indices,
        _memory(),
    )
    assert torch.allclose(first, second, atol=1e-7, rtol=0)


def test_observed_minus_one_affects_radio_energy():
    indices = torch.tensor([[0, 1]], dtype=torch.long)
    query = torch.tensor([[-1.0, 0.0]])
    observed = torch.tensor([[True, False]])
    missing = torch.tensor([[False, False]])
    first = marked_radio_energy(query, observed, indices, _memory())
    second = marked_radio_energy(query, missing, indices, _memory())
    assert not torch.allclose(first, second)


def test_temperature_matched_reference_rejects_wrong_temperature():
    reference = fit_temperature_radio_reference({
        1.0: torch.tensor([0.1, 0.2, 0.3]),
        1.5: torch.tensor([0.2, 0.4, 0.6]),
    })
    values = torch.tensor([0.15, 0.25])
    result = radio_compatibility(values, reference, temperature=1.0)
    assert torch.all((result >= 0) & (result <= 1))
    with pytest.raises(KeyError):
        radio_compatibility(values, reference, temperature=1.25)


def test_compatibility_is_monotone_in_profiled_objective():
    reference = fit_temperature_radio_reference({
        1.0: torch.tensor([0.0, 1.0, 2.0, 3.0])
    })
    values = torch.tensor([0.5, 1.5, 2.5])
    result = radio_compatibility(values, reference, temperature=1.0)
    assert bool((result[:-1] >= result[1:]).all())


def test_profile_api_has_no_target_or_label_argument():
    parameters = inspect.signature(profile_radio_nuisance).parameters
    assert "target" not in parameters
    assert "target_coords" not in parameters
    assert "label" not in parameters


def test_profiled_objective_is_finite_and_uses_only_temporary_variables():
    query = torch.tensor([[-0.9, 0.1]], dtype=torch.float32)
    observed = torch.tensor([[True, True]])
    indices = torch.tensor([[0, 1]], dtype=torch.long)
    prior = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    result = profile_radio_nuisance(
        query,
        observed,
        indices,
        _memory(),
        prior,
        steps=2,
    )
    assert result["energy"].shape == prior.shape
    assert result["profiled_objective"].shape == (1,)
    assert torch.isfinite(result["profiled_objective"]).all()
    assert result["style_offset"].requires_grad is False
    assert result["style_scale"].requires_grad is False


def test_temperature_reference_checkpoint_roundtrip():
    reference = fit_temperature_radio_reference({
        0.5: torch.tensor([0.1, 0.2, 0.4]),
        1.0: torch.tensor([0.2, 0.5, 0.7]),
    })
    payload = {
        str(key): value.clone()
        for key, value in reference.references.items()
    }
    restored = temperature_radio_reference_from_payload(payload)
    for temperature in (0.5, 1.0):
        assert torch.equal(
            reference.references[temperature],
            restored.references[temperature],
        )
    with pytest.raises(KeyError):
        restored.for_temperature(
            1.5, device=torch.device("cpu"), dtype=torch.float32
        )
