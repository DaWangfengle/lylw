from __future__ import annotations

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v19_r0_accuracy_recovery as experiment


def test_member_paths_keep_member_one_inherited():
    assert experiment._member_checkpoint(1) == experiment.INHERITED_MEMBER
    assert "member_02_seed19301.pt" in str(
        experiment._member_checkpoint(2)
    )
    assert "member_16_seed19315.pt" in str(
        experiment._member_checkpoint(16)
    )


def test_member_one_cannot_be_retrained(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "load_registry",
        lambda *args, **kwargs: {"registry_sha256": "x"},
    )
    with pytest.raises(ValueError, match="inherited"):
        experiment.train_member(1, device_name="cpu")


def test_paired_difference_orientation_is_candidate_minus_reference():
    candidate = np.asarray([1.0, 2.0, 4.0])
    reference = np.asarray([2.0, 2.0, 3.0])
    result = experiment._paired(candidate, reference)
    assert result["candidate_minus_reference_m"] == pytest.approx(0.0)
    assert result["fraction_groups_candidate_better"] == pytest.approx(1 / 3)


def test_metric_payload_reports_spatial_minus_mean():
    keys = [
        (0, 0, 0.0, 0.0),
        (0, 0, 0.0, 0.0),
        (0, 0, 1.0, 0.0),
        (0, 0, 1.0, 0.0),
    ]
    scored = {
        "posterior_mean_error": torch.tensor([4.0, 6.0, 8.0, 10.0]),
        "predictive_spatial_median_error": torch.tensor(
            [3.0, 5.0, 7.0, 9.0]
        ),
        "energy_score": torch.tensor([2.0, 2.0, 4.0, 4.0]),
        "nll": torch.tensor([5.0, 5.0, 7.0, 7.0]),
    }
    metrics, vectors = experiment._metric_payload(scored, keys)
    assert metrics["posterior_mean_mae_m"] == pytest.approx(7.0)
    assert metrics["predictive_spatial_median_mae_m"] == pytest.approx(6.0)
    assert metrics["decision_delta_spatial_minus_mean"][
        "candidate_minus_reference_m"
    ] == pytest.approx(-1.0)
    assert vectors["posterior_mean_mae"].tolist() == [5.0, 9.0]
