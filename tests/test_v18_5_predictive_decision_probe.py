from __future__ import annotations

import pytest
import torch

from fhm_dapp_tim.fhm_dapp.v18_5_predictive_decision_probe import (
    _paired_group_summary,
)


def test_predictive_decision_probe_uses_group_paired_delta():
    keys = [
        (0, 0, 0.0, 0.0),
        (0, 0, 0.0, 0.0),
        (0, 0, 1.0, 0.0),
        (0, 0, 1.0, 0.0),
    ]
    candidate = torch.tensor([2.0, 4.0, 8.0, 10.0])
    reference = torch.tensor([3.0, 5.0, 7.0, 9.0])
    result = _paired_group_summary(candidate, reference, keys)
    assert result["candidate_minus_reference_m"] == pytest.approx(0.0)
    assert result["coordinate_groups"] == 2
    assert result["fraction_groups_candidate_better"] == pytest.approx(0.5)
