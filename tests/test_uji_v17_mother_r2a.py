from __future__ import annotations

import inspect

import pytest

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as r0_execution
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2a as r2a_execution
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2a_protocol import (
    ADDITIONAL_MEMBER_INDICES,
    ADDITIONAL_MEMBER_SEEDS,
    BASE_MEMBER_COUNT,
    PANEL_MEMBER_COUNT,
    checkpoint_path,
)


def test_r2a_extends_the_panel_without_redefining_existing_members():
    assert BASE_MEMBER_COUNT == 5
    assert PANEL_MEMBER_COUNT == 10
    assert ADDITIONAL_MEMBER_INDICES == (6, 7, 8, 9, 10)
    assert ADDITIONAL_MEMBER_SEEDS == (17306, 17307, 17308, 17309, 17310)
    assert "member_06_seed17306.pt" in str(checkpoint_path(6))
    assert "member_10_seed17310.pt" in str(checkpoint_path(10))
    with pytest.raises(ValueError):
        checkpoint_path(5)
    with pytest.raises(ValueError):
        checkpoint_path(11)


def test_r2a_execution_reuses_both_exact_r0_training_stages():
    source = inspect.getsource(r2a_execution.train_member)
    assert "r0_execution._fit_schedule" in source
    assert "r0_execution._fit_refit" in source
    assert "development_localization_predictions\": 0" in source
    assert r0_execution.MAXIMUM_EPOCHS == 80
    assert r0_execution.PATIENCE == 10


def test_r2a_has_no_development_or_outer_evaluation_entry_point():
    assert not hasattr(r2a_execution, "evaluate")
    source = inspect.getsource(r2a_execution.main)
    assert 'sub.add_parser("train-member")' in source
    assert 'sub.add_parser("check-artifacts")' in source
    assert 'sub.add_parser("evaluate")' not in source
