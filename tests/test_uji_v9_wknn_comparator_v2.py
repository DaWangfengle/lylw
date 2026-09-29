from __future__ import annotations

import inspect

import pytest

from fhm_dapp_tim.fhm_dapp import uji_v9_wknn_comparator_v2 as comparator


def test_execution_contract_has_only_global_label_free_predictor():
    contract = comparator._assert_global_predictor_contract()

    assert contract["candidate_scope"] == "global"
    assert contract["building_or_floor_conditioning"] is False
    assert contract["exact_match_fixture_passed"] is True
    assert contract["inverse_distance_fixture_passed"] is True
    assert contract["predictor_parameters"] == [
        "train_x",
        "train_y",
        "test_x",
        "k_grid",
        "chunk_size",
    ]


def test_result_paths_are_exact_prefixed_and_non_overwriting(tmp_path):
    expected = tmp_path / "wknn_postlock_development_selection_v2.json"

    assert comparator._validate_output_path(expected, expected, "development") == expected

    with pytest.raises(ValueError, match="output path is not frozen"):
        comparator._validate_output_path(
            tmp_path / "postlock_wknn_development_selection_v2.json",
            expected,
            "development",
        )

    expected.touch()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        comparator._validate_output_path(expected, expected, "development")


def test_development_selection_cannot_load_v9_outer_losses():
    source = inspect.getsource(comparator.run_development_selection)

    assert "_load_v9_outer_group_losses" not in source
    assert "post_lock_timing_boundary" not in source
    assert "v9_outer_result" not in source


def test_v2_a1_amendment_is_naming_only_in_document():
    text = comparator.AMENDMENT_DOCUMENT.read_text(encoding="utf-8")

    assert "wknn_postlock_development_selection_v2.json" in text
    assert "wknn_postlock_outer_comparator_v2.json" in text
    assert "No dataset role, representation, prototype construction" in text


def test_cli_defaults_admit_only_the_prefixed_v2_a1_result_names():
    development = comparator._parse_args(["development"])
    outer = comparator._parse_args(["outer"])

    assert development.output == str(comparator.DEFAULT_DEVELOPMENT)
    assert outer.output == str(comparator.DEFAULT_OUTER)
    assert development.output.split("\\")[-1].startswith("wknn_postlock_")
    assert outer.output.split("\\")[-1].startswith("wknn_postlock_")
