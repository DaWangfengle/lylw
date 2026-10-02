from __future__ import annotations

import numpy as np

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    load_tampere,
    load_tuji1,
    load_uji,
    load_uts,
)


def _assert_model_scale(data) -> None:
    scaled = data.model_rss()
    assert scaled.shape == data.rss_dbm.shape
    assert np.isfinite(scaled).all()
    assert float(scaled.min()) >= -1.0 - 1e-7
    assert float(scaled.max()) <= 1.0 + 1e-7
    missing = ~np.isfinite(data.rss_dbm)
    assert np.all(scaled[missing] == -1.0)


def test_uji_adapter_uses_training_reference_for_preexposed_validation() -> None:
    train, reference = load_uji("train")
    validation, reused = load_uji(
        "preexposed_official",
        reference=reference,
    )
    assert len(train.coords_m) == 19937
    assert len(validation.coords_m) == 1111
    assert len(train.ap_ids) == 520
    assert validation.ap_ids == train.ap_ids
    assert reused is reference
    assert validation.split == "preexposed_official"
    assert not validation.metadata["extra_evaluation_aps_ignored"]
    _assert_model_scale(train)
    _assert_model_scale(validation)


def test_uts_adapter_never_expands_feature_space_from_test() -> None:
    train, reference = load_uts("train")
    test, reused = load_uts("test", reference=reference)
    assert len(train.coords_m) == 9108
    assert len(test.coords_m) == 388
    assert len(train.ap_ids) == 589
    assert test.ap_ids == train.ap_ids
    assert reused is reference
    assert not test.metadata["extra_evaluation_aps_ignored"]
    assert not test.metadata["unseen_buildings"]
    assert not test.metadata["unseen_floors"]
    _assert_model_scale(train)
    _assert_model_scale(test)


def test_tampere_adapter_matches_actual_four_file_release() -> None:
    train, reference = load_tampere("train")
    test, reused = load_tampere("test", reference=reference)
    assert len(train.coords_m) == 446
    assert len(test.coords_m) == 982
    assert len(train.ap_ids) == 489
    assert test.ap_ids == train.ap_ids
    assert reused is reference
    assert train.metadata["floor_height_mapping"]
    assert not test.metadata["unseen_floor_heights"]
    _assert_model_scale(train)
    _assert_model_scale(test)


def test_tuji1_adapter_combines_five_devices_without_test_ap_leakage() -> None:
    train, reference = load_tuji1("train")
    test, reused = load_tuji1("test", reference=reference)
    assert len(train.coords_m) == 4595
    assert len(test.coords_m) == 4304
    assert len(train.ap_ids) > 0
    assert test.ap_ids == train.ap_ids
    assert reused is reference
    assert len(reference.device_mapping) == 5
    assert np.all(train.device >= 0)
    assert np.all(test.device >= 0)
    _assert_model_scale(train)
    _assert_model_scale(test)


def test_tuji1_cross_device_holdout_does_not_register_heldout_device() -> None:
    training_devices = ("A12", "POCO", "S7", "tabS7")
    train, reference = load_tuji1(
        "train",
        devices=training_devices,
    )
    heldout, _ = load_tuji1(
        "test",
        devices=("S20",),
        reference=reference,
    )
    assert set(reference.device_mapping) == set(training_devices)
    assert np.all(heldout.device == -1)
    assert heldout.metadata["unseen_devices"] == ["S20"]
    assert heldout.ap_ids == train.ap_ids
    _assert_model_scale(heldout)
