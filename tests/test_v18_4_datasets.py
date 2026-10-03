from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fhm_dapp_tim"
    / "fhm_dapp"
    / "v18_4_datasets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v18_4_datasets_standalone",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
DATASETS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATASETS
SPEC.loader.exec_module(DATASETS)

load_tampere = DATASETS.load_tampere
load_tuji1 = DATASETS.load_tuji1
load_uji = DATASETS.load_uji
load_uts = DATASETS.load_uts
load_all_training_summaries = DATASETS.load_all_training_summaries


def _assert_model_scale(data) -> None:
    scaled = data.model_rss()
    assert scaled.shape == data.rss_dbm.shape
    assert np.isfinite(scaled).all()
    assert float(scaled.min()) >= -1.0 - 1e-7
    assert float(scaled.max()) <= 1.0 + 1e-7
    missing = ~np.isfinite(data.rss_dbm)
    assert np.all(scaled[missing] == -1.0)


def test_uji_training_adapter_matches_uploaded_training_source() -> None:
    train, reference = load_uji("train")
    assert len(train.coords_m) == 19937
    assert len(train.ap_ids) == 520
    assert train.split == "train"
    assert reference.dataset == "UJIIndoorLoc"
    assert not train.metadata["unseen_buildings"]
    assert not train.metadata["unseen_floors"]
    _assert_model_scale(train)


def test_uts_training_adapter_matches_uploaded_training_source() -> None:
    train, reference = load_uts("train")
    assert len(train.coords_m) == 9108
    assert len(train.ap_ids) == 589
    assert train.split == "train"
    assert reference.dataset == "UTSIndoorLoc"
    assert len(np.unique(train.floor)) == 16
    _assert_model_scale(train)


def test_tampere_training_adapter_matches_actual_training_files() -> None:
    train, reference = load_tampere("train")
    assert len(train.coords_m) == 446
    assert len(train.ap_ids) == 489
    assert train.split == "train"
    assert reference.dataset == "Tampere"
    assert train.metadata["floor_height_mapping"] == {
        "0.0": 0,
        "3.5": 1,
        "7.0": 2,
    }
    assert np.all(train.device == -1)
    _assert_model_scale(train)


def test_tuji1_training_adapter_combines_five_training_devices() -> None:
    train, reference = load_tuji1("train")
    assert len(train.coords_m) == 4595
    assert len(train.ap_ids) == 259
    assert train.split == "train"
    assert reference.dataset == "TUJI1"
    assert len(reference.device_mapping) == 5
    assert np.all(train.device >= 0)
    assert not train.metadata["ignored_evaluation_ap_ids"]
    _assert_model_scale(train)


def test_tuji1_lodo_training_dictionary_excludes_heldout_device() -> None:
    training_devices = ("A12", "POCO", "S7", "tabS7")
    train, reference = load_tuji1(
        "train",
        devices=training_devices,
    )
    assert set(reference.device_mapping) == set(training_devices)
    assert "S20" not in reference.device_mapping
    assert set(train.metadata["devices"]) == set(training_devices)
    assert np.all(train.device >= 0)
    _assert_model_scale(train)


def test_default_dataset_summary_is_training_only() -> None:
    summary = load_all_training_summaries()
    assert set(summary) == {
        "UJIIndoorLoc",
        "UTSIndoorLoc",
        "Tampere",
        "TUJI1",
    }
    for dataset, splits in summary.items():
        assert set(splits) == {"train"}, dataset

