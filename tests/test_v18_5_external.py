from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import v18_5_external as external
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import _canonical_sha
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import DatasetReference
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG


def _fake_uji_freeze() -> dict[str, object]:
    return {
        "source_commit_sha": "a" * 40,
        "freeze_sha256": "freeze-hash",
        "lambda_spatial": 0.25,
        "posterior_temperature": 1.5,
        "epsilon_kl": 0.05,
        "scale_multiplier": 1.25,
    }


def test_external_specs_register_exactly_eight_protocols():
    specs = external.external_specs()
    assert set(specs) == {
        "uts",
        "tampere",
        "tuji-pooled",
        "tuji-lodo-A12",
        "tuji-lodo-POCO",
        "tuji-lodo-S20",
        "tuji-lodo-S7",
        "tuji-lodo-tabS7",
    }
    for device in external.TUJI_DEVICES:
        spec = specs[f"tuji-lodo-{device}"]
        assert spec.heldout_device == device
        assert device not in spec.training_devices
        assert spec.evaluation_devices == (device,)
        assert len(spec.training_devices) == len(external.TUJI_DEVICES) - 1


def test_dataset_reference_json_roundtrip():
    reference = DatasetReference(
        dataset="demo",
        ap_ids=("a", "b"),
        building_mapping={"B": 0},
        floor_mapping={"F": 0},
        device_mapping={"phone": 0},
        user_mapping={"user": 0},
        extra={"note": [1, 2]},
    )
    restored = external._reference_from_json(
        external._reference_payload_json(reference)
    )
    assert restored == reference


def test_external_calibration_grid_is_exactly_192_and_keeps_identity_epsilon():
    assert (
        len(V185_CONFIG["temperature_grid"])
        * len(V185_CONFIG["epsilon_kl_grid"])
        * len(V185_CONFIG["scale_grid"])
        == 192
    )
    assert 0.0 in V185_CONFIG["epsilon_kl_grid"]


def test_prepare_uts_is_training_only(tmp_path: Path):
    events = []
    manifest_path = tmp_path / "uts_manifest.json"
    with patch.object(
        external, "load_uji_method_freeze", return_value=_fake_uji_freeze()
    ), patch.object(
        external, "_current_head", return_value="b" * 40
    ), patch.object(
        external,
        "_load_test",
        side_effect=AssertionError("prepare must not load test data"),
    ):
        manifest = external.prepare_external(
            "uts",
            manifest_path=manifest_path,
        )
    assert manifest["dataset"] == "UTSIndoorLoc"
    assert manifest["training_rows"] == 9108
    assert manifest["aps"] == 589
    assert manifest["formal_test_performance_evaluated"] is False
    assert manifest["external_calibration_rule"]["cells"] == 192
    assert manifest["uji_frozen_lambda_spatial"] == 0.25
    # UJI scalar calibration is recorded only for traceability; external
    # calibration still contains all 192 registered cells.
    assert manifest["uji_selected_calibration_for_reference_only"] == {
        "temperature": 1.5,
        "epsilon_kl": 0.05,
        "scale_multiplier": 1.25,
    }
    assert manifest_path.exists()


def test_prepare_tampere_is_training_only(tmp_path: Path):
    with patch.object(
        external, "load_uji_method_freeze", return_value=_fake_uji_freeze()
    ), patch.object(
        external, "_current_head", return_value="b" * 40
    ), patch.object(
        external,
        "_load_test",
        side_effect=AssertionError("prepare must not load test data"),
    ):
        manifest = external.prepare_external(
            "tampere",
            manifest_path=tmp_path / "tampere_manifest.json",
        )
    assert manifest["dataset"] == "Tampere"
    assert manifest["training_rows"] == 446
    assert manifest["aps"] == 489
    assert len(manifest["training_files"]) == 2


def test_prepare_tuji_lodo_excludes_heldout_device_everywhere(tmp_path: Path):
    with patch.object(
        external, "load_uji_method_freeze", return_value=_fake_uji_freeze()
    ), patch.object(
        external, "_current_head", return_value="b" * 40
    ), patch.object(
        external,
        "_load_test",
        side_effect=AssertionError("prepare must not load test data"),
    ):
        manifest = external.prepare_external(
            "tuji-lodo-A12",
            manifest_path=tmp_path / "lodo_manifest.json",
        )
    assert manifest["heldout_device"] == "A12"
    assert "A12" not in manifest["training_devices"]
    assert manifest["evaluation_devices"] == ["A12"]
    assert "A12" not in manifest["dataset_reference"]["device_mapping"]
    assert len(manifest["training_files"]) == 4
    assert manifest["training_rows"] > 0


def test_uji_freeze_validation_rejects_geometry_correction(
    tmp_path: Path,
):
    checkpoint = tmp_path / "uji.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "protocol_version": external.UJI_PROTOCOL_VERSION,
        "source_commit_sha": "a" * 40,
        "registry_sha256": "reg",
        "development_result_sha256": "dev",
        "calibration_result_sha256": "cal",
        "lambda_spatial": 0.25,
        "selected_epoch": 12,
        "posterior_temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
        "decoder": "posterior_mean_of_component_locations",
        "correction": "radio_only_exact_KL_projection",
        "geometry_posterior_correction": True,
        "selection_refit_checkpoint": str(checkpoint),
        "selection_refit_checkpoint_sha256": external._sha256_file(checkpoint),
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    payload["freeze_sha256"] = _canonical_sha(payload)
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with patch.object(external, "_method_source_compatible"):
        with pytest.raises(ValueError, match="geometry"):
            external.load_uji_method_freeze(path)


def test_evaluate_writes_lock_before_test_loader(tmp_path: Path):
    events: list[str] = []
    source = tmp_path / "test.csv"
    source.write_text("placeholder", encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    result_path = tmp_path / "result.json"

    freeze = {
        "training_freeze_sha256": "train-freeze",
        "uji_method_freeze_sha256": "uji-freeze",
        "checkpoint_sha256": "checkpoint",
        "checkpoint_path": "unused.pt",
        "manifest_sha256": "manifest",
        "execution_source_commit_sha": "c" * 40,
        "lambda_spatial": 0.25,
        "selected_epoch": 12,
        "posterior_temperature": 1.0,
        "epsilon_kl": 0.0,
        "scale_multiplier": 1.0,
    }
    model = SimpleNamespace(
        anchor_detection_prob=torch.ones(2, 2),
        anchor_rss_center=torch.zeros(2, 2),
        anchor_rss_scale=torch.ones(2, 2),
    )
    checkpoint = {
        "memory": {
            "prototype_rss": torch.zeros(2, 2),
            "prototype_observed_mask": torch.ones(2, 2, dtype=torch.bool),
        },
        "origin": torch.zeros(2),
    }
    reference = DatasetReference(
        dataset="UTSIndoorLoc",
        ap_ids=("a", "b"),
        building_mapping={"0": 0},
        floor_mapping={"0": 0},
        device_mapping={"0": 0},
        user_mapping={"0": 0},
        extra={},
    )
    fake_test = SimpleNamespace(metadata={})

    def write_new(path, payload):
        if Path(path) == lock_path:
            events.append("lock")
        elif Path(path) == result_path:
            events.append("result")

    def load_test(*args, **kwargs):
        events.append("test_load")
        return fake_test

    def distribution(*args, **kwargs):
        return {
            "probability": torch.tensor([[1.0]]),
            "component_location": torch.zeros(1, 1, 2),
            "component_scale": torch.ones(1, 1, 2),
            "posterior_mean": torch.zeros(1, 2),
            "correction": None,
        }

    with patch.object(
        external, "load_external_training_freeze", return_value=freeze
    ), patch.object(
        external, "_test_paths", return_value=[source]
    ), patch.object(
        external,
        "_load_external_checkpoint",
        return_value=(model, reference, object(), checkpoint),
    ), patch.object(
        external, "v185_distribution", side_effect=distribution
    ), patch.object(
        external, "_write_json_new", side_effect=write_new
    ), patch.object(
        external, "_load_test", side_effect=load_test
    ), patch.object(
        external, "_file_manifest", return_value=[]
    ), patch.object(
        external, "_evaluate_reserved_test", return_value={"mae": 1.0}
    ), patch.object(
        external, "_current_head", return_value="c" * 40
    ):
        result = external.evaluate_test_once(
            "uts",
            training_freeze_path=tmp_path / "freeze.json",
            lock_path=lock_path,
            result_path=result_path,
            device_name="cpu",
        )

    assert events.index("lock") < events.index("test_load")
    assert events.index("test_load") < events.index("result")
    assert result["formal_test_performance_evaluated"] is True


def test_external_source_contains_no_geometry_or_gm_deployment():
    source = (
        Path(__file__).resolve().parents[1]
        / "fhm_dapp_tim"
        / "fhm_dapp"
        / "v18_5_external.py"
    ).read_text(encoding="utf-8")
    assert "weighted_component_geometric_median" not in source
    assert "geometry_reliability" not in source
    assert '"geometry_posterior_correction": False' in source
