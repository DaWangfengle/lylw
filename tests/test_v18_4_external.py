from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fhm_dapp_tim.fhm_dapp.uji_v18_4_development import _canonical_sha
from fhm_dapp_tim.fhm_dapp.v18_4_external import (
    TUJI_DEVICES,
    TUJI_LODO_ROLE_SEEDS,
    _current_head,
    _reference_from_json,
    _reference_payload_json,
    external_specs,
    load_external_manifest,
    prepare_external,
)
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import DatasetReference


def _fake_uji_freeze(path: Path) -> Path:
    payload = {
        "protocol_version": "test",
        "source_commit_sha": _current_head(),
        "lambda_ot": 0.25,
        "epsilon_kl": 0.05,
        "uji_official_validation_preexposed": True,
        "uji_official_validation_used_for_v184_selection": False,
        "external_provided_tests_preexposed_for_adapter_validation": True,
        "external_test_performance_used_for_v184_selection": False,
    }
    payload["freeze_sha256"] = _canonical_sha(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_external_current_head_checks_tracked_tree_against_head() -> None:
    with patch(
        "fhm_dapp_tim.fhm_dapp.v18_4_external.subprocess.check_output",
        return_value="b" * 40,
    ), patch(
        "fhm_dapp_tim.fhm_dapp.v18_4_external.subprocess.run",
        return_value=SimpleNamespace(returncode=1),
    ) as run:
        try:
            _current_head()
        except RuntimeError as exc:
            assert "tracked working-tree changes" in str(exc)
        else:
            raise AssertionError("tracked source difference must fail closed")
        assert run.call_args.args[0] == [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
        ]


def test_external_specs_register_all_reserved_protocols() -> None:
    specs = external_specs()
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
    assert len(set(TUJI_LODO_ROLE_SEEDS.values())) == len(TUJI_DEVICES)
    for device in TUJI_DEVICES:
        spec = specs[f"tuji-lodo-{device}"]
        assert spec.heldout_device == device
        assert device not in spec.training_devices
        assert spec.evaluation_devices == (device,)
        assert len(spec.training_devices) == len(TUJI_DEVICES) - 1


def test_dataset_reference_json_roundtrip() -> None:
    reference = DatasetReference(
        dataset="demo",
        ap_ids=("a", "b"),
        building_mapping={"0": 0},
        floor_mapping={"1": 0},
        device_mapping={"phone": 0},
        user_mapping={"u": 0},
        extra={"x": [1, 2]},
    )
    restored = _reference_from_json(_reference_payload_json(reference))
    assert restored == reference


def test_prepare_external_uts_uses_training_only(tmp_path: Path) -> None:
    uji = _fake_uji_freeze(tmp_path / "uji_freeze.json")
    manifest_path = tmp_path / "uts_manifest.json"
    manifest = prepare_external(
        "uts",
        uji_freeze_path=uji,
        manifest_path=manifest_path,
    )
    assert manifest["dataset"] == "UTSIndoorLoc"
    assert manifest["training_rows"] == 9108
    assert manifest["aps"] == 589
    assert manifest["formal_test_performance_evaluated"] is False
    assert manifest["roles"]["fit"]["row_count"] > 0
    assert manifest["roles"]["development"]["row_count"] > 0
    assert manifest["roles"]["calibration"]["row_count"] > 0

    loaded = load_external_manifest(
        manifest_path,
        uji_freeze_path=uji,
    )
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]


def test_prepare_external_tampere_uses_training_only(tmp_path: Path) -> None:
    uji = _fake_uji_freeze(tmp_path / "uji_freeze.json")
    manifest = prepare_external(
        "tampere",
        uji_freeze_path=uji,
        manifest_path=tmp_path / "tampere_manifest.json",
    )
    assert manifest["dataset"] == "Tampere"
    assert manifest["training_rows"] == 446
    assert manifest["aps"] == 489
    assert manifest["formal_test_performance_evaluated"] is False
    assert len(manifest["training_files"]) == 2


def test_prepare_external_tuji_lodo_excludes_heldout_device(
    tmp_path: Path,
) -> None:
    uji = _fake_uji_freeze(tmp_path / "uji_freeze.json")
    manifest = prepare_external(
        "tuji-lodo-A12",
        uji_freeze_path=uji,
        manifest_path=tmp_path / "tuji_lodo_manifest.json",
    )
    assert manifest["dataset"] == "TUJI1"
    assert manifest["heldout_device"] == "A12"
    assert "A12" not in manifest["training_devices"]
    assert manifest["evaluation_devices"] == ["A12"]
    assert len(manifest["training_devices"]) == 4
    assert len(manifest["training_files"]) == 4
    assert "A12" not in manifest["reference"]["device_mapping"]
    assert manifest["formal_test_performance_evaluated"] is False
    assert manifest["training_rows"] > 0
    assert manifest["aps"] > 0
