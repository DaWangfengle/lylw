from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "fhm_dapp_tim" / "fhm_dapp"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

DATA_SPEC = importlib.util.spec_from_file_location(
    "v18_4_datasets",
    MODULE_DIR / "v18_4_datasets.py",
)
assert DATA_SPEC is not None and DATA_SPEC.loader is not None
DATA = importlib.util.module_from_spec(DATA_SPEC)
sys.modules[DATA_SPEC.name] = DATA
DATA_SPEC.loader.exec_module(DATA)

PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "v18_4_protocol",
    MODULE_DIR / "v18_4_protocol.py",
)
assert PROTOCOL_SPEC is not None and PROTOCOL_SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(PROTOCOL_SPEC)
sys.modules[PROTOCOL_SPEC.name] = PROTOCOL
PROTOCOL_SPEC.loader.exec_module(PROTOCOL)


def test_registered_role_manifest_uses_only_training_sources() -> None:
    manifest = PROTOCOL.prepare_manifest()
    assert manifest["development_fraction"] == 0.15
    assert manifest["calibration_fraction"] == 0.15
    expected_groups = {
        "UJIIndoorLoc": 933,
        "UTSIndoorLoc": 1466,
        "Tampere": 446,
        "TUJI1": 919,
    }
    expected_rows = {
        "UJIIndoorLoc": 19937,
        "UTSIndoorLoc": 9108,
        "Tampere": 446,
        "TUJI1": 4595,
    }
    for name, total_groups in expected_groups.items():
        record = manifest["datasets"][name]
        audit = record["audit"]
        assert record["source_split"] == "train"
        assert audit["pairwise_disjoint"] is True
        assert audit["exact_union"] is True
        assert (
            audit["coordinate_groups"]["fit"]
            + audit["coordinate_groups"]["development"]
            + audit["coordinate_groups"]["calibration"]
            == total_groups
        )
        assert (
            audit["rows"]["fit"]
            + audit["rows"]["development"]
            + audit["rows"]["calibration"]
            == expected_rows[name]
        )
        assert all(
            value == 0
            for value in audit["missing_hull_support"].values()
        )


def test_development_and_calibration_retain_all_training_strata() -> None:
    manifest = PROTOCOL.prepare_manifest()
    for record in manifest["datasets"].values():
        missing = record["audit"]["missing_strata"]
        assert missing["fit"] == []
        assert missing["development"] == []
        assert missing["selection_refit"] == []
        assert missing["calibration"] == []


def test_role_construction_is_deterministic() -> None:
    data, _ = DATA.load_uji("train")
    first = PROTOCOL.construct_roles(
        data,
        seed=PROTOCOL.ROLE_SEEDS["UJIIndoorLoc"],
    )
    second = PROTOCOL.construct_roles(
        data,
        seed=PROTOCOL.ROLE_SEEDS["UJIIndoorLoc"],
    )
    assert first == second


def test_uji_is_development_only_and_external_tests_are_reserved() -> None:
    manifest = PROTOCOL.prepare_manifest()
    assert (
        manifest["datasets"]["UJIIndoorLoc"]["evaluation_status"]
        == "development_only_preexposed_official"
    )
    for name in ("UTSIndoorLoc", "Tampere", "TUJI1"):
        assert (
            manifest["datasets"][name]["evaluation_status"]
            == "provided_test_reserved_no_model_selection"
        )


def test_tuji1_leave_one_device_out_protocol_is_fixed() -> None:
    manifest = PROTOCOL.prepare_manifest()
    cross = manifest["tuji1_cross_device_protocol"]
    assert cross["devices"] == ["A12", "POCO", "S20", "S7", "tabS7"]
    assert "leave-one-device-out" in cross["rule"]
