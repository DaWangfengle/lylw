from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "fhm_dapp_tim" / "fhm_dapp"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location(
    "v18_4_capacity_audit",
    MODULE_DIR / "v18_4_capacity_audit.py",
)
assert SPEC is not None and SPEC.loader is not None
CAPACITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CAPACITY
SPEC.loader.exec_module(CAPACITY)


def test_registered_lmax_contains_primary_quota_and_all_positive_targets() -> None:
    manifest = CAPACITY.capacity_manifest()
    assert manifest["primary_proposal_quota"] == 80
    assert manifest["registered_l_max"] == 128
    assert manifest["all_capacity_contracts_pass"] is True
    assert manifest["failed_capacity_contracts"] == []

    for roles in manifest["datasets"].values():
        for record in roles.values():
            assert record["capacity_sufficient"] is True
            assert (
                record["required_capacity_with_primary_quota"]
                <= manifest["registered_l_max"]
            )
            assert record["capacity_margin"] >= 0

    for record in manifest["tuji1_leave_one_device_out"].values():
        for role_record in record["roles"].values():
            assert role_record["capacity_sufficient"] is True
            assert role_record["capacity_margin"] >= 0
            assert (
                role_record["required_capacity_with_primary_quota"]
                <= manifest["registered_l_max"]
            )


def test_full_training_capacity_is_at_least_fit_capacity() -> None:
    manifest = CAPACITY.capacity_manifest()
    for roles in manifest["datasets"].values():
        assert (
            roles["full_training"]["coordinate_groups"]
            >= roles["selection_refit"]["coordinate_groups"]
            >= roles["fit"]["coordinate_groups"]
        )


def test_tuji_lodo_excludes_exactly_one_device() -> None:
    manifest = CAPACITY.capacity_manifest()
    devices = {"A12", "POCO", "S20", "S7", "tabS7"}
    seeds = set()
    for held_out, record in manifest["tuji1_leave_one_device_out"].items():
        assert set(record["training_devices"]) == devices - {held_out}
        seeds.add(record["role_seed"])
        roles = record["roles"]
        assert (
            roles["full_training"]["coordinate_groups"]
            >= roles["selection_refit"]["coordinate_groups"]
            >= roles["fit"]["coordinate_groups"]
        )
    assert len(seeds) == len(devices)
