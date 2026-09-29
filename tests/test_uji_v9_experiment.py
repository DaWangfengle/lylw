from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fhm_dapp_tim.fhm_dapp import uji_v9_experiment as v9


def _synthetic_coordinate_data() -> tuple[set[tuple], dict]:
    unique = []
    for building in (0, 1):
        for floor in (0, 1):
            for x in range(6):
                for y in range(6):
                    unique.append((building, floor, float(x), float(y)))
    keys = []
    phone = []
    building = []
    floor = []
    for key in unique:
        for phone_id in (0, 1):
            keys.append(key)
            phone.append(phone_id)
            building.append(key[0])
            floor.append(key[1])
    return set(unique), {
        "keys": keys,
        "phone": np.asarray(phone),
        "building": np.asarray(building),
        "floor": np.asarray(floor),
    }


def test_nested_coordinate_roles_are_disjoint_complete_and_hull_covered() -> None:
    all_keys, data = _synthetic_coordinate_data()
    roles, _details = v9.construct_coordinate_roles(all_keys)
    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["development"])
    assert roles["selection_refit"] | roles["development"] == roles["outer_refit"]
    assert roles["outer_refit"].isdisjoint(roles["outer_test"])
    assert roles["outer_refit"] | roles["outer_test"] == all_keys
    diagnostics = v9.audit_coordinate_roles(roles, data)
    assert diagnostics["exact_union_identities"] is True
    assert all(value == 0 for value in diagnostics["hull_checks"].values())
    for role in roles:
        assert len(
            diagnostics["roles"][role]["coordinate_counts_by_building_floor"]
        ) == 4


def test_registry_excludes_official_file_and_binds_design(tmp_path: Path) -> None:
    output = tmp_path / "registry.json"
    registry = v9.register_protocol(output=output)
    loaded = v9.load_registry(output)
    assert loaded["registry_sha256"] == registry["registry_sha256"]
    assert loaded["official_validation_file_in_scope"] is False
    encoded = json.dumps(loaded).lower()
    assert "validationdata.csv" not in encoded
    assert loaded["roles"]["outer_refit"]["coordinate_key_sha256"] != loaded[
        "roles"
    ]["outer_test"]["coordinate_key_sha256"]


def test_registry_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "registry.json"
    v9.register_protocol(output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["joint_grid"]["K"] = [1]
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        v9.load_registry(output)


def test_outer_access_requires_an_existing_freeze(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="forbidden"):
        v9.load_selection_freeze(
            tmp_path / "missing-freeze.json",
            registry={"registry_sha256": "unused"},
        )
