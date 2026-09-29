from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp import uji_v18_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_single import (
    V18_CONFIG,
    delaunay_neighbor_targets,
    validate_v18_config,
)


def _synthetic_grid() -> set[tuple[int, int, float, float]]:
    keys = set()
    for building in range(2):
        for floor in range(2):
            for x in range(7):
                for y in range(7):
                    keys.add((building, floor, float(x), float(y)))
    return keys


def _synthetic_data(keys: set[tuple[int, int, float, float]]) -> dict:
    ordered = sorted(keys)
    return {
        "keys": ordered,
        "building": np.asarray([key[0] for key in ordered], dtype=np.int64),
        "floor": np.asarray([key[1] for key in ordered], dtype=np.int64),
        "phone": np.zeros(len(ordered), dtype=np.int64),
    }


def test_v18_contract_is_strictly_single_member() -> None:
    validate_v18_config(V18_CONFIG)
    assert V18_CONFIG["residual_experts"] == 1
    assert V18_CONFIG["candidate_reranker"] is True
    assert V18_CONFIG["prototype_metric"] == "none"
    assert V18_CONFIG["scan_feature_mode"] == "bounded_maximal_invariant"

    bad = dict(V18_CONFIG)
    bad["residual_experts"] = 2
    with pytest.raises(ValueError, match="exactly one residual expert"):
        validate_v18_config(bad)


def test_v18_delaunay_targets_remove_duplicate_self_and_keep_strata() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [20.0, 0.0],
            [22.0, 0.0],
        ],
        dtype=torch.float32,
    )
    buildings = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    floors = torch.zeros(6, dtype=torch.long)
    geometry = build_delaunay_geometry(coords, buildings, floors)
    indices, weights = delaunay_neighbor_targets(geometry)

    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-6)
    for anchor in range(6):
        active = indices[anchor][weights[anchor] > 0]
        assert bool((buildings[active] == buildings[anchor]).all())
        assert bool((floors[active] == floors[anchor]).all())
        if len(torch.unique(active)) > 1 or int(active[0]) != anchor:
            assert not bool((active == anchor).any())


def test_v18_nested_coordinate_roles_are_exact_and_hull_covered() -> None:
    keys = _synthetic_grid()
    roles, _ = exp.construct_coordinate_roles(keys)
    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["development"])
    assert roles["selection_refit"] | roles["development"] == roles["outer_refit"]
    assert roles["outer_refit"].isdisjoint(roles["outer_test"])
    assert roles["outer_refit"] | roles["outer_test"] == keys
    audit = exp.audit_coordinate_roles(roles, _synthetic_data(keys))
    assert audit["exact_union_identities"] is True
    assert all(value == 0 for value in audit["hull_checks"].values())


def test_v18_outer_access_is_blocked_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="forbidden"):
        exp.load_freeze(
            tmp_path / "missing.json",
            registry={"registry_sha256": "unused"},
        )


def test_v18_source_does_not_reference_official_validation_file() -> None:
    source = Path(exp.__file__).read_text(encoding="utf-8").lower()
    assert "validationdata.csv" not in source
    assert "projectedensemble" not in source
    assert "alpha_consensus" not in source
