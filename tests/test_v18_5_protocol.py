import numpy as np

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import CanonicalFingerprintData
from fhm_dapp_tim.fhm_dapp.v18_4_protocol import ROLE_SEEDS as V184_ROLE_SEEDS
from fhm_dapp_tim.fhm_dapp.v18_5_protocol import (
    ROLE_SEEDS,
    audit_roles,
    construct_roles,
)


def _data():
    coords = []
    building = []
    floor = []
    # Two 4x4 strata provide hull + interior groups.
    for b in range(2):
        for x in range(4):
            for y in range(4):
                coords.append([x + 10 * b, y])
                building.append(b)
                floor.append(0)
    n = len(coords)
    return CanonicalFingerprintData(
        dataset="UJIIndoorLoc",
        split="train",
        rss_dbm=np.full((n, 2), -80.0, dtype=np.float32),
        coords_m=np.asarray(coords, dtype=np.float32),
        building=np.asarray(building, dtype=np.int64),
        floor=np.asarray(floor, dtype=np.int64),
        device=np.zeros(n, dtype=np.int64),
        user=np.zeros(n, dtype=np.int64),
        timestamp_s=np.arange(n, dtype=np.float64),
        ap_ids=("AP0", "AP1"),
        metadata={},
    )


def test_v185_uses_new_preregistered_role_seeds():
    assert ROLE_SEEDS["UJIIndoorLoc"] == 185501
    assert ROLE_SEEDS["UJIIndoorLoc"] != V184_ROLE_SEEDS["UJIIndoorLoc"]


def test_grouped_roles_are_disjoint_complete_and_keep_hull_support():
    data = _data()
    roles = construct_roles(data, seed=ROLE_SEEDS["UJIIndoorLoc"])
    audit = audit_roles(data, roles)
    assert audit["pairwise_disjoint"] is True
    assert audit["exact_union"] is True
    assert not any(audit["missing_hull_support"].values())


def test_role_construction_is_deterministic_for_registered_seed():
    data = _data()
    first = construct_roles(data, seed=ROLE_SEEDS["UJIIndoorLoc"])
    second = construct_roles(data, seed=ROLE_SEEDS["UJIIndoorLoc"])
    assert first == second
