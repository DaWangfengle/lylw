from __future__ import annotations

import numpy as np

from fhm_dapp_tim.fhm_dapp.train_anchor_v2 import FROZEN_TRAINING
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_protocol import (
    MEMBER_SEEDS,
    R0_MEMBER_COUNT,
    construct_coordinate_roles,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0 import (
    _coordinate_group_vector,
    _materialize_csv,
    _paired_group_bootstrap,
)


def _synthetic_grid() -> set[tuple[int, int, float, float]]:
    keys = set()
    for building in range(2):
        for floor in range(2):
            for x in range(9):
                for y in range(9):
                    keys.add((building, floor, float(x), float(y)))
    return keys


def test_v17_nested_coordinate_roles_are_exact_and_disjoint() -> None:
    all_keys = _synthetic_grid()
    roles, _ = construct_coordinate_roles(all_keys)
    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["development"])
    assert roles["selection_refit"] | roles["development"] == roles["outer_refit"]
    assert roles["outer_refit"].isdisjoint(roles["outer_test"])
    assert roles["outer_refit"] | roles["outer_test"] == all_keys


def test_v17_r0_is_the_five_member_historical_prefix() -> None:
    assert R0_MEMBER_COUNT == 5
    assert len(MEMBER_SEEDS) == 12
    assert list(MEMBER_SEEDS[:R0_MEMBER_COUNT]) == list(range(17301, 17306))
    assert FROZEN_TRAINING["scan_feature_mode"] == "legacy"
    assert FROZEN_TRAINING["prototype_metric"] == "l2"
    assert FROZEN_TRAINING["target_neighbors"] == 8
    assert FROZEN_TRAINING["self_target_weight"] == 0.5
    assert FROZEN_TRAINING["domain_adversarial_weight"] == 0.1
    assert FROZEN_TRAINING["inference_top_k"] == 24


def test_coordinate_macro_averages_scans_within_physical_coordinate() -> None:
    keys = [
        (0, 0, 0.0, 0.0),
        (0, 0, 0.0, 0.0),
        (0, 0, 1.0, 0.0),
    ]
    vector, ordered = _coordinate_group_vector([1.0, 3.0, 8.0], keys)
    assert ordered == [(0, 0, 0.0, 0.0), (0, 0, 1.0, 0.0)]
    assert np.array_equal(vector, np.asarray([2.0, 8.0]))
    assert float(vector.mean()) == 5.0


def test_group_bootstrap_uses_projected_minus_prior_direction() -> None:
    prior = np.asarray([3.0, 4.0, 5.0, 6.0])
    projected = prior - 1.0
    result = _paired_group_bootstrap(projected, prior)
    assert result["mean_difference_m"] == -1.0
    assert result["ci95_m"] == [-1.0, -1.0]
    assert result["probability_of_improvement"] == 1.0


def test_role_csv_materialization_preserves_registered_source_lines(tmp_path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "role.csv"
    source.write_bytes(b"a,b\r\n0,x\r\n1,y\r\n2,z\r\n")
    _materialize_csv(source, destination, np.asarray([0, 2], dtype=np.int64))
    assert destination.read_bytes() == b"a,b\r\n0,x\r\n2,z\r\n"
