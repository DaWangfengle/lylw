import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 import (
    EPOCHS,
    PRIMARY_CHECKPOINT,
    select_checkpoint_epochs,
    split_cal,
)
from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import (
    neighbor_contamination,
    same_floor_neighbors,
)
from fhm_dapp_tim.fhm_dapp.train_anchor import _neighbor_targets


def test_checkpoint_selection_replays_legacy_stop_and_skips_warmup():
    scores = [9.30] + [9.40] * 19 + [4.90] * 59 + [3.90]
    history = [
        {"epoch": epoch, "dev_mae_m": score}
        for epoch, score in enumerate(scores, start=1)
    ]
    selected = select_checkpoint_epochs(history)
    assert selected["legacy_early_stop"]["selected_epoch"] == 1
    assert selected["legacy_early_stop"]["simulated_stop_epoch"] == 20
    assert selected["best_dev_epoch_1_80"]["selected_epoch"] == 80
    assert selected[PRIMARY_CHECKPOINT]["selected_epoch"] == 80
    assert EPOCHS == 80


def test_checkpoint_selection_rejects_incomplete_or_nonfinite_history():
    with pytest.raises(ValueError, match="exactly 80"):
        select_checkpoint_epochs([{"epoch": 1, "dev_mae_m": 1.0}])
    history = [{"epoch": epoch, "dev_mae_m": float(epoch)}
               for epoch in range(1, EPOCHS + 1)]
    history[10]["dev_mae_m"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        select_checkpoint_epochs(history)


def test_cal_split_is_physical_site_disjoint():
    keys = []
    for building, floor in ((0, 0), (0, 1)):
        for site in range(12):
            key = (building, floor, float(site), float(site % 3))
            keys.extend([key, key])
    fit, evaluation = split_cal(np.arange(len(keys)), keys, seed=20261116)
    fit_sites = {keys[int(i)] for i in fit}
    eval_sites = {keys[int(i)] for i in evaluation}
    assert fit_sites
    assert eval_sites
    assert fit_sites.isdisjoint(eval_sites)
    assert set(fit).isdisjoint(set(evaluation))


def test_same_floor_is_exact_noop_when_every_anchor_has_same_bf():
    coords = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [5., 1.], [4., 3.]])
    buildings = torch.zeros(5, dtype=torch.long)
    floors = torch.zeros(5, dtype=torch.long)
    old_ids, old_weights = _neighbor_targets(
        coords, buildings, neighbor_count=4, temperature_m=8.0)
    new_ids, new_weights = same_floor_neighbors(
        coords, buildings, floors, count=4, temperature_m=8.0)
    assert torch.equal(old_ids, new_ids)
    assert torch.allclose(old_weights, new_weights, atol=1e-7, rtol=1e-7)
    assert neighbor_contamination(new_ids, new_weights, buildings, floors) == 0.0


def test_same_floor_neighbors_remove_cross_floor_mass():
    coords = torch.tensor([[0., 0.], [0., 0.], [1., 0.], [2., 0.]])
    buildings = torch.tensor([0, 0, 0, 0])
    floors = torch.tensor([0, 1, 0, 1])
    ids, weights = same_floor_neighbors(
        coords, buildings, floors, count=3, temperature_m=8.0)
    assert torch.allclose(weights.sum(1), torch.ones(4))
    assert neighbor_contamination(ids, weights, buildings, floors) == 0.0
    assert torch.all(buildings[ids] == buildings[:, None])
    assert torch.all(floors[ids] == floors[:, None])
