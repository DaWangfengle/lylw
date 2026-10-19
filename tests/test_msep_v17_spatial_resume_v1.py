"""Synthetic checks of the authorized sampling repair and frozen-parameter resume."""
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as study
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import energy_score_mc


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_installed_sampler_beta_zero_is_exact_and_positive_law_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(17)
    b, n, e = 414, 1052, 5000
    q = torch.softmax(torch.tensor(rng.normal(size=(b, n)), dtype=torch.float32, device=device), 1)
    mu = torch.tensor(rng.normal(size=(b, n, 2)) * 30, dtype=torch.float32, device=device)
    sigma, y = torch.ones_like(mu), torch.zeros((b, 2), device=device)
    extended_q = torch.cat((q, q.new_zeros((b, e))), 1)
    extended_mu = torch.cat((mu, mu.new_zeros((b, e, 2))), 1)
    extended_sigma = torch.ones_like(extended_mu)
    before = [x.clone() for x in (extended_q, extended_mu, extended_sigma)]
    original = energy_score_mc(q, mu, sigma, y, draws=32, seed=18201)
    parent = study.zero_mass_safe_energy_score(q, mu, sigma, y, draws=32, seed=18201)
    expanded = study.zero_mass_safe_energy_score(extended_q, extended_mu, extended_sigma, y, draws=32, seed=18201)
    assert torch.equal(original, parent)
    assert torch.equal(parent, expanded)
    assert all(torch.equal(a, b) for a, b in zip(before, (extended_q, extended_mu, extended_sigma)))


def test_authorized_resume_uses_saved_selection_and_never_refits(tmp_path):
    """Stop at the first scoring pack: no data/model inference is performed."""
    seed, variant, dataset = 17301, "raw_neighbors", "synthetic"
    folder = tmp_path / "frozen" / f"{dataset}_seed{seed}" / variant
    folder.mkdir(parents=True)
    selection = {"joint_calibrated_base": {"temperature": .7, "theta": [.5, .25]},
                 "structure1": {}, "2B": {"beta": .35, "gamma_space": .01},
                 "2A_beta_only_control": {"beta": .2, "gamma_space": 0.}}
    study.write_json(folder / "CAL_FIT_SELECTION.json", selection)
    study.write_json(folder / "FROZEN_RUN.lock", {"original": True})
    keys = [(0, 0, 1., 2.), (0, 0, 3., 4.)]
    cal_eval = np.array([1], dtype=np.int64)
    study.write_json(folder / "CAL_EVAL.lock", {
        "selection_sha256": study.sha(folder / "CAL_FIT_SELECTION.json"),
        "cal_eval_key_sha256": study.key_sha([keys[1]])})
    study.write_json(tmp_path / "USER_AUTHORIZATION_20261019.json", {"test_fixture": True})
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    model = torch.nn.Linear(2, 2)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    ctx = {"initial": model, "keys": keys, "cal_eval": cal_eval}
    from types import SimpleNamespace
    args = SimpleNamespace(output=str(tmp_path), root=str(tmp_path), device="cpu")

    class ScoringReached(Exception):
        pass

    def capture_pack(context, actual_model, rows, **kwargs):
        assert context is ctx and rows is cal_eval
        raise ScoringReached

    with patch.object(study, "reconstruct", return_value=ctx), \
         patch.object(study, "checkpoint", return_value=ckpt), \
         patch.object(study, "make_pack", side_effect=capture_pack) as pack, \
         patch.object(study, "grid_calibrate", side_effect=AssertionError("must not refit")), \
         patch.object(study, "select_s1", side_effect=AssertionError("must not reselect")):
        with pytest.raises(ScoringReached):
            study.frozen_one(args, dataset, seed, variant, resume=True)
    assert pack.call_count == 1
    for name, data in before.items():
        assert (folder / name).read_bytes() == data
    resumed = json.loads((folder / "AUTHORIZED_RESUME.lock").read_text())
    assert resumed["refit_performed"] is False
