"""Pure synthetic proof for a proposed repair; never loads experiment data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fhm_dapp_tim.fhm_dapp.msep_single_posterior import energy_score_mc
from fhm_dapp_tim.fhm_dapp.msep_v17_spatial_operators_v1 import joint_bf_xy_nll

spec = importlib.util.spec_from_file_location("beta_zero_candidate", Path(__file__).with_name("beta_zero_repair_candidate.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def fixture(device):
    rng = np.random.default_rng(17)
    b, n, e = 414, 1052, 5000
    q = torch.softmax(torch.tensor(rng.normal(size=(b, n)), dtype=torch.float32, device=device), 1)
    mu = torch.tensor(rng.normal(size=(b, n, 2)) * 30, dtype=torch.float32, device=device)
    sigma = torch.ones_like(mu)
    y = torch.zeros((b, 2), device=device)
    exq = torch.cat((q, q.new_zeros((b, e))), 1)
    exmu = torch.cat((mu, mu.new_zeros((b, e, 2))), 1)
    exsigma = torch.ones_like(exmu)
    return q, mu, sigma, y, exq, exmu, exsigma


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_original_sampler_reproduces_cuda_zero_suffix_failure():
    q, mu, sigma, y, eq, em, es = fixture("cuda")
    assert float((q.cumsum(1) - eq.cumsum(1)[:, :q.shape[1]]).abs().max()) > 0
    a = energy_score_mc(q, mu, sigma, y, draws=32, seed=18201)
    b = energy_score_mc(eq, em, es, y, draws=32, seed=18201)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_proposed_sampler_restores_exact_beta_zero_identity_without_mutation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    q, mu, sigma, y, eq, em, es = fixture(device)
    originals = [v.clone() for v in (eq, em, es)]
    a = candidate.zero_mass_safe_energy_score(q, mu, sigma, y, draws=32, seed=18201)
    b = candidate.zero_mass_safe_energy_score(eq, em, es, y, draws=32, seed=18201)
    assert torch.equal(a, b)
    assert all(torch.equal(a, b) for a, b in zip(originals, (eq, em, es)))


def test_positive_weight_predictive_mixture_is_bitwise_unchanged():
    q, mu, sigma, y, _, _, _ = fixture("cpu")
    expected = energy_score_mc(q, mu, sigma, y, draws=32, seed=18201)
    actual = candidate.zero_mass_safe_energy_score(q, mu, sigma, y, draws=32, seed=18201)
    assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_large_zero_suffix_joint_nll_still_masks_every_virtual_component():
    q, mu, sigma, y, eq, em, es = fixture("cuda")
    batch, n = q.shape
    b = torch.zeros(n, dtype=torch.long, device="cuda")
    f = torch.arange(n, device="cuda") % 3
    ids = torch.arange(n, device="cuda")[None].expand(batch, -1)
    exids = torch.cat((ids, ids.new_zeros((batch, eq.shape[1] - n))), 1)
    truth = torch.zeros(batch, dtype=torch.long, device="cuda")
    a = joint_bf_xy_nll(q, ids, mu, sigma, y, truth, truth, b, f)
    z = joint_bf_xy_nll(eq, exids, em, es, y, truth, truth, b, f)
    assert torch.allclose(a, z)
