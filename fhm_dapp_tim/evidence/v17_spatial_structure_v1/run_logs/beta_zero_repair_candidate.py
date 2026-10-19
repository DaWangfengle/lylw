"""Proposed engineering-only sampling repair. Not installed in frozen code."""
from __future__ import annotations

import torch
from fhm_dapp_tim.fhm_dapp.msep_single_posterior import energy_score_mc


def zero_mass_safe_energy_score(weights, centers, scales, target, *, draws=32, seed=19602):
    """Ignore globally zero-mass components only inside the Monte Carlo sampler.

    No positive probability, component parameter, random stream or predictive law
    is changed. Inputs retain the required compact N+E representation. CUDA's
    float32 cumsum can otherwise change physical CDF rounding when zero columns
    are appended; this can flip a discrete sampled ID under common uniforms.
    """
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    active = (weights > 0).any(dim=0)
    assert active.any()
    if not bool(active.all()):
        weights, centers, scales = weights[:, active], centers[:, active], scales[:, active]
    return energy_score_mc(weights, centers, scales, target, draws=draws, seed=seed)
