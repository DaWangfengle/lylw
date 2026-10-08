"""Registered, development-only G1 geometry-support pilot for V17.

This is not an outer-test or official-validation experiment. Geometry weights
use ONLY training anchor coordinates/building/floor, never query labels.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "v17_g1_geometry_support_development_v1_20261008"
PANEL_SIZE = 25
SEEDS = tuple(range(17301, 17326))
# Historical R0 inner member_fit/early_stop epochs; fixed BEFORE G1 outcomes.
EPOCHS = (65, 69, 56, 67, 62, 60, 52, 74, 79, 62, 65, 57, 64,
          61, 50, 74, 59, 73, 60, 68, 52, 58, 53, 61, 56)
GAMMAS = (0.0, 0.35, 0.70)
RADIUS_M = 10.0
FOLD_SEED = 18111
PLACEBO_SEED = 18112
BOOTSTRAP_SEED = 18113
SCORING_SEED = 18114
BOOTSTRAP_REPETITIONS = 5000
CONFIDENCE = 0.95
ES_CALIBRATION_TOLERANCE_M = 0.02
MIN_CONFIRM_GAIN_M = 0.15
EXPECTED_TRAIN_SHA256 = "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168"
EXPECTED_SELECTION_ANCHORS = 595
EXPECTED_DEVELOPMENT_COORDINATES = 151


def _stable_key(seed: int, key: tuple) -> bytes:
    return hashlib.sha256((str(seed) + ":" + repr(tuple(key))).encode("utf-8")).digest()


def split_development_keys(keys: set[tuple], *, seed: int = FOLD_SEED) -> tuple[set[tuple], set[tuple]]:
    """Stratified fixed split by physical coordinate, independent of RSS/outcomes."""
    strata: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for key in keys:
        if len(key) != 4:
            raise ValueError("coordinate key must be building/floor/x/y")
        strata[(int(key[0]), int(key[1]))].append(tuple(key))
    calibration, confirmation = set(), set()
    for stratum, group in sorted(strata.items()):
        ordered = sorted(group, key=lambda key: (_stable_key(seed, key), key))
        if len(ordered) == 1:
            # Exact singleton assignment is frozen via hash, not outcomes.
            destination = calibration if _stable_key(seed + 1, ordered[0])[0] % 2 == 0 else confirmation
            destination.add(ordered[0])
            continue
        cut = max(1, min(len(ordered) - 1, len(ordered) // 2))
        calibration.update(ordered[:cut])
        confirmation.update(ordered[cut:])
    if not calibration or not confirmation or calibration & confirmation or calibration | confirmation != keys:
        raise ValueError("invalid G1 coordinate fold partition")
    return calibration, confirmation


def support_log_density(
    coords: torch.Tensor,
    buildings: torch.Tensor,
    floors: torch.Tensor,
    *,
    radius_m: float = RADIUS_M,
) -> torch.Tensor:
    """log(1 + number of other anchors within radius and same building/floor).

    The self-anchor is never counted; identical coordinates in distinct
    anchor slots count as neighbors unless the slots are identical.
    """
    xy = torch.as_tensor(coords, dtype=torch.float64, device="cpu")
    b = torch.as_tensor(buildings, dtype=torch.long, device="cpu").reshape(-1)
    f = torch.as_tensor(floors, dtype=torch.long, device="cpu").reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2 or not (len(xy) == len(b) == len(f)):
        raise ValueError("invalid anchor geometry shapes")
    if not math.isfinite(radius_m) or radius_m <= 0 or not torch.isfinite(xy).all():
        raise ValueError("invalid radius or nonfinite anchor coordinates")
    distance = torch.cdist(xy, xy)
    same = (b[:, None] == b[None, :]) & (f[:, None] == f[None, :])
    near = (distance <= radius_m) & same
    near.fill_diagonal_(False)
    density = 1 + near.sum(dim=1)
    return density.to(torch.float64).log()


def placebo_log_density(
    log_density: torch.Tensor, buildings: torch.Tensor, floors: torch.Tensor,
    *, seed: int = PLACEBO_SEED
) -> torch.Tensor:
    """Permute density assignments only within matching building/floor."""
    value = torch.as_tensor(log_density, dtype=torch.float64).cpu().clone()
    b = torch.as_tensor(buildings).cpu().tolist()
    f = torch.as_tensor(floors).cpu().tolist()
    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (building, floor) in enumerate(zip(b, f)):
        strata[(int(building), int(floor))].append(i)
    gen = np.random.default_rng(seed)
    for _, positions in sorted(strata.items()):
        permutation = gen.permutation(len(positions))
        value[positions] = value[torch.as_tensor(positions, dtype=torch.long)[permutation]]
    return value


def corrected_probability(
    probability: torch.Tensor, log_density: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Apply inverse local support density to atomic anchor probability.

    Input P may be [E, batch, anchors] or [batch, anchors].
    gamma=0 returns the SAME tensor, preserving the exact V17 baseline.
    """
    if not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError("gamma must be nonnegative finite")
    if probability.ndim not in (2, 3) or probability.shape[-1] != len(log_density):
        raise ValueError("prior/density shape mismatch")
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("invalid atomic prior")
    if gamma == 0.0:
        return probability
    log_d = torch.as_tensor(log_density, dtype=probability.dtype, device=probability.device)
    if not torch.isfinite(log_d).all():
        raise ValueError("nonfinite log density")
    # This changes probabilities, not trained neural/prototype logits or residual laws.
    weighted = probability.double() * torch.exp(
        -float(gamma) * (log_d.double() - log_d.double().mean())
    )
    denom = weighted.sum(dim=-1, keepdim=True)
    if not torch.isfinite(denom).all() or not (denom > 0).all():
        raise ValueError("G1 density correction lost posterior mass")
    return (weighted / denom).to(dtype=probability.dtype)


def choose_gamma(calibration_metrics: dict[str, dict[str, float]]) -> float:
    """Frozen MAE-first, energy-guarded choice. Never inspect confirmation."""
    base = calibration_metrics["gamma_0"]["energy_score_2d_m"]
    eligible = []
    for gamma in GAMMAS:
        key = gamma_name(gamma)
        mae = calibration_metrics[key]["mae_2d_m"]
        energy = calibration_metrics[key]["energy_score_2d_m"]
        if math.isfinite(mae) and math.isfinite(energy) and energy <= base + ES_CALIBRATION_TOLERANCE_M:
            eligible.append((mae, gamma))
    return min(eligible)[1] if eligible else 0.0


def gamma_name(gamma: float) -> str:
    if gamma not in GAMMAS:
        raise ValueError("gamma not in preregistered grid")
    return "gamma_" + ("%g" % gamma).replace(".", "p")


def confirm_gate(delta_mae: float, upper_mae: float,
                 delta_es: float, upper_es: float, placebo_delta_mae: float) -> dict:
    """Decision is conjunctive; exploratory results are never outer confirmation."""
    checks = {
        "mae_gain_at_least_0p15m": delta_mae <= -MIN_CONFIRM_GAIN_M,
        "mae_familywise_upper_below_zero": upper_mae < 0.0,
        "energy_familywise_upper_not_positive": upper_es <= 0.0,
        "geometry_better_than_permuted_density": delta_mae < placebo_delta_mae,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "comparison": "candidate_minus_exact_v17_baseline",
            "note": "development-only evidence, not official or outer confirmation"}
