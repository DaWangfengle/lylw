# 从 evaluate_anchor_robust_ensemble.py 提取，自包含 fhm_dapp_tim 实验
"""鲁棒集成聚合：Huber M-estimator、加权几何中位数、配对 bootstrap.

从 evaluate_anchor_robust_ensemble.py 提取的聚合函数，仅依赖 numpy.
保留原始代码逻辑，不修改函数行为.
"""

from __future__ import annotations

import numpy as np


def weighted_geometric_median(
    predictions: np.ndarray,
    weights: np.ndarray,
    iterations: int = 32,
) -> np.ndarray:
    """加权几何中位数 (Weiszfeld 迭代).

    Args:
        predictions: [M, N, 2] M 个成员的预测坐标.
        weights: [M, N] 每个成员每个样本的权重.
        iterations: 最大迭代次数.

    Returns:
        [N, 2] 聚合后的坐标.
    """
    estimate = np.sum(
        predictions * weights[:, :, None], axis=0)
    for _ in range(iterations):
        distance = np.linalg.norm(
            predictions - estimate[None, :, :], axis=2)
        effective = weights / np.maximum(distance, 1e-4)
        updated = np.sum(
            predictions * effective[:, :, None], axis=0
        ) / np.maximum(effective.sum(axis=0)[:, None], 1e-12)
        if np.max(np.linalg.norm(updated - estimate, axis=1)) < 1e-5:
            estimate = updated
            break
        estimate = updated
    return estimate


def huber_precision_center(
    predictions: np.ndarray,
    weights: np.ndarray,
    iterations: int = 16,
) -> np.ndarray:
    """Canonical Huber M-estimator with inverse-variance base precision.

    Fixed Huber fusion used by the final protocol: initialize from the
    inverse-variance weighted mean,
    每轮用 Huber 权重 (1.345 * median_scale / distance 截断) 重新加权，
    直到收敛。

    Args:
        predictions: [M, N, 2] M 个成员的预测坐标.
        weights: [M, N] 基础精度权重 (通常 = 1/radial_scale²).
        iterations: 最大迭代次数.

    Returns:
        [N, 2] Huber 聚合后的坐标.
    """
    estimate = np.sum(
        predictions * weights[:, :, None], axis=0)
    for _ in range(iterations):
        distance = np.linalg.norm(
            predictions - estimate[None, :, :], axis=2)
        robust_scale = np.median(distance, axis=0)
        threshold = 1.345 * np.maximum(robust_scale, 1e-3)
        huber = np.minimum(
            1.0, threshold[None, :] / np.maximum(distance, 1e-6))
        effective = weights * huber
        effective /= np.maximum(
            effective.sum(axis=0, keepdims=True), 1e-12)
        updated = np.sum(
            predictions * effective[:, :, None], axis=0)
        if np.max(np.linalg.norm(updated - estimate, axis=1)) < 1e-5:
            estimate = updated
            break
        estimate = updated
    return estimate


def paired_bootstrap(
    candidate_errors: np.ndarray,
    baseline_errors: np.ndarray,
    repetitions: int = 10000,
    seed: int = 20260612,
) -> dict:
    """配对 bootstrap 检验 candidate vs baseline 的误差差.

    Args:
        candidate_errors: [N] candidate 的每样本误差.
        baseline_errors: [N] baseline 的每样本误差.
        repetitions: bootstrap 重复次数.
        seed: 随机种子.

    Returns:
        dict 含 mean_difference_m, ci95_m, probability_of_improvement,
        repetitions, seed.
    """
    difference = candidate_errors - baseline_errors
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        indices = rng.integers(
            0, len(difference), size=(count, len(difference)))
        bootstrap[start:start + count] = difference[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_difference_m": float(difference.mean()),
        "ci95_m": [float(lower), float(upper)],
        "probability_of_improvement": float(
            (bootstrap < 0).mean()),
        "repetitions": repetitions,
        "seed": seed,
    }


