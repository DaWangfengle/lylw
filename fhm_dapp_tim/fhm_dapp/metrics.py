# 从 evaluate_anchor_ensemble.py 提取，自包含 fhm_dapp_tim 实验
"""坐标误差指标：MAE、RMSE、中位数、P90、P95.

从 evaluate_anchor_ensemble.py 提取，仅依赖 numpy.
"""

from __future__ import annotations

import numpy as np


def coordinate_metrics(
    coordinates: np.ndarray, target: np.ndarray
) -> tuple[dict, np.ndarray]:
    errors = np.linalg.norm(coordinates - target, axis=1)
    return (
        {
            "mae": float(errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_error": float(np.median(errors)),
            "p90_error": float(np.quantile(errors, 0.90)),
            "p95_error": float(np.quantile(errors, 0.95)),
        },
        errors,
    )


