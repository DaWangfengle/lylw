"""Numerically stable deterministic WKNN primitives for the V9 comparator.

The method semantics match the three historical V4.4 WKNN representations.
This copy exists because physical UJI coordinates are metre-scale values near
``4.86e6`` and must not be averaged in float32 precision.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


METHODS = ("raw_wknn", "diff_wknn", "rank_wknn")
K_GRID = (1, 3, 5, 8, 12, 20)
VISIBILITY_THRESHOLD = -0.75


def visible_centered_feature(values: np.ndarray) -> np.ndarray:
    """Visibility mask plus visible-mean-centered normalized RSS."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows,aps]")
    visible = x > VISIBILITY_THRESHOLD
    count = visible.sum(axis=1, keepdims=True)
    center = np.divide(
        np.where(visible, x, 0.0).sum(axis=1, keepdims=True),
        np.maximum(count, 1),
    )
    centered = np.where(visible, x - center, 0.0)
    return np.concatenate((visible.astype(np.float64), centered), axis=1)


def rank_feature(values: np.ndarray) -> np.ndarray:
    """Visibility mask plus stable normalized ranks among visible RSS values."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows,aps]")
    visible = x > VISIBILITY_THRESHOLD
    rank = np.zeros_like(x, dtype=np.float64)
    for row in range(len(x)):
        indices = np.flatnonzero(visible[row])
        if len(indices) == 0:
            continue
        order = indices[np.argsort(x[row, indices], kind="stable")]
        if len(order) == 1:
            rank[row, order] = 1.0
        else:
            rank[row, order] = np.arange(len(order), dtype=np.float64) / float(
                len(order) - 1
            )
    return np.concatenate((visible.astype(np.float64), rank), axis=1)


def features(values: np.ndarray, method: str) -> np.ndarray:
    """Return the fixed V9 comparator representation for one named baseline."""

    if method == "raw_wknn":
        return np.asarray(values, dtype=np.float64)
    if method == "diff_wknn":
        return visible_centered_feature(values)
    if method == "rank_wknn":
        return rank_feature(values)
    raise ValueError(f"unknown baseline: {method}")


def coordinate_prototypes(
    rss: np.ndarray,
    raw_xy: np.ndarray,
    phones: np.ndarray,
    keys: list[tuple],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """One equal-device fingerprint and float64 coordinate per physical group."""

    grouped: dict[tuple, list[int]] = {}
    for row in np.asarray(indices, dtype=np.int64):
        grouped.setdefault(keys[int(row)], []).append(int(row))
    if not grouped:
        raise ValueError("baseline reference role contains no coordinate")
    rss64 = np.asarray(rss, dtype=np.float64)
    xy64 = np.asarray(raw_xy, dtype=np.float64)
    phone_array = np.asarray(phones, dtype=np.int64)
    prototypes = []
    targets = []
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        rows = np.asarray(grouped[key], dtype=np.int64)
        phone_summaries = []
        for phone in sorted(map(int, np.unique(phone_array[rows]))):
            phone_rows = rows[phone_array[rows] == phone]
            phone_summaries.append(np.median(rss64[phone_rows], axis=0))
        prototypes.append(np.mean(np.stack(phone_summaries), axis=0))
        targets.append(np.mean(xy64[rows], axis=0, dtype=np.float64))
    return np.stack(prototypes), np.stack(targets), ordered_keys


def predict_k_grid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    k_grid: tuple[int, ...] = K_GRID,
    chunk_size: int = 256,
) -> dict[int, np.ndarray]:
    """Predict every fixed ``k`` with stable global inverse-distance WKNN."""

    if max(k_grid) > len(train_x):
        raise ValueError("baseline k exceeds the prototype count")
    outputs = {
        int(k): np.empty((len(test_x), 2), dtype=np.float64) for k in k_grid
    }
    train = np.asarray(train_x, dtype=np.float64)
    train_norm = np.sum(train * train, axis=1)
    for start in range(0, len(test_x), int(chunk_size)):
        stop = min(start + int(chunk_size), len(test_x))
        block = np.asarray(test_x[start:stop], dtype=np.float64)
        distance2 = (
            np.sum(block * block, axis=1)[:, None]
            + train_norm[None, :]
            - 2.0 * block @ train.T
        )
        np.maximum(distance2, 0.0, out=distance2)
        neighbors = np.argsort(distance2, axis=1, kind="stable")[:, : max(k_grid)]
        local_distance = np.take_along_axis(distance2, neighbors, axis=1)
        weights = 1.0 / np.maximum(np.sqrt(local_distance), 1e-8)
        neighbor_targets = train_y[neighbors]
        weighted_sum = np.cumsum(neighbor_targets * weights[:, :, None], axis=1)
        weight_sum = np.cumsum(weights, axis=1)
        for k in k_grid:
            outputs[int(k)][start:stop] = weighted_sum[:, int(k) - 1] / weight_sum[
                :, int(k) - 1, None
            ]
    return outputs


def group_macro_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    keys: list[tuple],
    indices: np.ndarray,
) -> tuple[float, dict[tuple, float], np.ndarray]:
    """Compute row errors and the exact-coordinate macro mean."""

    errors = np.linalg.norm(predictions - targets, axis=1)
    grouped: dict[tuple, list[float]] = {}
    for row, error in zip(np.asarray(indices, dtype=np.int64), errors):
        grouped.setdefault(keys[int(row)], []).append(float(error))
    per_group = {key: float(np.mean(values)) for key, values in grouped.items()}
    return float(np.mean(list(per_group.values()))), per_group, errors

