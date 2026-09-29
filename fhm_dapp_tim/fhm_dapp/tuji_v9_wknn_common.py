"""Deterministic, device-balanced WKNN primitives for the TUJI1 comparator.

This module deliberately remains separate from :mod:`v9_wknn_common`, whose
``-0.75`` visibility threshold is part of the frozen UJI protocol.  TUJI1 uses
the V9 missing-event convention: a detected AP is normalized RSS greater than
``-0.999`` and the missing sentinel is exactly ``-1``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


METHODS = ("raw_wknn", "diff_wknn", "rank_wknn")
K_GRID = (1, 3, 5, 8, 12, 20, 32)
VISIBILITY_THRESHOLD = -0.999


def visible_centered_feature(values: np.ndarray) -> np.ndarray:
    """Return visibility plus visible-mean-centred normalized RSS."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows, aps]")
    visible = x > VISIBILITY_THRESHOLD
    count = visible.sum(axis=1, keepdims=True)
    center = np.divide(
        np.where(visible, x, 0.0).sum(axis=1, keepdims=True),
        np.maximum(count, 1),
    )
    centered = np.where(visible, x - center, 0.0)
    return np.concatenate((visible.astype(np.float64), centered), axis=1)


def rank_feature(values: np.ndarray) -> np.ndarray:
    """Return visibility plus stable normalized ranks among visible RSS."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("RSS features must have shape [rows, aps]")
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
    """Return one fixed TUJI1 comparator representation."""

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
    keys: Sequence[tuple],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Make one equal-device RSS prototype per exact physical coordinate."""

    grouped: dict[tuple, list[int]] = {}
    for row in np.asarray(indices, dtype=np.int64):
        grouped.setdefault(tuple(keys[int(row)]), []).append(int(row))
    if not grouped:
        raise ValueError("baseline reference role contains no coordinate")
    rss64 = np.asarray(rss, dtype=np.float64)
    xy64 = np.asarray(raw_xy, dtype=np.float64)
    phone_array = np.asarray(phones, dtype=np.int64)
    if rss64.ndim != 2 or xy64.ndim != 2 or xy64.shape[1] != 2:
        raise ValueError("TUJI prototypes require [rows, aps] RSS and [rows, 2] coordinates")
    prototypes: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ordered_keys = sorted(grouped)
    for key in ordered_keys:
        rows = np.asarray(grouped[key], dtype=np.int64)
        device_medians = []
        for phone in sorted(map(int, np.unique(phone_array[rows]))):
            phone_rows = rows[phone_array[rows] == phone]
            device_medians.append(np.median(rss64[phone_rows], axis=0))
        prototypes.append(np.mean(np.stack(device_medians), axis=0))
        coordinate_rows = xy64[rows]
        target = np.mean(coordinate_rows, axis=0, dtype=np.float64)
        if not np.allclose(coordinate_rows, target[None, :], rtol=0.0, atol=1e-6):
            raise ValueError("one physical-coordinate group has inconsistent targets")
        targets.append(target)
    return np.stack(prototypes), np.stack(targets), ordered_keys


def predict_k_grid(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    k_grid: Sequence[int] = K_GRID,
    chunk_size: int = 256,
) -> dict[int, np.ndarray]:
    """Predict every fixed k using global inverse-distance WKNN."""

    fixed_grid = tuple(map(int, k_grid))
    if not fixed_grid or min(fixed_grid) < 1:
        raise ValueError("k grid must contain positive integers")
    if max(fixed_grid) > len(train_x):
        raise ValueError("baseline k exceeds the prototype count")
    train = np.asarray(train_x, dtype=np.float64)
    targets = np.asarray(train_y, dtype=np.float64)
    queries = np.asarray(test_x, dtype=np.float64)
    if train.ndim != 2 or targets.shape != (len(train), 2) or queries.ndim != 2:
        raise ValueError("invalid WKNN feature or target shape")
    outputs = {k: np.empty((len(queries), 2), dtype=np.float64) for k in fixed_grid}
    train_norm = np.sum(train * train, axis=1)
    maximum = max(fixed_grid)
    for start in range(0, len(queries), int(chunk_size)):
        stop = min(start + int(chunk_size), len(queries))
        block = queries[start:stop]
        distance2 = (
            np.sum(block * block, axis=1)[:, None]
            + train_norm[None, :]
            - 2.0 * block @ train.T
        )
        np.maximum(distance2, 0.0, out=distance2)
        neighbors = np.argsort(distance2, axis=1, kind="stable")[:, :maximum]
        local_distance = np.take_along_axis(distance2, neighbors, axis=1)
        weights = 1.0 / np.maximum(np.sqrt(local_distance), 1e-8)
        neighbor_targets = targets[neighbors]
        weighted_sum = np.cumsum(neighbor_targets * weights[:, :, None], axis=1)
        weight_sum = np.cumsum(weights, axis=1)
        for k in fixed_grid:
            outputs[k][start:stop] = weighted_sum[:, k - 1] / weight_sum[:, k - 1, None]
    return outputs


def group_macro_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    keys: Sequence[tuple],
    indices: np.ndarray,
) -> tuple[float, dict[tuple, float], np.ndarray]:
    """Return row errors and exact-coordinate macro errors in sorted-key form."""

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("predictions and targets must share shape [rows, 2]")
    errors = np.linalg.norm(prediction - target, axis=1)
    grouped: dict[tuple, list[float]] = {}
    for row, error in zip(np.asarray(indices, dtype=np.int64), errors):
        grouped.setdefault(tuple(keys[int(row)]), []).append(float(error))
    per_group = {key: float(np.mean(grouped[key])) for key in sorted(grouped)}
    return float(np.mean(list(per_group.values()))), per_group, errors
