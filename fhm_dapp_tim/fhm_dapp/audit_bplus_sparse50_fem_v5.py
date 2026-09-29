"""Same-budget 50% sparse-front audit for the B+ spatial principle.

This development-only experiment geometrically retains half of the registered
development-core coordinates, trains one raw M0 front exclusively on those
coordinates, and evaluates only scans at the unmeasured half.  A nested
fit/early-stop/calibration split inside the retained survey selects the front
epoch and generalized-update temperature before the unmeasured role is used.

Candidate quadrature is generated solely from the measured coordinates.  A
Delaunay finite-element mesh supplies piecewise-linear basis functions and
area weights.  The audit compares the current IDW continuation with a unified
FEM continuation of the front log density, AP occurrence logits, and
conditional RSS centers.  E=1; no scan random effect, DANN branch, ensemble,
learned gate, or Top-K projection is present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial import ConvexHull, Delaunay, QhullError
import torch
import torch.nn.functional as F

from fhm_dapp_tim.fhm_dapp.anchor import location_keys, raw_coords, resolve_path
from fhm_dapp_tim.fhm_dapp.audit_bplus_localization_bridge_v5 import (
    _ap_fallbacks,
    _append_rows,
    _distribution_rows,
    _finish_rows,
    _paired_coordinate_delta,
    _parameter_label,
    candidate_transport_matrix,
    marked_hurdle_loss,
    summarize_rows,
)
from fhm_dapp_tim.fhm_dapp.audit_sparse_radiomap_v5 import (
    LocationKey,
    _global_detection_background,
    _natural_parameter_predictions,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.protocol_v4 import (
    coordinate_axis_partition,
    validate_registry,
)
from fhm_dapp_tim.fhm_dapp.train_anchor_v4 import (
    _configure_cuda_reproducibility,
    _fit_fixed,
    _fit_schedule,
)


def _csv_tuple(text: str, cast) -> tuple:
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _stable_rank(seed: int, salt: str, key: LocationKey) -> bytes:
    return hashlib.sha256(
        f"{int(seed)}|{salt}|{repr(key)}".encode("utf-8")
    ).digest()


def _hull_local_indices(xy: np.ndarray) -> np.ndarray:
    count = len(xy)
    if count <= 2:
        return np.arange(count, dtype=np.int64)
    centered = xy - xy.mean(axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered, tol=1e-10) < 2:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        projection = centered @ vh[0]
        return np.asarray([int(np.argmin(projection)), int(np.argmax(projection))])
    try:
        return np.asarray(ConvexHull(xy).vertices, dtype=np.int64)
    except QhullError:
        distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        first, second = np.unravel_index(int(np.argmax(distance)), distance.shape)
        return np.asarray(sorted({int(first), int(second)}), dtype=np.int64)


def geometry_support_with_hull(
    keys: set[LocationKey],
    fraction: float,
    *,
    seed: int,
) -> tuple[set[LocationKey], dict]:
    """Space-filling survey that keeps every stratum's geometric hull."""

    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must lie in (0,1)")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in keys:
        strata.setdefault(key[:2], []).append(key)
    selected: set[LocationKey] = set()
    details = {}
    for stratum, values in sorted(strata.items()):
        ordered = sorted(values)
        xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
        target = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
        hull = set(map(int, _hull_local_indices(xy)))
        target = min(len(ordered) - 1, max(target, len(hull)))
        local_selected = set(hull)
        if not local_selected:
            local_selected.add(
                min(
                    range(len(ordered)),
                    key=lambda i: _stable_rank(seed, "survey-start", ordered[i]),
                )
            )
        minimum_distance = np.full(len(ordered), np.inf, dtype=np.float64)
        for index in local_selected:
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(xy - xy[index], axis=1),
            )
        while len(local_selected) < target:
            available = [i for i in range(len(ordered)) if i not in local_selected]
            maximum = max(float(minimum_distance[i]) for i in available)
            tied = [
                i
                for i in available
                if math.isclose(float(minimum_distance[i]), maximum, abs_tol=1e-12)
            ]
            chosen = min(
                tied,
                key=lambda i: _stable_rank(
                    seed + 1009 * stratum[0] + 9176 * stratum[1],
                    f"survey-tie-{len(local_selected)}",
                    ordered[i],
                ),
            )
            local_selected.add(chosen)
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(xy - xy[chosen], axis=1),
            )
        chosen_keys = {ordered[i] for i in local_selected}
        selected.update(chosen_keys)
        details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "available_coordinates": int(len(ordered)),
            "retained_coordinates": int(len(chosen_keys)),
            "mandatory_hull_coordinates": int(len(hull)),
        }
    if not selected or selected == keys:
        raise RuntimeError("survey split did not leave measured and unmeasured roles")
    return selected, details


def internal_support_roles(
    support_keys: set[LocationKey],
    *,
    seed: int,
    early_stop_fraction: float,
    calibration_fraction: float,
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Create geometry-only nested roles while keeping hull nodes in fit."""

    if early_stop_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("nested validation fractions must be positive")
    if early_stop_fraction + calibration_fraction >= 0.5:
        raise ValueError("nested validation fractions leave too little fitting support")
    strata: dict[tuple[int, int], list[LocationKey]] = {}
    for key in support_keys:
        strata.setdefault(key[:2], []).append(key)
    roles = {"fit": set(), "early_stop": set(), "calibration": set()}
    details = {}
    for stratum, values in sorted(strata.items()):
        ordered = sorted(values)
        xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
        hull = {ordered[i] for i in _hull_local_indices(xy)}
        interior = [key for key in ordered if key not in hull]
        ranked = sorted(
            interior,
            key=lambda key: _stable_rank(
                seed + 7919 * stratum[0] + 104729 * stratum[1],
                "nested-role",
                key,
            ),
        )
        maximum_holdout = max(0, len(ranked) - 1)
        stop_count = min(
            maximum_holdout,
            max(1, int(round(len(ordered) * early_stop_fraction)))
            if len(ranked) >= 2
            else 0,
        )
        remaining = maximum_holdout - stop_count
        calibration_count = min(
            remaining,
            max(1, int(round(len(ordered) * calibration_fraction)))
            if remaining >= 1
            else 0,
        )
        stop = set(ranked[:stop_count])
        calibration = set(ranked[stop_count : stop_count + calibration_count])
        fit = set(ordered) - stop - calibration
        if not hull.issubset(fit):
            raise RuntimeError("nested split removed a mandatory hull coordinate")
        roles["fit"].update(fit)
        roles["early_stop"].update(stop)
        roles["calibration"].update(calibration)
        details[f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "fit": int(len(fit)),
            "early_stop": int(len(stop)),
            "calibration": int(len(calibration)),
            "hull_in_fit": int(len(hull)),
        }
    if any(len(roles[name]) == 0 for name in roles):
        raise RuntimeError("nested survey split produced an empty role")
    if roles["fit"] & roles["early_stop"] or roles["fit"] & roles["calibration"]:
        raise RuntimeError("nested survey roles overlap")
    if roles["early_stop"] & roles["calibration"]:
        raise RuntimeError("nested validation roles overlap")
    if set.union(*roles.values()) != support_keys:
        raise RuntimeError("nested survey roles do not partition the support")
    return roles, details


@dataclass(frozen=True)
class FEMQuadrature:
    support_keys: tuple[LocationKey, ...]
    candidate_keys: tuple[LocationKey, ...]
    candidate_xy: np.ndarray
    candidate_building: np.ndarray
    candidate_floor: np.ndarray
    quadrature_weight: np.ndarray
    barycentric: np.ndarray
    nodal_mass: np.ndarray
    stiffness_blocks: tuple[tuple[np.ndarray, np.ndarray], ...]
    diagnostics: dict

    def validate(self) -> None:
        nodes = len(self.support_keys)
        candidates = len(self.candidate_keys)
        if self.candidate_xy.shape != (candidates, 2):
            raise ValueError("candidate_xy has the wrong shape")
        if self.barycentric.shape != (candidates, nodes):
            raise ValueError("barycentric matrix has the wrong shape")
        if self.nodal_mass.shape != (nodes,):
            raise ValueError("nodal_mass has the wrong shape")
        if self.quadrature_weight.shape != (candidates,):
            raise ValueError("quadrature_weight has the wrong shape")
        if not np.isfinite(self.barycentric).all() or not np.isfinite(
            self.quadrature_weight
        ).all():
            raise ValueError("FEM quadrature contains non-finite values")
        if np.any(self.quadrature_weight <= 0.0) or np.any(self.nodal_mass <= 0.0):
            raise ValueError("FEM masses must be positive")
        if not np.allclose(self.barycentric.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("FEM basis rows do not form a partition of unity")


def _triangle_stiffness(xy: np.ndarray) -> tuple[float, np.ndarray]:
    first_edge = xy[1] - xy[0]
    second_edge = xy[2] - xy[0]
    signed_twice_area = float(
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    )
    area = 0.5 * abs(signed_twice_area)
    if area <= 1e-10:
        raise ValueError("degenerate FEM triangle")
    denominator = signed_twice_area
    gradient_x = np.asarray(
        [xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]],
        dtype=np.float64,
    ) / denominator
    gradient_y = np.asarray(
        [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]],
        dtype=np.float64,
    ) / denominator
    stiffness = area * (
        np.outer(gradient_x, gradient_x) + np.outer(gradient_y, gradient_y)
    )
    return area, stiffness


def build_fem_quadrature(support_keys: list[LocationKey]) -> FEMQuadrature:
    """Build three-point triangle quadrature and first-order FEM bases."""

    if len(set(support_keys)) != len(support_keys):
        raise ValueError("support keys must be unique")
    nodes = len(support_keys)
    barycentric_rows = []
    candidate_xy = []
    candidate_building = []
    candidate_floor = []
    candidate_keys = []
    quadrature_weight = []
    nodal_mass = np.zeros(nodes, dtype=np.float64)
    blocks = []
    diagnostics = {"strata": {}, "quadrature_rule": "three-point triangle Gauss"}
    triangle_rule = np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )
    strata = sorted({key[:2] for key in support_keys})
    for stratum in strata:
        global_indices = np.asarray(
            [i for i, key in enumerate(support_keys) if key[:2] == stratum],
            dtype=np.int64,
        )
        xy = np.asarray(
            [[support_keys[i][2], support_keys[i][3]] for i in global_indices],
            dtype=np.float64,
        )
        centered = xy - xy.mean(axis=0, keepdims=True)
        rank = np.linalg.matrix_rank(centered, tol=1e-10)
        local_stiffness = np.zeros((len(global_indices), len(global_indices)))
        stratum_measure = 0.0
        element_count = 0
        if len(global_indices) >= 3 and rank >= 2:
            triangulation = Delaunay(xy)
            for simplex in triangulation.simplices:
                simplex = np.asarray(simplex, dtype=np.int64)
                triangle_xy = xy[simplex]
                try:
                    area, stiffness = _triangle_stiffness(triangle_xy)
                except ValueError:
                    continue
                element_count += 1
                stratum_measure += area
                local_stiffness[np.ix_(simplex, simplex)] += stiffness
                nodal_mass[global_indices[simplex]] += area / 3.0
                for bary in triangle_rule:
                    point = bary @ triangle_xy
                    row = np.zeros(nodes, dtype=np.float64)
                    row[global_indices[simplex]] = bary
                    barycentric_rows.append(row)
                    candidate_xy.append(point)
                    candidate_building.append(stratum[0])
                    candidate_floor.append(stratum[1])
                    candidate_keys.append(
                        (
                            int(stratum[0]),
                            int(stratum[1]),
                            round(float(point[0]), 8),
                            round(float(point[1]), 8),
                        )
                    )
                    quadrature_weight.append(area / 3.0)
            geometry = "two_dimensional_triangles"
        else:
            if len(global_indices) < 2:
                raise ValueError(f"stratum {stratum} has fewer than two support nodes")
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            projection = centered @ vh[0]
            order = np.argsort(projection, kind="stable")
            line_rule = (
                0.5 * (1.0 - 1.0 / math.sqrt(3.0)),
                0.5 * (1.0 + 1.0 / math.sqrt(3.0)),
            )
            for left, right in zip(order[:-1], order[1:]):
                length = float(np.linalg.norm(xy[right] - xy[left]))
                if length <= 1e-10:
                    continue
                element_count += 1
                stratum_measure += length
                local = np.asarray([left, right], dtype=np.int64)
                local_stiffness[np.ix_(local, local)] += (
                    np.asarray([[1.0, -1.0], [-1.0, 1.0]]) / length
                )
                nodal_mass[global_indices[local]] += length / 2.0
                for t in line_rule:
                    point = (1.0 - t) * xy[left] + t * xy[right]
                    row = np.zeros(nodes, dtype=np.float64)
                    row[global_indices[left]] = 1.0 - t
                    row[global_indices[right]] = t
                    barycentric_rows.append(row)
                    candidate_xy.append(point)
                    candidate_building.append(stratum[0])
                    candidate_floor.append(stratum[1])
                    candidate_keys.append(
                        (
                            int(stratum[0]),
                            int(stratum[1]),
                            round(float(point[0]), 8),
                            round(float(point[1]), 8),
                        )
                    )
                    quadrature_weight.append(length / 2.0)
            geometry = "one_dimensional_segments_unit_width"
        if element_count == 0:
            raise RuntimeError(f"stratum {stratum} produced no FEM element")
        blocks.append((global_indices, local_stiffness))
        diagnostics["strata"][f"building_{stratum[0]}_floor_{stratum[1]}"] = {
            "nodes": int(len(global_indices)),
            "elements": int(element_count),
            "domain_measure": float(stratum_measure),
            "geometry": geometry,
        }
    mesh = FEMQuadrature(
        support_keys=tuple(support_keys),
        candidate_keys=tuple(candidate_keys),
        candidate_xy=np.asarray(candidate_xy, dtype=np.float64),
        candidate_building=np.asarray(candidate_building, dtype=np.int64),
        candidate_floor=np.asarray(candidate_floor, dtype=np.int64),
        quadrature_weight=np.asarray(quadrature_weight, dtype=np.float64),
        barycentric=np.asarray(barycentric_rows, dtype=np.float64),
        nodal_mass=nodal_mass,
        stiffness_blocks=tuple(blocks),
        diagnostics={
            **diagnostics,
            "support_nodes": int(nodes),
            "quadrature_candidates": int(len(candidate_keys)),
            "total_domain_measure": float(np.sum(quadrature_weight)),
        },
    )
    mesh.validate()
    return mesh


def harmonic_complete_rss_nodes(
    center: np.ndarray,
    available: np.ndarray,
    fallback_center: np.ndarray,
    mesh: FEMQuadrature,
) -> tuple[np.ndarray, dict]:
    """Dirichlet-energy completion of missing conditional-RSS node values."""

    completed = np.asarray(center, dtype=np.float64).copy()
    available = np.asarray(available, dtype=bool)
    solved_cells = 0
    fallback_cells = 0
    for global_indices, stiffness in mesh.stiffness_blocks:
        for ap in range(completed.shape[1]):
            observed = available[global_indices, ap]
            missing_local = np.flatnonzero(~observed)
            if len(missing_local) == 0:
                continue
            observed_local = np.flatnonzero(observed)
            if len(observed_local) == 0:
                completed[global_indices[missing_local], ap] = fallback_center[ap]
                fallback_cells += len(missing_local)
                continue
            lhs = stiffness[np.ix_(missing_local, missing_local)].copy()
            rhs = -stiffness[np.ix_(missing_local, observed_local)] @ completed[
                global_indices[observed_local], ap
            ]
            ridge = max(1e-10, 1e-9 * float(np.mean(np.diag(lhs))))
            lhs.flat[:: len(lhs) + 1] += ridge
            try:
                solution = linalg.solve(
                    lhs, rhs, assume_a="sym", check_finite=False
                )
            except linalg.LinAlgError:
                solution = np.full(len(missing_local), fallback_center[ap])
                fallback_cells += len(missing_local)
            else:
                solved_cells += len(missing_local)
            completed[global_indices[missing_local], ap] = np.clip(
                solution, -1.0, 1.0
            )
    if not np.isfinite(completed).all():
        raise RuntimeError("harmonic RSS completion produced non-finite values")
    return completed, {
        "harmonically_completed_node_ap_cells": int(solved_cells),
        "global_fallback_node_ap_cells": int(fallback_cells),
        "completion_rule": "FEM Dirichlet-energy minimizer with observed nodes fixed",
    }


def build_spatial_fields(
    *,
    rss: torch.Tensor,
    phone_ids: np.ndarray,
    coordinate_keys: list[LocationKey],
    training_indices: np.ndarray,
    support_keys: list[LocationKey],
    radio_memory: dict,
    mesh: FEMQuadrature,
    neighbors: int,
    inverse_distance_power: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    raw_probability = radio_memory["detection_prob"].to(torch.float64)
    device_count = radio_memory["device_counts"].to(torch.float64)
    background = _global_detection_background(
        rss, phone_ids, coordinate_keys, training_indices
    ).to(torch.float64)
    probability = (
        raw_probability * device_count[:, None] + background[None, :]
    ) / (device_count[:, None] + 1.0)
    probability_np = probability.clamp(1e-5, 1.0 - 1e-5).numpy()
    center = radio_memory["rss_center"].to(torch.float64).numpy()
    scale = radio_memory["rss_scale"].to(torch.float64).numpy()
    available = raw_probability.numpy() > 0.0
    fallback_center, fallback_scale = _ap_fallbacks(center, scale, available)
    anchor_center = center.copy()
    anchor_missing = ~available
    anchor_center[anchor_missing] = np.broadcast_to(
        fallback_center[None, :], anchor_center.shape
    )[anchor_missing]
    prediction = _natural_parameter_predictions(
        list(mesh.candidate_keys),
        support_keys,
        probability_np,
        center,
        available,
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
    )
    fields = {
        "anchor": {
            "detection_probability": torch.from_numpy(
                probability_np.astype(np.float32)
            ),
            "rss_center": torch.from_numpy(anchor_center.astype(np.float32)),
            "rss_scale": torch.from_numpy(fallback_scale.astype(np.float32)),
        }
    }
    fallback_diagnostics = {"anchor": float(anchor_missing.mean())}
    for source, name in (("nearest", "nearest"), ("smooth", "idw")):
        candidate_center = prediction[f"{source}_center"].copy()
        missing = ~np.isfinite(candidate_center)
        candidate_center[missing] = np.broadcast_to(
            fallback_center[None, :], candidate_center.shape
        )[missing]
        fallback_diagnostics[name] = float(missing.mean())
        fields[name] = {
            "detection_probability": torch.from_numpy(
                prediction[f"{source}_probability"].astype(np.float32)
            ),
            "rss_center": torch.from_numpy(candidate_center.astype(np.float32)),
            "rss_scale": torch.from_numpy(fallback_scale.astype(np.float32)),
        }
    completed_center, harmonic_diagnostics = harmonic_complete_rss_nodes(
        center, available, fallback_center, mesh
    )
    eta = np.log(probability_np) - np.log1p(-probability_np)
    fem_probability = 1.0 / (1.0 + np.exp(-(mesh.barycentric @ eta)))
    fem_center = mesh.barycentric @ completed_center
    fields["fem"] = {
        "detection_probability": torch.from_numpy(fem_probability.astype(np.float32)),
        "rss_center": torch.from_numpy(fem_center.astype(np.float32)),
        "rss_scale": torch.from_numpy(fallback_scale.astype(np.float32)),
    }
    return fields, {
        "conditional_center_global_fallback_fraction": fallback_diagnostics,
        "fem_harmonic_completion": harmonic_diagnostics,
        "scale_estimator": "AP-wise median support scale, spatially fixed",
    }


@torch.no_grad()
def evaluate_spatial_panel(
    *,
    model,
    radio_memory: dict,
    coord_origin: np.ndarray,
    training_indices: np.ndarray,
    evaluation_rows: np.ndarray,
    rss: torch.Tensor,
    raw_xy: np.ndarray,
    building: np.ndarray,
    floor: np.ndarray,
    phone_ids: np.ndarray,
    coordinate_keys: list[LocationKey],
    lambda_options: dict[str, tuple[float, ...]],
    neighbors: int,
    inverse_distance_power: float,
    degrees_freedom: float,
    batch_size: int,
    candidate_chunk_size: int,
    credible_levels: tuple[float, ...],
) -> dict:
    device = next(model.parameters()).device
    support_keys = list(radio_memory["coordinate_keys"])
    mesh = build_fem_quadrature(support_keys)
    fields_cpu, field_diagnostics = build_spatial_fields(
        rss=rss,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        training_indices=training_indices,
        support_keys=support_keys,
        radio_memory=radio_memory,
        mesh=mesh,
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
    )
    fields = {
        name: {key: value.to(device) for key, value in memory.items()}
        for name, memory in fields_cpu.items()
    }
    idw_transport = torch.from_numpy(
        candidate_transport_matrix(
            support_keys,
            list(mesh.candidate_keys),
            neighbors=neighbors,
            inverse_distance_power=inverse_distance_power,
        )
    ).to(device)
    fem_basis = torch.from_numpy(mesh.barycentric.astype(np.float32)).to(device)
    nodal_mass = torch.from_numpy(mesh.nodal_mass.astype(np.float32)).to(device)
    quadrature_weight = torch.from_numpy(
        mesh.quadrature_weight.astype(np.float32)
    ).to(device)
    log_quadrature_weight = torch.log(quadrature_weight).unsqueeze(0)
    origin = torch.as_tensor(coord_origin, device=device, dtype=torch.float32)
    candidate_xy = torch.from_numpy(mesh.candidate_xy.astype(np.float32)).to(device)
    candidate_xy = candidate_xy - origin.unsqueeze(0)
    candidate_building = torch.from_numpy(mesh.candidate_building).to(device)
    candidate_floor = torch.from_numpy(mesh.candidate_floor).to(device)
    height = 4.0
    candidate_support_3d = torch.cat(
        (
            candidate_xy,
            height * candidate_floor.to(candidate_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    candidate_pair = torch.cdist(candidate_support_3d, candidate_support_3d)
    anchor_xy = model.anchor_coords.to(device)
    anchor_building = model.anchor_buildings.to(device)
    anchor_floor = model.anchor_floors.to(device)
    anchor_support_3d = torch.cat(
        (
            anchor_xy,
            height * anchor_floor.to(anchor_xy.dtype).unsqueeze(1),
        ),
        dim=1,
    )
    anchor_pair = torch.cdist(anchor_support_3d, anchor_support_3d)
    configurations = {
        "front_anchor": {"variant": "front_anchor"},
        "front_idw": {"variant": "front_idw"},
        "front_fem": {"variant": "front_fem"},
    }
    for variant, values in lambda_options.items():
        if variant not in ("anchor_full", "idw_full", "fem_nearest", "fem_full"):
            raise ValueError(f"unknown update variant: {variant}")
        for value in values:
            configurations[f"{variant}__lambda_{_parameter_label(value)}"] = {
                "variant": variant,
                "lambda": float(value),
            }
    output_parts = {name: {} for name in configurations}
    rss_device = rss.to(device)
    for start in range(0, len(evaluation_rows), int(batch_size)):
        rows = evaluation_rows[start : start + int(batch_size)]
        query = rss_device[rows]
        logits = model.retrieval_logits(query)[0]
        anchor_probability = torch.softmax(logits, dim=1)
        node_log_density = F.log_softmax(logits, dim=1) - torch.log(
            nodal_mass
        ).unsqueeze(0)
        idw_log_prior = F.log_softmax(
            node_log_density @ idw_transport.t() + log_quadrature_weight, dim=1
        )
        fem_log_prior = F.log_softmax(
            node_log_density @ fem_basis.t() + log_quadrature_weight, dim=1
        )
        target_xy = (
            torch.as_tensor(raw_xy[rows], device=device, dtype=torch.float32)
            - origin.unsqueeze(0)
        )
        target_building = torch.as_tensor(building[rows], device=device, dtype=torch.long)
        target_floor = torch.as_tensor(floor[rows], device=device, dtype=torch.long)
        for name, probability, support_xy, support_building, support_floor, pair in (
            (
                "front_anchor",
                anchor_probability,
                anchor_xy,
                anchor_building,
                anchor_floor,
                anchor_pair,
            ),
            (
                "front_idw",
                torch.exp(idw_log_prior),
                candidate_xy,
                candidate_building,
                candidate_floor,
                candidate_pair,
            ),
            (
                "front_fem",
                torch.exp(fem_log_prior),
                candidate_xy,
                candidate_building,
                candidate_floor,
                candidate_pair,
            ),
        ):
            _append_rows(
                output_parts[name],
                _distribution_rows(
                    probability,
                    support_xy,
                    support_building,
                    support_floor,
                    target_xy,
                    target_building,
                    target_floor,
                    pair_distance_3d=pair,
                    floor_height_m=height,
                    true_candidate=None,
                    credible_levels=credible_levels,
                ),
            )
        losses = {}
        for field_name in ("anchor", "nearest", "idw", "fem"):
            memory = fields[field_name]
            losses[field_name] = marked_hurdle_loss(
                query,
                memory["detection_probability"],
                memory["rss_center"],
                memory["rss_scale"],
                degrees_freedom=degrees_freedom,
                scan_effect_tau=0.0,
                quadrature_order=9,
                candidate_chunk_size=candidate_chunk_size,
            )
        for name, specification in configurations.items():
            variant = specification["variant"]
            if variant.startswith("front_"):
                continue
            if variant == "anchor_full":
                log_prior = F.log_softmax(logits, dim=1)
                loss = losses["anchor"]
                support_xy = anchor_xy
                support_building = anchor_building
                support_floor = anchor_floor
                pair_distance = anchor_pair
            elif variant == "idw_full":
                log_prior = idw_log_prior
                loss = losses["idw"]
                support_xy = candidate_xy
                support_building = candidate_building
                support_floor = candidate_floor
                pair_distance = candidate_pair
            elif variant == "fem_nearest":
                log_prior = fem_log_prior
                loss = losses["nearest"]
                support_xy = candidate_xy
                support_building = candidate_building
                support_floor = candidate_floor
                pair_distance = candidate_pair
            else:
                log_prior = fem_log_prior
                loss = losses["fem"]
                support_xy = candidate_xy
                support_building = candidate_building
                support_floor = candidate_floor
                pair_distance = candidate_pair
            probability = torch.softmax(
                log_prior - loss / float(specification["lambda"]), dim=1
            )
            _append_rows(
                output_parts[name],
                _distribution_rows(
                    probability,
                    support_xy,
                    support_building,
                    support_floor,
                    target_xy,
                    target_building,
                    target_floor,
                    pair_distance_3d=pair_distance,
                    floor_height_m=height,
                    true_candidate=None,
                    credible_levels=credible_levels,
                ),
            )
        processed = min(start + int(batch_size), len(evaluation_rows))
        if start == 0 or processed == len(evaluation_rows) or (
            (start // int(batch_size)) + 1
        ) % 10 == 0:
            print(
                json.dumps(
                    {
                        "sparse50_evaluation_rows": int(processed),
                        "total_rows": int(len(evaluation_rows)),
                        "mesh_candidates": int(len(mesh.candidate_keys)),
                    }
                ),
                flush=True,
            )
    row_outputs = {name: _finish_rows(parts) for name, parts in output_parts.items()}
    summaries = {
        name: summarize_rows(values, evaluation_rows, coordinate_keys, phone_ids)
        for name, values in row_outputs.items()
    }
    return {
        "evaluation_rows": evaluation_rows,
        "row_outputs": row_outputs,
        "summaries": summaries,
        "configurations": configurations,
        "mesh_diagnostics": mesh.diagnostics,
        "field_diagnostics": field_diagnostics,
    }


def select_lambdas(panel: dict, lambda_grid: tuple[float, ...]) -> dict:
    selected = {}
    for variant in ("anchor_full", "idw_full", "fem_nearest", "fem_full"):
        scores = {}
        for value in lambda_grid:
            name = f"{variant}__lambda_{_parameter_label(value)}"
            scores[name] = panel["summaries"][name]["energy_score_3d_m"][
                "coordinate_macro"
            ]["mean"]
        chosen = min(
            scores,
            key=lambda name: (
                scores[name],
                -float(panel["configurations"][name]["lambda"]),
            ),
        )
        selected[variant] = {
            "configuration": chosen,
            "lambda": float(panel["configurations"][chosen]["lambda"]),
            "calibration_energy_score_3d_m": float(scores[chosen]),
            "all_scores": {name: float(score) for name, score in scores.items()},
        }
    return selected


def _rows_for_keys(
    keys: list[LocationKey],
    phone_ids: np.ndarray,
    allowed_phones: tuple[int, ...],
    selected_keys: set[LocationKey],
) -> np.ndarray:
    allowed = set(map(int, allowed_phones))
    return np.asarray(
        [
            row
            for row, phone in enumerate(phone_ids)
            if int(phone) in allowed and keys[row] in selected_keys
        ],
        dtype=np.int64,
    )


def _serializable_panel(panel: dict, names: dict[str, str]) -> dict:
    return {
        "evaluation_rows": int(len(panel["evaluation_rows"])),
        "mesh_diagnostics": panel["mesh_diagnostics"],
        "field_diagnostics": panel["field_diagnostics"],
        "summaries": {
            label: panel["summaries"][configuration]
            for label, configuration in names.items()
        },
    }


def run_sparse50(
    *,
    train_data: str | Path,
    registry_path: str | Path,
    geometry_seed: int,
    front_seed: int,
    retained_fraction: float,
    early_stop_fraction: float,
    calibration_fraction: float,
    lambda_grid: tuple[float, ...],
    neighbors: int,
    inverse_distance_power: float,
    degrees_freedom: float,
    batch_size: int,
    candidate_chunk_size: int,
    credible_levels: tuple[float, ...],
    device_name: str,
    threads: int,
    log_every: int,
    checkpoint_output: str | Path | None,
) -> dict:
    torch.set_num_threads(max(1, min(int(threads), torch.get_num_threads())))
    if torch.cuda.is_available():
        _configure_cuda_reproducibility()
        torch.set_float32_matmul_precision("highest")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    data_path = resolve_path(str(train_data)).resolve()
    registry_file = resolve_path(str(registry_path)).resolve()
    registry = validate_registry(registry_file, train_data=data_path)
    (
        rss,
        coords_norm,
        building_t,
        floor_t,
        _space_t,
        _user_t,
        phone_t,
        _timestamp_t,
        stats,
    ) = load_dataset(data_path, device="cpu")
    raw_xy = raw_coords(coords_norm, building_t, stats)
    building = building_t.numpy()
    floor_values = floor_t.numpy()
    phone_ids = phone_t.numpy()
    coordinate_keys = location_keys(raw_xy, building, floor_values)
    development_phones = tuple(map(int, registry["device_development_phone_ids"]))
    partition = coordinate_axis_partition(
        coordinate_keys, phone_ids, development_phones
    )
    development_keys = set(partition["development_core"])
    support_keys, support_details = geometry_support_with_hull(
        development_keys, retained_fraction, seed=geometry_seed
    )
    evaluation_keys = development_keys - support_keys
    nested, nested_details = internal_support_roles(
        support_keys,
        seed=geometry_seed + 1_000_003,
        early_stop_fraction=early_stop_fraction,
        calibration_fraction=calibration_fraction,
    )
    fit_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["fit"]
    )
    stop_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["early_stop"]
    )
    calibration_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, nested["calibration"]
    )
    calibration_train_keys = nested["fit"] | nested["early_stop"]
    calibration_train_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, calibration_train_keys
    )
    support_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, support_keys
    )
    evaluation_rows = _rows_for_keys(
        coordinate_keys, phone_ids, development_phones, evaluation_keys
    )
    all_role_rows = {
        "fit": fit_rows,
        "early_stop": stop_rows,
        "calibration": calibration_rows,
        "unmeasured_evaluation": evaluation_rows,
    }
    role_names = list(all_role_rows)
    for i, left in enumerate(role_names):
        for right in role_names[i + 1 :]:
            if len(np.intersect1d(all_role_rows[left], all_role_rows[right])):
                raise RuntimeError(f"sparse50 row roles overlap: {left}, {right}")

    selected_epoch, schedule_history = _fit_schedule(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor_values,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        source_stats=stats,
        fit_idx=fit_rows,
        stop_idx=stop_rows,
        mechanism="M0",
        seed=front_seed,
        device=device,
        log_every=log_every,
    )
    (
        calibration_model,
        calibration_memory,
        _calibration_centered,
        _calibration_anchor,
        _calibration_stats,
        calibration_origin,
        _calibration_phones,
        calibration_history,
    ) = _fit_fixed(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor_values,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        source_stats=stats,
        train_idx=calibration_train_rows,
        mechanism="M0",
        seed=front_seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    calibration_panel = evaluate_spatial_panel(
        model=calibration_model,
        radio_memory=calibration_memory,
        coord_origin=calibration_origin,
        training_indices=calibration_train_rows,
        evaluation_rows=calibration_rows,
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor_values,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        lambda_options={
            variant: lambda_grid
            for variant in ("anchor_full", "idw_full", "fem_nearest", "fem_full")
        },
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
        degrees_freedom=degrees_freedom,
        batch_size=batch_size,
        candidate_chunk_size=candidate_chunk_size,
        credible_levels=credible_levels,
    )
    selected_lambdas = select_lambdas(calibration_panel, lambda_grid)
    del calibration_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (
        final_model,
        final_memory,
        _final_centered,
        _final_anchor,
        final_stats,
        final_origin,
        final_training_phones,
        final_history,
    ) = _fit_fixed(
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor_values,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        source_stats=stats,
        train_idx=support_rows,
        mechanism="M0",
        seed=front_seed,
        epochs=selected_epoch,
        device=device,
        log_every=log_every,
    )
    final_panel = evaluate_spatial_panel(
        model=final_model,
        radio_memory=final_memory,
        coord_origin=final_origin,
        training_indices=support_rows,
        evaluation_rows=evaluation_rows,
        rss=rss,
        raw_xy=raw_xy,
        building=building,
        floor=floor_values,
        phone_ids=phone_ids,
        coordinate_keys=coordinate_keys,
        lambda_options={
            variant: (selected_lambdas[variant]["lambda"],)
            for variant in ("anchor_full", "idw_full", "fem_nearest", "fem_full")
        },
        neighbors=neighbors,
        inverse_distance_power=inverse_distance_power,
        degrees_freedom=degrees_freedom,
        batch_size=batch_size,
        candidate_chunk_size=candidate_chunk_size,
        credible_levels=credible_levels,
    )
    selected_names = {
        "front_anchor": "front_anchor",
        "front_idw": "front_idw",
        "front_fem": "front_fem",
        **{
            variant: f"{variant}__lambda_{_parameter_label(record['lambda'])}"
            for variant, record in selected_lambdas.items()
        },
    }
    final_summary = {
        label: final_panel["summaries"][name]
        for label, name in selected_names.items()
    }
    paired = {}
    for left, right in (
        ("anchor_full", "front_anchor"),
        ("front_idw", "front_anchor"),
        ("front_fem", "front_idw"),
        ("fem_nearest", "front_fem"),
        ("fem_nearest", "anchor_full"),
        ("fem_full", "fem_nearest"),
        ("fem_full", "idw_full"),
    ):
        paired[f"{left}_vs_{right}"] = {
            metric: _paired_coordinate_delta(
                final_panel["row_outputs"][selected_names[left]][metric],
                final_panel["row_outputs"][selected_names[right]][metric],
                evaluation_rows,
                coordinate_keys,
            )
            for metric in ("error_2d_m", "energy_score_3d_m")
        }

    checkpoint_path = None
    if checkpoint_output is not None:
        checkpoint_path = resolve_path(str(checkpoint_output)).resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite sparse50 checkpoint: {checkpoint_path}")
        torch.save(
            {
                "artifact_status": "v5_bplus_sparse50_development_member",
                "selection_authority": False,
                "model_state": {
                    key: value.detach().cpu()
                    for key, value in final_model.state_dict().items()
                },
                "radio_memory": {
                    key: value.detach().cpu()
                    for key, value in final_memory.items()
                    if isinstance(value, torch.Tensor)
                },
                "coordinate_keys": [list(key) for key in final_memory["coordinate_keys"]],
                "coord_origin": torch.as_tensor(final_origin),
                "stats": final_stats,
                "training_indices": torch.as_tensor(support_rows),
                "training_phone_ids": list(final_training_phones),
                "geometry_seed": int(geometry_seed),
                "front_seed": int(front_seed),
                "selected_epoch": int(selected_epoch),
                "selected_lambdas": selected_lambdas,
                "registry_sha256": registry["registry_sha256"],
            },
            checkpoint_path,
        )

    return {
        "artifact_status": "v5_bplus_sparse50_fem_development_audit",
        "selection_authority": False,
        "protected_coordinate_confirmation_evaluated": False,
        "protected_confirmation_phones_evaluated": False,
        "registered_outer_test_outcomes_evaluated": False,
        "train_data": str(data_path),
        "registry": str(registry_file),
        "registry_sha256": registry["registry_sha256"],
        "device": str(device),
        "geometry_seed": int(geometry_seed),
        "front_seed": int(front_seed),
        "selected_epoch": int(selected_epoch),
        "geometry": {
            "development_coordinates": int(len(development_keys)),
            "retained_coordinates": int(len(support_keys)),
            "unmeasured_coordinates": int(len(evaluation_keys)),
            "realized_retained_fraction": float(len(support_keys) / len(development_keys)),
            "support_by_stratum": support_details,
            "nested_support_roles": {
                name: int(len(values)) for name, values in nested.items()
            },
            "nested_by_stratum": nested_details,
            "row_roles": {name: int(len(rows)) for name, rows in all_role_rows.items()},
        },
        "design": {
            "member_count_E": 1,
            "retained_fraction": float(retained_fraction),
            "support_selection": "geometry-only hull-preserving farthest-point survey",
            "candidate_domain": "measured-coordinate Delaunay mesh only",
            "candidate_quadrature": "three-point Gaussian rule per triangle",
            "front_density_conversion": "categorical mass divided by FEM nodal area",
            "idw_control": {
                "neighbors": int(neighbors),
                "inverse_distance_power": float(inverse_distance_power),
            },
            "fem_operator": (
                "piecewise-linear nodal basis; missing conditional RSS uses the "
                "Dirichlet-energy minimizer"
            ),
            "student_degrees_freedom": float(degrees_freedom),
            "lambda_grid": list(map(float, lambda_grid)),
            "lambda_selection": "nested calibration coordinate-macro 3-D energy score",
            "discrete_anchor_control": (
                "the identical marked likelihood updates the raw anchor posterior"
            ),
            "no_scan_effect": True,
            "no_top_k": True,
            "no_ensemble": True,
        },
        "selected_lambdas": selected_lambdas,
        "schedule": {
            "epochs_run": int(len(schedule_history)),
            "selected_epoch": int(selected_epoch),
            "best_early_stop_energy_score_3d_m": float(
                min(
                    record["early_stop"]["coordinate_group_macro_energy_score_3d_m"]
                    for record in schedule_history
                )
            ),
        },
        "calibration": _serializable_panel(
            calibration_panel,
            {
                "front_anchor": "front_anchor",
                "front_idw": "front_idw",
                "front_fem": "front_fem",
                **{
                    variant: record["configuration"]
                    for variant, record in selected_lambdas.items()
                },
            },
        ),
        "unmeasured_evaluation": {
            "rows": int(len(evaluation_rows)),
            "coordinates": int(len(evaluation_keys)),
            "mesh_diagnostics": final_panel["mesh_diagnostics"],
            "field_diagnostics": final_panel["field_diagnostics"],
            "selected_summaries": final_summary,
            "paired_coordinate_comparisons": paired,
        },
        "training": {
            "schedule_history": schedule_history,
            "calibration_refit_history": calibration_history,
            "final_refit_history": final_history,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        },
        "interpretation_contract": {
            "positive_same_budget_bridge": (
                "A full continuous method must improve the corrected discrete-anchor "
                "control on unmeasured coordinates under the identical 50% survey budget."
            ),
            "fem_admission": (
                "FEM is admitted only if it improves IDW spatial scores, or is "
                "practically tied while removing IDW neighborhood/power choices."
            ),
            "scope": (
                "This is development evidence for one geometry design and one member; "
                "it is not confirmation or an ensemble/member-count result."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B+ same-budget 50% sparse-front FEM audit"
    )
    parser.add_argument(
        "--train-data", default="data/UJIndoorLoc/trainingData.csv"
    )
    parser.add_argument(
        "--registry", default="evidence/v44/registry_v4_4.json"
    )
    parser.add_argument("--geometry-seed", type=int, default=5201)
    parser.add_argument("--front-seed", type=int, default=4101)
    parser.add_argument("--retained-fraction", type=float, default=0.50)
    parser.add_argument("--early-stop-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument(
        "--lambda-grid", default="0.015625,0.03125,0.0625,0.125,0.25,0.5"
    )
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--inverse-distance-power", type=float, default=2.0)
    parser.add_argument("--student-df", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-chunk-size", type=int, default=48)
    parser.add_argument("--credible-levels", default="0.5,0.8,0.9,0.95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-output", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lambda_grid = _csv_tuple(args.lambda_grid, float)
    credible_levels = _csv_tuple(args.credible_levels, float)
    if any(not math.isfinite(value) or value <= 0.0 for value in lambda_grid):
        raise ValueError("all lambda values must be finite and positive")
    if any(not 0.0 < value < 1.0 for value in credible_levels):
        raise ValueError("credible levels must lie inside (0,1)")
    payload = run_sparse50(
        train_data=args.train_data,
        registry_path=args.registry,
        geometry_seed=args.geometry_seed,
        front_seed=args.front_seed,
        retained_fraction=args.retained_fraction,
        early_stop_fraction=args.early_stop_fraction,
        calibration_fraction=args.calibration_fraction,
        lambda_grid=lambda_grid,
        neighbors=args.neighbors,
        inverse_distance_power=args.inverse_distance_power,
        degrees_freedom=args.student_df,
        batch_size=args.batch_size,
        candidate_chunk_size=args.candidate_chunk_size,
        credible_levels=credible_levels,
        device_name=args.device,
        threads=args.threads,
        log_every=args.log_every,
        checkpoint_output=args.checkpoint_output,
    )
    output = resolve_path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sparse50 artifact: {output}")
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact_status": payload["artifact_status"],
                "output": str(output),
                "geometry": payload["geometry"],
                "selected_lambdas": payload["selected_lambdas"],
                "paired_coordinate_comparisons": payload["unmeasured_evaluation"][
                    "paired_coordinate_comparisons"
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
