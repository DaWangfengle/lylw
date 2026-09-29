"""Execute the registered V17-R1 frozen-front backend identification.

The module has no outer-test or official-validation entry point.  It first
materializes one immutable development cache from the five R0 members, then
compares two nuisance operators and two point actions on exactly that cache.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.candidates import prepare_candidates
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
    spatial_median_from_draws,
)
from fhm_dapp_tim.fhm_dapp.consensus import (
    alpha_consensus_prior,
    weighted_component_mean,
)
from fhm_dapp_tim.fhm_dapp.dapp import (
    alpha_proximal_weights,
    project_alpha_proximal_hurdle,
    robust_hurdle_anchor_energy,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _save_torch_new,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.marked_orbit_v10 import (
    nuisance_posterior_diagnostics,
    tensor_legendre_uniform_quadrature,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import (
    ProjectedEnsemble,
    _compact_consensus_candidates,
    _member_weight_tensor,
)
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r1_protocol import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CACHE_BATCH_SIZE,
    CONSENSUS_ALPHA,
    DEFAULT_CACHE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    LAMBDA_VALUE,
    LOG_GAIN_BOUNDS,
    MAE_NONINFERIORITY_MARGIN_M,
    MARGINAL_BATCH_SIZE,
    MEMBER_COUNT,
    OFFSET_BOUNDS,
    POSTERIOR_VARIANTS,
    PREDICTIVE_DRAWS_PER_STREAM,
    PREDICTIVE_SEED,
    PROFILE_LEARNING_RATE,
    PROFILE_LOG_GAIN_WEIGHT,
    PROFILE_OFFSET_WEIGHT,
    PROFILE_STEPS,
    PROTOCOL_VERSION,
    PROXIMAL_ALPHA,
    QUADRATURE_ORDER,
    R0_PARITY_TOLERANCE_M,
    SCORING_BATCH_SIZE,
    STUDENT_DF,
    TOP_K,
    TOP_L,
    VARIANTS,
    load_registry,
)


def _validate_output(registry: Mapping[str, object], field: str, path: Path) -> None:
    allowed = Path(registry["allowed_outputs"][field]).resolve()
    if path.resolve() != allowed:
        raise ValueError(f"{field} path is outside the R1 registry")
    if path.exists():
        raise FileExistsError(path)


def _coordinate_groups(
    keys: Sequence[tuple],
) -> tuple[torch.Tensor, tuple[tuple, ...]]:
    ordered = tuple(sorted(set(map(tuple, keys))))
    mapping = {key: index for index, key in enumerate(ordered)}
    return (
        torch.tensor([mapping[tuple(key)] for key in keys], dtype=torch.long),
        ordered,
    )


def _group_vector(
    row_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    values = torch.as_tensor(row_values).detach().cpu().double()
    groups = torch.as_tensor(group_ids).detach().cpu().long()
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("row values and coordinate groups must be aligned")
    total = torch.zeros(int(group_count), dtype=torch.float64)
    count = torch.zeros(int(group_count), dtype=torch.float64)
    total.scatter_add_(0, groups, values)
    count.scatter_add_(0, groups, torch.ones_like(values))
    if bool((count <= 0).any()):
        raise RuntimeError("an empty coordinate group entered R1")
    return total / count


def balanced_affine_energy_grid(
    detection_prob: torch.Tensor,
    rss_center: torch.Tensor,
    rss_scale: torch.Tensor,
    query_values: torch.Tensor,
    query_mask: torch.Tensor,
    reliability: torch.Tensor,
    nodes: torch.Tensor,
    *,
    degrees_freedom: float = STUDENT_DF,
) -> torch.Tensor:
    """Evaluate the unchanged balanced hurdle loss on affine grid nodes."""

    node_tensor = torch.as_tensor(
        nodes, device=query_values.device, dtype=query_values.dtype
    )
    if node_tensor.ndim != 2 or node_tensor.shape[1] != 2 or len(node_tensor) == 0:
        raise ValueError("affine nodes must have nonempty shape [H,2]")
    if not bool(torch.isfinite(node_tensor).all()):
        raise ValueError("affine nodes must be finite")
    values = []
    for offset, log_gain in node_tensor:
        values.append(
            robust_hurdle_anchor_energy(
                detection_prob,
                rss_center,
                rss_scale,
                query_values,
                query_mask,
                reliability,
                style_offset=offset,
                style_scale=torch.exp(log_gain),
                visible_factor=1.0,
                invisible_factor=1.0,
                distribution="student_t",
                degrees_freedom=float(degrees_freedom),
                score="balanced_log",
            )
        )
    return torch.stack(values, dim=2)


def marginal_effective_loss(
    energy_grid: torch.Tensor,
    quadrature_weights: torch.Tensor,
    *,
    lambda_value: float = LAMBDA_VALUE,
) -> torch.Tensor:
    """Return ``-lambda log E_h exp(-loss/lambda)`` per row and anchor."""

    if energy_grid.ndim != 3 or not energy_grid.is_floating_point():
        raise ValueError("energy_grid must be floating [B,K,H]")
    weights = torch.as_tensor(
        quadrature_weights,
        device=energy_grid.device,
        dtype=energy_grid.dtype,
    )
    if weights.shape != (energy_grid.shape[2],):
        raise ValueError("quadrature weights and energy nodes disagree")
    if not bool(torch.isfinite(energy_grid).all()):
        raise ValueError("energy grid must be finite")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise ValueError("quadrature weights must be finite and positive")
    temperature = float(lambda_value)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("lambda_value must be finite and positive")
    weights = weights / weights.sum()
    return -temperature * torch.logsumexp(
        weights.log()[None, None, :] - energy_grid / temperature,
        dim=2,
    )


def _paired_simultaneous_family(
    differences: Mapping[str, np.ndarray],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Compute max-centered paired upper bounds using common group draws."""

    names = tuple(differences)
    if not names:
        raise ValueError("a simultaneous family cannot be empty")
    arrays = [np.asarray(differences[name], dtype=np.float64) for name in names]
    shape = arrays[0].shape
    if len(shape) != 1 or len(arrays[0]) < 2 or any(x.shape != shape for x in arrays):
        raise ValueError("family differences must be aligned group vectors")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("family differences must be finite")
    matrix = np.stack(arrays, axis=1)
    means = matrix.mean(axis=0)
    draws = np.empty((int(repetitions), len(names)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, int(repetitions), 250):
        stop = min(start + 250, int(repetitions))
        index = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        draws[start:stop] = matrix[index].mean(axis=1)
    maximum_centered = (draws - means[None, :]).max(axis=1)
    critical = float(np.quantile(maximum_centered, float(confidence)))
    comparisons = {}
    for column, name in enumerate(names):
        comparisons[name] = {
            "mean_difference_m": float(means[column]),
            "paired_percentile_ci95_m": [
                float(np.quantile(draws[:, column], 0.025)),
                float(np.quantile(draws[:, column], 0.975)),
            ],
            "simultaneous_upper_bound_m": float(means[column] + critical),
            "bootstrap_probability_difference_below_zero": float(
                np.mean(draws[:, column] < 0.0)
            ),
        }
    return {
        "difference_orientation": "left_named_quantity_minus_right_named_quantity",
        "comparisons": comparisons,
        "critical_value_m": critical,
        "confidence": float(confidence),
        "family_size": len(names),
        "coordinate_groups": int(len(matrix)),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "common_coordinate_group_resamples": True,
    }


@torch.no_grad()
def build_cache(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_CACHE,
    device_name: str = "cuda",
) -> dict:
    """Build the sole R1 development cache from the frozen R0 front."""

    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_cache", destination)
    selection_csv = Path(registry["data_contract"]["selection_refit_csv"])
    development_csv = Path(registry["data_contract"]["development_csv"])
    checkpoints = [Path(item["path"]) for item in registry["frozen_checkpoints"]]
    selection_role = _role_record(selection_csv)
    development_role = _role_record(development_csv)
    device = torch.device(device_name)

    with _installed_adapters([selection_role, development_role]):
        ensemble = ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device=device_name,
            dataset="UJI",
        )
        loaded = load_dataset(
            str(development_csv),
            dataset_type="UJI",
            device="cpu",
            reference_stats=ensemble.reference["stats"],
        )
    rss = loaded[0].float().cpu()
    exact = load_uji_v16(development_csv)
    if len(rss) != registry["data_contract"]["development_role"]["row_count"]:
        raise ValueError("R1 development row count changed")
    if not torch.equal(rss, exact["rss"]):
        raise ValueError("reference-stat loading changed development RSS")
    if not np.array_equal(development_role["building"], exact["building"]):
        raise ValueError("development building labels changed")
    if not np.array_equal(development_role["floor"], exact["floor"]):
        raise ValueError("development floor labels changed")

    query = rss.to(device)
    candidates_list = []
    shared_detection, shared_center, shared_scale = ensemble.anchor_memory
    for model, _checkpoint in ensemble.checkpoints:
        model.eval()
        candidate = prepare_candidates(
            model,
            query,
            shared_detection,
            shared_center,
            shared_scale,
            TOP_L,
            0.75,
            8.0,
            0.75,
            candidate_building_conditioning=False,
            include_measurement_stats=False,
        )
        candidates_list.append((model, _compact_consensus_candidates(candidate)))
    if len(candidates_list) != MEMBER_COUNT:
        raise RuntimeError("R1 member panel changed")
    consensus_query = candidates_list[0][1]["query"]
    if any(
        not torch.equal(consensus_query, item[1]["query"])
        for item in candidates_list[1:]
    ):
        raise RuntimeError("member seen-AP query spaces are not identical")
    precision = _member_weight_tensor(
        torch.stack(
            [candidate["radial_scale"] for _model, candidate in candidates_list],
            dim=0,
        ),
        "inverse_scale",
    )
    initial_stack = torch.stack(
        [candidate["initial_weights"] for _model, candidate in candidates_list],
        dim=0,
    )
    index_stack = torch.stack(
        [candidate["candidate_indices"] for _model, candidate in candidates_list],
        dim=0,
    )
    anchor_count = int(ensemble.checkpoints[0][0].n_anchors)
    consensus_indices, consensus_prior = alpha_consensus_prior(
        initial_stack,
        index_stack,
        precision,
        anchor_count,
        TOP_K,
        epsilon=1e-6,
        alpha=CONSENSUS_ALPHA,
    )
    member_locations = []
    member_scales = []
    for model, candidate in candidates_list:
        location, scale = model._components(
            candidate["embedding"], consensus_indices
        )
        member_locations.append(location)
        member_scales.append(scale)

    origin = np.asarray(ensemble.reference["coord_origin"], dtype=np.float64)
    target = torch.from_numpy(
        development_role["raw_xy"] - origin[None, :]
    ).double()
    group_ids, group_keys = _coordinate_groups(exact["keys"])
    expected_group_keys = tuple(
        tuple(value)
        for value in registry["data_contract"]["development_role"][
            "coordinate_keys"
        ]
    )
    if group_keys != expected_group_keys:
        raise ValueError("development coordinate identities changed")
    if len(rss) > CACHE_BATCH_SIZE:
        raise ValueError("registered one-batch R1 candidate contract was exceeded")
    randomness = common_student_t_randomness(
        len(rss),
        draws=PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=STUDENT_DF,
        seed=PREDICTIVE_SEED,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "role": "development",
        "development_csv_sha256": _sha256_file(development_csv),
        "selection_refit_csv_sha256": _sha256_file(selection_csv),
        "member_paths": tuple(str(path.resolve()) for path in checkpoints),
        "member_sha256": tuple(_sha256_file(path) for path in checkpoints),
        "coordinate_group_ids": group_ids,
        "coordinate_group_keys": group_keys,
        "target_xy": target,
        "target_building": exact["building_t"].long(),
        "target_floor": exact["floor_t"].long(),
        "query": consensus_query.detach().cpu(),
        "consensus_anchor_indices": consensus_indices.detach().cpu(),
        "consensus_prior": consensus_prior.detach().cpu(),
        "member_weights": precision.transpose(0, 1).detach().cpu(),
        "member_locations": torch.stack(member_locations, dim=1).detach().cpu(),
        "member_scales": torch.stack(member_scales, dim=1).detach().cpu(),
        "shared_radio_memory": {
            "detection_prob": shared_detection.detach().cpu(),
            "rss_center": shared_center.detach().cpu(),
            "rss_scale": shared_scale.detach().cpu(),
        },
        "coord_origin": torch.from_numpy(origin.copy()),
        "randomness": randomness,
        "construction_contract": {
            "cache_batch_size": CACHE_BATCH_SIZE,
            "actual_candidate_batches": 1,
            "E": MEMBER_COUNT,
            "L": TOP_L,
            "K": TOP_K,
            "candidate_building_conditioning": False,
            "member_weight_mode": "inverse_scale",
            "measurement_pool_mode": "shared",
            "development_predictions": int(len(rss)),
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
    }
    _save_torch_new(destination, payload)
    return {
        "cache": str(destination),
        "cache_sha256": _sha256_file(destination),
        "rows": int(len(rss)),
        "coordinate_groups": int(len(group_keys)),
        "members": MEMBER_COUNT,
        "anchors": anchor_count,
        "support_width": TOP_K,
        "seen_aps": int(consensus_query.shape[1]),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def load_cache(
    path: str | Path = DEFAULT_CACHE,
    *,
    registry: Mapping[str, object],
) -> dict:
    source = Path(path).resolve()
    if source != Path(registry["allowed_outputs"]["development_cache"]).resolve():
        raise ValueError("cache path is outside the R1 registry")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V17-R1 cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R1 cache registry mismatch")
    if payload.get("role") != "development":
        raise ValueError("R1 cache role changed")
    if payload.get("development_csv_sha256") != registry["data_contract"][
        "development_csv_sha256"
    ]:
        raise ValueError("R1 cache development binding changed")
    if payload.get("selection_refit_csv_sha256") != registry["data_contract"][
        "selection_refit_csv_sha256"
    ]:
        raise ValueError("R1 cache selection binding changed")
    expected_member_hashes = tuple(
        item["sha256"] for item in registry["frozen_checkpoints"]
    )
    if tuple(payload.get("member_sha256", ())) != expected_member_hashes:
        raise ValueError("R1 cache member panel changed")

    rows = int(registry["data_contract"]["development_role"]["row_count"])
    groups = int(
        registry["data_contract"]["development_role"]["coordinate_count"]
    )
    expected_group_keys = tuple(
        tuple(value)
        for value in registry["data_contract"]["development_role"][
            "coordinate_keys"
        ]
    )
    if tuple(map(tuple, payload["coordinate_group_keys"])) != expected_group_keys:
        raise ValueError("R1 cache coordinate keys changed")
    query = payload["query"]
    indices = payload["consensus_anchor_indices"]
    prior = payload["consensus_prior"]
    member_weight = payload["member_weights"]
    locations = payload["member_locations"]
    scales = payload["member_scales"]
    target = payload["target_xy"]
    group_ids = payload["coordinate_group_ids"]
    if query.ndim != 2 or len(query) != rows:
        raise ValueError("R1 cache query shape changed")
    if indices.shape != (rows, TOP_K) or prior.shape != (rows, TOP_K):
        raise ValueError("R1 cache consensus support shape changed")
    if member_weight.shape != (rows, MEMBER_COUNT):
        raise ValueError("R1 cache member-weight shape changed")
    expected_components = (rows, MEMBER_COUNT, TOP_K, 2)
    if locations.shape != expected_components or scales.shape != expected_components:
        raise ValueError("R1 cache continuous component shape changed")
    if target.shape != (rows, 2) or group_ids.shape != (rows,):
        raise ValueError("R1 cache target/group shape changed")
    tensors = (query, prior, member_weight, locations, scales, target)
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("R1 cache contains non-finite tensors")
    if bool((indices < 0).any()):
        raise ValueError("R1 cache contains negative support identities")
    if not torch.allclose(prior.sum(dim=1), torch.ones(rows), atol=2e-5):
        raise ValueError("R1 consensus prior is not normalized")
    if not torch.allclose(member_weight.sum(dim=1), torch.ones(rows), atol=2e-5):
        raise ValueError("R1 member weights are not normalized")
    if bool((prior < 0).any()) or bool((member_weight <= 0).any()):
        raise ValueError("R1 probability cache has invalid mass")
    if bool((scales <= 0).any()):
        raise ValueError("R1 component scales are not positive")
    radio = payload["shared_radio_memory"]
    radio_shape = radio["detection_prob"].shape
    if (
        len(radio_shape) != 2
        or radio["rss_center"].shape != radio_shape
        or radio["rss_scale"].shape != radio_shape
        or radio_shape[1] != query.shape[1]
    ):
        raise ValueError("R1 shared radio memory shape changed")
    if bool((indices >= radio_shape[0]).any()):
        raise ValueError("R1 support identity exceeds shared radio memory")
    if int(group_ids.min()) != 0 or int(group_ids.max()) != groups - 1:
        raise ValueError("R1 coordinate-group IDs changed")
    counts = torch.bincount(group_ids.long(), minlength=groups)
    if bool((counts <= 0).any()):
        raise ValueError("R1 cache has an empty coordinate group")
    randomness = payload["randomness"]
    expected_draw_shape = (rows, PREDICTIVE_DRAWS_PER_STREAM)
    for key in ("uniform_1", "uniform_2"):
        if randomness[key].shape != expected_draw_shape:
            raise ValueError("R1 uniform randomness shape changed")
    for key in ("noise_1", "noise_2"):
        if randomness[key].shape != (*expected_draw_shape, 2):
            raise ValueError("R1 Student-t randomness shape changed")
    return payload


def _historical_profile_weights(cache: dict, device: torch.device) -> torch.Tensor:
    prior = cache["consensus_prior"].to(device)
    indices = cache["consensus_anchor_indices"].to(device)
    query = cache["query"].to(device)
    radio = cache["shared_radio_memory"]
    detection = radio["detection_prob"].to(device)[indices]
    center = radio["rss_center"].to(device)[indices]
    scale = radio["rss_scale"].to(device)[indices]
    mask = query > -0.999
    values = torch.where(mask, query, torch.zeros_like(query))
    reliability = torch.ones(query.shape[1], device=device, dtype=query.dtype)
    with torch.enable_grad():
        weights = project_alpha_proximal_hurdle(
            initial_weights=prior,
            detection_prob=detection,
            rss_center=center,
            rss_scale=scale,
            query_values=values,
            query_mask=mask,
            reliability=reliability,
            prior_weight=LAMBDA_VALUE,
            alpha=PROXIMAL_ALPHA,
            steps=PROFILE_STEPS,
            learning_rate=PROFILE_LEARNING_RATE,
            style_offset_max=max(abs(value) for value in OFFSET_BOUNDS),
            style_scale_max_log=max(abs(value) for value in LOG_GAIN_BOUNDS),
            style_offset_weight=PROFILE_OFFSET_WEIGHT,
            style_scale_weight=PROFILE_LOG_GAIN_WEIGHT,
            visible_factor=1.0,
            invisible_factor=1.0,
            hurdle_distribution="student_t",
            hurdle_df=STUDENT_DF,
            hurdle_score="balanced_log",
            evidence_temperature_mode="none",
            return_calibration=False,
        )
    return weights.detach().cpu()


@torch.no_grad()
def _marginal_weights(
    cache: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    quadrature = tensor_legendre_uniform_quadrature(
        order=QUADRATURE_ORDER,
        offset_bounds=OFFSET_BOUNDS,
        log_gain_bounds=LOG_GAIN_BOUNDS,
        device=device,
        dtype=torch.float32,
    )
    nodes = quadrature["nodes"]
    node_weight = quadrature["weights"]
    if not isinstance(nodes, torch.Tensor) or not isinstance(node_weight, torch.Tensor):
        raise TypeError("R1 quadrature tensors are missing")
    output = []
    nuisance_offset = []
    nuisance_log_gain = []
    nuisance_effective_nodes = []
    effective_loss_mean = []
    rows = len(cache["consensus_prior"])
    radio = cache["shared_radio_memory"]
    detection_memory = radio["detection_prob"].to(device)
    center_memory = radio["rss_center"].to(device)
    scale_memory = radio["rss_scale"].to(device)
    reliability = torch.ones(
        cache["query"].shape[1], device=device, dtype=torch.float32
    )
    for start in range(0, rows, MARGINAL_BATCH_SIZE):
        stop = min(start + MARGINAL_BATCH_SIZE, rows)
        prior = cache["consensus_prior"][start:stop].to(device)
        indices = cache["consensus_anchor_indices"][start:stop].to(device)
        query = cache["query"][start:stop].to(device)
        mask = query > -0.999
        values = torch.where(mask, query, torch.zeros_like(query))
        energy = balanced_affine_energy_grid(
            detection_memory[indices],
            center_memory[indices],
            scale_memory[indices],
            values,
            mask,
            reliability,
            nodes,
            degrees_freedom=STUDENT_DF,
        )
        effective = marginal_effective_loss(
            energy, node_weight, lambda_value=LAMBDA_VALUE
        )
        posterior = alpha_proximal_weights(
            prior,
            effective,
            LAMBDA_VALUE,
            PROXIMAL_ALPHA,
        )
        # This is an interpretable conditional induced by the exponential
        # nuisance kernel, not a second Bayesian update of the scan.
        conditional = torch.softmax(
            node_weight.log()[None, None, :] - energy / LAMBDA_VALUE,
            dim=2,
        )
        nuisance_probability = torch.sum(
            posterior[:, :, None] * conditional, dim=1
        )
        diagnostic = nuisance_posterior_diagnostics(nuisance_probability, nodes)
        output.append(posterior.cpu())
        nuisance_offset.append(diagnostic["mean_offset"].cpu())
        nuisance_log_gain.append(diagnostic["mean_log_gain"].cpu())
        nuisance_effective_nodes.append(diagnostic["effective_nodes"].cpu())
        effective_loss_mean.append(effective.mean(dim=1).cpu())
    offsets = torch.cat(nuisance_offset)
    gains = torch.cat(nuisance_log_gain)
    nodes_used = torch.cat(nuisance_effective_nodes)
    loss_mean = torch.cat(effective_loss_mean)
    diagnostic_payload = {
        "quadrature_order_per_axis": QUADRATURE_ORDER,
        "quadrature_nodes": int(QUADRATURE_ORDER**2),
        "mean_posterior_offset": float(offsets.mean()),
        "mean_posterior_log_gain": float(gains.mean()),
        "mean_effective_nuisance_nodes": float(nodes_used.mean()),
        "q10_effective_nuisance_nodes": float(torch.quantile(nodes_used, 0.10)),
        "q90_effective_nuisance_nodes": float(torch.quantile(nodes_used, 0.90)),
        "mean_anchor_effective_loss": float(loss_mean.mean()),
    }
    return torch.cat(output), diagnostic_payload


@torch.no_grad()
def _score_posterior(
    cache: dict,
    anchor_probability: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = len(anchor_probability)
    member_weight = cache["member_weights"]
    member_location = cache["member_locations"]
    member_scale = cache["member_scales"]
    if anchor_probability.shape != (rows, TOP_K):
        raise ValueError("R1 posterior support shape changed")
    joint = member_weight[:, :, None] * anchor_probability[:, None, :]
    consensus_location = weighted_component_mean(
        member_location.permute(1, 0, 2, 3),
        member_weight.transpose(0, 1),
    )
    exact_mean = torch.sum(
        anchor_probability[:, :, None] * consensus_location, dim=1
    )
    energy_parts = []
    median_parts = []
    for start in range(0, rows, SCORING_BATCH_SIZE):
        stop = min(start + SCORING_BATCH_SIZE, rows)
        probability = joint[start:stop].reshape(stop - start, -1).to(device)
        location = member_location[start:stop].reshape(
            stop - start, -1, 2
        ).to(device)
        scale = member_scale[start:stop].reshape(
            stop - start, -1, 2
        ).to(device)
        randomness = {
            key: (
                value[start:stop].to(device)
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for key, value in cache["randomness"].items()
        }
        energy, first, second = continuous_energy_score_mc(
            probability,
            location,
            scale,
            cache["target_xy"][start:stop].to(device),
            randomness,
        )
        median = spatial_median_from_draws(first, second)
        energy_parts.append(energy.cpu())
        median_parts.append(median.cpu())
    row_energy = torch.cat(energy_parts)
    spatial_median = torch.cat(median_parts)
    target = cache["target_xy"]
    return {
        "row_energy": row_energy,
        "D0_action": exact_mean,
        "D1_action": spatial_median,
        "D0_error": torch.linalg.vector_norm(exact_mean - target, dim=1),
        "D1_error": torch.linalg.vector_norm(spatial_median - target, dim=1),
    }


def _metric_record(
    row_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> tuple[dict[str, float | int], np.ndarray]:
    group = _group_vector(row_values, group_ids, group_count).numpy()
    values = row_values.detach().cpu().double().numpy()
    return (
        {
            "coordinate_macro_m": float(group.mean()),
            "row_mean_m": float(values.mean()),
            "row_median_m": float(np.median(values)),
            "row_p90_m": float(np.quantile(values, 0.90)),
            "rows": int(len(values)),
            "coordinate_groups": int(len(group)),
        },
        group,
    )


def _r0_parity_gate(
    cache: dict,
    scored: Mapping[str, Mapping[str, torch.Tensor]],
    registry: Mapping[str, object],
) -> dict[str, object]:
    binding = registry["bindings"]["r0_result"]
    r0 = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    if r0.get("result_sha256") != binding["result_sha256"]:
        raise ValueError("R0 result self-hash binding changed")
    errors = r0["legacy_evaluation_payload"]["errors_m"]
    expected_prior = np.asarray(errors["baseline_consensus"], dtype=np.float64)
    expected_profile = np.asarray(errors["projected_consensus"], dtype=np.float64)
    actual_prior = scored["P0"]["D0_error"].double().numpy()
    actual_profile = scored["N0"]["D0_error"].double().numpy()
    if expected_prior.shape != actual_prior.shape or expected_profile.shape != actual_profile.shape:
        raise ValueError("R0 and R1 parity vectors have different shapes")
    prior_abs = np.abs(actual_prior - expected_prior)
    profile_abs = np.abs(actual_profile - expected_profile)
    passed = bool(
        prior_abs.max(initial=0.0) <= R0_PARITY_TOLERANCE_M
        and profile_abs.max(initial=0.0) <= R0_PARITY_TOLERANCE_M
    )
    return {
        "tolerance_m": R0_PARITY_TOLERANCE_M,
        "P0D0_vs_r0_consensus_prior_max_abs_m": float(prior_abs.max(initial=0.0)),
        "P0D0_vs_r0_consensus_prior_mean_abs_m": float(prior_abs.mean()),
        "N0D0_vs_r0_projected_max_abs_m": float(profile_abs.max(initial=0.0)),
        "N0D0_vs_r0_projected_mean_abs_m": float(profile_abs.mean()),
        "passed": passed,
    }


def evaluate(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_path: str | Path = DEFAULT_CACHE,
    output: str | Path = DEFAULT_RESULT,
    device_name: str = "cuda",
) -> dict:
    """Evaluate all six registered readouts once on development."""

    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_result", destination)
    cache_source = Path(cache_path).resolve()
    cache = load_cache(cache_source, registry=registry)
    device = torch.device(device_name)

    posterior = {
        "P0": cache["consensus_prior"].clone(),
        "N0": _historical_profile_weights(cache, device),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    posterior["N1"], marginal_diagnostics = _marginal_weights(cache, device)
    if tuple(posterior) != POSTERIOR_VARIANTS:
        raise RuntimeError("R1 posterior order changed")
    for value in posterior.values():
        if not torch.allclose(value.sum(dim=1), torch.ones(len(value)), atol=2e-5):
            raise RuntimeError("R1 posterior is not normalized")

    scored = {
        name: _score_posterior(cache, value, device=device)
        for name, value in posterior.items()
    }
    parity = _r0_parity_gate(cache, scored, registry)
    if not parity["passed"]:
        raise RuntimeError(
            "R1 stopped because the frozen N0D0/P0D0 path did not reproduce R0"
        )

    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    posterior_metrics = {}
    energy_groups: dict[str, np.ndarray] = {}
    mae_groups: dict[str, np.ndarray] = {}
    row_vectors = {"energy_score": {}, "euclidean_error": {}}
    for name in POSTERIOR_VARIANTS:
        energy_record, energy_group = _metric_record(
            scored[name]["row_energy"], group_ids, group_count
        )
        energy_groups[name] = energy_group
        row_vectors["energy_score"][name] = scored[name]["row_energy"].tolist()
        action_metrics = {}
        for decoder in ("D0", "D1"):
            variant = f"{name}{decoder}"
            record, group = _metric_record(
                scored[name][f"{decoder}_error"], group_ids, group_count
            )
            action_metrics[decoder] = record
            mae_groups[variant] = group
            row_vectors["euclidean_error"][variant] = scored[name][
                f"{decoder}_error"
            ].tolist()
        posterior_metrics[name] = {
            "energy_score_2d_m": energy_record,
            "actions": action_metrics,
            "mean_anchor_total_variation_from_P0": float(
                0.5
                * torch.abs(posterior[name] - posterior["P0"])
                .sum(dim=1)
                .mean()
            ),
        }
    if tuple(mae_groups) != VARIANTS:
        raise RuntimeError("R1 variant order changed")

    energy_difference = {
        "N0-P0": energy_groups["N0"] - energy_groups["P0"],
        "N1-P0": energy_groups["N1"] - energy_groups["P0"],
        "N1-N0": energy_groups["N1"] - energy_groups["N0"],
    }
    decoder_main = np.mean(
        np.stack(
            [
                mae_groups[f"{name}D1"] - mae_groups[f"{name}D0"]
                for name in POSTERIOR_VARIANTS
            ],
            axis=0,
        ),
        axis=0,
    )
    nuisance_action_main = 0.5 * (
        (mae_groups["N1D0"] - mae_groups["N0D0"])
        + (mae_groups["N1D1"] - mae_groups["N0D1"])
    )
    mae_difference = {
        "P0D1-P0D0": mae_groups["P0D1"] - mae_groups["P0D0"],
        "N0D0-P0D0": mae_groups["N0D0"] - mae_groups["P0D0"],
        "N0D1-P0D0": mae_groups["N0D1"] - mae_groups["P0D0"],
        "N1D0-P0D0": mae_groups["N1D0"] - mae_groups["P0D0"],
        "N1D1-P0D0": mae_groups["N1D1"] - mae_groups["P0D0"],
        "N0D1-N0D0": mae_groups["N0D1"] - mae_groups["N0D0"],
        "N1D1-N1D0": mae_groups["N1D1"] - mae_groups["N1D0"],
        "N1-N0_action_main_effect": nuisance_action_main,
        "D1-D0_decoder_main_effect": decoder_main,
    }
    energy_family = _paired_simultaneous_family(energy_difference)
    mae_family = _paired_simultaneous_family(mae_difference)

    marked_gates = {}
    eligible = []
    for name in ("N0", "N1"):
        energy_ok = bool(
            energy_family["comparisons"][f"{name}-P0"][
                "simultaneous_upper_bound_m"
            ]
            < 0.0
        )
        action_bounds = {
            decoder: mae_family["comparisons"][f"{name}{decoder}-P0D0"][
                "simultaneous_upper_bound_m"
            ]
            for decoder in ("D0", "D1")
        }
        action_ok = any(
            value < MAE_NONINFERIORITY_MARGIN_M
            for value in action_bounds.values()
        )
        passed = bool(energy_ok and action_ok)
        if passed:
            eligible.append(name)
        marked_gates[name] = {
            "energy_improvement_passed": energy_ok,
            "action_simultaneous_upper_bounds_m": action_bounds,
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
            "at_least_one_action_noninferior": action_ok,
            "eligible": passed,
        }
    marginal_advantage = bool(
        energy_family["comparisons"]["N1-N0"]["simultaneous_upper_bound_m"]
        < 0.0
    )
    decoder_advantage = bool(
        mae_family["comparisons"]["D1-D0_decoder_main_effect"][
            "simultaneous_upper_bound_m"
        ]
        < 0.0
    )
    if not eligible:
        selected_posterior = "P0"
    elif len(eligible) == 1:
        selected_posterior = eligible[0]
    else:
        selected_posterior = "N1" if marginal_advantage else "N0"
    selected_specific_decoder_advantage = bool(
        mae_family["comparisons"][
            f"{selected_posterior}D1-{selected_posterior}D0"
        ]["simultaneous_upper_bound_m"]
        < 0.0
    )
    selected_decoder = (
        "D1"
        if decoder_advantage and selected_specific_decoder_advantage
        else "D0"
    )
    selected_variant = f"{selected_posterior}{selected_decoder}"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "v17_r1_development_complete",
        "scientific_status": (
            "iterative UJI development after known R0 outcome; not independent confirmation"
        ),
        "registry": str(Path(registry_path).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "cache": str(cache_source),
        "cache_sha256": _sha256_file(cache_source),
        "query_role": "development",
        "development_predictions": int(len(cache["target_xy"])),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
        "r0_parity_gate": parity,
        "posterior_metrics": posterior_metrics,
        "inference": {
            "energy_family": energy_family,
            "mae_family": mae_family,
        },
        "decision_gates": {
            "marked_posterior": marked_gates,
            "marginalization_advantage": marginal_advantage,
            "spatial_median_advantage": decoder_advantage,
            "selected_posterior_spatial_median_advantage": (
                selected_specific_decoder_advantage
            ),
            "eligible_marked_posteriors": eligible,
        },
        "registered_selection": {
            "posterior": selected_posterior,
            "decoder": selected_decoder,
            "variant": selected_variant,
            "rule": registry["decision_gates"],
        },
        "marginal_nuisance_diagnostics": marginal_diagnostics,
        "coordinate_group_keys": [list(key) for key in cache["coordinate_group_keys"]],
        "group_vectors": {
            "energy_score": {
                name: values.tolist() for name, values in energy_groups.items()
            },
            "euclidean_error": {
                name: values.tolist() for name, values in mae_groups.items()
            },
        },
        "row_vectors": row_vectors,
    }
    payload["result_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(destination, payload)
    return payload


def check_artifacts(
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, object]:
    registry = load_registry(registry_path)
    records = {}
    for field in ("development_cache", "development_result"):
        path = Path(registry["allowed_outputs"][field])
        records[field] = {
            "path": str(path.resolve()),
            "exists": path.exists(),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
    return {
        "registry_sha256": registry["registry_sha256"],
        "artifacts": records,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache_parser = sub.add_parser("build-cache")
    cache_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache_parser.add_argument("--output", default=str(DEFAULT_CACHE))
    cache_parser.add_argument("--device", default="cuda")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate_parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    evaluate_parser.add_argument("--output", default=str(DEFAULT_RESULT))
    evaluate_parser.add_argument("--device", default="cuda")
    check_parser = sub.add_parser("check-artifacts")
    check_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args(argv)
    if args.command == "build-cache":
        result = build_cache(
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
        )
    elif args.command == "evaluate":
        result = evaluate(
            registry_path=args.registry,
            cache_path=args.cache,
            output=args.output,
            device_name=args.device,
        )
    else:
        result = check_artifacts(args.registry)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
