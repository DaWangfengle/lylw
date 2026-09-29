"""Execute V17-R2C joint information-budget saturation on development."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import continuous_energy_score_mc
from fhm_dapp_tim.fhm_dapp.consensus import alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b import load_cache as load_r2b_cache
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import (
    load_registry as load_r2b_registry,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2c_protocol import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE,
    CONFIGURATIONS,
    CONSENSUS_ALPHA_GRID,
    CONSENSUS_EPSILON,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    ENERGY_EQUIVALENCE_MARGIN_M,
    FINAL_MINIMUM_MEMBERS,
    HISTORICAL_CONFIGURATION,
    HISTORICAL_VARIANT,
    MAE_NONINFERIORITY_MARGIN_M,
    MEMBER_CEILING_WITH_TWO_LATER_VALUES,
    MEMBER_GRID,
    PROTOCOL_VERSION,
    SCORING_BATCH_SIZE,
    SUPPORT_K_GRID,
    SUPPORT_L_GRID,
    SUPPORT_AUDIT_REPRODUCTION_TOLERANCE,
    VARIANTS,
    balanced_member_subsets,
    load_registry,
    variant_name,
)


def _validate_output(registry: Mapping[str, object], path: Path) -> None:
    allowed = Path(registry["allowed_outputs"]["development_result"]).resolve()
    if path.resolve() != allowed:
        raise ValueError("development result path is outside the R2C registry")
    if path.exists():
        raise FileExistsError(path)


def _group_vector(
    row_values: torch.Tensor, group_ids: torch.Tensor, group_count: int
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
        raise RuntimeError("an empty coordinate group entered R2C")
    return total / count


def studentized_max_t_family(
    differences: Mapping[str, np.ndarray],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
    orientation: str,
) -> dict[str, object]:
    """Heteroscedastic max-t upper bounds using common coordinate resamples."""

    names = tuple(differences)
    if not names:
        raise ValueError("a max-t family cannot be empty")
    arrays = [np.asarray(differences[name], dtype=np.float64) for name in names]
    shape = arrays[0].shape
    if len(shape) != 1 or len(arrays[0]) < 2 or any(value.shape != shape for value in arrays):
        raise ValueError("max-t differences must be aligned group vectors")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("max-t differences must be finite")
    matrix = np.stack(arrays, axis=1)
    means = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix))
    safe_se = np.where(standard_errors > 1e-12, standard_errors, 1.0)
    draws = np.empty((int(repetitions), len(names)), dtype=np.float64)
    maxima = np.empty(int(repetitions), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, int(repetitions), 250):
        stop = min(start + 250, int(repetitions))
        index = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        sampled = matrix[index].mean(axis=1)
        draws[start:stop] = sampled
        centered_t = (sampled - means[None, :]) / safe_se[None, :]
        centered_t[:, standard_errors <= 1e-12] = 0.0
        maxima[start:stop] = centered_t.max(axis=1)
    critical = float(np.quantile(maxima, float(confidence)))
    upper = means + critical * standard_errors
    comparisons = {}
    for column, name in enumerate(names):
        comparisons[name] = {
            "mean_difference_m": float(means[column]),
            "standard_error_m": float(standard_errors[column]),
            "paired_percentile_ci95_m": [
                float(np.quantile(draws[:, column], 0.025)),
                float(np.quantile(draws[:, column], 0.975)),
            ],
            "simultaneous_upper_bound_m": float(upper[column]),
            "bootstrap_probability_difference_below_zero": float(
                np.mean(draws[:, column] < 0.0)
            ),
        }
    return {
        "difference_orientation": orientation,
        "comparisons": comparisons,
        "critical_max_t": critical,
        "confidence": float(confidence),
        "family_size": len(names),
        "coordinate_groups": int(len(matrix)),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "studentized": True,
        "common_coordinate_group_resamples": True,
    }


def _metric_record(
    row_values: torch.Tensor, group_ids: torch.Tensor, group_count: int
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


def _member_top_l(
    probability: torch.Tensor,
    member_location: torch.Tensor,
    member_scale: torch.Tensor,
    L: int,
    *,
    student_df: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(probability, int(L), dim=2, sorted=True)
    retained_mass = values.sum(dim=2)
    weights = values / retained_mass[:, :, None].clamp_min(1e-30)
    gather = indices[:, :, :, None].expand(-1, -1, -1, 2)
    locations = torch.gather(member_location, 2, gather)
    scales = torch.gather(member_scale, 2, gather)
    mean = torch.sum(weights[:, :, :, None] * locations, dim=2)
    component_variance = scales.square() * float(student_df) / (float(student_df) - 2.0)
    radial_second = torch.sum(
        weights
        * (
            (locations - mean[:, :, None, :]).square().sum(dim=3)
            + component_variance.sum(dim=3)
        ),
        dim=2,
    )
    radial = radial_second.clamp_min(1e-8).sqrt()
    return weights, indices, radial, retained_mass


def _joint_consensus_law(
    member_prior: torch.Tensor,
    member_indices: torch.Tensor,
    member_radial: torch.Tensor,
    full_location: torch.Tensor,
    full_scale: torch.Tensor,
    subset: tuple[int, ...],
    *,
    K: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subset_index = torch.tensor(subset, device=member_prior.device, dtype=torch.long)
    priors = member_prior.index_select(0, subset_index)
    indices = member_indices.index_select(0, subset_index)
    radial = member_radial.index_select(0, subset_index)
    member_weight = radial.square().clamp_min(1e-8).reciprocal()
    member_weight = member_weight / member_weight.sum(dim=0, keepdim=True).clamp_min(1e-30)
    consensus_indices, consensus_probability = alpha_consensus_prior(
        priors,
        indices,
        member_weight,
        full_location.shape[2],
        int(K),
        epsilon=CONSENSUS_EPSILON,
        alpha=float(alpha),
    )
    locations = full_location.index_select(0, subset_index)
    scales = full_scale.index_select(0, subset_index)
    gather = consensus_indices[None, :, :, None].expand(len(subset), -1, -1, 2)
    selected_location = torch.gather(locations, 2, gather)
    selected_scale = torch.gather(scales, 2, gather)
    joint = (
        member_weight.transpose(0, 1)[:, :, None]
        * consensus_probability[:, None, :]
    )
    return (
        joint.reshape(len(consensus_probability), -1),
        selected_location.permute(1, 0, 2, 3).reshape(
            len(consensus_probability), -1, 2
        ),
        selected_scale.permute(1, 0, 2, 3).reshape(
            len(consensus_probability), -1, 2
        ),
    )


@torch.no_grad()
def _score_joint_law(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    randomness: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    action = torch.sum(probability[:, :, None] * location, dim=1)
    mae = torch.linalg.vector_norm(action.double() - target, dim=1).cpu()
    energy_parts = []
    for start in range(0, len(target), SCORING_BATCH_SIZE):
        stop = min(start + SCORING_BATCH_SIZE, len(target))
        batch_randomness = {
            key: (
                value[start:stop]
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for key, value in randomness.items()
        }
        energy, _first, _second = continuous_energy_score_mc(
            probability[start:stop],
            location[start:stop],
            scale[start:stop],
            target[start:stop],
            batch_randomness,
        )
        energy_parts.append(energy.cpu().double())
    return torch.cat(energy_parts), mae.double()


def select_joint_information_budget(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    energy_groups: Mapping[str, np.ndarray],
    mae_family: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config_by_name = {
        variant_name(E, L, K, alpha): (E, L, K, alpha)
        for E, L, K, alpha in CONFIGURATIONS
    }
    eligible = [HISTORICAL_VARIANT]
    for name in VARIANTS:
        if name == HISTORICAL_VARIANT:
            continue
        if mae_family["comparisons"][name]["simultaneous_upper_bound_m"] <= (
            MAE_NONINFERIORITY_MARGIN_M
        ):
            eligible.append(name)
    multi_eligible = [name for name in eligible if config_by_name[name][0] >= FINAL_MINIMUM_MEMBERS]
    if not multi_eligible:
        raise RuntimeError("no multi-member R2C configuration passed the MAE gate")
    global_best = min(
        multi_eligible,
        key=lambda name: (
            metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
            VARIANTS.index(name),
        ),
    )
    energy_differences = {
        name: energy_groups[name] - energy_groups[global_best]
        for name in multi_eligible
        if name != global_best
    }
    if energy_differences:
        energy_family = studentized_max_t_family(
            energy_differences,
            seed=BOOTSTRAP_SEED,
            orientation="candidate_minus_eligible_global_energy_minimum",
        )
    else:
        energy_family = {
            "difference_orientation": "candidate_minus_eligible_global_energy_minimum",
            "comparisons": {},
            "critical_max_t": 0.0,
            "confidence": CONFIDENCE,
            "family_size": 0,
            "coordinate_groups": len(energy_groups[global_best]),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "studentized": True,
            "common_coordinate_group_resamples": True,
        }
    platform = [global_best]
    platform.extend(
        name
        for name in multi_eligible
        if name != global_best
        and energy_family["comparisons"][name]["simultaneous_upper_bound_m"]
        <= ENERGY_EQUIVALENCE_MARGIN_M
    )

    order = {name: index for index, name in enumerate(VARIANTS)}

    def information_key(name: str) -> tuple:
        E, L, K, alpha = config_by_name[name]
        return (
            E * L + K,
            E,
            L,
            K,
            metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
            abs(alpha - 0.25),
            order[name],
        )

    ceiling_confirmed = [
        name
        for name in platform
        if config_by_name[name][0] <= MEMBER_CEILING_WITH_TWO_LATER_VALUES
    ]
    provisional = min(ceiling_confirmed, key=information_key) if ceiling_confirmed else None
    member_extension = provisional is None
    support_extension = False
    selected = provisional
    if provisional is not None:
        E, L, K, _alpha = config_by_name[provisional]
        support_extension = L == max(SUPPORT_L_GRID) or K == max(SUPPORT_K_GRID)
        if support_extension:
            selected = None
    if selected is None:
        selected_payload = None
    else:
        E, L, K, alpha = config_by_name[selected]
        selected_payload = {
            "variant": selected,
            "E": E,
            "L": L,
            "K": K,
            "alpha": alpha,
            "information_budget_E_times_L_plus_K": E * L + K,
            "later_member_counts": [value for value in MEMBER_GRID if value > E],
        }
    profiles = {}
    for E in MEMBER_GRID:
        candidates = [name for name in eligible if config_by_name[name][0] == E]
        if not candidates:
            profiles[str(E)] = None
            continue
        best = min(
            candidates,
            key=lambda name: (
                metrics[name]["energy_score_2d_m"]["coordinate_macro_m"],
                order[name],
            ),
        )
        profiles[str(E)] = {
            "variant": best,
            "energy_score_coordinate_macro_m": metrics[best]["energy_score_2d_m"][
                "coordinate_macro_m"
            ],
        }
    decision = {
        "eligible_global_energy_minimum": global_best,
        "mae_eligible_variants": eligible,
        "energy_equivalence_platform": platform,
        "provisional_minimum_information_variant": provisional,
        "registered_selection": selected_payload,
        "member_information_saturated": not member_extension,
        "support_information_saturated": (
            None if provisional is None else not support_extension
        ),
        "requires_member_panel_extension": member_extension,
        "requires_support_grid_extension": support_extension,
        "may_access_outer_test": False,
        "profiled_best_by_E": profiles,
    }
    return decision, energy_family


@torch.no_grad()
def evaluate(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, destination)
    r2b_registry = load_r2b_registry(registry["bindings"]["r2b_registry"]["path"])
    cache = load_r2b_cache(
        registry["bindings"]["r2b_cache"]["path"], registry=r2b_registry
    )
    device = torch.device(device_name)
    beta_neural = registry["frozen_atomic_law"]["beta_neural"]
    beta_prototype = registry["frozen_atomic_law"]["beta_prototype"]
    neural = cache["member_neural_logits"].to(device)
    distance = cache["prototype_distance"].to(device)
    full_probability = torch.softmax(
        float(beta_neural) * neural - float(beta_prototype) * distance[None, :, :],
        dim=2,
    )
    full_location = cache["member_locations"].to(device)
    full_scale = cache["member_scales"].to(device)
    target = cache["target_xy"].to(device)
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cache["randomness"].items()
    }
    rows = len(target)
    row_energy = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}
    row_mae = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}
    retained_audit = {}
    parity_row_mae = None

    for L in SUPPORT_L_GRID:
        member_prior, member_indices, member_radial, retained_mass = _member_top_l(
            full_probability, full_location, full_scale, L
        )
        retained_audit[str(L)] = {
            "mean_retained_mass": float(retained_mass.mean()),
            "q10_retained_mass": float(torch.quantile(retained_mass, 0.10)),
            "median_retained_mass": float(torch.quantile(retained_mass, 0.50)),
            "q90_retained_mass": float(torch.quantile(retained_mass, 0.90)),
            "mean_reverse_kl_nats": float(
                (-retained_mass.clamp_min(1e-30).log()).mean()
            ),
        }
        for E in MEMBER_GRID:
            subsets = balanced_member_subsets(E)
            alphas = (0.25,) if E == 1 else CONSENSUS_ALPHA_GRID
            valid_k = tuple(K for K in SUPPORT_K_GRID if K <= L)
            for subset in subsets:
                for alpha in alphas:
                    maximum_k = max(valid_k)
                    maximum_probability, maximum_location, maximum_scale = (
                        _joint_consensus_law(
                            member_prior,
                            member_indices,
                            member_radial,
                            full_location,
                            full_scale,
                            subset,
                            K=maximum_k,
                            alpha=alpha,
                        )
                    )
                    member_count = len(subset)
                    probability_grid = maximum_probability.reshape(
                        rows, member_count, maximum_k
                    )
                    location_grid = maximum_location.reshape(
                        rows, member_count, maximum_k, 2
                    )
                    scale_grid = maximum_scale.reshape(
                        rows, member_count, maximum_k, 2
                    )
                    for K in valid_k:
                        probability = probability_grid[:, :, :K]
                        probability = probability / probability.sum(
                            dim=(1, 2), keepdim=True
                        ).clamp_min(1e-30)
                        probability = probability.reshape(rows, -1)
                        location = location_grid[:, :, :K, :].reshape(
                            rows, -1, 2
                        )
                        scale = scale_grid[:, :, :K, :].reshape(rows, -1, 2)
                        energy, mae = _score_joint_law(
                            probability, location, scale, target, randomness
                        )
                        name = variant_name(E, L, K, alpha)
                        divisor = float(len(subsets))
                        row_energy[name] += energy / divisor
                        row_mae[name] += mae / divisor
            print(json.dumps({
                "completed_L": L,
                "completed_E": E,
                "balanced_subsets": len(subsets),
                "alphas": len(alphas),
                "K_values": list(valid_k),
            }))

        if L == HISTORICAL_CONFIGURATION[1]:
            probability, location, scale = _joint_consensus_law(
                member_prior,
                member_indices,
                member_radial,
                full_location,
                full_scale,
                (0, 1, 2, 3, 4),
                K=HISTORICAL_CONFIGURATION[2],
                alpha=HISTORICAL_CONFIGURATION[3],
            )
            action = torch.sum(probability[:, :, None] * location, dim=1)
            parity_row_mae = torch.linalg.vector_norm(
                action.double() - target, dim=1
            ).cpu()

    if parity_row_mae is None:
        raise RuntimeError("R2C parity configuration was not evaluated")
    registered_audit = registry["unlabelled_support_audit_at_registration"][
        "diagnostic"
    ]
    audit_differences = {
        f"L{L}_{field}": abs(retained_audit[str(L)][field] - expected)
        for L in SUPPORT_L_GRID
        for field, expected in registered_audit[str(L)].items()
    }
    maximum_audit_difference = max(audit_differences.values())
    audit_tolerance = float(
        registry["unlabelled_support_audit_at_registration"][
            "evaluation_reproduction_tolerance"
        ]
    )
    if maximum_audit_difference > audit_tolerance:
        raise RuntimeError("R2C failed to reproduce its unlabelled support audit")
    r1_result = json.loads(
        Path(registry["bindings"]["r1_result"]["path"]).read_text(encoding="utf-8")
    )
    expected_parity = torch.tensor(
        r1_result["row_vectors"]["euclidean_error"]["P0D0"], dtype=torch.float64
    )
    if expected_parity.shape != parity_row_mae.shape:
        raise ValueError("R1/R2C parity vectors have different shapes")
    parity_abs = (parity_row_mae - expected_parity).abs()
    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    observed_parity_macro = float(
        _group_vector(parity_row_mae, group_ids, group_count).mean()
    )
    expected_parity_macro = float(
        _group_vector(expected_parity, group_ids, group_count).mean()
    )
    macro_difference = observed_parity_macro - expected_parity_macro
    parity_gate = {
        "r2c_coordinate_macro_mae_m": observed_parity_macro,
        "r1_coordinate_macro_mae_m": expected_parity_macro,
        "signed_coordinate_macro_mae_difference_m": macro_difference,
        "absolute_coordinate_macro_mae_difference_m": abs(macro_difference),
        "maximum_absolute_error_m": float(parity_abs.max()),
        "mean_absolute_error_m": float(parity_abs.mean()),
        "tolerance_m": registry["parity_gate"][
            "absolute_coordinate_macro_mae_tolerance_m"
        ],
        "gate_estimand": "absolute_coordinate_macro_mae_difference_m",
        "passed": bool(
            abs(macro_difference)
            <= registry["parity_gate"][
                "absolute_coordinate_macro_mae_tolerance_m"
            ]
        ),
    }
    if not parity_gate["passed"]:
        raise RuntimeError("R2C failed the frozen R1 P0D0 parity gate")

    metrics = {}
    energy_groups = {}
    mae_groups = {}
    for name in VARIANTS:
        energy_record, energy_group = _metric_record(
            row_energy[name], group_ids, group_count
        )
        mae_record, mae_group = _metric_record(row_mae[name], group_ids, group_count)
        metrics[name] = {
            "energy_score_2d_m": energy_record,
            "exact_mixture_mean_mae_m": mae_record,
        }
        energy_groups[name] = energy_group
        mae_groups[name] = mae_group

    mae_family = studentized_max_t_family(
        {
            name: mae_groups[name] - mae_groups[HISTORICAL_VARIANT]
            for name in VARIANTS
            if name != HISTORICAL_VARIANT
        },
        seed=BOOTSTRAP_SEED,
        orientation="candidate_minus_balanced_historical_configuration",
    )
    decision, energy_family = select_joint_information_budget(
        metrics, energy_groups, mae_family
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "r2b_cache_sha256": _sha256_file(registry["bindings"]["r2b_cache"]["path"]),
        "metrics": metrics,
        "coordinate_group_vectors": {
            "energy_score_2d_m": {
                name: energy_groups[name].tolist() for name in VARIANTS
            },
            "exact_mixture_mean_mae_m": {
                name: mae_groups[name].tolist() for name in VARIANTS
            },
        },
        "simultaneous_inference": {
            "mae_candidate_minus_balanced_historical": mae_family,
            "energy_candidate_minus_eligible_global_minimum": energy_family,
        },
        "registered_decision": decision,
        "r1_parity_gate": parity_gate,
        "retained_mass_diagnostic": {
            "values": retained_audit,
            "maximum_absolute_registration_difference": maximum_audit_difference,
            "reproduction_tolerance": audit_tolerance,
            "passed": True,
        },
        "development_scan_predictions": rows,
        "registered_configurations": len(VARIANTS),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def check_result(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict:
    registry = load_registry(registry_path)
    source = Path(result_path).resolve()
    if source != Path(registry["allowed_outputs"]["development_result"]).resolve():
        raise ValueError("result path is outside the R2C registry")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("result_sha256") != _canonical_json_sha256(
        result, omit="result_sha256"
    ):
        raise ValueError("R2C result self-hash failed")
    if result.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R2C result and registry differ")
    if result.get("r2b_cache_sha256") != registry["bindings"]["r2b_cache"][
        "file_sha256"
    ]:
        raise ValueError("R2C result and R2B cache differ")
    return {
        "registry_sha256": registry["registry_sha256"],
        "result_sha256": result["result_sha256"],
        "registered_decision": result["registered_decision"],
        "parity_gate": result["r1_parity_gate"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    score.add_argument("--output", default=str(DEFAULT_RESULT))
    score.add_argument("--device", default="cuda")
    check = sub.add_parser("check-result")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    check.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        payload = evaluate(
            registry_path=args.registry, output=args.output, device_name=args.device
        )
    else:
        payload = check_result(registry_path=args.registry, result_path=args.result)
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
