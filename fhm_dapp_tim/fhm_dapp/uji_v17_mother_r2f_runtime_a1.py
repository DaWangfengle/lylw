"""Deterministic execution amendment for the frozen V17-R2F experiment.

The registered R2F scientific contract is unchanged.  This runtime replaces
the CUDA sparse-union implementation of the already-defined alpha-consensus
operator with its dense executable definition.  The two operators are
algebraically identical, but the sparse CUDA path uses ``scatter_add_`` and is
not bitwise reproducible.  That microscopic probability jitter can change a
categorical common-random-number draw and therefore made the registered R2E
energy-score continuity gate impossible to reproduce at 1e-5 m.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as registered_r2f
from fhm_dapp_tim.fhm_dapp.consensus import dense_alpha_consensus_prior
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = (
    ROOT
    / "evidence"
    / "v17_acrhap_mother"
    / "amendment_v17_r2f_a1_deterministic_consensus.json"
)
AMENDMENT_PROTOCOL = "uji_v17_r2f_runtime_a1_deterministic_dense_v1_20260822"


def _load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("amendment_sha256") != _canonical_json_sha256(
        payload, omit="amendment_sha256"
    ):
        raise ValueError("V17-R2F-A1 amendment self-hash failed")
    if payload.get("adapter_protocol") != AMENDMENT_PROTOCOL:
        raise ValueError("V17-R2F-A1 amendment protocol changed")
    if payload.get("scientific_change") is not False:
        raise ValueError("V17-R2F-A1 must remain scientifically neutral")
    if payload.get("selection_authority_for_v17_r2f") is not False:
        raise ValueError("V17-R2F-A1 cannot redefine the R2F selection rule")
    if payload.get("may_modify_v17_r2f_model") is not False:
        raise ValueError("V17-R2F-A1 cannot modify the R2F model")

    for field in (
        "runtime_source",
        "test_source",
        "amendment_note",
        "registered_execution_source",
        "registered_protocol_source",
        "registered_design",
        "registered_registry",
        "development_cache",
    ):
        record = payload["bindings"][field]
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2F-A1 binding changed: {field}")
    if Path(payload["bindings"]["runtime_source"]["path"]).resolve() != Path(
        __file__
    ).resolve():
        raise ValueError("V17-R2F-A1 runtime path changed")

    registry_record = payload["bindings"]["registered_registry"]
    registry = registered_r2f.load_registry(registry_record["path"])
    if registry.get("registry_sha256") != registry_record["registry_sha256"]:
        raise ValueError("V17-R2F-A1 registry identity changed")
    if (
        Path(payload["bindings"]["development_cache"]["path"]).resolve()
        != Path(registry["allowed_outputs"]["development_cache"]).resolve()
    ):
        raise ValueError("V17-R2F-A1 cache path differs from the registry")
    return payload


@contextlib.contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(
            previous, warn_only=previous_warn_only
        )


def _dense_projected_joint_law(
    member_prior: torch.Tensor,
    member_indices: torch.Tensor,
    member_radial: torch.Tensor,
    full_location: torch.Tensor,
    full_scale: torch.Tensor,
    subset: tuple[int, ...],
    *,
    construction_k: int,
    output_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the registered law through its deterministic dense definition.

    ``construction_k`` reproduces the frozen R2E evaluation order: construct
    the largest registered consensus support for a given L, then apply the
    exact reverse-KL Top-K projection and renormalize.  For R2F this is
    algebraically the same K=192 law and changes no scientific parameter.
    """

    if not subset:
        raise ValueError("member subset cannot be empty")
    if not 0 < int(output_k) <= int(construction_k) <= full_location.shape[2]:
        raise ValueError("invalid deterministic consensus support")
    subset_index = torch.tensor(
        tuple(int(value) for value in subset),
        device=member_prior.device,
        dtype=torch.long,
    )
    priors = member_prior.index_select(0, subset_index)
    indices = member_indices.index_select(0, subset_index)
    radial = member_radial.index_select(0, subset_index)
    member_weight = radial.square().clamp_min(1e-8).reciprocal()
    member_weight = member_weight / member_weight.sum(
        dim=0, keepdim=True
    ).clamp_min(1e-30)
    consensus_indices, consensus_probability = dense_alpha_consensus_prior(
        priors,
        indices,
        member_weight,
        full_location.shape[2],
        int(construction_k),
        epsilon=registered_r2f.r2e_execution.CONSENSUS_EPSILON,
        alpha=registered_r2f.CONSENSUS_ALPHA,
    )
    if output_k < construction_k:
        consensus_indices = consensus_indices[:, :output_k]
        consensus_probability = consensus_probability[:, :output_k]
        consensus_probability = consensus_probability / consensus_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-30)

    locations = full_location.index_select(0, subset_index)
    scales = full_scale.index_select(0, subset_index)
    gather = consensus_indices[None, :, :, None].expand(
        len(subset), -1, -1, 2
    )
    selected_location = torch.gather(locations, 2, gather)
    selected_scale = torch.gather(scales, 2, gather)
    joint = (
        member_weight.transpose(0, 1)[:, :, None]
        * consensus_probability[:, None, :]
    )
    rows = len(consensus_probability)
    return (
        joint.reshape(rows, -1),
        selected_location.permute(1, 0, 2, 3).reshape(rows, -1, 2),
        selected_scale.permute(1, 0, 2, 3).reshape(rows, -1, 2),
    )


def _endpoint_record(observed: np.ndarray, expected: np.ndarray) -> dict:
    absolute = np.abs(observed - expected)
    return {
        "r2f_a1_coordinate_macro_m": float(observed.mean()),
        "r2e_coordinate_macro_m": float(expected.mean()),
        "absolute_coordinate_macro_difference_m": float(
            abs(observed.mean() - expected.mean())
        ),
        "maximum_absolute_group_difference_m": float(
            absolute.max(initial=0.0)
        ),
        "mean_absolute_group_difference_m": float(absolute.mean()),
    }


def _assert_repeat_equal(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]
) -> dict:
    labels = ("probability", "location", "scale", "energy", "mae")
    checks = {
        label: bool(torch.equal(left, right))
        for label, left, right in zip(labels, first, second)
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError("deterministic dense consensus repeat check failed")
    return checks


def check_contract(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    require_result_absent: bool = True,
) -> dict:
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = registered_r2f.load_registry(registry_path)
    cache_path = amendment["bindings"]["development_cache"]["path"]
    cache = registered_r2f.load_cache(cache_path, registry=registry)
    result_path = Path(registry["allowed_outputs"]["development_result"])
    if require_result_absent and result_path.exists():
        raise FileExistsError("R2F development result already exists")
    if cache["construction_contract"].get(
        "old_twenty_member_tensor_prefix_reused_exactly"
    ) is not True:
        raise ValueError("R2F cache did not preserve the R2E tensor prefix")
    return {
        "adapter_protocol": AMENDMENT_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "cache_sha256": _sha256_file(cache_path),
        "members": int(cache["construction_contract"]["members"]),
        "rows": int(cache["construction_contract"]["rows"]),
        "r2e_prefix_tensor_parity": True,
        "scientific_change": False,
        "result_absent": not result_path.exists(),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


@torch.no_grad()
def evaluate(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    device_name: str = "cuda",
) -> dict:
    contract = check_contract(
        amendment_path=amendment_path, require_result_absent=True
    )
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = registered_r2f.load_registry(registry_path)
    cache_path = amendment["bindings"]["development_cache"]["path"]
    destination = Path(registry["allowed_outputs"]["development_result"]).resolve()
    registered_r2f._validate_output(registry, "development_result", destination)
    cache = registered_r2f.load_cache(cache_path, registry=registry)
    device = torch.device(device_name)

    with _deterministic_algorithms():
        beta_n = registry["frozen_predictive_law"]["beta_neural"]
        beta_p = registry["frozen_predictive_law"]["beta_prototype"]
        full_probability = registered_r2f._atomic_probability_panel(
            cache["member_neural_logits"].to(device),
            cache["prototype_distance"].to(device),
            beta_n,
            beta_p,
        )
        full_location = cache["member_locations"].to(device)
        full_scale = cache["member_scales"].to(device)
        target = cache["target_xy"].to(device)
        randomness = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in cache["randomness"].items()
        }
        rows = len(target)
        row_energy = {
            name: torch.zeros(rows, dtype=torch.float64)
            for name in registered_r2f.VARIANTS
        }
        row_mae = {
            name: torch.zeros(rows, dtype=torch.float64)
            for name in registered_r2f.VARIANTS
        }
        retained_audit = {}

        # Candidate results use the complete 25-member panel.  Every subset is
        # evaluated through the same deterministic dense definition.
        for L in registered_r2f.SUPPORT_L_GRID:
            member_prior, member_indices, member_radial, retained_mass = (
                registered_r2f._member_top_l(
                    full_probability, full_location, full_scale, L
                )
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
            for E in registered_r2f.MEMBER_GRID:
                subsets = registered_r2f.balanced_member_subsets(E)
                for subset in subsets:
                    probability, location, scale = _dense_projected_joint_law(
                        member_prior,
                        member_indices,
                        member_radial,
                        full_location,
                        full_scale,
                        subset,
                        construction_k=L,
                        output_k=registered_r2f.SUPPORT_K,
                    )
                    energy, mae = registered_r2f._score_joint_law(
                        probability, location, scale, target, randomness
                    )
                    name = registered_r2f.variant_name(E, L)
                    divisor = float(len(subsets))
                    row_energy[name] += energy / divisor
                    row_mae[name] += mae / divisor
                print(
                    json.dumps(
                        {
                            "completed_L": L,
                            "completed_E": E,
                            "balanced_subsets": len(subsets),
                            "K": registered_r2f.SUPPORT_K,
                            "consensus_execution": "deterministic_dense_definition",
                        }
                    )
                )

        # Continuity is recomputed on the exact old 20-member prefix.  L=595
        # follows the old max-K then project path and remains a score-level
        # parity gate.  The legacy sparse L=192 energy draw is diagnostic only;
        # its exact-mixture-mean MAE remains an effective parity endpoint.
        old_probability = registered_r2f.r2e_execution._atomic_probability_panel(
            cache["member_neural_logits"][: registered_r2f.BASE_MEMBER_COUNT].to(
                device
            ),
            cache["prototype_distance"].to(device),
            beta_n,
            beta_p,
        )
        old_location = full_location[: registered_r2f.BASE_MEMBER_COUNT]
        old_scale = full_scale[: registered_r2f.BASE_MEMBER_COUNT]
        continuity_rows = {}
        repeatability = {}
        old_subset = tuple(range(registered_r2f.BASE_MEMBER_COUNT))
        for L in registered_r2f.SUPPORT_L_GRID:
            prior, indices, radial, _mass = registered_r2f._member_top_l(
                old_probability, old_location, old_scale, L
            )
            first_law = _dense_projected_joint_law(
                prior,
                indices,
                radial,
                old_location,
                old_scale,
                old_subset,
                construction_k=L,
                output_k=registered_r2f.SUPPORT_K,
            )
            first_energy, first_mae = registered_r2f._score_joint_law(
                *first_law, target, randomness
            )
            second_law = _dense_projected_joint_law(
                prior,
                indices,
                radial,
                old_location,
                old_scale,
                old_subset,
                construction_k=L,
                output_k=registered_r2f.SUPPORT_K,
            )
            second_energy, second_mae = registered_r2f._score_joint_law(
                *second_law, target, randomness
            )
            reference_name = (
                registered_r2f.R2E_COMPACT_REFERENCE
                if L == 192
                else registered_r2f.R2E_GLOBAL_REFERENCE
            )
            repeatability[reference_name] = _assert_repeat_equal(
                (*first_law, first_energy, first_mae),
                (*second_law, second_energy, second_mae),
            )
            continuity_rows[reference_name] = {
                "energy": first_energy,
                "mae": first_mae,
            }

    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    r2e_result = json.loads(
        Path(registry["bindings"]["r2e_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    continuity_endpoints = {}
    for reference_name, endpoint_rows in continuity_rows.items():
        continuity_endpoints[reference_name] = {}
        for endpoint, result_key in (
            ("energy", "energy_score_2d_m"),
            ("mae", "exact_mixture_mean_mae_m"),
        ):
            observed = registered_r2f._group_vector(
                endpoint_rows[endpoint], group_ids, group_count
            ).numpy()
            expected = np.asarray(
                r2e_result["coordinate_group_vectors"][result_key][reference_name],
                dtype=np.float64,
            )
            continuity_endpoints[reference_name][endpoint] = _endpoint_record(
                observed, expected
            )

    tolerance = registered_r2f.R2E_PARITY_TOLERANCE_M
    effective_checks = {
        "r2e_tensor_prefix_bitwise_equal": True,
        "compact_exact_mixture_mean_macro_within_tolerance": (
            continuity_endpoints[registered_r2f.R2E_COMPACT_REFERENCE]["mae"]
            ["absolute_coordinate_macro_difference_m"]
            <= tolerance
        ),
        "global_energy_macro_within_tolerance": (
            continuity_endpoints[registered_r2f.R2E_GLOBAL_REFERENCE]["energy"]
            ["absolute_coordinate_macro_difference_m"]
            <= tolerance
        ),
        "global_exact_mixture_mean_macro_within_tolerance": (
            continuity_endpoints[registered_r2f.R2E_GLOBAL_REFERENCE]["mae"]
            ["absolute_coordinate_macro_difference_m"]
            <= tolerance
        ),
        "dense_repeatability_exact": all(
            record["passed"] for record in repeatability.values()
        ),
    }
    continuity_gate = {
        "status": "amended_after_legacy_sparse_cuda_gate_failed_without_result",
        "reference_variants": [
            registered_r2f.R2E_COMPACT_REFERENCE,
            registered_r2f.R2E_GLOBAL_REFERENCE,
        ],
        "endpoints": continuity_endpoints,
        "repeatability": repeatability,
        "effective_checks": effective_checks,
        "tolerance_m": tolerance,
        "compact_energy_status": (
            "reported_diagnostic_only_because_the_bound_R2E_sparse_CUDA_score_"
            "is_not_bitwise_reproducible"
        ),
        "passed": all(effective_checks.values()),
    }
    if not continuity_gate["passed"]:
        raise RuntimeError("R2F-A1 failed its deterministic continuity gate")

    metrics = {}
    energy_groups = {}
    mae_groups = {}
    for name in registered_r2f.VARIANTS:
        energy_record, energy_group = registered_r2f._metric_record(
            row_energy[name], group_ids, group_count
        )
        mae_record, mae_group = registered_r2f._metric_record(
            row_mae[name], group_ids, group_count
        )
        metrics[name] = {
            "energy_score_2d_m": energy_record,
            "exact_mixture_mean_mae_m": mae_record,
        }
        energy_groups[name] = energy_group
        mae_groups[name] = mae_group

    r2c = json.loads(
        Path(registry["bindings"]["r2c_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    historical_mae = np.asarray(
        r2c["coordinate_group_vectors"]["exact_mixture_mean_mae_m"]
        [registered_r2f.HISTORICAL_REFERENCE],
        dtype=np.float64,
    )
    mae_family = registered_r2f.studentized_max_t_family(
        {
            name: mae_groups[name] - historical_mae
            for name in registered_r2f.VARIANTS
        },
        repetitions=registered_r2f.BOOTSTRAP_REPETITIONS,
        seed=registered_r2f.BOOTSTRAP_SEED,
        confidence=registered_r2f.CONFIDENCE,
        orientation="candidate_minus_frozen_R2C_balanced_historical_configuration",
    )
    decision, energy_family = registered_r2f._select_information_budget(
        metrics, energy_groups, mae_family
    )
    result = {
        "protocol_version": registered_r2f.PROTOCOL_VERSION,
        "runtime_protocol": AMENDMENT_PROTOCOL,
        "registry_sha256": registry["registry_sha256"],
        "cache_sha256": _sha256_file(cache_path),
        "execution_amendment": {
            "amendment_sha256": amendment["amendment_sha256"],
            "scientific_change": False,
            "selection_authority_for_v17_r2f": False,
            "consensus_execution": "deterministic_dense_executable_definition",
            "legacy_failed_attempt_created_result": False,
        },
        "metrics": metrics,
        "coordinate_group_vectors": {
            "energy_score_2d_m": {
                name: energy_groups[name].tolist()
                for name in registered_r2f.VARIANTS
            },
            "exact_mixture_mean_mae_m": {
                name: mae_groups[name].tolist()
                for name in registered_r2f.VARIANTS
            },
        },
        "simultaneous_inference": {
            "mae_candidate_minus_frozen_historical": mae_family,
            "energy_candidate_minus_eligible_global_minimum": energy_family,
        },
        "registered_decision": decision,
        "r2e_continuity_gate": continuity_gate,
        "retained_mass_diagnostic": retained_audit,
        "finite_member_support_sentinel": {
            "anchors": registered_r2f.ANCHOR_COUNT,
            "L": registered_r2f.ANCHOR_COUNT,
            "member_projection_truncation": False,
            "consensus_K": registered_r2f.SUPPORT_K,
        },
        "development_scan_predictions": rows,
        "registered_configurations": len(registered_r2f.VARIANTS),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def check_result(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
) -> dict:
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = registered_r2f.load_registry(registry_path)
    source = Path(registry["allowed_outputs"]["development_result"])
    result = json.loads(source.read_text(encoding="utf-8"))
    registered_check = registered_r2f.check_result(
        registry_path=registry_path, result_path=source
    )
    execution = result.get("execution_amendment", {})
    if result.get("runtime_protocol") != AMENDMENT_PROTOCOL:
        raise ValueError("R2F-A1 runtime protocol changed in result")
    if execution.get("amendment_sha256") != amendment["amendment_sha256"]:
        raise ValueError("R2F-A1 result amendment binding changed")
    if execution.get("scientific_change") is not False:
        raise ValueError("R2F-A1 result claims a scientific change")
    if result["r2e_continuity_gate"].get("passed") is not True:
        raise ValueError("R2F-A1 deterministic continuity gate failed")
    return {
        "runtime_protocol": AMENDMENT_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "result_sha256": result["result_sha256"],
        "registered_decision": registered_check["registered_decision"],
        "continuity_gate": result["r2e_continuity_gate"],
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-contract")
    check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    score = subparsers.add_parser("evaluate")
    score.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    score.add_argument("--device", default="cuda")
    result_check = subparsers.add_parser("check-result")
    result_check.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    args = parser.parse_args(argv)
    if args.command == "check-contract":
        result = check_contract(amendment_path=args.amendment)
    elif args.command == "evaluate":
        result = evaluate(
            amendment_path=args.amendment,
            device_name=args.device,
        )
    elif args.command == "check-result":
        result = check_result(amendment_path=args.amendment)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
