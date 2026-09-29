"""Final deterministic execution amendment for the frozen V17-R2F experiment.

R2F-A1 established a deterministic dense execution, but incorrectly retained
score-level parity to historical Monte Carlo energy artifacts.  A fixed
categorical CRN estimator is discontinuous in microscopic probability changes,
so that is not a valid numerical-continuity criterion.  A2 instead audits the
predictive operator itself on every development row, retains analytic mixture-
mean parity, and requires exact repeatability of the deterministic score.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as registered_r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f_runtime_a1 as runtime_a1
from fhm_dapp_tim.fhm_dapp.consensus import (
    alpha_consensus_prior,
    dense_alpha_consensus_prior,
)
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
    / "amendment_v17_r2f_a2_operator_continuity.json"
)
AMENDMENT_PROTOCOL = "uji_v17_r2f_runtime_a2_operator_continuity_v1_20260822"
LAW_EQUIVALENCE_ATOL = 1e-6
LAW_AUDIT_CHUNK_SIZE = 64


def _load_amendment(path: str | Path = DEFAULT_AMENDMENT) -> dict:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("amendment_sha256") != _canonical_json_sha256(
        payload, omit="amendment_sha256"
    ):
        raise ValueError("V17-R2F-A2 amendment self-hash failed")
    if payload.get("adapter_protocol") != AMENDMENT_PROTOCOL:
        raise ValueError("V17-R2F-A2 amendment protocol changed")
    if payload.get("scientific_change") is not False:
        raise ValueError("V17-R2F-A2 must remain scientifically neutral")
    if payload.get("selection_authority_for_v17_r2f") is not False:
        raise ValueError("V17-R2F-A2 cannot redefine R2F selection")
    if payload.get("may_modify_v17_r2f_model") is not False:
        raise ValueError("V17-R2F-A2 cannot modify the R2F model")
    if payload.get("operator_equivalence_atol") != LAW_EQUIVALENCE_ATOL:
        raise ValueError("V17-R2F-A2 law-equivalence tolerance changed")

    for field in (
        "runtime_source",
        "test_source",
        "amendment_note",
        "a1_runtime_source",
        "a1_test_source",
        "a1_amendment_note",
        "a1_amendment",
        "registered_execution_source",
        "registered_protocol_source",
        "registered_design",
        "registered_registry",
        "development_cache",
    ):
        record = payload["bindings"][field]
        if _sha256_file(record["path"]) != record["file_sha256"]:
            raise ValueError(f"V17-R2F-A2 binding changed: {field}")
    if Path(payload["bindings"]["runtime_source"]["path"]).resolve() != Path(
        __file__
    ).resolve():
        raise ValueError("V17-R2F-A2 runtime path changed")

    registry_record = payload["bindings"]["registered_registry"]
    registry = registered_r2f.load_registry(registry_record["path"])
    if registry.get("registry_sha256") != registry_record["registry_sha256"]:
        raise ValueError("V17-R2F-A2 registry identity changed")
    if (
        Path(payload["bindings"]["development_cache"]["path"]).resolve()
        != Path(registry["allowed_outputs"]["development_cache"]).resolve()
    ):
        raise ValueError("V17-R2F-A2 cache path differs from the registry")
    return payload


def _operator_equivalence_audit(
    member_prior: torch.Tensor,
    member_indices: torch.Tensor,
    member_radial: torch.Tensor,
    *,
    anchor_count: int,
    top_k: int,
    chunk_size: int = LAW_AUDIT_CHUNK_SIZE,
    atol: float = LAW_EQUIVALENCE_ATOL,
) -> dict:
    """Compare sparse optimization and dense definition as anchor measures.

    The comparison runs on CPU in bounded chunks, avoiding CUDA atomic-add
    order.  Top-K order itself is not treated as identity because equal-weight
    boundary anchors may permute; both results are scattered onto global anchor
    IDs before the probability-measure difference is computed.
    """

    priors = member_prior.detach().cpu()
    indices = member_indices.detach().cpu()
    radial = member_radial.detach().cpu()
    rows = priors.shape[1]
    maximum_probability_difference = 0.0
    maximum_row_l1_difference = 0.0
    index_mismatch_rows = 0
    for start in range(0, rows, int(chunk_size)):
        stop = min(start + int(chunk_size), rows)
        chunk_prior = priors[:, start:stop]
        chunk_indices = indices[:, start:stop]
        chunk_radial = radial[:, start:stop]
        member_weight = chunk_radial.square().clamp_min(1e-8).reciprocal()
        member_weight = member_weight / member_weight.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-30)
        sparse_indices, sparse_probability = alpha_consensus_prior(
            chunk_prior,
            chunk_indices,
            member_weight,
            int(anchor_count),
            int(top_k),
            epsilon=registered_r2f.r2e_execution.CONSENSUS_EPSILON,
            alpha=registered_r2f.CONSENSUS_ALPHA,
        )
        dense_indices, dense_probability = dense_alpha_consensus_prior(
            chunk_prior,
            chunk_indices,
            member_weight,
            int(anchor_count),
            int(top_k),
            epsilon=registered_r2f.r2e_execution.CONSENSUS_EPSILON,
            alpha=registered_r2f.CONSENSUS_ALPHA,
        )
        index_mismatch_rows += int(
            (sparse_indices != dense_indices).any(dim=1).sum()
        )
        sparse_measure = torch.zeros(
            stop - start,
            int(anchor_count),
            dtype=sparse_probability.dtype,
        ).scatter_(1, sparse_indices, sparse_probability)
        dense_measure = torch.zeros_like(sparse_measure).scatter_(
            1, dense_indices, dense_probability
        )
        absolute = (sparse_measure - dense_measure).abs()
        maximum_probability_difference = max(
            maximum_probability_difference, float(absolute.max())
        )
        maximum_row_l1_difference = max(
            maximum_row_l1_difference,
            float(absolute.sum(dim=1).max()),
        )
    result = {
        "execution_device": "cpu",
        "rows": int(rows),
        "chunk_size": int(chunk_size),
        "index_mismatch_rows_due_to_equal_or_near_equal_top_k_order": int(
            index_mismatch_rows
        ),
        "maximum_anchor_probability_difference": maximum_probability_difference,
        "maximum_row_l1_difference": maximum_row_l1_difference,
        "absolute_tolerance": float(atol),
        "passed": maximum_probability_difference <= float(atol),
    }
    if not result["passed"]:
        raise RuntimeError("sparse and dense alpha-consensus laws are not equivalent")
    return result


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
        "operator_equivalence_atol": LAW_EQUIVALENCE_ATOL,
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
    check_contract(amendment_path=amendment_path, require_result_absent=True)
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registered_registry"]["path"]
    registry = registered_r2f.load_registry(registry_path)
    cache_path = amendment["bindings"]["development_cache"]["path"]
    destination = Path(registry["allowed_outputs"]["development_result"]).resolve()
    registered_r2f._validate_output(registry, "development_result", destination)
    cache = registered_r2f.load_cache(cache_path, registry=registry)
    device = torch.device(device_name)

    with runtime_a1._deterministic_algorithms():
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
                    probability, location, scale = (
                        runtime_a1._dense_projected_joint_law(
                            member_prior,
                            member_indices,
                            member_radial,
                            full_location,
                            full_scale,
                            subset,
                            construction_k=L,
                            output_k=registered_r2f.SUPPORT_K,
                        )
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
        compact_operator_audit = None
        old_subset = tuple(range(registered_r2f.BASE_MEMBER_COUNT))
        for L in registered_r2f.SUPPORT_L_GRID:
            prior, indices, radial, _mass = registered_r2f._member_top_l(
                old_probability, old_location, old_scale, L
            )
            if L == 192:
                compact_operator_audit = _operator_equivalence_audit(
                    prior,
                    indices,
                    radial,
                    anchor_count=registered_r2f.ANCHOR_COUNT,
                    top_k=registered_r2f.SUPPORT_K,
                )
            first_law = runtime_a1._dense_projected_joint_law(
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
            second_law = runtime_a1._dense_projected_joint_law(
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
            repeatability[reference_name] = runtime_a1._assert_repeat_equal(
                (*first_law, first_energy, first_mae),
                (*second_law, second_energy, second_mae),
            )
            continuity_rows[reference_name] = {
                "energy": first_energy,
                "mae": first_mae,
            }

    if compact_operator_audit is None:
        raise RuntimeError("compact operator-equivalence audit was not run")
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
            continuity_endpoints[reference_name][endpoint] = (
                runtime_a1._endpoint_record(observed, expected)
            )

    tolerance = registered_r2f.R2E_PARITY_TOLERANCE_M
    effective_checks = {
        "r2e_tensor_prefix_bitwise_equal": True,
        "compact_sparse_dense_probability_measure_equivalent": (
            compact_operator_audit["passed"]
        ),
        "compact_exact_mixture_mean_macro_within_tolerance": (
            continuity_endpoints[registered_r2f.R2E_COMPACT_REFERENCE]["mae"]
            ["absolute_coordinate_macro_difference_m"]
            <= tolerance
        ),
        "global_exact_mixture_mean_macro_within_tolerance": (
            continuity_endpoints[registered_r2f.R2E_GLOBAL_REFERENCE]["mae"]
            ["absolute_coordinate_macro_difference_m"]
            <= tolerance
        ),
        "dense_probability_and_score_repeatability_exact": all(
            record["passed"] for record in repeatability.values()
        ),
    }
    continuity_gate = {
        "status": "operator_level_continuity_after_two_aborted_score_gate_attempts",
        "reference_variants": [
            registered_r2f.R2E_COMPACT_REFERENCE,
            registered_r2f.R2E_GLOBAL_REFERENCE,
        ],
        "operator_equivalence": compact_operator_audit,
        "endpoints": continuity_endpoints,
        "repeatability": repeatability,
        "effective_checks": effective_checks,
        "mae_macro_tolerance_m": tolerance,
        "historical_energy_endpoints_status": (
            "reported_diagnostic_only_because_fixed_CRN_categorical_MC_scores_"
            "are_discontinuous_under_microscopic_equivalent_probability_changes"
        ),
        "passed": all(effective_checks.values()),
    }
    if not continuity_gate["passed"]:
        raise RuntimeError("R2F-A2 failed its operator-level continuity gate")

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
            "operator_continuity_not_historical_MC_identity": True,
            "legacy_failed_attempts_created_result": False,
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
        raise ValueError("R2F-A2 runtime protocol changed in result")
    if execution.get("amendment_sha256") != amendment["amendment_sha256"]:
        raise ValueError("R2F-A2 result amendment binding changed")
    if execution.get("scientific_change") is not False:
        raise ValueError("R2F-A2 result claims a scientific change")
    if result["r2e_continuity_gate"].get("passed") is not True:
        raise ValueError("R2F-A2 operator-level continuity gate failed")
    if result["r2e_continuity_gate"]["operator_equivalence"].get("passed") is not True:
        raise ValueError("R2F-A2 operator-equivalence audit failed")
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
