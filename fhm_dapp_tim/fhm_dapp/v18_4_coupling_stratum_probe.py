"""Final write-once V18.4 coupling/stratum diagnostic; frozen inference only.

Three registered checkpoint/role packages, eight registered temperatures and
thirteen correction cells per temperature. Alternative geometry energies are
oracle diagnostics only and never feed correction or empirical references.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch

from . import v18_4_mechanism_probe as previous
from .coherent_marked_v9 import common_student_t_randomness, continuous_energy_score_mc
from .uji_v18_4_development import (
    CALIBRATION_TEMPERATURE_GRID, MODEL_SEED, PREDICTIVE_DRAWS, SCORING_SEED,
    _coordinate_group_ids, _load_model_checkpoint, _load_registered_data,
    _prediction_entropy, _reference_from_payload, _row_nll, _set_seed,
)
from .v18_4_core import (
    kl_p_to_q, normalize_probability, reliability_bounded_posterior_correction,
    weighted_component_geometric_median,
)
from .v18_4_model import (
    V184_CONFIG, _masked_geometry_energy, apply_correction,
    raw_correction_energies, v184_distribution,
)

REPO_ROOT = previous.REPO_ROOT
EVIDENCE = previous.EVIDENCE
FROZEN_SOURCE_SHA = previous.FROZEN_SOURCE_SHA
PROBE_VERSION = "v18_4_final_read_only_coupling_stratum_probe_v1"
TEMPERATURES = (.50, .75, 1., 1.25, 1.50, 2., 2.50, 3.)
EPSILONS = (0., .025, .050, .100)
MODES = ("none", "radio_only", "geometry_only", "full")
METRICS = ("energy_score_m", "nll", "posterior_mean_mae_m")
GEOMETRIES = ("G1_anchor_conditional", "G2_component_conditional",
              "G3_anchor_mass_preserving", "G4_component_mass_preserving")
PRIMARY = ("fit_development", "refit_calibration")
SEMANTIC = {"fit_development": 1., "refit_calibration": 1.5}
KL_TOLERANCE = previous.KL_TOLERANCE
# Fixed before execution: same CUDA functions/order, aggregate comparison.
REPRODUCTION_ATOL = 1e-6
REPRODUCTION_RTOL = 1e-7
OUTPUT_JSON = EVIDENCE / "COUPLING_STRATUM_PROBE_V18_4.json"
OUTPUT_MD = EVIDENCE / "COUPLING_STRATUM_PROBE_V18_4.md"
EXECUTION_JSON = EVIDENCE / "run_audit/COUPLING_STRATUM_PROBE_EXECUTION_V18_4.json"
EXECUTION_MD = EVIDENCE / "run_audit/COUPLING_STRATUM_PROBE_EXECUTION_V18_4.md"
OUTPUTS = (OUTPUT_JSON, OUTPUT_MD, EXECUTION_JSON, EXECUTION_MD)
MODULE = "fhm_dapp_tim/fhm_dapp/v18_4_coupling_stratum_probe.py"
TEST_MODULE = "tests/test_v18_4_coupling_stratum_probe.py"
TEST_FILES = (*previous.TEST_FILES, TEST_MODULE)
PRIOR_EXPECTED = dict(zip((p.relative_to(REPO_ROOT).as_posix() for p in previous.OUTPUTS), (
    "790b667892d4b21ec8eb062d6fc27c9cf6d63196d680d300648bf94c1a983d67",
    "31affba0655b847e0c2fd742169a4355eb1657c765ea72db6e3577c0be90b627",
    "591c47388d2ba8588b6b6cdaa751af31b40dd2c17430e276aa73bd0ebe9cad40",
    "31332d041b50dd650d3349362f0dd48a0264e35ff00fec3a46272e9e7f19767e",
)))
PRIOR_CANONICAL = "75bd468302f3defdcb845d27bdc176716267ba0edb3d73d04bcd76d33ccfc1aa"
SCENARIOS = (
    {"name": "fit_development", "role": "development", "reference_role": "fit",
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_lambda_0p10.pt"},
    {"name": "fit_calibration", "role": "calibration", "reference_role": "fit",
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_lambda_0p10.pt"},
    {"name": "refit_calibration", "role": "calibration", "reference_role": "selection_refit",
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_selection_refit.pt"},
)
require = previous.require
assert_finite = previous.assert_finite
sha256_file = previous.sha256_file
seal = previous.seal
json_bytes = previous.json_bytes
write_once = previous.write_once
assert_unused = previous.assert_unused
tensor_state_hash = previous.tensor_state_hash
randomness_hash = previous.randomness_hash
ReadOnlyModelGuard = previous.ReadOnlyModelGuard
table = previous.markdown_table


class FileOpenAudit(previous.FileOpenAudit):
    """Inherited dataset guard plus a ban on writes to every source/test."""

    def __init__(self, root: Path = REPO_ROOT):
        super().__init__(root)
        self.allowed_outputs = {previous.normalized_path(
            self.root / path.relative_to(REPO_ROOT)) for path in OUTPUTS}

    def check(self, path, mode=None, flags: int = 0):
        if not isinstance(path, int):
            full = previous.normalized_path(path)
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and Path(full).suffix.lower() == ".py" and full.startswith(
                    previous.normalized_path(self.root) + os.sep):
                self.denied.append({"path": full, "reason": "all source/test writes forbidden"})
                raise RuntimeError("file-open audit denied source/test write forbidden: " + full)
        super().check(path, mode, flags)


def validate_prior_payloads(prior: Mapping, execution: Mapping) -> dict:
    require(previous.canonical_hash(prior, "result_sha256") == prior.get("result_sha256") == PRIOR_CANONICAL,
            "prior result canonical hash mismatch")
    require(previous.canonical_hash(execution, "execution_record_sha256") == execution.get("execution_record_sha256"),
            "prior execution canonical hash mismatch")
    require(execution.get("status") == "PASS" and execution.get("probe_exit_code") == 0,
            "prior execution did not PASS")
    require(execution.get("result_self_hash") == PRIOR_CANONICAL, "prior execution/result link mismatch")
    for payload in (prior, execution):
        require(payload.get("frozen_source_commit_sha") == FROZEN_SOURCE_SHA, "prior frozen source link mismatch")
    for path in previous.OUTPUTS[:2]:
        rel = path.relative_to(REPO_ROOT).as_posix()
        require(execution["output_sha256"].get(rel) == PRIOR_EXPECTED[rel], "prior execution/output byte link mismatch")
    require(all(prior.get(key) is False for key in previous.BOUNDARIES), "prior probe boundary flags mismatch")
    return {"status": "PASS", "file_byte_sha256": dict(PRIOR_EXPECTED),
            "result_sha256": PRIOR_CANONICAL, "execution_record_sha256": execution["execution_record_sha256"],
            "execution_status": "PASS", "canonical_hashes_and_output_cross_links_verified": True}


def verify_integrity() -> tuple[dict, dict, dict]:
    integrity, registry = previous.verify_integrity()
    for relative, expected in PRIOR_EXPECTED.items():
        require(sha256_file(REPO_ROOT / relative) == expected, "prior file-byte hash mismatch: " + relative)
    prior = json.loads(previous.OUTPUT_JSON.read_text(encoding="utf-8"))
    execution = json.loads(previous.EXECUTION_JSON.read_text(encoding="utf-8"))
    integrity["prior_mechanism_validation"] = validate_prior_payloads(prior, execution)
    extra = {}
    for relative in ("fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py", "tests/test_v18_4_mechanism_probe.py"):
        expected = execution["gate"]["tested_file_sha256"][relative]
        require(sha256_file(REPO_ROOT / relative) == expected, "prior tested source changed: " + relative)
        extra[relative] = expected
    # Preserve all pre-existing Python source/tests, including those beyond the
    # formal frozen list. This does not read a dataset or an external result.
    for directory in (REPO_ROOT / "fhm_dapp_tim", REPO_ROOT / "tests"):
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative not in (MODULE, TEST_MODULE) and "__pycache__" not in path.parts:
                extra[relative] = sha256_file(path)
    integrity["all_pre_existing_python_sha256"] = extra
    require(tuple(CALIBRATION_TEMPERATURE_GRID) == TEMPERATURES, "registered temperature grid changed")
    require(tuple(float(x) for x in V184_CONFIG["epsilon_kl_grid"]) == EPSILONS[1:], "registered epsilon grid changed")
    return integrity, registry, prior


def cell_key(temperature: float, mode: str, epsilon: float) -> str:
    return f"T{temperature:g}|{mode}|eps{epsilon:g}"


def cell_definitions() -> list[dict]:
    return [{"temperature": t, "mode": mode, "epsilon_kl": eps, "scale_multiplier": 1.}
            for t in TEMPERATURES for mode in MODES
            for eps in ((0.,) if mode == "none" else EPSILONS)]


def assert_surface_complete(cells: Mapping) -> None:
    expected = {cell_key(c["temperature"], c["mode"], c["epsilon_kl"]) for c in cell_definitions()}
    require(len(cells) == 104 and set(cells) == expected, "missing/extra/duplicate surface cell")


def surface_posteriors(q: torch.Tensor, mask: torch.Tensor, production: Mapping) -> dict:
    result = {}
    zero_energy, zero_reliability = torch.zeros_like(q), torch.zeros(len(q), device=q.device, dtype=q.dtype)
    zero = torch.zeros(len(q), device=q.device, dtype=torch.float64)
    result[("none", 0.)] = {"posterior": q, "rho": zero, "realized_kl": zero,
                            "epsilon_kl_effective": zero, "radio_reliability": zero_reliability,
                            "geometry_reliability": zero_reliability}
    for mode in MODES[1:]:
        er = production["radio_energy"] if mode != "geometry_only" else zero_energy
        eg = production["geometry_energy"] if mode != "radio_only" else zero_energy
        rr = production["radio_reliability"] if mode != "geometry_only" else zero_reliability
        rg = production["geometry_reliability"] if mode != "radio_only" else zero_reliability
        for eps in EPSILONS:
            correction = dict(production) if mode == "full" and eps == .025 else reliability_bounded_posterior_correction(
                q, er, eg, radio_reliability=rr, geometry_reliability=rg, epsilon_kl=eps, mask=mask)
            correction.update(radio_energy=er, geometry_energy=eg)
            result[(mode, eps)] = correction
            if mode in ("radio_only", "geometry_only"):
                inactive = "geometry" if mode == "radio_only" else "radio"
                require(bool((correction[inactive + "_energy"] == 0).all()) and
                        bool((correction[inactive + "_reliability"] == 0).all()), "inactive source was not zero")
            require(torch.equal(correction["epsilon_kl_effective"], torch.maximum(rr, rg).double() * eps),
                    "effective production reliability budget mismatch")
    require(len(result) == 13, "posterior surface is incomplete")
    require(bool((q[~mask] == 0).all()), "q_T padding received mass")
    for (mode, eps), correction in result.items():
        assert_finite(correction, f"{mode}@{eps}")
        p = correction["posterior"]
        require(bool((p >= 0).all()) and bool((p[~mask] == 0).all()), "invalid posterior/padding mass")
        require(bool(torch.allclose(p.double().sum(1), torch.ones(len(p), device=p.device, dtype=torch.float64),
                                   atol=4 * torch.finfo(p.dtype).eps, rtol=0)), "posterior not normalized")
        actual_kl = kl_p_to_q(p.double(), q.double(), mask).clamp(min=0)
        require(bool((actual_kl <= correction["epsilon_kl_effective"] + KL_TOLERANCE).all()), "KL budget violated")
        require(bool(torch.allclose(actual_kl, correction["realized_kl"], atol=1e-12, rtol=1e-9)), "reported KL mismatch")
        if eps == 0:
            require(torch.equal(p, q), "epsilon=0 identity failed")
            require(bool((correction["realized_kl"] == 0).all()) and bool((correction["rho"] == 0).all()),
                    "zero-budget KL/rho not zero")
    require(torch.equal(result[("full", .025)]["posterior"], production["posterior"]), "full production differs")
    return result


def sign_classification(values: Sequence[float]) -> dict:
    require(len(values) == 8, "sign map needs all eight temperatures")
    assert_finite(values)
    classification = ("SIGN_STABLE_BENEFICIAL" if all(v < 0 for v in values) else
                      "SIGN_STABLE_HARMFUL" if all(v > 0 for v in values) else "TEMPERATURE_DEPENDENT")
    return {"deltas": list(values), "sign_vector": [int(np.sign(v)) for v in values], "classification": classification}


def coupling_maps(scenarios: Mapping) -> dict:
    reports, joint = {}, False
    for name, scenario in scenarios.items():
        rows = []
        for mode in MODES[1:]:
            for eps in EPSILONS[1:]:
                metrics = {metric: sign_classification([scenario["cells"][cell_key(t, mode, eps)]
                           ["paired_correction_effect"][metric]["mean_delta"] for t in TEMPERATURES]) for metric in METRICS}
                classes = {m["classification"] for m in metrics.values()}
                separable = len(classes) == 1 and "TEMPERATURE_DEPENDENT" not in classes
                classification = "SEPARABLE" if separable else "NONSEPARABLE_FROM_TEMPERATURE"
                if name in PRIMARY and mode in ("radio_only", "full") and not separable:
                    joint = True
                rows.append({"mode": mode, "epsilon_kl": eps, "metrics": metrics, "classification": classification})
        reports[name] = rows
    return {"temperature_order": list(TEMPERATURES), "scenarios": reports,
            "temperature_and_correction_require_joint_selection": joint,
            "separable_rule": "all three metrics stable in the same direction across all eight T"}


def intrinsically_informative(scenarios: Mapping) -> dict:
    reports = {}
    for name in PRIMARY:
        sources = scenarios[name]["source_informativeness"]
        correlations = [sources[f"{t:g}"]["radio"]["mean_row_spearman"] for t in TEMPERATURES]
        fractions = [sources[f"{t:g}"]["radio"]["fraction_oracle_gain_positive"] for t in TEMPERATURES]
        correlation, fraction = float(np.median(correlations)), float(np.median(fractions))
        reports[name] = {"median_across_T_mean_row_spearman": correlation,
                         "median_across_T_fraction_oracle_gain_positive": fraction,
                         "passes": correlation > .30 and fraction > .75}
    return {"primary_scenarios": reports, "RADIO_INTRINSICALLY_INFORMATIVE": all(r["passes"] for r in reports.values())}


def robust_common_region(scenarios: Mapping, mode: str) -> dict:
    require(mode in ("radio_only", "full"), "unsupported robustness gate mode")
    reports = []
    for eps in EPSILONS[1:]:
        beneficial = {}
        for name in PRIMARY:
            beneficial[name] = [t for t in TEMPERATURES if (
                scenarios[name]["cells"][cell_key(t, mode, eps)]["paired_correction_effect"]["energy_score_m"]["mean_delta"] < 0
                and scenarios[name]["cells"][cell_key(t, mode, eps)]["paired_correction_effect"]["nll"]["mean_delta"] < 0
                and Decimal(str(scenarios[name]["cells"][cell_key(t, mode, eps)]["paired_correction_effect"]
                                ["posterior_mean_mae_m"]["mean_delta"])) <= Decimal("0.05"))]
        common = [t for t in TEMPERATURES if all(t in beneficial[n] for n in PRIMARY)]
        adjacent = [[a, b] for a, b in zip(TEMPERATURES, TEMPERATURES[1:]) if a in common and b in common]
        reports.append({"epsilon_kl": eps, "beneficial_temperatures": beneficial,
                        "common_temperatures": common, "adjacent_common_pairs": adjacent,
                        "ROBUST_COMMON_REGION": bool(adjacent)})
    return {"mode": mode, "primary_scenarios": list(PRIMARY), "epsilon_reports": reports,
            "any_robust_common_region": any(r["ROBUST_COMMON_REGION"] for r in reports)}


def role_shift_decomposition(scenarios: Mapping) -> dict:
    s1, s2 = (scenarios[n] for n in ("fit_development", "fit_calibration"))
    for key in ("checkpoint_sha256", "reference_tensor_sha256", "memory_anchors", "lambda_spatial", "selected_epoch"):
        require(s1["definition"][key] == s2["definition"][key], "role shift changed frozen FIT package: " + key)
    records = {}
    for key, cell in s1["cells"].items():
        other = s2["cells"][key]
        metric_delta = {m: other["metrics"][m] - cell["metrics"][m] for m in
                        (*METRICS, "entropy_nats", "effective_anchor_count", "top1_probability")}
        gain_delta = {m: other["paired_correction_effect"][m]["mean_delta"] -
                        cell["paired_correction_effect"][m]["mean_delta"] for m in METRICS}
        records[key] = {"role_shift_metric_delta": metric_delta, "role_shift_in_correction_effect": gain_delta}
    return {"comparison": "S2 fit_calibration - S1 fit_development", "same_FIT_package_verified": True,
            "paired_statistical_comparison": False, "cells": records}


def refit_decomposition(scenarios: Mapping, group_cache: Mapping) -> dict:
    s2, s3 = (scenarios[n] for n in ("fit_calibration", "refit_calibration"))
    for key in ("row_index_sha256", "coordinate_group_order_sha256", "rows", "coordinate_groups"):
        require(s2["definition"][key] == s3["definition"][key], "refit comparison rows/groups differ: " + key)
    records = {}
    for key, cell in s2["cells"].items():
        none_key = cell_key(cell["temperature"], "none", 0.)
        metrics, gains = {}, {}
        for metric in METRICS:
            a, b = (group_cache[n][key][metric] for n in ("fit_calibration", "refit_calibration"))
            base_a, base_b = (group_cache[n][none_key][metric] for n in ("fit_calibration", "refit_calibration"))
            groups = list(range(len(a)))
            metrics[metric] = previous.paired_statistics(b, a, groups)
            gains[metric] = previous.paired_statistics(b - base_b, a - base_a, groups)
        records[key] = {"paired_refit_metric_delta": metrics, "paired_refit_effect_on_correction": gains}
    return {"comparison": "S3 refit_calibration - S2 fit_calibration", "same_rows_and_group_order_verified": True,
            "changed_package": ["learned model state", "anchor memory", "correction reference"], "cells": records}


def stratum_mass(q, mask, building, floor):
    require(q.shape == mask.shape == building.shape == floor.shape, "stratum shape mismatch")
    same = (building[:, :, None] == building[:, None, :]) & (floor[:, :, None] == floor[:, None, :])
    pair = same & mask[:, :, None] & mask[:, None, :]
    return (q[:, None, :] * pair.to(q.dtype)).sum(2)


def geometry_energy(q, locations, mask, building, floor, *, spatial_scale_m,
                    sigma_scale: float, margin_scale: float, conditional: bool):
    """Same production mask/kernel/margin law, optionally retain stratum mass."""
    prior = q.detach() * mask.to(q.dtype)
    prior = prior / prior.sum(1, keepdim=True).clamp(min=1e-12)
    if conditional:
        return _masked_geometry_energy(prior, locations, mask, building, floor,
            spatial_scale_m=spatial_scale_m, sigma_scale=sigma_scale, margin_scale=margin_scale)
    q = prior * mask.to(prior.dtype)
    q = q / q.sum(1, keepdim=True).clamp(min=1e-12)
    pair = mask[:, :, None] & mask[:, None, :] & (building[:, :, None] == building[:, None, :]) & (
        floor[:, :, None] == floor[:, None, :])
    spatial_scale = torch.as_tensor(spatial_scale_m, dtype=locations.dtype, device=locations.device).clamp(min=1e-8)
    sigma, margin = spatial_scale * sigma_scale, spatial_scale * margin_scale
    dist2 = torch.cdist(locations, locations).square()
    log_support = q[:, None, :].clamp(min=1e-30).log() - .5 * dist2 / sigma.square().clamp(min=1e-12)
    log_support = torch.where(pair, log_support, torch.full_like(log_support, -1e6))
    soft_distance = (-2 * sigma.square() * torch.logsumexp(log_support, 2)).clamp(min=0).sqrt()
    energy = (torch.relu(soft_distance - margin) / spatial_scale).square()
    energy = torch.where(mask, energy, torch.zeros_like(energy))
    assert_finite(energy, "diagnostic geometry")
    require(bool((energy >= 0).all()), "negative diagnostic geometry energy")
    return energy


def geometry_variants(q, anchors, components, mask, building, floor, *, spatial_scale_m, production_geometry):
    kwargs = {"spatial_scale_m": spatial_scale_m, "sigma_scale": float(V184_CONFIG["geometry_sigma_scale"]),
              "margin_scale": float(V184_CONFIG["geometry_margin_scale"])}
    result = {GEOMETRIES[0]: production_geometry}
    for name, coords, conditional in ((GEOMETRIES[1], components, True), (GEOMETRIES[2], anchors, False),
                                       (GEOMETRIES[3], components, False)):
        result[name] = geometry_energy(q, coords, mask, building, floor, conditional=conditional, **kwargs)
    assert_finite(result)
    return result


def distribution_summary(values, *, displacement=False) -> dict:
    values = previous.as_array(values)
    result = {"mean": float(values.mean()), "median": float(np.median(values))}
    for quantile in ((.75, .90, .95) if displacement else (.10, .25, .75, .90)):
        result[f"p{int(100 * quantile)}"] = float(np.quantile(values, quantile))
    if displacement:
        for threshold in (5, 10, 20):
            result[f"fraction_gt_{threshold}m"] = float((values > threshold).mean())
    return result


def geometry_stratum_audit(energy, q, mask, building, floor, anchor_distance,
                           component_distance, residual, candidate_indices) -> dict:
    q, energy = np.asarray(q), np.asarray(energy)
    mask = np.asarray(mask, dtype=bool)
    masses = np.zeros_like(q)
    row_records, selected, mapped = [], [], []
    for row in range(len(q)):
        valid = np.flatnonzero(mask[row])
        strata = {}
        for idx in valid:
            strata.setdefault((int(building[row, idx]), int(floor[row, idx])), []).append(int(idx))
        for indices in strata.values():
            masses[row, indices] = q[row, indices].sum()
        argmin = int(valid[np.argmin(energy[row, valid])])
        map_index = int(valid[np.argmax(q[row, valid])])
        selected.append(float(component_distance[row, argmin]))
        mapped.append(float(component_distance[row, map_index]))
        argmin_stratum = (int(building[row, argmin]), int(floor[row, argmin]))
        row_records.append({"role_row_offset": row, "argmin_support_index": argmin,
            "argmin_anchor_index": int(candidate_indices[row, argmin]), "argmin_geometry_energy": float(energy[row, argmin]),
            "argmin_stratum": list(argmin_stratum), "argmin_stratum_mass": float(masses[row, argmin]),
            "argmin_q_T_probability": float(q[row, argmin]),
            "argmin_stratum_is_q_T_MAP_stratum": argmin_stratum == (int(building[row, map_index]), int(floor[row, map_index])),
            "argmin_component_error_m": float(component_distance[row, argmin]),
            "argmin_anchor_error_m": float(anchor_distance[row, argmin]),
            "argmin_anchor_component_residual_m": float(residual[row, argmin]),
            "unique_building_floor_strata": len(strata),
            "strata_min_geometry_energy_le_1e_8": sum(float(energy[row, indices].min()) <= 1e-8 for indices in strata.values())})
    argmin_mass = previous.as_array([r["argmin_stratum_mass"] for r in row_records])
    selected, mapped = previous.as_array(selected), previous.as_array(mapped)
    low_mass = {}
    for threshold in (.01, .05, .10):
        subset = argmin_mass < threshold
        low_mass[f"{threshold:g}"] = {"fraction": float(subset.mean()), "row_count": int(subset.sum()),
            "mean_argmin_component_error_m": float(selected[subset].mean()) if subset.any() else None}
    return {"argmin_stratum_mass": distribution_summary(argmin_mass), "low_mass_subsets": low_mass,
            "fraction_argmin_in_q_T_MAP_stratum": float(np.mean([r["argmin_stratum_is_q_T_MAP_stratum"] for r in row_records])),
            "fraction_argmin_beats_q_T_MAP_component": float((selected < mapped).mean()),
            "mean_argmin_component_error_m": float(selected.mean()), "mean_q_T_MAP_component_error_m": float(mapped.mean()),
            "mean_unique_strata_per_support": float(np.mean([r["unique_building_floor_strata"] for r in row_records])),
            "mean_strata_with_near_zero_min_energy": float(np.mean([r["strata_min_geometry_energy_le_1e_8"] for r in row_records])),
            "row_records": row_records}


def geometry_hypothesis_gates(semantic: Mapping) -> dict:
    h1, h2 = {}, {}
    for name in PRIMARY:
        variants = semantic[name]["variants"]
        mass_pairs, coordinate_pairs = {}, {}
        for a, b in ((GEOMETRIES[0], GEOMETRIES[2]), (GEOMETRIES[1], GEOMETRIES[3])):
            old, new = variants[a], variants[b]
            error_old, error_new = map(lambda x: Decimal(str(x)),
                                      (old["mean_argmin_component_error_m"], new["mean_argmin_component_error_m"]))
            decrease = new["low_mass_subsets"]["0.05"]["fraction"] < old["low_mass_subsets"]["0.05"]["fraction"]
            improves = error_old > 0 and error_new <= Decimal("0.75") * error_old
            mass_pairs[b + "_vs_" + a] = {"argmin_error_improves_at_least_25_percent": improves,
                "low_mass_fraction_decreases": decrease, "passes": improves and decrease}
        for a, b in ((GEOMETRIES[0], GEOMETRIES[1]), (GEOMETRIES[2], GEOMETRIES[3])):
            old, new = variants[a], variants[b]
            correlation = Decimal(str(new["mean_row_spearman"])) - Decimal(str(old["mean_row_spearman"])) >= Decimal("0.05")
            error_old, error_new = map(lambda x: Decimal(str(x)),
                                      (old["mean_argmin_component_error_m"], new["mean_argmin_component_error_m"]))
            error = error_old > 0 and error_new <= Decimal("0.90") * error_old
            coordinate_pairs[b + "_vs_" + a] = {"spearman_improves_at_least_0p05": correlation,
                "argmin_error_improves_at_least_10_percent": error, "passes": correlation or error}
        h1[name] = {"pairs": mass_pairs, "passes": any(r["passes"] for r in mass_pairs.values())}
        h2[name] = {"pairs": coordinate_pairs, "passes": any(r["passes"] for r in coordinate_pairs.values())}
    mass_supported, mismatch_supported = all(r["passes"] for r in h1.values()), all(r["passes"] for r in h2.values())
    return {"H1": {"scenarios": h1, "supported": mass_supported, "conclusion":
                   "stratum_mass_loss_supported" if mass_supported else "stratum_mass_loss_not_clearly_supported"},
            "H2": {"scenarios": h2, "supported": mismatch_supported, "conclusion":
                   "anchor_component_mismatch_supported" if mismatch_supported else "anchor_component_mismatch_not_clearly_supported"}}


def row_surface_metrics(p, locations, scales, target, mask, model, shared_draws) -> dict:
    """One production ES and one production GM; mean is the sole point action."""
    es, _, _ = continuous_energy_score_mc(p, locations, scales, target, shared_draws)
    mean = (p[:, :, None] * locations).sum(1)
    gm = weighted_component_geometric_median(p, locations, mask=mask)
    entropy = _prediction_entropy(p)
    values = {"energy_score_m": es, "nll": _row_nll(p, locations, scales, target, model),
              "posterior_mean_mae_m": torch.linalg.vector_norm(mean - target, dim=1),
              "entropy_nats": entropy, "effective_anchor_count": entropy.exp(),
              "top1_probability": p.amax(1), "gm_displacement_m": torch.linalg.vector_norm(gm - mean, dim=1)}
    # Exact production tail mass formula; avoids its redundant second GM call.
    normalized = normalize_probability(p, mask)
    rows = torch.arange(len(p), device=p.device)
    mode_location = locations[rows, normalized.argmax(1)]
    mode_distance = torch.linalg.vector_norm(locations - mode_location[:, None, :], dim=2)
    truth_distance = torch.linalg.vector_norm(locations - target[:, None, :], dim=2)
    for radius in (10, 20, 40):
        values[f"truth_far_mass_{radius}m"] = (normalized * (truth_distance > radius).to(p.dtype)).sum(1)
        values[f"mode_far_mass_{radius}m"] = (normalized * (mode_distance > radius).to(p.dtype)).sum(1)
    assert_finite(values)
    return values


def summarize_cell(values: Mapping, groups: Sequence) -> dict:
    metrics = {}
    for name in METRICS:
        mean, se = previous.mean_se(previous.group_means(values[name], groups))
        metrics[name], metrics[name + "_se"] = mean, se
    point = previous.decoder_metrics(values["posterior_mean_mae_m"], groups)
    metrics.update({key: value for key, value in point.items() if key.startswith("row_")})
    for name in ("entropy_nats", "effective_anchor_count", "top1_probability", "realized_kl",
                 "epsilon_kl_effective", "rho", "radio_reliability", "geometry_reliability"):
        metrics[name] = float(values[name].mean())
    metrics["mean_gm_displacement_from_mean_m"] = float(values["gm_displacement_m"].mean())
    metrics["p90_gm_displacement_from_mean_m"] = float(np.quantile(values["gm_displacement_m"], .90))
    for name in values:
        if name.startswith(("truth_far_mass_", "mode_far_mass_")):
            metrics[name] = float(values[name].mean())
    return metrics


def compare_shared_cell(actual: Mapping, expected: Mapping, label: str) -> list[dict]:
    checks = []
    for metric, reference in expected.items():
        require(metric in actual, "shared-cell metric missing: " + metric)
        value = float(actual[metric])
        tolerance = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(float(reference))
        error = abs(value - float(reference))
        require(math.isfinite(value) and error <= tolerance, f"shared prior cell reproduction failed: {label}.{metric}")
        checks.append({"cell": label, "metric": metric, "expected": float(reference), "actual": value,
                       "absolute_error": error, "fixed_tolerance": tolerance, "passes": True})
    return checks


def prior_reproduction_checks(name: str, temperature: float, cells: Mapping, prior: Mapping) -> list[dict]:
    if name not in SEMANTIC or temperature != SEMANTIC[name]:
        return []
    prior_name = "development_native" if name == "fit_development" else "calibration_frozen"
    checks = []
    for mode in MODES:
        eps = 0. if mode == "none" else .025
        old = prior["scenarios"][prior_name]["modes"][mode]
        expected = {}
        for key, value in old["probabilistic_metrics"].items():
            mapped = key.removeprefix("coordinate_macro_").removeprefix("mean_")
            expected[mapped] = value
        for key, value in old["decoders"]["posterior_mean"].items():
            mapped = {"coordinate_macro_mae_m": "posterior_mean_mae_m",
                      "coordinate_macro_mae_se_m": "posterior_mean_mae_m_se"}.get(key, key)
            expected[mapped] = value
        expected.update({key: value for key, value in old["tail_diagnostics"].items()
                         if key.startswith(("truth_far_mass_", "mode_far_mass_")) or
                         key in ("mean_gm_displacement_from_mean_m", "p90_gm_displacement_from_mean_m")})
        key = cell_key(temperature, mode, eps)
        checks.extend(compare_shared_cell(cells[key]["metrics"], expected, name + ":" + key))
    if name == "fit_development":
        development = json.loads((EVIDENCE / "uji_development_v18_4.json").read_text(encoding="utf-8"))
        selected = development["selected"]
        expected = {metric: selected[metric] for metric in METRICS}
        checks.extend(compare_shared_cell(cells[cell_key(1., "full", .025)]["metrics"], expected, "formal_development.selected"))
    return checks


def mismatch_audit(geometry, q, mask, anchor_distance, component_distance, residual, stratum_report) -> dict:
    anchor = previous.source_energy_alignment(geometry, anchor_distance, q, mask)
    component = previous.source_energy_alignment(geometry, component_distance, q, mask)
    return {"mean_row_spearman_anchor_truth_distance": anchor["mean_row_spearman"],
            "mean_row_spearman_component_truth_distance": component["mean_row_spearman"],
            "component_minus_anchor_mean_spearman": component["mean_row_spearman"] - anchor["mean_row_spearman"],
            "argmin_anchor_error_m": anchor["source_argmin_candidate_mean_error_m"],
            "argmin_component_error_m": component["source_argmin_candidate_mean_error_m"],
            "residual_displacement_all_valid_candidates_m": distribution_summary(np.asarray(residual)[mask], displacement=True),
            "residual_displacement_geometry_argmin_m": distribution_summary(
                [r["argmin_anchor_component_residual_m"] for r in stratum_report["row_records"]], displacement=True)}


def run_scenario(scenario: Mapping, registry, data, roles, rss_all, device, authorization, prior) -> tuple[dict, dict]:
    started = time.perf_counter()
    rows = roles[scenario["role"]]
    groups = _coordinate_group_ids(data, rows)
    ordered_groups = list(dict.fromkeys(groups))
    require(len(rows) == registry["roles"][scenario["role"]]["row_count"] and
            len(ordered_groups) == registry["roles"][scenario["role"]]["coordinate_count"], "role row/group mismatch")
    checkpoint = torch.load(REPO_ROOT / scenario["checkpoint"], map_location="cpu", weights_only=False)
    reference_validation = previous.validate_checkpoint(checkpoint, scenario, registry)
    model = _load_model_checkpoint(checkpoint, device=device)
    reference = _reference_from_payload(checkpoint["correction_reference"]).to(device)
    ref_before = tensor_state_hash(checkpoint["correction_reference"])
    centered = data.coords_m - checkpoint["origin"].cpu().numpy()
    randomness = common_student_t_randomness(len(rows), draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V184_CONFIG["student_df"]), seed=SCORING_SEED)
    draw_hash = randomness_hash(randomness)
    definitions = {cell_key(c["temperature"], c["mode"], c["epsilon_kl"]): c for c in cell_definitions()}
    cells, group_cache, strata, source_reports, mismatches, semantic = {}, {}, {}, {}, {}, None
    reproduction, batch_provenance, temperature_states = [], [], []
    counts = {"base_model_calls": 0, "raw_correction_energies": 0, "production_apply_correction": 0,
              "energy_score_calls": 0, "identity_metric_reuse_count": 0, "posterior_cells": 0}
    guard = ReadOnlyModelGuard(model, authorization)
    print(f"START {scenario['name']}: {len(rows)} rows / {len(ordered_groups)} groups / all 8 T", flush=True)
    with guard, torch.no_grad():
        for temperature in TEMPERATURES:
            t_started = time.perf_counter()
            state_before = tensor_state_hash(model.state_dict())
            collected, diagnostics, variants_collected = {}, {}, {}

            def collect(key, name, value):
                assert_finite(value, name)
                collected.setdefault(key, {}).setdefault(name, []).append(value.detach().cpu().double().numpy())

            def diagnostic(name, value):
                assert_finite(value, name)
                diagnostics.setdefault(name, []).append(value.detach().cpu().numpy())

            print(f"TEMPERATURE {scenario['name']} T={temperature:g} START", flush=True)
            batch_count = 0
            for start in range(0, len(rows), int(V184_CONFIG["batch_size"])):
                stop = min(start + int(V184_CONFIG["batch_size"]), len(rows))
                role_rows = rows[start:stop]
                rss = rss_all[role_rows].to(device)
                distribution = v184_distribution(model, rss, final_temperature=temperature, scale_multiplier=1.)
                counts["base_model_calls"] += 1
                q, mask = distribution["calibrated_base_probability"], distribution["candidate_mask"]
                locations, scales = distribution["component_locations"], distribution["component_scales"]
                assert_finite(distribution, "base distribution")
                require(bool((scales > 0).all()), "component scale not positive")
                snapshot_fields = ("candidate_indices", "candidate_mask", "calibrated_base_probability", "component_locations", "component_scales")
                snapshot = tensor_state_hash({k: distribution[k] for k in snapshot_fields})
                with guard.radio_batch(len(role_rows)):
                    raw = raw_correction_energies(model, rss, q, distribution)
                counts["raw_correction_energies"] += 1
                production = apply_correction(base_probability=q, candidate_mask=mask, raw=raw,
                                              reference=reference, epsilon_kl=.025)
                counts["production_apply_correction"] += 1
                source_hash = tensor_state_hash({**raw, "radio_energy": production["radio_energy"],
                                                 "geometry_energy": production["geometry_energy"]})
                posterior_cells = surface_posteriors(q, mask, production)
                indices = distribution["candidate_indices"]
                anchors = model.anchor_coords[indices]
                buildings, floors = model.anchor_buildings[indices], model.anchor_floors[indices]
                variants = (geometry_variants(q, anchors, locations, mask, buildings, floors,
                    spatial_scale_m=model.v184_spatial_scale_m, production_geometry=raw["geometry_energy_raw"])
                    if SEMANTIC.get(scenario["name"]) == temperature else {})
                # No evaluation truth participates in any posterior or energy.
                target = torch.from_numpy(centered[role_rows].astype(np.float32)).to(device)
                shared_draws = {k: (v[start:stop].to(device) if isinstance(v, torch.Tensor) else v)
                                for k, v in randomness.items()}
                batch_draw_hash = randomness_hash(shared_draws)
                distance = torch.linalg.vector_norm(locations - target[:, None, :], dim=2)
                base_expected_distance = (q * distance).sum(1)
                identity_metrics = None
                for (mode, eps), correction in posterior_cells.items():
                    p = correction["posterior"]
                    key = cell_key(temperature, mode, eps)
                    if mode == "none" or eps > 0:
                        require(randomness_hash(shared_draws) == batch_draw_hash, "shared Student-t draws changed")
                        values = row_surface_metrics(p, locations, scales, target, mask, model, shared_draws)
                        counts["energy_score_calls"] += 1
                        if mode == "none":
                            identity_metrics = values
                    else:
                        require(torch.equal(p, q) and identity_metrics is not None, "identity reuse not valid")
                        values = identity_metrics
                        counts["identity_metric_reuse_count"] += 1
                    for metric, value in values.items():
                        collect(key, metric, value)
                    for metric in ("realized_kl", "epsilon_kl_effective", "rho", "radio_reliability", "geometry_reliability"):
                        collect(key, metric, correction[metric])
                    if mode in ("radio_only", "geometry_only") and eps == .025:
                        collect(key, "oracle_gain_m", base_expected_distance - (p * distance).sum(1))
                    counts["posterior_cells"] += 1
                require(randomness_hash(shared_draws) == batch_draw_hash, "Energy Score mutated shared draws")
                for name, value in {"q": q, "mask": mask, "building": buildings, "floor": floors,
                    "indices": indices, "geometry": raw["geometry_energy_raw"], "radio": production["radio_energy"],
                    "geometry_normalized": production["geometry_energy"], "distance": distance,
                    "anchor_distance": torch.linalg.vector_norm(anchors - target[:, None, :], dim=2),
                    "residual": torch.linalg.vector_norm(locations - anchors, dim=2)}.items():
                    diagnostic(name, value)
                for name, energy in variants.items():
                    variants_collected.setdefault(name, []).append(energy.cpu().numpy())
                require(snapshot == tensor_state_hash({k: distribution[k] for k in snapshot_fields}), "support/components/q_T mutated")
                require(source_hash == tensor_state_hash({**raw, "radio_energy": production["radio_energy"],
                    "geometry_energy": production["geometry_energy"]}), "production source energies mutated")
                guard.assert_model_clean()
                require(tensor_state_hash(model.state_dict()) == state_before, "model state mutated during batch")
                batch_provenance.append({"temperature": temperature, "role_start": start, "role_stop": stop,
                    "shared_draw_sha256": batch_draw_hash, "support_component_q_T_sha256": snapshot,
                    "production_energy_sha256": source_hash, "base_calls": 1, "raw_calls": 1,
                    "production_apply_calls": 1, "posterior_cells": 13, "ES_same_draw_evaluations": 10,
                    "identity_metric_reuses": 3})
                batch_count += 1
                if batch_count % 5 == 0 or stop == len(rows):
                    print(f"PROGRESS {scenario['name']} T={temperature:g}: {stop}/{len(rows)} rows; {time.perf_counter()-t_started:.1f}s", flush=True)
            arrays = {key: {metric: np.concatenate(parts) for metric, parts in values.items()}
                      for key, values in collected.items()}
            for key, values in arrays.items():
                require(all(len(value) == len(rows) for value in values.values()), "metric rows missing")
                definition = definitions[key]
                none = arrays[cell_key(temperature, "none", 0.)]
                effect = {m: previous.paired_statistics(values[m], none[m], groups) for m in METRICS}
                cells[key] = {**definition, "metrics": summarize_cell(values, groups), "paired_correction_effect": effect}
                group_cache[key] = {m: previous.group_means(values[m], groups) for m in METRICS}
            joined = {name: previous.concat_support_arrays(parts) for name, parts in diagnostics.items()}
            q_cpu, mask_cpu = joined["q"], joined["mask"].astype(bool)
            stratum = geometry_stratum_audit(joined["geometry"], q_cpu, mask_cpu, joined["building"], joined["floor"],
                joined["anchor_distance"], joined["distance"], joined["residual"], joined["indices"])
            strata[f"{temperature:g}"] = stratum
            mismatches[f"{temperature:g}"] = mismatch_audit(joined["geometry"], q_cpu, mask_cpu,
                joined["anchor_distance"], joined["distance"], joined["residual"], stratum)
            source_reports[f"{temperature:g}"] = {}
            for source in ("radio", "geometry"):
                energy = joined["radio" if source == "radio" else "geometry_normalized"]
                alignment = previous.source_energy_alignment(energy, joined["distance"], q_cpu, mask_cpu)
                source_values = arrays[cell_key(temperature, source + "_only", .025)]
                gain, reliability = source_values["oracle_gain_m"], source_values[source + "_reliability"]
                source_reports[f"{temperature:g}"][source] = {**alignment,
                    "mean_expected_distance_oracle_gain_m": float(gain.mean()),
                    "fraction_oracle_gain_positive": float((gain > 0).mean()),
                    "spearman_reliability_oracle_gain": previous.spearman(reliability, gain), "oracle_epsilon_kl": .025}
            if variants_collected:
                variant_reports = {}
                for name, parts in variants_collected.items():
                    energy = previous.concat_support_arrays(parts)
                    audit = (stratum if name == GEOMETRIES[0] else geometry_stratum_audit(energy, q_cpu, mask_cpu,
                        joined["building"], joined["floor"], joined["anchor_distance"], joined["distance"], joined["residual"], joined["indices"]))
                    alignment = previous.source_energy_alignment(energy, joined["distance"], q_cpu, mask_cpu)
                    # Mandatory production G1 row records are already retained;
                    # alternative geometry definitions need aggregates only.
                    variant_reports[name] = {**{k: v for k, v in audit.items() if k != "row_records"}, **alignment,
                        "used_to_change_posterior": False, "reliability_fitted": False}
                semantic = {"temperature": temperature, "variants": variant_reports}
            reproduction.extend(prior_reproduction_checks(scenario["name"], temperature, cells, prior))
            state_after = tensor_state_hash(model.state_dict())
            require(state_before == state_after == guard.before_hash, "model state differs between temperatures")
            temperature_states.append({"temperature": temperature, "before_sha256": state_before,
                "after_sha256": state_after, "all_parameter_grads_None": True, "runtime_seconds": time.perf_counter()-t_started})
            print(f"TEMPERATURE {scenario['name']} T={temperature:g} PASS: 13 cells; state unchanged; {time.perf_counter()-t_started:.1f}s", flush=True)
    assert_surface_complete(cells)
    require(randomness_hash(randomness) == draw_hash, "full-role Student-t randomness changed")
    ref_after = tensor_state_hash({name: getattr(reference, name) for name in checkpoint["correction_reference"]})
    require(ref_before == ref_after == tensor_state_hash(checkpoint["correction_reference"]), "correction reference changed")
    batches = len(batch_provenance)
    require(counts["base_model_calls"] == counts["raw_correction_energies"] == counts["production_apply_correction"] == batches,
            "base/raw/apply call counts incorrect")
    require(counts["energy_score_calls"] == 10*batches and counts["identity_metric_reuse_count"] == 3*batches
            and counts["posterior_cells"] == 13*batches, "surface/randomness reuse cell counts incorrect")
    group_order_hash = hashlib.sha256(json_bytes({"ordered_coordinate_groups": [str(g) for g in ordered_groups]})).hexdigest()
    report = {"definition": {**scenario, "scale_multiplier": 1., "lambda_spatial": .1, "selected_epoch": 16,
        "rows": len(rows), "coordinate_groups": len(ordered_groups), "coordinate_group_order_sha256": group_order_hash,
        "row_index_sha256": registry["roles"][scenario["role"]]["row_index_sha256"],
        "checkpoint_sha256": previous.FORMAL_EXPECTED[scenario["checkpoint"]], **reference_validation},
        "cells": cells, "source_informativeness": source_reports, "geometry_stratum_mass_audit": strata,
        "geometry_coordinate_space_mismatch": mismatches, "semantic_geometry_diagnostics": semantic,
        "shared_prior_cell_reproduction": reproduction, "model_integrity": guard.verification,
        "model_state_per_temperature": temperature_states, "reference_before_sha256": ref_before, "reference_after_sha256": ref_after,
        "production_call_counts": counts, "shared_Student_t_randomness": {"seed": SCORING_SEED, "draws": PREDICTIVE_DRAWS,
            "degrees_freedom": float(V184_CONFIG["student_df"]), "full_role_sha256": draw_hash,
            "generation_calls": 1, "unchanged": True, "reused_across_all_temperatures_modes_epsilons": True,
            "batch_provenance": batch_provenance}, "runtime_seconds": time.perf_counter()-started}
    assert_finite(report)
    print(f"DONE {scenario['name']}: {report['runtime_seconds']:.1f}s; 104 cells; model unchanged", flush=True)
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report, group_cache


def gate_receipt() -> dict:
    raw = os.environ.get("V184_COUPLING_GATE_RECEIPT")
    require(raw is not None, "successful compile/full-regression receipt required")
    receipt = json.loads(raw)
    require(receipt["compile"]["exit_code"] == 0 and receipt["pytest"]["exit_code"] == 0, "compile/regression gate failed")
    tests = receipt["pytest"]
    require(tests["test_files"] == list(TEST_FILES), "full accepted regression list mismatch")
    require(tests["passed"] == 128 + tests["new_tests"] and all(tests[k] == 0 for k in
            ("failed", "errors", "skipped", "xfailed", "xpassed")), "regression count/status mismatch")
    require(receipt["human_authorization"].get("radio_temporary_adam_allowed") is True, "recorded nuisance exception missing")
    expected_paths = {MODULE, TEST_MODULE, *TEST_FILES, *previous.FORMAL_EXPECTED,
                      "fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py"}
    require(expected_paths <= set(receipt["tested_file_sha256"]), "tested source/artifact hashes missing")
    for relative, expected in receipt["tested_file_sha256"].items():
        require(sha256_file(REPO_ROOT / relative) == expected, "source/artifact changed after gate: " + relative)
    return receipt


def format_ci(value: Mapping) -> str:
    a, b = value["normal_95_ci"]
    return f"{value['mean_delta']:+.6f} [{a:+.6f}, {b:+.6f}]"


def scientific_markdown(result: Mapping) -> str:
    texts = ["# V18.4 final read-only coupling × stratum audit\n",
        f"Evidence HEAD: `{result['current_evidence_commit_sha']}`；frozen source: `{FROZEN_SOURCE_SHA}`。",
        "3 个预注册场景 × 8 个温度 × 13 个单元 = **312 单元**；全部角色行；Student-t scale 固定 1。",
        "ES/NLL/后验均值 MAE 按坐标组等权；SE 为样本组均值的标准误，配对 CI 为 ±1.96 SE。其余汇总按行等权。"
        "δ=修正−同温度 none，负数改善。ε=0 精确恒等控制复用指标，但保留各模式真实可靠性。"
        "GM 仅用于位移/尾部诊断；点指标仅评价后验均值。",
        "## 1. 场景及完整性\n"]
    scenarios = result["scenarios"]
    texts.append(table(("场景", "Checkpoint/reference", "Rows", "Groups", "Reference rows", "Memory anchors"),
        [[n, s["definition"]["reference_role"], s["definition"]["rows"], s["definition"]["coordinate_groups"],
          s["definition"]["reference_rows"], s["definition"]["memory_anchors"]] for n, s in scenarios.items()]))
    texts.append(f"完整性 PASS：{result['integrity']['verified_source_file_count']} 份冻结源码、正式字节哈希/自哈希/交叉链接、"
        "两份 checkpoint、前轮四份产物/自哈希/PASS 状态和模型状态均验证；既有 Python 源码及产物字节保持。")
    for mode, title in (("radio_only", "2. Radio-only"), ("full", "3. Full correction")):
        texts.append("## " + title + "：T × ε 配对热表\n")
        for name, scenario in scenarios.items():
            for metric in METRICS:
                texts.append(f"### {name} — δ {metric}\n")
                texts.append(table(("T", "ε=0", "ε=.025", "ε=.050", "ε=.100"), [[f"{t:g}", "0 (identity)"] + [
                    f"{scenario['cells'][cell_key(t,mode,eps)]['paired_correction_effect'][metric]['mean_delta']:+.6f}"
                    for eps in EPSILONS[1:]] for t in TEMPERATURES]))
    texts.append("## 4. Geometry-only 温度符号汇总\n")
    symbol = {-1: "−", 0: "0", 1: "+"}
    rows = []
    for name, entries in result["temperature_correction_map"]["scenarios"].items():
        for entry in entries:
            if entry["mode"] == "geometry_only":
                rows.append([name, entry["epsilon_kl"], *["".join(symbol[s] for s in entry["metrics"][m]["sign_vector"]) for m in METRICS], entry["classification"]])
    texts.append("符号依次对应 T=.5/.75/1/1.25/1.5/2/2.5/3；三个指标须全温度稳定且方向一致才判 SEPARABLE。")
    texts.append(table(("Scenario", "ε", "ES signs", "NLL signs", "Mean MAE signs", "Gate"), rows))
    texts.append("## 5. Robust common-region gates\n")
    rows = []
    for key in ("radio_robust_common_region", "full_robust_common_region"):
        gate = result[key]
        for entry in gate["epsilon_reports"]:
            rows.append([gate["mode"], entry["epsilon_kl"], entry["beneficial_temperatures"][PRIMARY[0]],
                entry["beneficial_temperatures"][PRIMARY[1]], entry["common_temperatures"], entry["adjacent_common_pairs"], entry["ROBUST_COMMON_REGION"]])
    texts.append("每个 T 要求 δES<0、δNLL<0、δmean-MAE≤.05m；同 ε 下两个主场景共同通过且至少两个相邻温度。")
    texts.append(table(("Mode", "ε", "S1 beneficial T", "S3 beneficial T", "Common", "Adjacent pairs", "PASS"), rows))
    texts.append(table(("Scenario", "Median mean-row radio Spearman across T", "Median gain-positive fraction", "Intrinsic PASS"),
        [[name, f"{r['median_across_T_mean_row_spearman']:.6f}", f"{r['median_across_T_fraction_oracle_gain_positive']:.6f}", r["passes"]]
         for name, r in result["radio_intrinsic_gate"]["primary_scenarios"].items()]))
    texts.append("## 6. Role shift：固定 FIT package\n")
    texts.append("S2−S1 为不同坐标组的描述性差；不称为配对统计比较。下表列 none 的指标差和 ε=.025 的修正效应差。全部 104 单元见 JSON。")
    for metric in METRICS:
        texts.append("### " + metric + "\n")
        rows = []
        for t in TEMPERATURES:
            base = result["role_shift_decomposition"]["cells"][cell_key(t,"none",0.)]["role_shift_metric_delta"][metric]
            shifts = [result["role_shift_decomposition"]["cells"][cell_key(t,m,.025)]["role_shift_in_correction_effect"][metric]
                      for m in MODES[1:]]
            rows.append([t, f"{base:+.6f}", *[f"{s:+.6f}" for s in shifts]])
        texts.append(table(("T", "none S2−S1", "radio gain S2−S1", "geometry gain S2−S1", "full gain S2−S1"), rows))
    texts.append("## 7. Refit package：相同 calibration 行\n")
    texts.append("S3−S2 包含学习状态、anchor memory 和 correction reference 的整体变化；以下是 T=1/1.5 的配对组差及 CI。全部单元及效应差 CI 见 JSON。")
    rows = []
    for t in (1., 1.5):
        for mode in ("none", "radio_only", "full"):
            key = cell_key(t,mode,0. if mode == "none" else .025)
            record = result["refit_effect_decomposition"]["cells"][key]
            rows.append([key, *[format_ci(record["paired_refit_metric_delta"][m]) for m in METRICS],
                         *[format_ci(record["paired_refit_effect_on_correction"][m]) for m in METRICS]])
    texts.append(table(("Cell", "refit ES δ [CI]", "refit NLL δ [CI]", "refit mean MAE δ [CI]",
                        "gain ES δ [CI]", "gain NLL δ [CI]", "gain mean MAE δ [CI]"), rows))
    texts.append("## 8. Geometry argmin 的 stratum mass\n")
    rows = []
    for name, scenario in scenarios.items():
        for t in TEMPERATURES:
            r = scenario["geometry_stratum_mass_audit"][f"{t:g}"]
            rows.append([name, t, f"{r['argmin_stratum_mass']['mean']:.6f}", f"{r['argmin_stratum_mass']['median']:.6f}",
                *[f"{r['low_mass_subsets'][f'{threshold:g}']['fraction']:.4%}" for threshold in (.01,.05,.10)],
                f"{r['mean_argmin_component_error_m']:.4f}", f"{r['mean_q_T_MAP_component_error_m']:.4f}",
                f"{r['fraction_argmin_in_q_T_MAP_stratum']:.4%}", f"{r['mean_strata_with_near_zero_min_energy']:.4f}"])
    texts.append(table(("Scenario", "T", "Mean mass", "Median mass", "<.01", "<.05", "<.10", "Argmin err m", "MAP err m", "MAP stratum", "Near-zero strata"), rows))
    texts.append("每行 argmin index、能量、mass、q_T、MAP stratum 一致性、两坐标误差、残差位移及 stratum 数见 JSON。Near-zero 固定为 E≤1e−8。")
    texts.append("## 9. Anchor/component 坐标对齐\n")
    rows = []
    for name, scenario in scenarios.items():
        for t in TEMPERATURES:
            r = scenario["geometry_coordinate_space_mismatch"][f"{t:g}"]
            rows.append([name, t, f"{r['mean_row_spearman_anchor_truth_distance']:.6f}",
                f"{r['mean_row_spearman_component_truth_distance']:.6f}", f"{r['component_minus_anchor_mean_spearman']:+.6f}",
                f"{r['argmin_anchor_error_m']:.4f}", f"{r['argmin_component_error_m']:.4f}",
                f"{r['residual_displacement_all_valid_candidates_m']['p90']:.4f}",
                f"{r['residual_displacement_geometry_argmin_m']['p90']:.4f}"])
    texts.append(table(("Scenario", "T", "ρ anchor", "ρ component", "B−A", "argmin anchor m", "argmin component m", "all residual P90 m", "argmin residual P90 m"), rows))
    texts.append("残差统计按所有有效候选等权（不是 q_T 权重）；另报告只选 geometry argmin 的逐行统计。完整分位数和 >5/10/20m 比例见 JSON。")
    texts.append("## 10. 四种 geometry source，仅语义温度诊断\n")
    rows = []
    for name, semantic in result["semantic_geometry_comparison"].items():
        for g, r in semantic["variants"].items():
            rows.append([name, semantic["temperature"], g, f"{r['mean_row_spearman']:.6f}", f"{r['median_row_spearman']:.6f}",
                f"{r['fraction_spearman_positive']:.4%}", f"{r['mean_argmin_component_error_m']:.4f}",
                f"{r['fraction_argmin_beats_q_T_MAP_component']:.4%}", f"{r['argmin_stratum_mass']['mean']:.6f}",
                *[f"{r['low_mass_subsets'][f'{th:g}']['fraction']:.4%}" for th in (.01,.05,.10)]])
    texts.append(table(("Scenario", "T", "G", "Mean ρ", "Median ρ", "ρ>0", "Argmin err m", "Beats MAP", "Mean mass", "<.01", "<.05", "<.10"), rows))
    texts.append("G1=anchor/conditional；G2=component/conditional；G3=anchor/mass-preserving；G4=component/mass-preserving。G2/G3/G4 未拟合可靠性、未改变后验、未用于选择方法。")
    texts.append("## 11. 预注册 H1/H2\n")
    rows = []
    for h, record in result["geometry_hypothesis_gates"].items():
        rows.append([h, record["scenarios"][PRIMARY[0]]["passes"], record["scenarios"][PRIMARY[1]]["passes"], record["conclusion"]])
    texts.append(table(("Hypothesis", "S1@T1 passes", "S3@T1.5 passes", "Both-scenario gate"), rows))
    texts.append("H1 每个场景至少一组质量保留比较同时达到 argmin 误差减幅≥25% 且 <.05 质量的 argmin 比例下降。H2 每个场景至少一组坐标替换比较达到 Spearman 增幅≥.05 或 argmin 误差减幅≥10%。边界固定。")
    texts.append("## 12. 证据、门禁与设计含义\n")
    conclusions = result["preregistered_conclusions"]
    texts.append(table(("Preregistered gate", "Result"), [[k, v] for k, v in conclusions.items()]))
    joint = result["temperature_correction_map"]["temperature_and_correction_require_joint_selection"]
    texts.append("**观察证据：** 完整曲面、配对 CI、角色/refit 差和 geometry/source 对齐均按固定网格得到；单个 argmin 的误差与全后验期望距离增益是不同诊断量。")
    texts.append("**门禁结论：** 上表由运行前固定规则计算；radio intrinsic gate 使用八个 T 的中位数。共同区域要求两主场景、同 ε、相邻 T。")
    texts.append("**设计含义：** " + ("后续设计需联合处理温度与修正强度。" if joint else "此固定网格未触发温度与修正联合选择门禁。") +
        "radio/full 的候选门禁仅作为 V18.5 设计审阅输入；H1/H2 是结构解释证据。须先科学审阅，再决定任何新方法或实验。")
    texts.append(f"共享前轮单元复现检查 PASS：{sum(len(s['shared_prior_cell_reproduction']) for s in scenarios.values())} 项，"
        f"固定容差 atol={REPRODUCTION_ATOL}, rtol={REPRODUCTION_RTOL}。全部模型状态逐位保持且网络 grads 为 None；"
        "仅已授权的 radio 临时 offset/log_gain 每批 6 步 Adam。仅 trainingData.csv 被读取；未访问 UJI validation/external，未训练，未实现 V18.5。")
    texts.append(f"Result canonical SHA256: `{result['result_sha256']}`。\n\n四份产物写入后硬停止，等待科学审阅。")
    return "\n\n".join(texts) + "\n"


def execution_markdown(record: Mapping) -> str:
    texts = ["# V18.4 coupling × stratum execution record\n", f"Status: **{record['status']}**。",
        f"Evidence HEAD: `{record.get('current_evidence_commit_sha')}`；frozen source: `{FROZEN_SOURCE_SHA}`。",
        "## 环境与门禁\n", "```json\n" + json.dumps(record.get("environment"), indent=2) + "\n```"]
    texts.append(table(("Step", "Exit", "Runtime seconds", "Tests", "Command"), [
        [step, record.get("gate",{}).get(step,{}).get("exit_code"), record.get("gate",{}).get(step,{}).get("runtime_seconds"),
         record.get("gate",{}).get(step,{}).get("passed",""), record.get("gate",{}).get(step,{}).get("command","")]
        for step in ("compile", "pytest")] + [["probe", record.get("probe_exit_code"), record.get("probe_runtime_seconds"), "", record["probe_command"]]]))
    texts.append("## 完整性、模型、数据访问\n")
    texts.append("```json\n" + json.dumps({k: record.get(k) for k in (
        "source_artifact_postcheck", "dataset_opens", "denied_file_opens", "model_integrity", "radio_temporary_adam_totals", "boundary_confirmations")},
        ensure_ascii=False, indent=2) + "\n```")
    texts.append("## 问题与处理\n")
    for item in record.get("problems_and_resolutions", []):
        texts.append("- " + item["problem"] + "\n  " + item["resolution"])
    if record.get("failure"):
        texts.append("```text\n" + record["failure"] + "\n```")
    texts.append("## 科学输出字节 SHA256\n")
    texts.append(table(("Path", "SHA256"), [[p, h] for p, h in record.get("output_sha256",{}).items()]))
    texts.append(f"Execution canonical SHA256: `{record['execution_record_sha256']}`。输出路径从未覆盖；结束后硬停止。")
    return "\n\n".join(texts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    assert_unused(OUTPUTS)
    receipt = gate_receipt()
    started = time.perf_counter()
    _set_seed(MODEL_SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    audit, scenarios, group_cache = FileOpenAudit(), {}, {}
    record = {"record_type": PROBE_VERSION, "status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
        "current_evidence_commit_sha": head, "frozen_source_commit_sha": FROZEN_SOURCE_SHA,
        "environment": previous.environment(device), "gate": receipt, "human_authorization": receipt["human_authorization"],
        "probe_command": f"{sys.executable} -m fhm_dapp_tim.fhm_dapp.v18_4_coupling_stratum_probe --device {args.device}",
        "problems_and_resolutions": receipt.get("problems_and_resolutions", [])}
    try:
        with audit:
            integrity, registry, prior = verify_integrity()
            record["integrity"] = integrity
            train = REPO_ROOT / "data/UJIndoorLoc/trainingData.csv"
            require(sha256_file(train) == registry["train_sha256"], "trainingData source hash mismatch")
            data, roles = _load_registered_data(registry)
            require(len(data.coords_m) == registry["training_rows"], "full training row count mismatch")
            rss_all = torch.from_numpy(data.model_rss())
            for definition in SCENARIOS:
                name = definition["name"]
                scenarios[name], group_cache[name] = run_scenario(definition, registry, data, roles, rss_all, device,
                    receipt["human_authorization"], prior)
            require(tuple(scenarios) == tuple(s["name"] for s in SCENARIOS), "exact scenario set/order mismatch")
            require(sum(len(s["cells"]) for s in scenarios.values()) == 312, "surface missing cells")
            after, _, _ = verify_integrity()
            require(after == integrity, "pre-existing sources/artifacts changed")
            for relative, expected in receipt["tested_file_sha256"].items():
                require(sha256_file(REPO_ROOT / relative) == expected, "tested file changed during probe: " + relative)
            require(sha256_file(train) == registry["train_sha256"], "trainingData changed")
            require(not audit.denied and all(o["allowed"] for o in audit.opens), "file-open boundary violated")
            coupling = coupling_maps(scenarios)
            intrinsic = intrinsically_informative(scenarios)
            radio, full = (robust_common_region(scenarios, mode) for mode in ("radio_only", "full"))
            radio["conclusion"] = ("radio_correction_candidate_supported" if radio["any_robust_common_region"] else
                "radio_evidence_informative_but_current_tilt_not_robust" if intrinsic["RADIO_INTRINSICALLY_INFORMATIVE"] else
                "radio_evidence_not_supported")
            full["conclusion"] = "full_correction_candidate_supported" if full["any_robust_common_region"] else "full_correction_not_robust"
            semantic = {n: scenarios[n]["semantic_geometry_diagnostics"] for n in PRIMARY}
            require(all(semantic[n]["temperature"] == SEMANTIC[n] and tuple(semantic[n]["variants"]) == GEOMETRIES for n in PRIMARY),
                    "semantic geometry diagnostics missing")
            hypotheses = geometry_hypothesis_gates(semantic)
            result = {"probe_version": PROBE_VERSION, "scientific_status": "FINAL_READ_ONLY_DIAGNOSTIC_STOP_FOR_SCIENTIFIC_REVIEW",
                "current_evidence_commit_sha": head, "frozen_source_commit_sha": FROZEN_SOURCE_SHA,
                "integrity": integrity, "source_artifact_integrity": "PASS", "prior_mechanism_validation": integrity["prior_mechanism_validation"],
                "dataset_opens": list(audit.opens), "train_sha256": registry["train_sha256"], **previous.BOUNDARIES,
                "model_buffers_modified": False, "temporary_radio_optimizer_used_under_existing_human_exception": True,
                "human_authorization": receipt["human_authorization"], "scenario_definitions": list(SCENARIOS),
                "temperature_grid": list(TEMPERATURES), "epsilon_kl_grid": list(EPSILONS), "correction_modes": list(MODES),
                "scale_multiplier": 1., "point_decoder": "posterior_mean", "surface_cell_count": 312,
                "conventions": {"macro": "equal coordinate groups; sample group SE; normal CI mean +/-1.96 SE",
                    "other_aggregates": "equal rows; candidate residuals equal valid candidates",
                    "delta": "corrected - none at same T; negative improves", "role_shift": "unpaired descriptive S2-S1",
                    "refit": "paired same-row coordinate group means S3-S2", "epsilon_zero": "bitwise production identity; score reuse",
                    "ties": "first valid candidate order; Spearman average ranks; undefined correlations ->0",
                    "near_zero_geometry": "raw geometry energy <=1e-8", "low_mass": "strict < .01/.05/.10",
                    "empty_threshold_subset": "row_count=0, fraction=0, mean_error=null",
                    "geometry_mass_preserving": "same global normalization/kernel/mask/sigma/margin; omit only conditional stratum renormalization",
                    "GM": "frozen production weighted_component_geometric_median, diagnostic only",
                    "KL_tolerance": KL_TOLERANCE, "shared_reproduction_atol": REPRODUCTION_ATOL, "shared_reproduction_rtol": REPRODUCTION_RTOL},
                "scenarios": scenarios, "temperature_correction_map": coupling, "radio_intrinsic_gate": intrinsic,
                "radio_robust_common_region": radio, "full_robust_common_region": full,
                "role_shift_decomposition": role_shift_decomposition(scenarios),
                "refit_effect_decomposition": refit_decomposition(scenarios, group_cache),
                "semantic_geometry_comparison": semantic, "geometry_hypothesis_gates": hypotheses,
                "preregistered_conclusions": {"temperature_and_correction_require_joint_selection":
                    coupling["temperature_and_correction_require_joint_selection"], "radio": radio["conclusion"],
                    "full": full["conclusion"], "H1": hypotheses["H1"]["conclusion"], "H2": hypotheses["H2"]["conclusion"]}}
            assert_finite(result)
            seal(result, "result_sha256")
            science_json, science_md = json_bytes(result), scientific_markdown(result).encode("utf-8")
            totals = {key: sum(s["model_integrity"][key] for s in scenarios.values()) for key in (
                "temporary_adam_initializations", "temporary_adam_steps", "temporary_nuisance_backward_calls", "radio_batches")}
            record.update(status="PASS", probe_exit_code=0, probe_runtime_seconds=time.perf_counter()-started,
                dataset_opens=list(audit.opens), denied_file_opens=list(audit.denied),
                boundary_confirmations={**previous.BOUNDARIES, "model_buffers_modified": False},
                source_artifact_postcheck="PASS: every frozen/prior/pre-existing Python/evidence/checkpoint/tested file unchanged",
                model_integrity={n: s["model_integrity"] for n, s in scenarios.items()},
                radio_temporary_adam_totals=totals, surface_cell_count=312,
                shared_prior_cell_reproduction_checks=sum(len(s["shared_prior_cell_reproduction"]) for s in scenarios.values()),
                output_sha256={OUTPUT_JSON.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(science_json).hexdigest(),
                               OUTPUT_MD.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(science_md).hexdigest()},
                result_self_hash=result["result_sha256"], finished_utc=datetime.now(timezone.utc).isoformat(), hard_stop_after_outputs=True)
            assert_finite(record)
            seal(record, "execution_record_sha256")
            contents = (science_json, science_md, json_bytes(record), execution_markdown(record).encode("utf-8"))
            assert_unused(OUTPUTS)
            for path, content in zip(OUTPUTS, contents):
                write_once(path, content)
            for path in OUTPUTS:
                print(f"OUTPUT {path.relative_to(REPO_ROOT).as_posix()} SHA256 {sha256_file(path)}", flush=True)
            print("PASS; HARD STOP for scientific review", flush=True)
    except Exception:
        record.update(status="FAIL_CLOSED", probe_exit_code=1, probe_runtime_seconds=time.perf_counter()-started,
            failure=traceback.format_exc(), dataset_opens=list(audit.opens), denied_file_opens=list(audit.denied),
            completed_scenarios=list(scenarios), model_integrity={n:s["model_integrity"] for n,s in scenarios.items()},
            finished_utc=datetime.now(timezone.utc).isoformat())
        seal(record, "execution_record_sha256")
        if not EXECUTION_JSON.exists() and not EXECUTION_MD.exists():
            write_once(EXECUTION_JSON, json_bytes(record))
            write_once(EXECUTION_MD, execution_markdown(record).encode("utf-8"))
        raise


if __name__ == "__main__":
    main()
