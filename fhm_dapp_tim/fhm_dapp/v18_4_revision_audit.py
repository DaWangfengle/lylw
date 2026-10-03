"""Outcome-free pre-run revision audit; no formal protocol or model selection.

Only UJI trainingData-derived fit/development scans and synthetic constructions
are used. CPU diagnostics and, when available, a small CUDA smoke are audited.
No checkpoint, registry, formal calibration/freeze, or test result is written.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    cap_anisotropic_residual_transform,
    kl_p_to_q,
    reliability_bounded_posterior_correction,
    spatial_energy_distance,
)
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import DATA_ROOT, REPO_ROOT
from fhm_dapp_tim.fhm_dapp.v18_4_model import (
    V184_CONFIG,
    base_posterior,
    candidate_support,
    fit_correction_reference,
    prototype_retrieval_score,
    raw_correction_energies,
    v184_distribution,
)
from fhm_dapp_tim.fhm_dapp.v18_4_realdata_smoke import run_smoke


EVIDENCE = REPO_ROOT / "fhm_dapp_tim" / "evidence" / "v18_4"
OUTPUT_JSON = EVIDENCE / "PRE_RUN_REVISION_AUDIT_V18_4.json"
OUTPUT_MD = EVIDENCE / "PRE_RUN_REVISION_AUDIT_V18_4.md"
KL_TOLERANCE = 1e-7
RESIDUAL_TOLERANCE_M = 1e-5
FORMAL_NAMES = (
    "uji_registry_v18_4.json", "uji_development_v18_4.json",
    "uji_calibration_v18_4.json", "uji_method_freeze_v18_4.json",
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _formal_snapshot() -> list[str]:
    paths = [EVIDENCE / name for name in FORMAL_NAMES]
    for root in (EVIDENCE / "external", REPO_ROOT / "fhm_dapp_tim" / "models" / "v18_4"):
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(str(path.relative_to(REPO_ROOT)) for path in paths if path.exists())


@contextmanager
def training_only_access_guard():
    """Audit Python file opens; deny all other dataset paths and formal writes."""
    root = os.path.normcase(os.path.abspath(DATA_ROOT))
    train = os.path.normcase(os.path.abspath(DATA_ROOT / "UJIndoorLoc" / "trainingData.csv"))
    evidence = os.path.normcase(os.path.abspath(EVIDENCE))
    counts: Counter[str] = Counter()
    state = {"active": True}

    def hook(event, args):
        if not state["active"] or event != "open" or isinstance(args[0], int):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(args[0]))))
        if path == root or path.startswith(root + os.sep):
            if path != train:
                raise RuntimeError(f"revision audit denied non-UJI-training data access: {path}")
            counts[path] += 1
        mode = args[1]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        )
        if writing and path.startswith(evidence + os.sep):
            if Path(path).name in FORMAL_NAMES or path.startswith(evidence + os.sep + "external" + os.sep):
                raise RuntimeError("revision audit denied a formal protocol artifact write")

    sys.addaudithook(hook)
    try:
        yield counts
    finally:
        state["active"] = False


def synthetic_math_checks(device: torch.device) -> dict:
    dtype = torch.float64
    q = torch.tensor([[.8, .2]], dtype=dtype, device=device)
    coords = torch.tensor([[[0., 0.], [2., 0.]]], dtype=dtype, device=device)
    target = torch.ones(1, 1, dtype=dtype, device=device)
    identical = spatial_energy_distance(q, coords, q, coords, spatial_scale=1.)
    near = spatial_energy_distance(q, coords, target, coords[:, :1], spatial_scale=1.)
    far = spatial_energy_distance(q, coords, target, torch.tensor([[[20., 0.]]], dtype=dtype, device=device), spatial_scale=1.)
    logits = torch.tensor([[1., 0., -1.]], dtype=dtype, device=device, requires_grad=True)
    x = torch.tensor([[[0., 0.], [2., 0.], [8., 0.]]], dtype=dtype, device=device)
    gradient_loss = spatial_energy_distance(torch.softmax(logits, dim=1), x,
        torch.tensor([[.7, .3]], dtype=dtype, device=device), coords + 1., spatial_scale=2.).sum()
    gradient_loss.backward()
    gradient_norm = torch.linalg.vector_norm(logits.grad)
    _require(all(bool(torch.isfinite(value).all()) for value in (identical, near, far, gradient_loss, logits.grad)), "non-finite spatial loss/gradient")
    _require(min(float(identical), float(near), float(far), float(gradient_loss)) >= -1e-10, "negative spatial energy distance")
    _require(abs(float(identical)) <= 1e-10 and float(far) > float(near), "spatial identity/order failed")
    _require(float(gradient_norm) > 0., "spatial gradient is zero")

    base = torch.tensor([[.55, .25, .20, 0.]] * 4, dtype=dtype, device=device)
    mask = torch.tensor([[True, True, True, False]] * 4, device=device)
    energy = torch.tensor([[8., 2., 0., 999.]] * 4, dtype=dtype, device=device)
    rr = torch.tensor([1., .5, .1, 0.], dtype=dtype, device=device)
    rg = .5 * rr
    corrected = reliability_bounded_posterior_correction(base, energy, 2. * energy,
        radio_reliability=rr, geometry_reliability=rg, epsilon_kl=.1, mask=mask)
    kl = kl_p_to_q(corrected["posterior"], base, mask)
    _require(bool((kl <= corrected["epsilon_kl_effective"] + KL_TOLERANCE).all()), "row KL exceeds effective budget")
    _require(bool((kl[:-1] >= kl[1:] - KL_TOLERANCE).all()), "reliability monotonicity failed")
    _require(torch.equal(corrected["posterior"][-1], base[-1]), "zero reliability is not identity")
    _require(bool((corrected["posterior"][~mask] == 0.).all()), "padding received posterior mass")
    _require(torch.allclose(corrected["combined_energy"][:3], corrected["combined_energy"][0].expand(3, -1), atol=1e-12, rtol=1e-12), "controlled direction changed")
    trust_rows = [{
        "epsilon_kl_max": .1, "radio_reliability": float(rr[row]), "geometry_reliability": float(rg[row]),
        **{key: float(corrected[key][row]) for key in ("epsilon_kl_effective", "rho", "realized_kl", "full_correction_kl", "radio_direction_weight", "geometry_direction_weight")},
    } for row in range(len(base))]

    oversized = torch.tensor([[[300., 0.], [0., 90.]], [[60., 30.], [30., 60.]]], dtype=dtype, device=device)
    capped = cap_anisotropic_residual_transform(oversized, max_residual_m=30.)
    raw_direction = torch.tensor([[2., 1.], [-3., 4.], [100., -100.], [0., 0.]], dtype=dtype, device=device)
    direction = torch.tanh(raw_direction)
    direction /= torch.maximum(torch.linalg.vector_norm(direction, dim=1, keepdim=True), torch.ones(4, 1, dtype=dtype, device=device))
    displacement = torch.einsum("aij,bj->abi", capped, direction)
    max_singular = float(torch.linalg.svdvals(capped).max())
    max_displacement = float(torch.linalg.vector_norm(displacement, dim=2).max())
    _require(max_singular <= 30. + RESIDUAL_TOLERANCE_M and max_displacement <= 30. + RESIDUAL_TOLERANCE_M, "residual cap violated")
    normal = torch.tensor([[[9., 1.], [1., 3.]]], dtype=dtype, device=device)
    _require(torch.equal(cap_anisotropic_residual_transform(normal, max_residual_m=30.), normal), "normal anisotropy changed")
    return {
        "device": str(device),
        "spatial_loss": {"identical_loss": float(identical), "near_loss": float(near), "far_loss": float(far),
            "gradient_loss": float(gradient_loss), "gradient_norm": float(gradient_norm), "gradient_finite_nonzero": True},
        "trust_reliability_rows": trust_rows,
        "same_direction_lower_reliability_no_larger_kl": True,
        "zero_reliability_bitwise_identity": True, "padding_exact_zero": True,
        "residual_cap": {"registered_max_residual_m": 30., "effective_max_singular_value_m": max_singular,
            "max_synthetic_displacement_m": max_displacement, "oversized_transform_test_passed": True,
            "normal_transform_unchanged": True},
    }


def real_training_checks(device_name: str) -> dict:
    smoke, context = run_smoke(device_name=device_name)
    model, rows, rss = context["model"], context["rows"], context["rss"]
    device = next(model.parameters()).device
    model.eval()
    model.zero_grad(set_to_none=True)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fit_rows = rows["fit"][:8]
    dev_rows = rows["development"][:8]

    def posterior_for(selected):
        scans = rss[selected].to(device)
        with torch.no_grad():
            invariant, embedding = model.retrieval_logits(scans)
            prototype = prototype_retrieval_score(model, scans)
            support = candidate_support(model, invariant, prototype)
            base = base_posterior(model, scans, embedding, invariant, support)
        return scans, invariant, prototype, support, base

    fit_scans, invariant, prototype, fit_support, fit_base = posterior_for(fit_rows)
    raw = raw_correction_energies(model, fit_scans, fit_base["probability"], fit_support)
    reference = fit_correction_reference(radio_energy_raw=raw["radio_energy_raw"], geometry_energy_raw=raw["geometry_energy_raw"],
        base_probability=fit_base["probability"], candidate_mask=fit_base["candidate_mask"],
        style_offset=raw["style_offset"], style_scale=raw["style_scale"])
    anchor = torch.as_tensor(context["sample_anchor"][fit_rows], dtype=torch.long, device=device)
    target_indices, target_weights = context["neighbours"][anchor], context["weights"][anchor]
    forced = candidate_support(model, invariant, prototype, force_indices=target_indices, force_weights=target_weights)
    covered = ((forced["candidate_indices"].unsqueeze(2) == target_indices.unsqueeze(1)) & forced["candidate_mask"].unsqueeze(2)).any(dim=1)
    _require(bool((covered | (target_weights <= 0.)).all()), "forced positive support dropped")
    retained = (covered.to(target_weights) * target_weights).sum(dim=1)
    _require(bool((retained >= 1. - 1e-6).all()), "forced positive mass lost")
    _require(not bool(((target_indices == anchor.unsqueeze(1)) & (target_weights > 0.)).any()), "positive self target retained")

    with torch.no_grad():
        corrected = v184_distribution(model, rss[dev_rows].to(device), correction_reference=reference,
            epsilon_kl=.05, final_temperature=1.5, scale_multiplier=1.25)
    for key in ("base_probability", "calibrated_base_probability", "candidate_weights", "component_locations", "component_scales"):
        _require(bool(torch.isfinite(corrected[key]).all()), f"non-finite real-data {key}")
    mask = corrected["candidate_mask"]
    _require(bool((corrected["candidate_weights"][~mask] == 0.).all()), "real-data padding mass")
    actual_kl = kl_p_to_q(corrected["candidate_weights"].double(), corrected["calibrated_base_probability"].double(), mask)
    effective = corrected["correction_epsilon_kl_effective"]
    _require(bool((actual_kl <= effective + KL_TOLERANCE).all()), "real-data row KL exceeds budget")
    _require(torch.allclose(actual_kl, corrected["correction_realized_kl"], atol=1e-10, rtol=1e-6), "realized KL reference mismatch")
    displacement = corrected["component_locations"] - model.anchor_coords[corrected["candidate_indices"]]
    max_displacement = float(torch.linalg.vector_norm(displacement, dim=2).max())
    max_singular = float(torch.linalg.svdvals(model.local_residual_transform.double()).max())
    _require(max_displacement <= 30. + RESIDUAL_TOLERANCE_M and max_singular <= 30. + RESIDUAL_TOLERANCE_M, "real-data residual cap violated")
    _require(all(parameter.grad is None for parameter in model.parameters()), "radio profiler touched model gradients")
    _require(all(torch.equal(value, state[key]) for key, value in model.state_dict().items()), "radio profiler changed model state")
    sizes = torch.cat([fit_support["natural_size"], corrected["natural_support_size"]])
    pools = torch.cat([fit_support["natural_pool_size_before_ceiling"], corrected["natural_pool_size_before_ceiling"]])
    truncated = torch.cat([fit_support["natural_truncated"], corrected["natural_truncated"]])
    return {
        "smoke": smoke, "reference_fit_rows": len(fit_rows), "reference_fit_temperature": 1.,
        "development_smoke_rows": len(dev_rows), "development_base_temperature": 1.5,
        "base_posterior_finite": True, "corrected_posterior_finite": True,
        "realized_kl_against_q_T": actual_kl.cpu().tolist(), "epsilon_kl_effective": effective.cpu().tolist(),
        "correction_model_gradients_absent": True, "correction_model_state_bitwise_unchanged": True,
        "effective_transform_max_singular_value_m": max_singular, "max_component_displacement_m": max_displacement,
        "support": {"sample_rows": len(sizes), "fit_sample_rows": len(fit_rows), "development_sample_rows": len(dev_rows),
            "natural_support_sizes": sizes.cpu().tolist(), "natural_pool_sizes_before_ceiling": pools.cpu().tolist(),
            "ceiling_truncation_fraction": float(truncated.double().mean()), "registered_ceiling": 128,
            "forced_positive_preservation": True, "forced_positive_retained_mass": retained.cpu().tolist(),
            "forced_support_sizes": forced["candidate_mask"].sum(dim=1).cpu().tolist(),
            "positive_self_mass": 0.},
    }


def _markdown(report: dict) -> str:
    lines = ["# V18.4 pre-run mathematical revision audit", "",
        f"Status: **{report['status']}**. Outcome-free diagnostics; no model selection.", "",
        "Only UJI trainingData-derived fit/development roles and synthetic constructions were used.",
        "No official validation or external test performance was accessed. No formal UJI protocol was run.", ""]
    for device_name in ("cpu", "cuda"):
        if device_name not in report.get("diagnostics", {}):
            continue
        diagnostics = report["diagnostics"][device_name]
        math_check, real = diagnostics["synthetic"], diagnostics["real_training"]
        spatial, smoke = math_check["spatial_loss"], real["smoke"]
        lines += [f"## {device_name.upper()}", "",
            f"Spatial loss: identical={spatial['identical_loss']:.12g}; near={spatial['near_loss']:.12g}; far={spatial['far_loss']:.12g}; finite nonzero logit-gradient norm={spatial['gradient_norm']:.12g}.", "",
            "| r_radio | r_geometry | max epsilon | effective epsilon | realized KL | rho |",
            "|---:|---:|---:|---:|---:|---:|"]
        for row in math_check["trust_reliability_rows"]:
            lines.append("| " + " | ".join(f"{row[key]:.9g}" for key in ("radio_reliability", "geometry_reliability", "epsilon_kl_max", "epsilon_kl_effective", "realized_kl", "rho")) + " |")
        lines += ["", "Lower reliability did not increase realized KL for a fixed direction. Zero reliability preserved q bitwise; padding remained exactly zero.", "",
            f"Real-data smoke: fit={smoke['fit_rows']} rows/{smoke['fit_coordinate_groups']} groups; development={smoke['development_rows']} rows/{smoke['development_coordinate_groups']} groups; anchors={smoke['anchors']}; APs={smoke['aps']}; s_C={smoke['spatial_scale_m']:.9g} m.", "",
            f"One {smoke['training_batch_rows']}-row forward/backward batch, zero model optimizer steps. All loss terms, model gradients, base and corrected posteriors were finite. Correction left model state bitwise unchanged and created no network gradients.", "",
            "Training-batch loss terms: " + "; ".join(f"{key}={value:.9g}" for key, value in smoke['loss_terms'].items() if key.endswith('loss') or key.endswith('nll')) + ".", "",
            f"Residual bound=30 m; real effective-transform maximum singular value={real['effective_transform_max_singular_value_m']:.9g} m; synthetic oversized-transform cap and displacement checks passed.", "",
            f"Natural support sizes on {real['support']['sample_rows']} fit/development scans: {real['support']['natural_support_sizes']}; ceiling truncation fraction={real['support']['ceiling_truncation_fraction']:.9g}. All forced positive targets were preserved.", ""]
    lines += ["## Revision semantics", "",
        "A. Exact dimensionless half energy distance replaces iterative Sinkhorn; one lambda_spatial grid remains {0.10,0.25,0.50}.",
        "B. Normalized reliability weights determine direction; max(r_radio,r_geometry) scales the registered maximum KL budget.",
        "C. q -> temperature q_T -> trust correction -> final posterior. Fitting references stay frozen at training T=1; deployment energies/mismatches and KL use q_T. Scale calibration changes only component scales.",
        "D. Local anisotropic Delaunay residual transforms are globally capped at 30 m, preserving their axes and singular-value ratio.", "",
        "This checks numerical/design contracts on small samples; it does not establish localization accuracy, trained-model convergence, or exhaustive natural-support sufficiency.",
        "Formal UJI prepare/develop/calibrate-freeze and all external formal evaluation remain pending review.", ""]
    if "failure" in report:
        lines += ["## Fail-closed failure", "", "```text", report["failure"]["traceback"], "```", ""]
    return "\n".join(lines)


def main() -> None:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("pre-run revision audit outputs already exist; never overwrite")
    before = _formal_snapshot()
    report = {"status": "RUNNING", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit_at_revision_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "working_tree_revision_audit": True, "python_executable": sys.executable, "torch_version": torch.__version__,
        "lambda_spatial_grid": list(V184_CONFIG["lambda_spatial_grid"]), "epsilon_kl_max_grid": list(V184_CONFIG["epsilon_kl_grid"]),
        "method_combinations": 9, "formal_uji_protocol_run": False, "external_formal_evaluation_run": False,
        "official_validation_accessed": False, "external_test_performance_accessed": False,
        "kl_numerical_tolerance": KL_TOLERANCE, "residual_numerical_tolerance_m": RESIDUAL_TOLERANCE_M,
        "formal_artifacts_before": before, "diagnostics": {}}
    failure = None
    try:
        with training_only_access_guard() as data_reads:
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            for name in devices:
                report["diagnostics"][name] = {"synthetic": synthetic_math_checks(torch.device(name)),
                    "real_training": real_training_checks(name)}
            train = DATA_ROOT / "UJIndoorLoc" / "trainingData.csv"
            report["training_source"] = {"path": train.relative_to(REPO_ROOT).as_posix(),
                "sha256": hashlib.sha256(train.read_bytes()).hexdigest()}
            report["audited_dataset_file_opens"] = dict(data_reads)
        after = _formal_snapshot()
        _require(before == after, "formal protocol artifact set changed")
        report["formal_artifacts_after"] = after
        report["status"] = "PASS"
    except Exception as exc:
        failure = exc
        report["status"] = "FAIL_CLOSED"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with OUTPUT_MD.open("x", encoding="utf-8") as handle:
        handle.write(_markdown(report))
    print(json.dumps({"status": report["status"], "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD),
        "devices": list(report["diagnostics"])}, indent=2))
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
