"""Outcome-free V18.5 pre-run audit.

The audit may open dataset TRAINING sources only.  It checks mathematical
contracts, explicit-mask/memory behavior, grouped roles, support capacity and
small UJI CPU/CUDA smoke.  It never creates a formal V18.5 checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    TUJI_DEVICE_FILES,
    dataset_summary,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import (
    exact_kl_energy_projection,
    select_joint_calibration,
)
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import (
    DATA_ROOT,
    build_v185_anchor_memory,
    load_tampere,
    load_tuji1,
    load_uji,
    load_uts,
    model_input,
)
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG
from fhm_dapp_tim.fhm_dapp.v18_5_protocol import (
    ROLE_SEEDS,
    TUJI_LODO_ROLE_SEEDS,
    construct_roles,
    coordinate_keys,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.v18_5_realdata_smoke import run_smoke

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_5"
DEFAULT_JSON = DEFAULT_EVIDENCE / "PRE_RUN_AUDIT_V18_5.json"
DEFAULT_MD = DEFAULT_EVIDENCE / "PRE_RUN_AUDIT_V18_5.md"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _synthetic_math_checks() -> dict[str, object]:
    q = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
    e = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    active = exact_kl_energy_projection(q, e, epsilon_kl=0.20)
    if not float(active["dual_beta"][0]) > 1.0:
        raise RuntimeError("audit failed to exercise beta>1")
    if abs(float(active["realized_kl"][0]) - 0.20) > 1e-9:
        raise RuntimeError("active KL budget not attained")

    zero = exact_kl_energy_projection(q, e, epsilon_kl=0.0)
    if not torch.equal(zero["posterior"], q):
        raise RuntimeError("epsilon=0 identity failed")

    tie_q = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    tie_e = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64)
    limit = exact_kl_energy_projection(tie_q, tie_e, epsilon_kl=1.0)
    expected = torch.tensor([[0.625, 0.375, 0.0]], dtype=torch.float64)
    if not torch.allclose(limit["posterior"], expected, atol=1e-14, rtol=0):
        raise RuntimeError("minimum-energy tie limit failed")

    base = torch.tensor(
        [[0.55, 0.25, 0.12, 0.08]], dtype=torch.float64
    )
    energy = torch.tensor(
        [[0.0, 1.0, 3.0, 7.0]], dtype=torch.float64
    )
    one = exact_kl_energy_projection(base, energy, epsilon_kl=0.08)
    two = exact_kl_energy_projection(
        base, 9.0 * energy + 77.0, epsilon_kl=0.08
    )
    if not torch.allclose(one["posterior"], two["posterior"], atol=1e-12):
        raise RuntimeError("energy affine invariance failed")

    cells = []
    for temperature in V185_CONFIG["temperature_grid"]:
        for epsilon in V185_CONFIG["epsilon_kl_grid"]:
            for scale in V185_CONFIG["scale_grid"]:
                cells.append({
                    "cell_id": f"{temperature}-{epsilon}-{scale}",
                    "group_nll": [1.0, 1.0, 1.0],
                    "group_energy_score": [2.0, 2.0, 2.0],
                    "posterior_mean_mae_m": 3.0,
                    "temperature": float(temperature),
                    "epsilon_kl": float(epsilon),
                    "scale_multiplier": float(scale),
                })
    if len(cells) != 192:
        raise RuntimeError("synthetic calibration surface is not 192 cells")
    selected = select_joint_calibration(cells)
    chosen = selected["selected_cell"]
    if (
        float(chosen["epsilon_kl"]) != 0.0
        or float(chosen["temperature"]) != 1.0
        or float(chosen["scale_multiplier"]) != 1.0
    ):
        raise RuntimeError("calibration parsimony tie law failed")

    return {
        "epsilon_zero_identity": True,
        "beta_greater_than_one": float(active["dual_beta"][0]),
        "active_realized_kl": float(active["realized_kl"][0]),
        "minimum_energy_tie_limit": True,
        "energy_affine_invariance": True,
        "synthetic_calibration_cells": 192,
        "synthetic_parsimony_choice": {
            "temperature": float(chosen["temperature"]),
            "epsilon_kl": float(chosen["epsilon_kl"]),
            "scale_multiplier": float(chosen["scale_multiplier"]),
        },
    }


def _capacity(memory) -> dict[str, object]:
    geometry = build_delaunay_geometry(
        memory["coords"], memory["building"], memory["floor"]
    )
    indices, weights = delaunay_neighbor_targets(geometry)
    positive_width = int((weights > 0).sum(dim=1).max())
    anchors = len(memory["coords"])
    primary = (
        min(int(V185_CONFIG["invariant_retrieval_k"]), anchors)
        + min(int(V185_CONFIG["prototype_retrieval_k"]), anchors)
        + min(int(V185_CONFIG["floor_retrieval_k"]), anchors)
    )
    required = primary + positive_width
    return {
        "anchors": int(anchors),
        "maximum_positive_target_width": positive_width,
        "conservative_required_capacity": int(required),
        "support_ceiling": int(V185_CONFIG["support_ceiling"]),
        "capacity_safe": bool(required <= int(V185_CONFIG["support_ceiling"])),
    }


def _audit_training_dataset(data, *, seed: int) -> dict[str, object]:
    roles = construct_roles(data, seed=int(seed))
    rows = row_indices_for_roles(data, roles)
    inputs = model_input(data)
    observed_minus_one = int(
        ((inputs.rss == -1.0) & inputs.observed_mask).sum()
    )
    raw = data.coords_m.astype(np.float32)

    keys = coordinate_keys(data)
    memories = {}
    for role in ("fit", "selection_refit"):
        origin = raw[rows[role]].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        centered = (raw - origin[None, :]).astype(np.float32)
        memory, _sample_anchor, diagnostics = build_v185_anchor_memory(
            inputs.rss,
            inputs.observed_mask,
            centered,
            data.building,
            data.floor,
            rows[role],
        )
        capacity = _capacity(memory)
        if not capacity["capacity_safe"]:
            raise RuntimeError(
                f"{data.dataset}/{role}: support capacity unsafe"
            )
        memories[role] = {
            "rows": int(len(rows[role])),
            "coordinate_groups": int(
                len({keys[int(index)] for index in rows[role].tolist()})
            ),
            "memory": diagnostics,
            "capacity": capacity,
        }

    summary = dataset_summary(data)
    return {
        "dataset": data.dataset,
        "split": data.split,
        "rows": int(len(data.coords_m)),
        "aps": int(len(data.ap_ids)),
        "observed_values_at_or_below_clip_min": int(
            summary["observed_values_at_or_below_clip_min"]
        ),
        "observed_numeric_minus_one_count": observed_minus_one,
        "visible_ap_mean": float(summary["visible_ap_mean"]),
        "visible_ap_min": int(summary["visible_ap_min"]),
        "visible_ap_max": int(summary["visible_ap_max"]),
        "roles": memories,
    }


def _training_dataset_audit(data_root: Path) -> dict[str, object]:
    loaders: dict[str, Callable] = {
        "UJIIndoorLoc": lambda: load_uji("train", data_root=data_root)[0],
        "UTSIndoorLoc": lambda: load_uts("train", data_root=data_root)[0],
        "Tampere": lambda: load_tampere("train", data_root=data_root)[0],
        "TUJI1": lambda: load_tuji1("train", data_root=data_root)[0],
    }
    result = {}
    for name, loader in loaders.items():
        result[name] = _audit_training_dataset(
            loader(), seed=ROLE_SEEDS[name]
        )

    devices = tuple(TUJI_DEVICE_FILES)
    lodo = {}
    for heldout in devices:
        training = tuple(value for value in devices if value != heldout)
        data, _ = load_tuji1(
            "train",
            devices=training,
            data_root=data_root,
        )
        lodo[heldout] = _audit_training_dataset(
            data,
            seed=TUJI_LODO_ROLE_SEEDS[heldout],
        )
        lodo[heldout]["training_devices"] = list(training)
        lodo[heldout]["heldout_device_not_loaded"] = heldout
    result["TUJI1_LODO"] = lodo
    return result


def _formal_artifact_absence() -> dict[str, bool]:
    paths = [
        DEFAULT_EVIDENCE / "uji_registry_v18_5.json",
        DEFAULT_EVIDENCE / "uji_development_v18_5.json",
        DEFAULT_EVIDENCE / "uji_calibration_v18_5.json",
        DEFAULT_EVIDENCE / "uji_method_freeze_v18_5.json",
        ROOT / "models" / "v18_5" / "uji_selection_refit.pt",
    ]
    result = {str(path.relative_to(ROOT)): not path.exists() for path in paths}
    if not all(result.values()):
        raise RuntimeError("formal V18.5 artifact already exists")
    return result


def run_audit(
    *,
    data_root: Path = DATA_ROOT,
    include_smoke: bool = True,
) -> dict[str, object]:
    result = {
        "audit_version": "v18_5_pre_run_v1_20261005",
        "scientific_status": "OUTCOME_FREE_PRE_RUN",
        "math": _synthetic_math_checks(),
        "training_only_datasets": _training_dataset_audit(data_root),
        "formal_artifacts_absent": _formal_artifact_absence(),
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
        "formal_v185_run": False,
    }
    if include_smoke:
        result["cpu_smoke"] = run_smoke("cpu")
        if torch.cuda.is_available():
            result["cuda_smoke"] = run_smoke("cuda")
        else:
            result["cuda_smoke"] = {"status": "not_available"}
    result["status"] = "PASS"
    return result


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# V18.5 Pre-run Audit",
        "",
        f"Status: **{payload['status']}**",
        "",
        "## Mathematical core",
        "",
        f"- beta>1 exercised: {payload['math']['beta_greater_than_one']:.6f}",
        f"- active KL: {payload['math']['active_realized_kl']:.9f}",
        "- epsilon=0 identity: PASS",
        "- energy affine invariance: PASS",
        "- minimum-energy tie limit: PASS",
        "- synthetic 192-cell selector: PASS",
        "",
        "## Training-only datasets",
        "",
    ]
    for name, record in payload["training_only_datasets"].items():
        if name == "TUJI1_LODO":
            continue
        lines.append(
            f"- {name}: rows={record['rows']}, APs={record['aps']}, "
            f"fit anchors={record['roles']['fit']['memory']['anchors']}, "
            f"selection-refit anchors="
            f"{record['roles']['selection_refit']['memory']['anchors']}"
        )
    lines.extend([
        "",
        "No UJI validation or external test performance was accessed.",
        "No formal V18.5 model was trained or frozen by this audit.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MD))
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    json_path = Path(args.json)
    md_path = Path(args.markdown)
    if json_path.exists() or md_path.exists():
        raise FileExistsError("V18.5 audit outputs are write-once")
    payload = run_audit(include_smoke=not args.skip_smoke)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "json": str(json_path),
        "json_sha256": _sha256_file(json_path),
        "markdown": str(md_path),
        "markdown_sha256": _sha256_file(md_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
