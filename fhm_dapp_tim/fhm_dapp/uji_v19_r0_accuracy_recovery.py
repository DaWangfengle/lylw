"""UJI V19-R0 accuracy-recovery experiment.

This is a development experiment, not a final benchmark evaluation.

It reuses the exact V18.5 FIT/DEVELOPMENT coordinate roles. Member 1 is the
frozen V18.5 lambda=0, epoch-27 checkpoint. Members 2-16 are independently
initialized copies of the same V18.5 model/training law, trained on the same
FIT rows for exactly 27 epochs with lambda_spatial=0.

The development evaluation changes only the inference capacity:
- every member is scored over the complete anchor dictionary;
- each member retains Top-L=192;
- V17-style inverse-radial member reliability;
- alpha=1 common anchor consensus with K=192;
- member-specific residual locations/scales remain separate;
- both posterior mean and full-predictive spatial median are reported.

No radio projection, temperature calibration, scale calibration, UJI official
validation or external test is allowed in R0.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import common_student_t_randomness
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import load_uji, model_input
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    DEFAULT_DEVELOPMENT as V185_DEVELOPMENT,
    DEFAULT_FREEZE as V185_FREEZE,
    DEFAULT_MODELS as V185_MODELS,
    DEFAULT_REGISTRY as V185_REGISTRY,
    _build_training_state,
    _canonical_sha,
    _coordinate_group_ids,
    _coordinate_macro,
    _coordinate_macro_se,
    _cpu_memory,
    _cpu_state,
    _load_checkpoint,
    _save_torch_new,
    _set_seed,
    _sha256_file,
    _sha256_indices,
    _train_epoch,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.v19_r0_core import (
    V19_R0_CONFIG,
    consensus_joint_distribution,
    dense_member_distribution,
    member_seeds,
    member_top_l,
    predictive_actions_and_scores,
    validate_v19_r0_config,
)


PROTOCOL_VERSION = "uji_v19_r0_accuracy_recovery_v1_20261006"

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_EVIDENCE = ROOT / "evidence" / "v19_r0"
DEFAULT_MODELS = ROOT / "models" / "v19_r0"
DEFAULT_CACHE = DEFAULT_MODELS / "development_cache"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v19_r0.json"
DEFAULT_RESULT = DEFAULT_EVIDENCE / "development_result_v19_r0.json"
INHERITED_MEMBER = V185_MODELS / "uji_lambda_0p00.pt"


def _git_state() -> dict[str, object]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    if len(sha) != 40:
        raise RuntimeError("invalid Git HEAD")
    if dirty == 1:
        raise RuntimeError("V19-R0 requires clean tracked source")
    if dirty != 0:
        raise RuntimeError("cannot inspect tracked source state")
    return {"source_commit_sha": sha, "tracked_source_dirty": False}


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _member_checkpoint(index: int) -> Path:
    seeds = member_seeds()
    member = int(index)
    if not 1 <= member <= len(seeds):
        raise ValueError("member index outside V19-R0 panel")
    if member == 1:
        return INHERITED_MEMBER
    return DEFAULT_MODELS / f"member_{member:02d}_seed{seeds[member - 1]}.pt"


def _member_cache(index: int) -> Path:
    seeds = member_seeds()
    member = int(index)
    if not 1 <= member <= len(seeds):
        raise ValueError("member index outside V19-R0 panel")
    return (
        DEFAULT_CACHE
        / f"member_{member:02d}_seed{seeds[member - 1]}_development.pt"
    )


def _validate_historical_v185() -> tuple[dict, dict, dict]:
    registry = _json(V185_REGISTRY)
    development = _json(V185_DEVELOPMENT)
    freeze = _json(V185_FREEZE)
    if registry.get("registry_sha256") != _canonical_sha(
        registry, omit="registry_sha256"
    ):
        raise ValueError("historical V18.5 registry self-hash mismatch")
    if development.get("development_result_sha256") != _canonical_sha(
        development, omit="development_result_sha256"
    ):
        raise ValueError("historical V18.5 development self-hash mismatch")
    if freeze.get("freeze_sha256") != _canonical_sha(
        freeze, omit="freeze_sha256"
    ):
        raise ValueError("historical V18.5 freeze self-hash mismatch")
    if development.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.5 development links another registry")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V18.5 freeze links another registry")

    selected = development["selected"]
    if float(selected["lambda_spatial"]) != 0.0:
        raise ValueError("V19-R0 requires the observed V18.5 lambda=0 result")
    if int(selected["selected_epoch"]) != int(
        V19_R0_CONFIG["fixed_epochs"]
    ):
        raise ValueError("V18.5 selected epoch no longer matches V19-R0")
    if float(freeze["lambda_spatial"]) != 0.0:
        raise ValueError("V18.5 freeze lambda mismatch")

    lambda_zero = next(
        record
        for record in development["lambda_reports"]
        if float(record["lambda_spatial"]) == 0.0
    )
    if int(lambda_zero["selected_epoch"]) != int(
        V19_R0_CONFIG["fixed_epochs"]
    ):
        raise ValueError("V18.5 lambda=0 checkpoint epoch mismatch")
    if not INHERITED_MEMBER.is_file():
        raise FileNotFoundError(INHERITED_MEMBER)
    if _sha256_file(INHERITED_MEMBER) != lambda_zero["checkpoint_sha256"]:
        raise ValueError("inherited V18.5 lambda=0 checkpoint hash mismatch")

    train_path = REPO_ROOT / str(registry["train_path"])
    if train_path.name != "trainingData.csv":
        raise ValueError("V19-R0 accepts the UJI trainingData registry only")
    if _sha256_file(train_path) != registry["train_sha256"]:
        raise ValueError("historical V18.5 trainingData hash mismatch")
    for name in ("fit", "development"):
        rows = np.asarray(registry["roles"][name]["rows"], dtype=np.int64)
        if _sha256_indices(rows) != registry["roles"][name][
            "row_index_sha256"
        ]:
            raise ValueError(f"historical V18.5 role hash mismatch: {name}")
    return registry, development, freeze


def prepare(
    *,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict[str, object]:
    validate_v19_r0_config()
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    v185, development, freeze = _validate_historical_v185()
    source = _git_state()
    seeds = member_seeds()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        **source,
        "scientific_status": (
            "post-V18.5 accuracy-recovery development experiment; "
            "not independent benchmark confirmation"
        ),
        "v18_5_registry_path": str(
            Path(V185_REGISTRY).relative_to(REPO_ROOT).as_posix()
        ),
        "v18_5_registry_file_sha256": _sha256_file(V185_REGISTRY),
        "v18_5_registry_sha256": v185["registry_sha256"],
        "v18_5_development_path": str(
            Path(V185_DEVELOPMENT).relative_to(REPO_ROOT).as_posix()
        ),
        "v18_5_development_file_sha256": _sha256_file(V185_DEVELOPMENT),
        "v18_5_development_sha256": development[
            "development_result_sha256"
        ],
        "v18_5_freeze_file_sha256": _sha256_file(V185_FREEZE),
        "v18_5_freeze_sha256": freeze["freeze_sha256"],
        "train_path": v185["train_path"],
        "train_sha256": v185["train_sha256"],
        "fit_role": copy.deepcopy(v185["roles"]["fit"]),
        "development_role": copy.deepcopy(
            v185["roles"]["development"]
        ),
        "inherited_member_path": str(
            INHERITED_MEMBER.relative_to(REPO_ROOT).as_posix()
        ),
        "inherited_member_sha256": _sha256_file(INHERITED_MEMBER),
        "inherited_member_seed": int(seeds[0]),
        "member_seeds": list(map(int, seeds)),
        "member_prefixes": list(
            map(int, V19_R0_CONFIG["member_prefixes"])
        ),
        "member_top_l": int(V19_R0_CONFIG["member_top_l"]),
        "consensus_top_k": int(V19_R0_CONFIG["consensus_top_k"]),
        "consensus_alpha": float(V19_R0_CONFIG["consensus_alpha"]),
        "consensus_epsilon": float(
            V19_R0_CONFIG["consensus_epsilon"]
        ),
        "fixed_lambda_spatial": 0.0,
        "fixed_epochs": int(V19_R0_CONFIG["fixed_epochs"]),
        "student_df": float(V19_R0_CONFIG["student_df"]),
        "predictive_draws": int(V19_R0_CONFIG["predictive_draws"]),
        "scoring_seed": int(V19_R0_CONFIG["scoring_seed"]),
        "v18_5_development_reference": {
            "posterior_mean_mae_m": float(
                development["selected"]["base_metrics"][
                    "posterior_mean_mae_m"
                ]
            ),
            "energy_score_m": float(
                development["selected"]["base_metrics"]["energy_score_m"]
            ),
            "nll": float(
                development["selected"]["base_metrics"]["nll"]
            ),
        },
        "radio_projection_used": False,
        "temperature_calibration_used": False,
        "scale_calibration_used": False,
        "uji_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    payload["registry_sha256"] = _canonical_sha(payload)
    _write_json_new(destination, payload)
    return payload


def load_registry(
    path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, object]:
    payload = _json(path)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V19-R0 registry")
    if payload.get("registry_sha256") != _canonical_sha(
        payload, omit="registry_sha256"
    ):
        raise ValueError("V19-R0 registry self-hash mismatch")
    if payload.get("source_commit_sha") != _git_state()[
        "source_commit_sha"
    ]:
        raise RuntimeError("V19-R0 source revision changed after prepare")
    v185, development, freeze = _validate_historical_v185()
    if payload["v18_5_registry_sha256"] != v185["registry_sha256"]:
        raise ValueError("V19-R0 links another V18.5 registry")
    if payload["v18_5_development_sha256"] != development[
        "development_result_sha256"
    ]:
        raise ValueError("V19-R0 links another V18.5 development result")
    if payload["v18_5_freeze_sha256"] != freeze["freeze_sha256"]:
        raise ValueError("V19-R0 links another V18.5 freeze")
    if payload["inherited_member_sha256"] != _sha256_file(
        INHERITED_MEMBER
    ):
        raise ValueError("V19-R0 inherited member changed")
    if list(map(int, payload["member_seeds"])) != list(member_seeds()):
        raise ValueError("V19-R0 member seeds changed")
    if int(payload["fixed_epochs"]) != int(V19_R0_CONFIG["fixed_epochs"]):
        raise ValueError("V19-R0 fixed epoch contract changed")
    return payload


def _data_and_rows(registry: Mapping[str, object]):
    train_path = REPO_ROOT / str(registry["train_path"])
    data, _ = load_uji("train", data_root=train_path.parents[1])
    fit = np.asarray(registry["fit_role"]["rows"], dtype=np.int64)
    development = np.asarray(
        registry["development_role"]["rows"], dtype=np.int64
    )
    if _sha256_indices(fit) != registry["fit_role"][
        "row_index_sha256"
    ]:
        raise ValueError("V19-R0 FIT rows changed")
    if _sha256_indices(development) != registry[
        "development_role"
    ]["row_index_sha256"]:
        raise ValueError("V19-R0 DEVELOPMENT rows changed")
    return data, fit, development


def train_member(
    member_index: int,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
) -> dict[str, object]:
    registry = load_registry(registry_path)
    index = int(member_index)
    seeds = member_seeds()
    if index == 1:
        raise ValueError(
            "member 1 is inherited from V18.5 and must not be retrained"
        )
    if not 2 <= index <= len(seeds):
        raise ValueError("member index outside trainable V19-R0 panel")
    output = _member_checkpoint(index)
    if output.exists():
        raise FileExistsError(output)

    data, fit_rows, _development_rows = _data_and_rows(registry)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(seeds[index - 1])
    _set_seed(seed)
    state = _build_training_state(
        data=data, row_indices=fit_rows, device=device
    )
    optimizer = torch.optim.AdamW(
        state["model"].parameters(),
        lr=float(V185_CONFIG["learning_rate"]),
        weight_decay=float(V185_CONFIG["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trajectory = []
    started = time.time()
    for epoch in range(1, int(V19_R0_CONFIG["fixed_epochs"]) + 1):
        record = _train_epoch(
            state=state,
            data=data,
            training_rows=fit_rows,
            lambda_spatial=0.0,
            optimizer=optimizer,
            device=device,
            generator=generator,
        )
        trajectory.append({"epoch": epoch, **record})

    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "member_index": index,
        "member_seed": seed,
        "training_role": "fit",
        "training_row_index_sha256": registry["fit_role"][
            "row_index_sha256"
        ],
        "lambda_spatial": 0.0,
        "epochs": int(V19_R0_CONFIG["fixed_epochs"]),
        "origin": torch.from_numpy(state["origin"]),
        "spatial_scale_m": float(state["spatial_scale_m"]),
        "n_devices": int(state["n_devices"]),
        "memory": _cpu_memory(state["memory"]),
        "memory_diagnostics": state["memory_diagnostics"],
        "model_state": _cpu_state(state["model"]),
        "trajectory": trajectory,
        "v18_5_model_config": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in V185_CONFIG.items()
        },
        "runtime_seconds": float(time.time() - started),
        "uji_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    _save_torch_new(output, checkpoint)
    return {
        "member_index": index,
        "seed": seed,
        "checkpoint": str(output),
        "checkpoint_sha256": _sha256_file(output),
        "epochs": int(V19_R0_CONFIG["fixed_epochs"]),
        "runtime_seconds": checkpoint["runtime_seconds"],
    }


def _validate_member_checkpoint(
    registry: Mapping[str, object],
    member_index: int,
) -> dict[str, object]:
    path = _member_checkpoint(member_index)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if member_index == 1:
        if _sha256_file(path) != registry["inherited_member_sha256"]:
            raise ValueError("inherited member hash mismatch")
        if float(payload["lambda_spatial"]) != 0.0:
            raise ValueError("inherited member lambda changed")
        if int(payload["selected_epoch"]) != int(
            V19_R0_CONFIG["fixed_epochs"]
        ):
            raise ValueError("inherited member epoch changed")
        return payload
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"member {member_index} protocol mismatch")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError(f"member {member_index} registry mismatch")
    if int(payload.get("member_index", -1)) != int(member_index):
        raise ValueError(f"member {member_index} identity mismatch")
    expected_seed = member_seeds()[member_index - 1]
    if int(payload.get("member_seed", -1)) != int(expected_seed):
        raise ValueError(f"member {member_index} seed mismatch")
    if float(payload.get("lambda_spatial", -1.0)) != 0.0:
        raise ValueError(f"member {member_index} lambda mismatch")
    if int(payload.get("epochs", -1)) != int(
        V19_R0_CONFIG["fixed_epochs"]
    ):
        raise ValueError(f"member {member_index} epoch mismatch")
    return payload


def cache_member(
    member_index: int,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    device_name: str = "cuda",
) -> dict[str, object]:
    registry = load_registry(registry_path)
    index = int(member_index)
    _validate_member_checkpoint(registry, index)
    output = _member_cache(index)
    if output.exists():
        raise FileExistsError(output)

    data, _fit_rows, development_rows = _data_and_rows(registry)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_path = _member_checkpoint(index)
    model, _neighbors, _weights = _load_checkpoint(
        checkpoint_path, device=device
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    inputs = model_input(data)
    origin = checkpoint["origin"].cpu().numpy().astype(np.float32)
    centered = data.coords_m.astype(np.float32) - origin[None, :]

    probability_parts = []
    location_parts = []
    scale_parts = []
    top_weight_parts = []
    top_index_parts = []
    radial_parts = []
    retained_parts = []
    batch_size = int(V185_CONFIG["batch_size"])
    started = time.time()
    model.eval()
    for start in range(0, len(development_rows), batch_size):
        rows = development_rows[start:start + batch_size]
        dense = dense_member_distribution(
            model,
            inputs.rss[rows].to(device),
            inputs.observed_mask[rows].to(device),
        )
        top = member_top_l(
            dense["probability"],
            dense["component_location"],
            dense["component_scale"],
            top_l=int(V19_R0_CONFIG["member_top_l"]),
            student_df=float(V19_R0_CONFIG["student_df"]),
        )
        probability_parts.append(dense["probability"].cpu())
        location_parts.append(dense["component_location"].cpu())
        scale_parts.append(dense["component_scale"].cpu())
        top_weight_parts.append(top["weights"].cpu())
        top_index_parts.append(top["indices"].cpu())
        radial_parts.append(top["radial_scale"].cpu())
        retained_parts.append(top["retained_mass"].cpu())

    cache = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "member_index": index,
        "member_seed": int(member_seeds()[index - 1]),
        "checkpoint_path": str(
            checkpoint_path.relative_to(REPO_ROOT).as_posix()
        ),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "development_row_index_sha256": registry[
            "development_role"
        ]["row_index_sha256"],
        "development_rows": int(len(development_rows)),
        "anchor_count": int(model.n_anchors),
        "origin": torch.from_numpy(origin.copy()),
        "target_centered": torch.from_numpy(
            centered[development_rows].copy()
        ),
        "probability": torch.cat(probability_parts),
        "component_location": torch.cat(location_parts),
        "component_scale": torch.cat(scale_parts),
        "top_l_weights": torch.cat(top_weight_parts),
        "top_l_indices": torch.cat(top_index_parts),
        "radial_scale": torch.cat(radial_parts),
        "retained_mass": torch.cat(retained_parts),
        "runtime_seconds": float(time.time() - started),
    }
    if cache["probability"].shape[1] < int(V19_R0_CONFIG["member_top_l"]):
        raise RuntimeError("anchor dictionary smaller than registered L=192")
    _save_torch_new(output, cache)
    return {
        "member_index": index,
        "cache": str(output),
        "cache_sha256": _sha256_file(output),
        "mean_top_l_retained_mass": float(
            cache["retained_mass"].double().mean()
        ),
        "minimum_top_l_retained_mass": float(
            cache["retained_mass"].double().min()
        ),
        "runtime_seconds": cache["runtime_seconds"],
    }


def _group_vector_tensor(
    values: torch.Tensor,
    keys,
) -> np.ndarray:
    groups: dict[tuple[int, int, float, float], list[float]] = {}
    for value, key in zip(values.double().tolist(), keys):
        groups.setdefault(key, []).append(float(value))
    return np.asarray(
        [np.mean(groups[key]) for key in sorted(groups)],
        dtype=np.float64,
    )


def _paired(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    mean = float(difference.mean())
    se = 0.0 if len(difference) <= 1 else float(
        difference.std(ddof=1) / math.sqrt(len(difference))
    )
    return {
        "candidate_minus_reference_m": mean,
        "se_m": se,
        "ci95_low_m": mean - 1.96 * se,
        "ci95_high_m": mean + 1.96 * se,
        "fraction_groups_candidate_better": float(
            np.mean(difference < 0.0)
        ),
    }


def _metric_payload(
    scored: Mapping[str, torch.Tensor],
    keys,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    vectors = {
        "posterior_mean_mae": _group_vector_tensor(
            scored["posterior_mean_error"], keys
        ),
        "predictive_spatial_median_mae": _group_vector_tensor(
            scored["predictive_spatial_median_error"], keys
        ),
        "energy_score": _group_vector_tensor(
            scored["energy_score"], keys
        ),
        "nll": _group_vector_tensor(scored["nll"], keys),
    }
    metrics = {
        "posterior_mean_mae_m": float(
            vectors["posterior_mean_mae"].mean()
        ),
        "posterior_mean_mae_se_m": float(
            vectors["posterior_mean_mae"].std(ddof=1)
            / math.sqrt(len(vectors["posterior_mean_mae"]))
        ),
        "predictive_spatial_median_mae_m": float(
            vectors["predictive_spatial_median_mae"].mean()
        ),
        "predictive_spatial_median_mae_se_m": float(
            vectors["predictive_spatial_median_mae"].std(ddof=1)
            / math.sqrt(len(vectors["predictive_spatial_median_mae"]))
        ),
        "energy_score_m": float(vectors["energy_score"].mean()),
        "energy_score_se_m": float(
            vectors["energy_score"].std(ddof=1)
            / math.sqrt(len(vectors["energy_score"]))
        ),
        "nll": float(vectors["nll"].mean()),
        "nll_se": float(
            vectors["nll"].std(ddof=1)
            / math.sqrt(len(vectors["nll"]))
        ),
        "decision_delta_spatial_minus_mean": _paired(
            vectors["predictive_spatial_median_mae"],
            vectors["posterior_mean_mae"],
        ),
    }
    return metrics, vectors


def evaluate(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_RESULT,
    device_name: str = "cuda",
) -> dict[str, object]:
    registry = load_registry(registry_path)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    data, _fit_rows, development_rows = _data_and_rows(registry)
    keys = _coordinate_group_ids(data, development_rows)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    caches = []
    for index in range(1, int(V19_R0_CONFIG["member_count"]) + 1):
        path = _member_cache(index)
        if not path.is_file():
            raise FileNotFoundError(path)
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"member {index} cache protocol mismatch")
        if cache.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"member {index} cache registry mismatch")
        if int(cache.get("member_index", -1)) != index:
            raise ValueError(f"member {index} cache identity mismatch")
        if cache.get("development_row_index_sha256") != registry[
            "development_role"
        ]["row_index_sha256"]:
            raise ValueError(f"member {index} cache row mismatch")
        caches.append(cache)

    anchor_counts = {int(cache["anchor_count"]) for cache in caches}
    if len(anchor_counts) != 1:
        raise ValueError("member anchor dictionaries differ")
    anchors = next(iter(anchor_counts))
    if anchors < int(V19_R0_CONFIG["consensus_top_k"]):
        raise RuntimeError("anchor dictionary smaller than K=192")
    origins = [cache["origin"].float() for cache in caches]
    if any(not torch.equal(origins[0], value) for value in origins[1:]):
        raise ValueError("member coordinate origins differ")

    randomness = common_student_t_randomness(
        len(development_rows),
        draws=int(V19_R0_CONFIG["predictive_draws"]),
        degrees_freedom=float(V19_R0_CONFIG["student_df"]),
        seed=int(V19_R0_CONFIG["scoring_seed"]),
    )
    batch_size = 32
    prefix_store = {
        int(E): {
            "posterior_mean_error": [],
            "predictive_spatial_median_error": [],
            "energy_score": [],
            "nll": [],
            "member_weight_entropy": [],
            "member_weight_max": [],
            "consensus_top1": [],
        }
        for E in V19_R0_CONFIG["member_prefixes"]
    }
    inherited_dense = {
        "posterior_mean_error": [],
        "predictive_spatial_median_error": [],
        "energy_score": [],
        "nll": [],
    }
    individual_mean_errors = {
        index: []
        for index in range(1, int(V19_R0_CONFIG["member_count"]) + 1)
    }
    member_point_disagreement = []
    pairwise_top1_agreement = []

    started = time.time()
    for start in range(0, len(development_rows), batch_size):
        stop = min(start + batch_size, len(development_rows))
        target = caches[0]["target_centered"][start:stop].to(device)
        local_randomness = {
            key: (
                value[start:stop].to(device)
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for key, value in randomness.items()
        }

        member_points = []
        member_top1 = []
        for member_index, cache in enumerate(caches, start=1):
            probability = cache["probability"][start:stop].to(device)
            component_location = cache[
                "component_location"
            ][start:stop].to(device)
            point = torch.sum(
                probability.unsqueeze(2) * component_location, dim=1
            )
            member_points.append(point)
            member_top1.append(probability.argmax(dim=1))
            individual_mean_errors[member_index].append(
                torch.linalg.vector_norm(point - target, dim=1).cpu()
            )
        point_panel = torch.stack(member_points)
        panel_center = point_panel.mean(dim=0)
        member_point_disagreement.append(
            torch.sqrt(
                (
                    (point_panel - panel_center.unsqueeze(0))
                    .square()
                    .sum(dim=2)
                    .mean(dim=0)
                ).clamp_min(0.0)
            ).cpu()
        )
        top1_panel = torch.stack(member_top1)
        pair_index = torch.triu_indices(
            len(caches), len(caches), offset=1, device=device
        )
        pairwise_top1_agreement.append(
            (
                top1_panel[pair_index[0]]
                == top1_panel[pair_index[1]]
            ).float().mean(dim=0).cpu()
        )

        dense_score = predictive_actions_and_scores(
            caches[0]["probability"][start:stop].to(device),
            caches[0]["component_location"][start:stop].to(device),
            caches[0]["component_scale"][start:stop].to(device),
            target,
            local_randomness,
            degrees_freedom=float(V19_R0_CONFIG["student_df"]),
        )
        for name in inherited_dense:
            inherited_dense[name].append(dense_score[name].cpu())

        top_weights = torch.stack(
            [cache["top_l_weights"][start:stop] for cache in caches]
        ).to(device)
        top_indices = torch.stack(
            [cache["top_l_indices"][start:stop] for cache in caches]
        ).to(device)
        radial = torch.stack(
            [cache["radial_scale"][start:stop] for cache in caches]
        ).to(device)
        location = torch.stack(
            [cache["component_location"][start:stop] for cache in caches]
        ).to(device)
        scale = torch.stack(
            [cache["component_scale"][start:stop] for cache in caches]
        ).to(device)

        for E in map(int, V19_R0_CONFIG["member_prefixes"]):
            joint = consensus_joint_distribution(
                top_weights,
                top_indices,
                radial,
                location,
                scale,
                members=E,
                top_k=int(V19_R0_CONFIG["consensus_top_k"]),
                alpha=float(V19_R0_CONFIG["consensus_alpha"]),
                epsilon=float(V19_R0_CONFIG["consensus_epsilon"]),
            )
            scored = predictive_actions_and_scores(
                joint["probability"],
                joint["component_location"],
                joint["component_scale"],
                target,
                local_randomness,
                degrees_freedom=float(V19_R0_CONFIG["student_df"]),
            )
            store = prefix_store[E]
            for name in (
                "posterior_mean_error",
                "predictive_spatial_median_error",
                "energy_score",
                "nll",
            ):
                store[name].append(scored[name].cpu())
            member_weight = joint["member_weight"]
            store["member_weight_entropy"].append(
                (
                    -member_weight
                    * member_weight.clamp_min(1e-30).log()
                ).sum(dim=0).cpu()
            )
            store["member_weight_max"].append(
                member_weight.max(dim=0).values.cpu()
            )
            store["consensus_top1"].append(
                joint["consensus_anchor_probability"].max(
                    dim=1
                ).values.cpu()
            )

    inherited_scored = {
        name: torch.cat(values)
        for name, values in inherited_dense.items()
    }
    inherited_metrics, inherited_vectors = _metric_payload(
        inherited_scored, keys
    )
    individual_member_metrics = {}
    for member_index, chunks in individual_mean_errors.items():
        values = torch.cat(chunks)
        vector = _group_vector_tensor(values, keys)
        individual_member_metrics[str(member_index)] = {
            "seed": int(member_seeds()[member_index - 1]),
            "dense_posterior_mean_mae_m": float(vector.mean()),
            "dense_posterior_mean_mae_se_m": float(
                vector.std(ddof=1) / math.sqrt(len(vector))
            ),
        }
    disagreement = torch.cat(member_point_disagreement).double()
    top1_agreement = torch.cat(pairwise_top1_agreement).double()
    diversity = {
        "mean_member_point_rms_disagreement_m": float(
            disagreement.mean()
        ),
        "p90_member_point_rms_disagreement_m": float(
            torch.quantile(disagreement, 0.90)
        ),
        "mean_pairwise_top1_anchor_agreement": float(
            top1_agreement.mean()
        ),
    }

    profiles: dict[str, object] = {}
    vectors_by_e: dict[int, dict[str, np.ndarray]] = {}
    for E, store in prefix_store.items():
        scored = {
            name: torch.cat(store[name])
            for name in (
                "posterior_mean_error",
                "predictive_spatial_median_error",
                "energy_score",
                "nll",
            )
        }
        metrics, vectors = _metric_payload(scored, keys)
        vectors_by_e[E] = vectors
        metrics.update({
            "E": E,
            "L": int(V19_R0_CONFIG["member_top_l"]),
            "K": int(V19_R0_CONFIG["consensus_top_k"]),
            "alpha": float(V19_R0_CONFIG["consensus_alpha"]),
            "mean_member_weight_entropy": float(
                torch.cat(store["member_weight_entropy"]).double().mean()
            ),
            "mean_max_member_weight": float(
                torch.cat(store["member_weight_max"]).double().mean()
            ),
            "mean_consensus_top1": float(
                torch.cat(store["consensus_top1"]).double().mean()
            ),
            "mean_member_top_l_retained_mass": float(
                torch.stack([
                    cache["retained_mass"] for cache in caches[:E]
                ]).double().mean()
            ),
        })
        profiles[str(E)] = metrics

    reference_vectors = vectors_by_e[1]
    paired_vs_e1 = {}
    for E in map(int, V19_R0_CONFIG["member_prefixes"]):
        if E == 1:
            continue
        paired_vs_e1[str(E)] = {
            "posterior_mean_mae": _paired(
                vectors_by_e[E]["posterior_mean_mae"],
                reference_vectors["posterior_mean_mae"],
            ),
            "predictive_spatial_median_mae": _paired(
                vectors_by_e[E]["predictive_spatial_median_mae"],
                reference_vectors["predictive_spatial_median_mae"],
            ),
            "energy_score": _paired(
                vectors_by_e[E]["energy_score"],
                reference_vectors["energy_score"],
            ),
        }

    e1 = profiles["1"]
    e1_vs_dense = {
        "posterior_mean_mae": _paired(
            vectors_by_e[1]["posterior_mean_mae"],
            inherited_vectors["posterior_mean_mae"],
        ),
        "predictive_spatial_median_mae": _paired(
            vectors_by_e[1]["predictive_spatial_median_mae"],
            inherited_vectors["predictive_spatial_median_mae"],
        ),
        "energy_score": _paired(
            vectors_by_e[1]["energy_score"],
            inherited_vectors["energy_score"],
        ),
    }

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "scientific_status": (
            "accuracy-recovery development result; no member count is selected "
            "and no final method is frozen by R0"
        ),
        "development_rows": int(len(development_rows)),
        "coordinate_groups": int(len(set(keys))),
        "inherited_v18_5_development_reference": registry[
            "v18_5_development_reference"
        ],
        "inherited_member_dense_all_anchor": inherited_metrics,
        "individual_member_dense_posterior_mean": individual_member_metrics,
        "member_diversity": diversity,
        "E1_top192_consensus": e1,
        "E1_top192_minus_dense_all_anchor": e1_vs_dense,
        "profiles": profiles,
        "paired_vs_E1": paired_vs_e1,
        "member_cache_sha256": {
            str(index): _sha256_file(_member_cache(index))
            for index in range(
                1, int(V19_R0_CONFIG["member_count"]) + 1
            )
        },
        "runtime_seconds": float(time.time() - started),
        "radio_projection_used": False,
        "temperature_calibration_used": False,
        "scale_calibration_used": False,
        "member_count_selected": False,
        "uji_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    result["result_sha256"] = _canonical_sha(result)
    _write_json_new(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")

    train = sub.add_parser("train-member")
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--device", default="cuda")

    cache = sub.add_parser("cache-member")
    cache.add_argument("--member-index", type=int, required=True)
    cache.add_argument("--device", default="cuda")

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare()
    elif args.command == "train-member":
        payload = train_member(
            args.member_index, device_name=args.device
        )
    elif args.command == "cache-member":
        payload = cache_member(
            args.member_index, device_name=args.device
        )
    else:
        payload = evaluate(device_name=args.device)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
