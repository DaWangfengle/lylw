"""Formal UJI development driver for V18.5 JRT-SP.

Protocol:
prepare
    Freeze a new V18.5 grouped-coordinate partition of UJI trainingData only.

develop
    Train lambda_spatial in {0,.1,.25,.5}.  Select epochs and the global
    lambda from the uncorrected T=1 base posterior only.

calibrate-freeze
    Refit the selected base method on selection_refit, fit one label-free radio
    compatibility reference per registered temperature on selection_refit, and
    jointly calibrate T x epsilon_KL x Student-t scale on calibration with the
    paired one-standard-error law.

No official UJI validation or external test split is opened here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import (
    posterior_mean,
    radio_trust_projection,
    select_joint_calibration,
    temperature_base_posterior,
)
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import (
    DATA_ROOT,
    CanonicalFingerprintData,
    build_v185_anchor_memory,
    load_uji,
    model_input,
)
from fhm_dapp_tim.fhm_dapp.v18_5_model import (
    V185_CONFIG,
    base_posterior_v185,
    build_v185_model,
    candidate_support_v185,
    prototype_retrieval_score_v185,
    training_spatial_scale_m,
    v185_distribution,
    v185_training_loss,
)
from fhm_dapp_tim.fhm_dapp.v18_5_protocol import (
    PROTOCOL_VERSION as ROLE_PROTOCOL_VERSION,
    ROLE_SEEDS,
    coordinate_keys,
    construct_roles,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    TemperatureRadioReference,
    fit_temperature_radio_reference,
    profile_radio_nuisance,
    radio_compatibility,
)

PROTOCOL_VERSION = "uji_v18_5_joint_radio_v1_20261005"
MODEL_SEED = 185511
SCORING_SEED = 185512
PREDICTIVE_DRAWS = 64

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_TRAIN = DATA_ROOT / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_5"
DEFAULT_MODELS = ROOT / "models" / "v18_5"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "uji_registry_v18_5.json"
DEFAULT_DEVELOPMENT = DEFAULT_EVIDENCE / "uji_development_v18_5.json"
DEFAULT_REFIT = DEFAULT_MODELS / "uji_selection_refit.pt"
DEFAULT_CALIBRATION = DEFAULT_EVIDENCE / "uji_calibration_v18_5.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "uji_method_freeze_v18_5.json"


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _git_source_state() -> dict[str, object]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode != 0
    if len(sha) != 40:
        raise RuntimeError("invalid Git HEAD")
    if dirty:
        raise RuntimeError("formal V18.5 stages require clean tracked source")
    return {"source_commit_sha": sha, "tracked_source_dirty": False}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, object], *, omit: str | None = None) -> str:
    value = copy.deepcopy(dict(payload))
    if omit:
        value.pop(omit, None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_indices(indices: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()


def _repo_relative(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _write_json_new(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_torch_new(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _cpu_memory(memory: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in memory.items()
    }


def _config_json() -> dict[str, object]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in V185_CONFIG.items()
    }


def prepare(
    *,
    train_path: str | Path = DEFAULT_TRAIN,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    path = Path(train_path).resolve()
    if path.name != "trainingData.csv":
        raise ValueError("V18.5 UJI prepare accepts trainingData.csv only")
    if Path(registry_path).exists():
        raise FileExistsError(registry_path)

    data, _ = load_uji("train", data_root=path.parents[1])
    roles = construct_roles(data, seed=ROLE_SEEDS["UJIIndoorLoc"])
    rows = row_indices_for_roles(data, roles)
    keys = coordinate_keys(data)
    role_payload = {}
    for name, indices in rows.items():
        groups = {keys[int(index)] for index in indices.tolist()}
        role_payload[name] = {
            "rows": list(map(int, indices.tolist())),
            "row_count": int(len(indices)),
            "coordinate_count": int(len(groups)),
            "row_index_sha256": _sha256_indices(indices),
        }

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "role_protocol_version": ROLE_PROTOCOL_VERSION,
        **_git_source_state(),
        "train_path": _repo_relative(path),
        "train_sha256": _sha256_file(path),
        "training_rows": int(len(data.coords_m)),
        "training_coordinate_groups": int(len(set(keys))),
        "aps": int(len(data.ap_ids)),
        "role_seed": int(ROLE_SEEDS["UJIIndoorLoc"]),
        "roles": role_payload,
        "lambda_spatial_grid": list(map(float, V185_CONFIG["lambda_spatial_grid"])),
        "temperature_grid": list(map(float, V185_CONFIG["temperature_grid"])),
        "epsilon_kl_grid": list(map(float, V185_CONFIG["epsilon_kl_grid"])),
        "scale_grid": list(map(float, V185_CONFIG["scale_grid"])),
        "model_config": _config_json(),
        "uji_official_validation_used_for_v185_selection": False,
        "external_test_performance_used_for_v185_selection": False,
        "scientific_note": (
            "V18.5 uses a fresh preregistered trainingData role seed after "
            "V18.4 mechanism development; this is not claimed as blind "
            "benchmark access."
        ),
    }
    registry["registry_sha256"] = _canonical_sha(registry)
    _write_json_new(registry_path, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry["registry_sha256"] != _canonical_sha(
        registry, omit="registry_sha256"
    ):
        raise ValueError("registry self-hash mismatch")
    train_path = REPO_ROOT / str(registry["train_path"])
    if _sha256_file(train_path) != registry["train_sha256"]:
        raise ValueError("trainingData hash mismatch")
    current = _git_source_state()
    if current["source_commit_sha"] != registry["source_commit_sha"]:
        raise RuntimeError("source revision differs from prepared registry")
    return registry


def _registered_data(registry: Mapping[str, object]):
    path = REPO_ROOT / str(registry["train_path"])
    data, _ = load_uji("train", data_root=path.parents[1])
    rows = {}
    for name, record in registry["roles"].items():
        indices = np.asarray(record["rows"], dtype=np.int64)
        if _sha256_indices(indices) != record["row_index_sha256"]:
            raise ValueError(f"role hash mismatch: {name}")
        rows[name] = indices
    return data, rows


def _coordinate_group_ids(data, indices):
    keys = coordinate_keys(data)
    return [keys[int(index)] for index in indices.tolist()]


def _group_vector(
    values: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    value = values.detach().cpu().double().numpy()
    groups: dict[tuple[int, int, float, float], list[float]] = {}
    for scalar, key in zip(value.tolist(), keys):
        groups.setdefault(key, []).append(float(scalar))
    ordered = sorted(groups)
    vector = np.asarray(
        [np.mean(groups[key]) for key in ordered], dtype=np.float64
    )
    return ordered, vector


def _coordinate_macro(values, keys) -> float:
    return float(_group_vector(values, keys)[1].mean())


def _coordinate_macro_se(values, keys) -> float:
    vector = _group_vector(values, keys)[1]
    return (
        0.0
        if len(vector) <= 1
        else float(vector.std(ddof=1) / math.sqrt(len(vector)))
    )


def _row_nll(probability, location, scale, target, model):
    component = model.student_log_prob(target, location, scale)
    return -torch.logsumexp(
        probability.clamp_min(1e-30).log() + component, dim=1
    )


def _entropy(probability):
    return -torch.sum(
        torch.where(
            probability > 0,
            probability * probability.clamp_min(1e-30).log(),
            torch.zeros_like(probability),
        ),
        dim=1,
    )


def _build_training_state(
    *,
    data: CanonicalFingerprintData,
    row_indices: np.ndarray,
    device: torch.device,
):
    inputs = model_input(data)
    raw = data.coords_m.astype(np.float32)
    origin = raw[row_indices].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    centered = (raw - origin[None, :]).astype(np.float32)
    memory, sample_anchor, memory_diagnostics = build_v185_anchor_memory(
        inputs.rss,
        inputs.observed_mask,
        centered,
        data.building,
        data.floor,
        row_indices,
    )
    spatial_scale = training_spatial_scale_m(memory)
    valid_device = data.device[row_indices]
    valid_device = valid_device[valid_device >= 0]
    n_devices = int(valid_device.max()) + 1 if len(valid_device) else 0
    n_buildings = int(memory["building"].max()) + 1
    model, _geometry, neighbours, weights = build_v185_model(
        memory=memory,
        n_aps=inputs.rss.shape[1],
        n_buildings=n_buildings,
        n_devices=n_devices,
        spatial_scale_m=spatial_scale,
        device=device,
    )
    return {
        "model": model,
        "memory": memory,
        "memory_diagnostics": memory_diagnostics,
        "sample_anchor": sample_anchor,
        "centered": centered,
        "origin": origin,
        "spatial_scale_m": spatial_scale,
        "neighbours": neighbours,
        "neighbour_weights": weights,
        "rss": inputs.rss,
        "observed_mask": inputs.observed_mask,
        "n_devices": n_devices,
    }


def _train_epoch(
    *,
    state,
    data,
    training_rows,
    lambda_spatial,
    optimizer,
    device,
    generator,
):
    model = state["model"]
    model.train()
    permutation = torch.randperm(
        len(training_rows), generator=generator
    ).numpy()
    totals: dict[str, float] = {}
    count = 0
    batch_size = int(V185_CONFIG["batch_size"])
    for start in range(0, len(permutation), batch_size):
        local = permutation[start:start + batch_size]
        rows = training_rows[local]
        anchor = torch.from_numpy(
            state["sample_anchor"][rows]
        ).long().to(device)
        if bool((anchor < 0).any()):
            raise RuntimeError("training row has no fitting anchor")
        rss = state["rss"][rows].to(device)
        mask = state["observed_mask"][rows].to(device)
        target = torch.from_numpy(
            state["centered"][rows].astype(np.float32)
        ).to(device)
        device_ids = torch.from_numpy(data.device[rows]).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        result = v185_training_loss(
            model,
            rss,
            mask,
            target,
            anchor,
            device_ids,
            state["neighbours"],
            state["neighbour_weights"],
            lambda_spatial=float(lambda_spatial),
        )
        if not bool(torch.isfinite(result["loss"])):
            raise RuntimeError("nonfinite V18.5 training loss")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(V185_CONFIG["gradient_clip"])
        )
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError("nonfinite V18.5 gradient")
        optimizer.step()
        width = len(rows)
        count += width
        for key, value in result.items():
            totals[key] = totals.get(key, 0.0) + float(value) * width
    return {key: value / count for key, value in totals.items()}


def _evaluate_base(
    *,
    state,
    data,
    row_indices,
    device,
    scoring_seed,
):
    model = state["model"]
    model.eval()
    keys = _coordinate_group_ids(data, row_indices)
    randomness = common_student_t_randomness(
        len(row_indices),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V185_CONFIG["student_df"]),
        seed=int(scoring_seed),
    )
    es_values, nll_values, mean_errors = [], [], []
    entropy, effective, top1 = [], [], []
    batch_size = int(V185_CONFIG["batch_size"])
    with torch.no_grad():
        for start in range(0, len(row_indices), batch_size):
            stop = min(start + batch_size, len(row_indices))
            rows = row_indices[start:stop]
            target = torch.from_numpy(
                state["centered"][rows].astype(np.float32)
            ).to(device)
            result = v185_distribution(
                model,
                state["rss"][rows].to(device),
                state["observed_mask"][rows].to(device),
                radio_memory={
                    "detection_prob": model.anchor_detection_prob,
                    "rss_center": model.anchor_rss_center,
                    "rss_scale": model.anchor_rss_scale,
                },
                radio_reference=None,
                temperature=1.0,
                epsilon_kl=0.0,
                scale_multiplier=1.0,
            )
            batch_randomness = {
                key: (
                    value[start:stop].to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in randomness.items()
            }
            es, _first, _second = continuous_energy_score_mc(
                result["probability"],
                result["component_location"],
                result["component_scale"],
                target,
                batch_randomness,
            )
            nll = _row_nll(
                result["probability"],
                result["component_location"],
                result["component_scale"],
                target,
                model,
            )
            mean_error = torch.linalg.vector_norm(
                result["posterior_mean"] - target, dim=1
            )
            ent = _entropy(result["probability"])
            es_values.append(es.cpu())
            nll_values.append(nll.cpu())
            mean_errors.append(mean_error.cpu())
            entropy.append(ent.cpu())
            effective.append(ent.exp().cpu())
            top1.append(result["probability"].max(dim=1).values.cpu())
    es = torch.cat(es_values)
    nll = torch.cat(nll_values)
    mean = torch.cat(mean_errors)
    return {
        "energy_score_m": _coordinate_macro(es, keys),
        "energy_score_se_m": _coordinate_macro_se(es, keys),
        "nll": _coordinate_macro(nll, keys),
        "nll_se": _coordinate_macro_se(nll, keys),
        "posterior_mean_mae_m": _coordinate_macro(mean, keys),
        "posterior_mean_mae_se_m": _coordinate_macro_se(mean, keys),
        "mean_entropy_nats": float(torch.cat(entropy).double().mean()),
        "mean_effective_anchors": float(torch.cat(effective).double().mean()),
        "mean_top1_probability": float(torch.cat(top1).double().mean()),
        "rows": int(len(row_indices)),
        "coordinate_groups": int(len(set(keys))),
    }


def _fit_one_lambda(
    *,
    lambda_spatial,
    data,
    rows,
    device,
):
    _set_seed(MODEL_SEED)
    state = _build_training_state(
        data=data, row_indices=rows["fit"], device=device
    )
    model = state["model"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V185_CONFIG["learning_rate"]),
        weight_decay=float(V185_CONFIG["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(MODEL_SEED)
    eligible = []
    trajectory = []
    for epoch in range(1, int(V185_CONFIG["maximum_epochs"]) + 1):
        train_metrics = _train_epoch(
            state=state,
            data=data,
            training_rows=rows["fit"],
            lambda_spatial=lambda_spatial,
            optimizer=optimizer,
            device=device,
            generator=generator,
        )
        record = {"epoch": epoch, "train": train_metrics}
        if epoch >= int(V185_CONFIG["minimum_epochs"]):
            metrics = _evaluate_base(
                state=state,
                data=data,
                row_indices=rows["development"],
                device=device,
                scoring_seed=SCORING_SEED,
            )
            record["development_base"] = metrics
            eligible.append({
                "epoch": epoch,
                **metrics,
                "state": _cpu_state(model),
            })
        trajectory.append(record)

    best_es = min(float(item["energy_score_m"]) for item in eligible)
    margin = float(V185_CONFIG["energy_tie_margin_m"])
    shortlist = [
        item for item in eligible
        if float(item["energy_score_m"]) <= best_es + margin
    ]
    selected = min(
        shortlist,
        key=lambda item: (
            float(item["nll"]),
            float(item["posterior_mean_mae_m"]),
            int(item["epoch"]),
        ),
    )
    model.load_state_dict(selected["state"], strict=True)
    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "fit_candidate",
        "lambda_spatial": float(lambda_spatial),
        "selected_epoch": int(selected["epoch"]),
        "origin": torch.from_numpy(state["origin"]),
        "spatial_scale_m": float(state["spatial_scale_m"]),
        "n_devices": int(state["n_devices"]),
        "memory": _cpu_memory(state["memory"]),
        "memory_diagnostics": state["memory_diagnostics"],
        "model_state": _cpu_state(model),
        "model_config": _config_json(),
    }
    report = {
        "lambda_spatial": float(lambda_spatial),
        "selected_epoch": int(selected["epoch"]),
        "selected_base_metrics": {
            key: value for key, value in selected.items()
            if key != "state"
        },
        "eligible_epoch_shortlist": [
            {key: value for key, value in item.items() if key != "state"}
            for item in shortlist
        ],
        "trajectory": trajectory,
        "anchors": int(len(state["memory"]["coords"])),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "spatial_scale_m": float(state["spatial_scale_m"]),
        "memory_diagnostics": state["memory_diagnostics"],
    }
    return checkpoint, report


def _select_global_lambda(reports):
    best_es = min(float(item["selected_base_metrics"]["energy_score_m"]) for item in reports)
    margin = float(V185_CONFIG["energy_tie_margin_m"])
    shortlist = [
        item for item in reports
        if float(item["selected_base_metrics"]["energy_score_m"])
        <= best_es + margin
    ]
    return min(
        shortlist,
        key=lambda item: (
            float(item["selected_base_metrics"]["nll"]),
            float(item["selected_base_metrics"]["posterior_mean_mae_m"]),
            float(item["lambda_spatial"]),
        ),
    )


def develop(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_path: str | Path = DEFAULT_DEVELOPMENT,
    model_dir: str | Path = DEFAULT_MODELS,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data, rows = _registered_data(registry)
    reports = []
    checkpoint_records = []
    started = time.time()
    for lam in map(float, V185_CONFIG["lambda_spatial_grid"]):
        checkpoint, report = _fit_one_lambda(
            lambda_spatial=lam,
            data=data,
            rows=rows,
            device=device,
        )
        tag = f"{lam:.2f}".replace(".", "p")
        path = Path(model_dir) / f"uji_lambda_{tag}.pt"
        _save_torch_new(path, checkpoint)
        report["checkpoint_path"] = _repo_relative(path)
        report["checkpoint_sha256"] = _sha256_file(path)
        reports.append(report)
        checkpoint_records.append({
            "lambda_spatial": lam,
            "path": _repo_relative(path),
            "sha256": _sha256_file(path),
        })
    selected = _select_global_lambda(reports)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "lambda_reports": reports,
        "checkpoints": checkpoint_records,
        "selected": {
            "lambda_spatial": float(selected["lambda_spatial"]),
            "selected_epoch": int(selected["selected_epoch"]),
            "base_metrics": selected["selected_base_metrics"],
        },
        "selection_law": (
            "ES minimum; <=0.05m ES shortlist; minimum NLL; minimum "
            "posterior-mean MAE; smaller lambda"
        ),
        "runtime_seconds": float(time.time() - started),
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    result["development_result_sha256"] = _canonical_sha(result)
    _write_json_new(output_path, result)
    return result


def _load_checkpoint(path: str | Path, *, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    memory = checkpoint["memory"]
    model, _geometry, neighbours, weights = build_v185_model(
        memory=memory,
        n_aps=int(memory["prototype_rss"].shape[1]),
        n_buildings=int(memory["building"].max()) + 1,
        n_devices=int(checkpoint["n_devices"]),
        spatial_scale_m=float(checkpoint["spatial_scale_m"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, neighbours, weights


def _train_fixed_epochs(
    *,
    lambda_spatial,
    epochs,
    data,
    row_indices,
    device,
):
    _set_seed(MODEL_SEED)
    state = _build_training_state(
        data=data, row_indices=row_indices, device=device
    )
    optimizer = torch.optim.AdamW(
        state["model"].parameters(),
        lr=float(V185_CONFIG["learning_rate"]),
        weight_decay=float(V185_CONFIG["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(MODEL_SEED)
    trajectory = []
    for _epoch in range(1, int(epochs) + 1):
        trajectory.append(_train_epoch(
            state=state,
            data=data,
            training_rows=row_indices,
            lambda_spatial=lambda_spatial,
            optimizer=optimizer,
            device=device,
            generator=generator,
        ))
    return state, trajectory


def _fit_temperature_references(
    *,
    state,
    row_indices,
    device,
) -> TemperatureRadioReference:
    model = state["model"]
    batch_size = int(V185_CONFIG["batch_size"])
    values = {float(t): [] for t in V185_CONFIG["temperature_grid"]}
    model.eval()
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        for start in range(0, len(row_indices), batch_size):
            rows = row_indices[start:start + batch_size]
            rss = state["rss"][rows].to(device)
            mask = state["observed_mask"][rows].to(device)
            with torch.no_grad():
                logits, embedding = model.retrieval_logits_v185(rss, mask)
                prototype = prototype_retrieval_score_v185(model, rss, mask)
                support = candidate_support_v185(
                    model, logits, prototype
                )
                base = base_posterior_v185(
                    model, rss, mask, embedding, logits, support
                )
                q_t = temperature_base_posterior(
                    base["probability"],
                    base["candidate_mask"],
                    temperature=temperature,
                )
            raw = profile_radio_nuisance(
                rss,
                mask,
                base["candidate_indices"],
                {
                    "detection_prob": model.anchor_detection_prob,
                    "rss_center": model.anchor_rss_center,
                    "rss_scale": model.anchor_rss_scale,
                },
                q_t,
                ap_reliability=model.v185_ap_reliability,
                degrees_freedom=float(V185_CONFIG["student_df"]),
                steps=int(V185_CONFIG["radio_profile_steps"]),
                learning_rate=float(V185_CONFIG["radio_profile_learning_rate"]),
                offset_max=float(V185_CONFIG["radio_profile_offset_max"]),
                log_scale_max=float(V185_CONFIG["radio_profile_log_scale_max"]),
                offset_weight=float(V185_CONFIG["radio_profile_offset_weight"]),
                log_scale_weight=float(V185_CONFIG["radio_profile_log_scale_weight"]),
            )
            values[temperature].append(raw["profiled_objective"].cpu())
    return fit_temperature_radio_reference({
        temperature: torch.cat(chunks)
        for temperature, chunks in values.items()
    })


def _reference_payload(reference: TemperatureRadioReference):
    return {
        str(temperature): values.detach().cpu()
        for temperature, values in reference.references.items()
    }


def _group_arrays(values, keys, ordered_groups):
    _, vector = _group_vector(values, keys)
    actual = sorted(set(keys))
    if actual != ordered_groups:
        raise RuntimeError("coordinate-group ordering mismatch")
    return vector


def _calibration_surface(
    *,
    state,
    data,
    row_indices,
    reference,
    device,
):
    model = state["model"]
    model.eval()
    keys = _coordinate_group_ids(data, row_indices)
    ordered_groups = sorted(set(keys))
    randomness = common_student_t_randomness(
        len(row_indices),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V185_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )
    cells = {}
    for temperature in map(float, V185_CONFIG["temperature_grid"]):
        for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
            for scale in map(float, V185_CONFIG["scale_grid"]):
                cell_id = f"T={temperature:.2f}|E={epsilon:.3f}|S={scale:.2f}"
                cells[cell_id] = {
                    "temperature": temperature,
                    "epsilon_kl": epsilon,
                    "scale_multiplier": scale,
                    "es": [],
                    "nll": [],
                    "mean_error": [],
                }

    batch_size = int(V185_CONFIG["batch_size"])
    for start in range(0, len(row_indices), batch_size):
        stop = min(start + batch_size, len(row_indices))
        rows = row_indices[start:stop]
        rss = state["rss"][rows].to(device)
        observed = state["observed_mask"][rows].to(device)
        target = torch.from_numpy(
            state["centered"][rows].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits, embedding = model.retrieval_logits_v185(rss, observed)
            prototype = prototype_retrieval_score_v185(
                model, rss, observed
            )
            support = candidate_support_v185(model, logits, prototype)
            base = base_posterior_v185(
                model, rss, observed, embedding, logits, support
            )
            location, base_scale = model._components(
                embedding, base["candidate_indices"]
            )
        batch_randomness = {
            key: (
                value[start:stop].to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in randomness.items()
        }

        for temperature in map(float, V185_CONFIG["temperature_grid"]):
            q_t = temperature_base_posterior(
                base["probability"],
                base["candidate_mask"],
                temperature=temperature,
            )
            raw = profile_radio_nuisance(
                rss,
                observed,
                base["candidate_indices"],
                {
                    "detection_prob": model.anchor_detection_prob,
                    "rss_center": model.anchor_rss_center,
                    "rss_scale": model.anchor_rss_scale,
                },
                q_t,
                ap_reliability=model.v185_ap_reliability,
                degrees_freedom=float(V185_CONFIG["student_df"]),
                steps=int(V185_CONFIG["radio_profile_steps"]),
                learning_rate=float(V185_CONFIG["radio_profile_learning_rate"]),
                offset_max=float(V185_CONFIG["radio_profile_offset_max"]),
                log_scale_max=float(V185_CONFIG["radio_profile_log_scale_max"]),
                offset_weight=float(V185_CONFIG["radio_profile_offset_weight"]),
                log_scale_weight=float(V185_CONFIG["radio_profile_log_scale_weight"]),
            )
            compatibility = radio_compatibility(
                raw["profiled_objective"],
                reference,
                temperature=temperature,
            )
            posterior_by_epsilon = {}
            for epsilon in map(float, V185_CONFIG["epsilon_kl_grid"]):
                if epsilon == 0.0:
                    posterior_by_epsilon[epsilon] = q_t
                else:
                    posterior_by_epsilon[epsilon] = radio_trust_projection(
                        q_t,
                        raw["energy"],
                        compatibility,
                        epsilon_kl=epsilon,
                        mask=base["candidate_mask"],
                    )["posterior"]

            for epsilon, probability in posterior_by_epsilon.items():
                mean = posterior_mean(
                    probability, location, base["candidate_mask"]
                )
                mean_error = torch.linalg.vector_norm(
                    mean - target, dim=1
                )
                for scale_multiplier in map(float, V185_CONFIG["scale_grid"]):
                    scaled = base_scale * scale_multiplier
                    es, _first, _second = continuous_energy_score_mc(
                        probability,
                        location,
                        scaled,
                        target,
                        batch_randomness,
                    )
                    nll = _row_nll(
                        probability, location, scaled, target, model
                    )
                    cell_id = (
                        f"T={temperature:.2f}|E={epsilon:.3f}|"
                        f"S={scale_multiplier:.2f}"
                    )
                    cells[cell_id]["es"].append(es.cpu())
                    cells[cell_id]["nll"].append(nll.cpu())
                    cells[cell_id]["mean_error"].append(mean_error.cpu())

    selector_cells = []
    surface = []
    for cell_id, record in cells.items():
        es = torch.cat(record.pop("es"))
        nll = torch.cat(record.pop("nll"))
        mean_error = torch.cat(record.pop("mean_error"))
        group_es = _group_arrays(es, keys, ordered_groups)
        group_nll = _group_arrays(nll, keys, ordered_groups)
        group_mean = _group_arrays(mean_error, keys, ordered_groups)
        summary = {
            "cell_id": cell_id,
            **record,
            "energy_score_m": float(group_es.mean()),
            "nll": float(group_nll.mean()),
            "posterior_mean_mae_m": float(group_mean.mean()),
        }
        surface.append(summary)
        selector_cells.append({
            **summary,
            "group_energy_score": group_es.tolist(),
            "group_nll": group_nll.tolist(),
        })
    if len(surface) != 192:
        raise RuntimeError(f"expected 192 calibration cells, got {len(surface)}")
    selected = select_joint_calibration(selector_cells)
    return {
        "surface": surface,
        "selection": selected,
        "coordinate_groups": int(len(ordered_groups)),
        "rows": int(len(row_indices)),
    }


def calibrate_freeze(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    development_path: str | Path = DEFAULT_DEVELOPMENT,
    refit_path: str | Path = DEFAULT_REFIT,
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> tuple[dict, dict]:
    registry = load_registry(registry_path)
    development = json.loads(Path(development_path).read_text(encoding="utf-8"))
    if development["development_result_sha256"] != _canonical_sha(
        development, omit="development_result_sha256"
    ):
        raise ValueError("development result self-hash mismatch")
    for path in (refit_path, calibration_path, freeze_path):
        if Path(path).exists():
            raise FileExistsError(path)
    data, rows = _registered_data(registry)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    selected = development["selected"]
    started = time.time()
    state, trajectory = _train_fixed_epochs(
        lambda_spatial=float(selected["lambda_spatial"]),
        epochs=int(selected["selected_epoch"]),
        data=data,
        row_indices=rows["selection_refit"],
        device=device,
    )
    reference = _fit_temperature_references(
        state=state,
        row_indices=rows["selection_refit"],
        device=device,
    )
    refit_checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "selection_refit",
        "lambda_spatial": float(selected["lambda_spatial"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "origin": torch.from_numpy(state["origin"]),
        "spatial_scale_m": float(state["spatial_scale_m"]),
        "n_devices": int(state["n_devices"]),
        "memory": _cpu_memory(state["memory"]),
        "memory_diagnostics": state["memory_diagnostics"],
        "model_state": _cpu_state(state["model"]),
        "temperature_radio_reference": _reference_payload(reference),
        "trajectory": trajectory,
        "model_config": _config_json(),
    }
    _save_torch_new(refit_path, refit_checkpoint)

    calibration_eval = _calibration_surface(
        state=state,
        data=data,
        row_indices=rows["calibration"],
        reference=reference,
        device=device,
    )
    selected_cell = calibration_eval["selection"]["selected_cell"]
    calibration = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development["development_result_sha256"],
        "selection_refit_checkpoint": _repo_relative(refit_path),
        "selection_refit_checkpoint_sha256": _sha256_file(refit_path),
        **calibration_eval,
        "selected": {
            "temperature": float(selected_cell["temperature"]),
            "epsilon_kl": float(selected_cell["epsilon_kl"]),
            "scale_multiplier": float(selected_cell["scale_multiplier"]),
            "nll": float(selected_cell["nll"]),
            "energy_score_m": float(selected_cell["energy_score_m"]),
            "posterior_mean_mae_m": float(
                selected_cell["posterior_mean_mae_m"]
            ),
        },
        "selection_law": (
            "paired NLL one-SE; paired ES one-SE; minimum posterior-mean "
            "MAE; smaller epsilon; T closest 1; scale closest 1"
        ),
        "runtime_seconds": float(time.time() - started),
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    calibration["calibration_result_sha256"] = _canonical_sha(calibration)
    _write_json_new(calibration_path, calibration)

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "source_commit_sha": registry["source_commit_sha"],
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development["development_result_sha256"],
        "calibration_result_sha256": calibration["calibration_result_sha256"],
        "lambda_spatial": float(selected["lambda_spatial"]),
        "selected_epoch": int(selected["selected_epoch"]),
        "posterior_temperature": float(selected_cell["temperature"]),
        "epsilon_kl": float(selected_cell["epsilon_kl"]),
        "scale_multiplier": float(selected_cell["scale_multiplier"]),
        "decoder": "posterior_mean_of_component_locations",
        "correction": "radio_only_exact_KL_projection",
        "geometry_posterior_correction": False,
        "selection_refit_checkpoint": _repo_relative(refit_path),
        "selection_refit_checkpoint_sha256": _sha256_file(refit_path),
        "official_validation_accessed": False,
        "external_test_performance_accessed": False,
    }
    freeze["freeze_sha256"] = _canonical_sha(freeze)
    _write_json_new(freeze_path, freeze)
    return calibration, freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    develop_parser = sub.add_parser("develop")
    develop_parser.add_argument("--device", default="cuda")
    calibrate_parser = sub.add_parser("calibrate-freeze")
    calibrate_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
    elif args.command == "develop":
        print(json.dumps(
            develop(device_name=args.device), indent=2, sort_keys=True
        ))
    else:
        calibration, freeze = calibrate_freeze(
            device_name=args.device
        )
        print(json.dumps(
            {"calibration": calibration["selected"], "freeze": freeze},
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
