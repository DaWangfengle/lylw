"""Formal UJI development driver for V18.4.

This module uses only UJI trainingData.csv and the fresh V18.4 coordinate-group
roles.  It never opens the pre-exposed official validation file.

Stages
------
prepare
    Freeze UJI training-source hash and fit/development/calibration row roles.

develop
    For each registered lambda_OT:
      * train on fit;
      * select epoch from the uncorrected base posterior on development;
      * fit label-free correction references on fit only;
      * evaluate the three registered epsilon_KL values on development.
    Select the global 3 x 3 method by the preregistered lexicographic rule.

calibrate-freeze
    Refit the selected method on selection_refit for the frozen epoch, fit the
    correction reference on selection_refit, and use calibration only for the
    registered final-posterior temperature x Student-t scale surface.

No external/provided test split is read by any stage in this module.
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
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.v18_4_core import (
    masked_softmax,
    tail_mass_diagnostics,
    weighted_component_geometric_median,
)
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import (
    DATA_ROOT,
    CanonicalFingerprintData,
    load_uji,
)
from fhm_dapp_tim.fhm_dapp.v18_4_model import (
    CorrectionReference,
    V184_CONFIG,
    apply_correction,
    base_posterior,
    build_v184_model,
    candidate_support,
    fit_correction_reference,
    prototype_retrieval_score,
    raw_correction_energies,
    training_spatial_scale_m,
    v184_distribution,
    v184_training_loss,
)
from fhm_dapp_tim.fhm_dapp.v18_4_protocol import (
    ROLE_SEEDS,
    construct_roles,
    coordinate_keys,
    row_indices_for_roles,
)


PROTOCOL_VERSION = "uji_v18_4_development_v1_20261002"
MODEL_SEED = 184411
SCORING_SEED = 184412
PREDICTIVE_DRAWS = 64
ENERGY_SHORTLIST_MARGIN_M = 0.05
CALIBRATION_TEMPERATURE_GRID = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
    2.50,
    3.00,
)
CALIBRATION_SCALE_GRID = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_TRAIN = DATA_ROOT / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v18_4"
DEFAULT_MODELS = ROOT / "models" / "v18_4"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "uji_registry_v18_4.json"
DEFAULT_DEVELOPMENT_RESULT = DEFAULT_EVIDENCE / "uji_development_v18_4.json"
DEFAULT_REFIT_CHECKPOINT = DEFAULT_MODELS / "uji_selection_refit.pt"
DEFAULT_CALIBRATION_RESULT = DEFAULT_EVIDENCE / "uji_calibration_v18_4.json"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "uji_method_freeze_v18_4.json"


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


def _repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(
            f"formal V18.4 artifact must live under repository root: {resolved}"
        ) from exc


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(payload: Mapping[str, object], *, omit: str | None = None) -> str:
    value = copy.deepcopy(dict(payload))
    if omit is not None:
        value.pop(omit, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_indices(indices: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _write_json_new(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _save_torch_new(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


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
    result = {}
    for key, value in V184_CONFIG.items():
        result[key] = list(value) if isinstance(value, tuple) else value
    return result


def prepare(
    *,
    train_path: str | Path = DEFAULT_TRAIN,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict:
    """Freeze V18.4 UJI role rows without opening official validation."""

    destination = Path(registry_path)
    if destination.exists():
        raise FileExistsError(destination)
    path = Path(train_path).resolve()
    if path.name != "trainingData.csv":
        raise ValueError("V18.4 UJI development must use trainingData.csv")

    data, _reference = load_uji("train", data_root=path.parents[1])
    roles = construct_roles(
        data,
        seed=ROLE_SEEDS["UJIIndoorLoc"],
    )
    rows = row_indices_for_roles(data, roles)
    keys = coordinate_keys(data)

    role_payload = {}
    for name, indices in rows.items():
        group_set = {
            keys[int(index)]
            for index in indices.tolist()
        }
        role_payload[name] = {
            "rows": list(map(int, indices.tolist())),
            "row_count": int(len(indices)),
            "row_index_sha256": _sha256_indices(indices),
            "coordinate_count": int(len(group_set)),
        }

    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "UJI is a V18.4 method-development benchmark; official "
            "validationData is pre-exposed and is not opened by this protocol"
        ),
        "official_validation_opened": False,
        "train_path": _repo_relative(path),
        "train_sha256": _sha256_file(path),
        "training_rows": int(len(data.coords_m)),
        "training_coordinate_groups": int(len(set(keys))),
        "aps": int(len(data.ap_ids)),
        "role_seed": int(ROLE_SEEDS["UJIIndoorLoc"]),
        "roles": role_payload,
        "lambda_ot_grid": list(map(float, V184_CONFIG["lambda_ot_grid"])),
        "epsilon_kl_grid": list(map(float, V184_CONFIG["epsilon_kl_grid"])),
        "model_config": _config_json(),
    }
    registry["registry_sha256"] = _canonical_sha(registry)
    _write_json_new(destination, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("registry_sha256") != _canonical_sha(
        registry,
        omit="registry_sha256",
    ):
        raise ValueError("V18.4 UJI registry self-hash mismatch")
    train = _resolve_repo_path(registry["train_path"])
    if _sha256_file(train) != registry["train_sha256"]:
        raise ValueError("UJI trainingData SHA no longer matches registry")
    if registry.get("official_validation_opened") is not False:
        raise ValueError("registry does not preserve official-validation boundary")
    return registry


def _load_registered_data(
    registry: Mapping[str, object],
) -> tuple[CanonicalFingerprintData, dict[str, np.ndarray]]:
    path = _resolve_repo_path(str(registry["train_path"]))
    data, _reference = load_uji("train", data_root=path.parents[1])
    rows = {}
    for name, record in registry["roles"].items():
        indices = np.asarray(record["rows"], dtype=np.int64)
        if _sha256_indices(indices) != record["row_index_sha256"]:
            raise ValueError(f"role row hash mismatch: {name}")
        rows[name] = indices
    return data, rows


def _coordinate_group_ids(
    data: CanonicalFingerprintData,
    indices: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    keys = coordinate_keys(data)
    return [keys[int(index)] for index in indices.tolist()]


def _coordinate_macro(
    values: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
) -> float:
    value = values.detach().cpu().double()
    if len(value) != len(keys):
        raise ValueError("metric/key row count mismatch")
    groups: dict[tuple[int, int, float, float], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    group_values = [
        value[torch.as_tensor(rows, dtype=torch.long)].mean()
        for rows in groups.values()
    ]
    return float(torch.stack(group_values).mean())


def _coordinate_macro_se(
    values: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
) -> float:
    value = values.detach().cpu().double()
    groups: dict[tuple[int, int, float, float], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    group_values = torch.stack(
        [
            value[torch.as_tensor(rows, dtype=torch.long)].mean()
            for rows in groups.values()
        ]
    )
    if len(group_values) <= 1:
        return 0.0
    return float(group_values.std(unbiased=True) / math.sqrt(len(group_values)))


def _row_nll(
    probability: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    log_component = model.student_log_prob(
        target,
        location,
        scale,
    )
    return -torch.logsumexp(
        probability.clamp(min=1e-30).log() + log_component,
        dim=1,
    )


def _prediction_entropy(probability: torch.Tensor) -> torch.Tensor:
    return -torch.sum(
        torch.where(
            probability > 0.0,
            probability * probability.clamp(min=1e-30).log(),
            torch.zeros_like(probability),
        ),
        dim=1,
    )


def _evaluate_base(
    *,
    model: torch.nn.Module,
    rss_all: torch.Tensor,
    coords_centered: np.ndarray,
    data: CanonicalFingerprintData,
    row_indices: np.ndarray,
    device: torch.device,
    include_gm: bool,
    batch_size: int,
    scoring_seed: int,
) -> dict[str, float]:
    """Evaluate uncorrected q_theta; no radio/geometry correction."""

    model.eval()
    keys = _coordinate_group_ids(data, row_indices)
    randomness = common_student_t_randomness(
        len(row_indices),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V184_CONFIG["student_df"]),
        seed=int(scoring_seed),
    )
    row_es = []
    row_nll = []
    row_gm_error = []
    row_mean_error = []
    entropy = []
    effective = []
    top1 = []
    support_size = []

    with torch.no_grad():
        for start in range(0, len(row_indices), int(batch_size)):
            stop = min(start + int(batch_size), len(row_indices))
            rows = row_indices[start:stop]
            rss = rss_all[rows].to(device)
            target = torch.from_numpy(
                coords_centered[rows].astype(np.float32)
            ).to(device)
            distribution = v184_distribution(model, rss)
            probability = distribution["candidate_weights"]
            location = distribution["component_locations"]
            scale = distribution["component_scales"]

            batch_randomness = {
                key: (
                    value[start:stop].to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in randomness.items()
            }
            es, _first, _second = continuous_energy_score_mc(
                probability,
                location,
                scale,
                target,
                batch_randomness,
            )
            row_es.append(es.cpu())
            row_nll.append(
                _row_nll(
                    probability,
                    location,
                    scale,
                    target,
                    model,
                ).cpu()
            )
            ent = _prediction_entropy(probability)
            entropy.append(ent.cpu())
            effective.append(ent.exp().cpu())
            top1.append(probability.max(dim=1).values.cpu())
            support_size.append(
                distribution["candidate_mask"].sum(dim=1).cpu()
            )
            if include_gm:
                gm = weighted_component_geometric_median(
                    probability,
                    location,
                    mask=distribution["candidate_mask"],
                )
                mean = torch.sum(
                    probability.unsqueeze(2) * location,
                    dim=1,
                )
                row_gm_error.append(
                    torch.linalg.vector_norm(gm - target, dim=1).cpu()
                )
                row_mean_error.append(
                    torch.linalg.vector_norm(mean - target, dim=1).cpu()
                )

    es_values = torch.cat(row_es)
    nll_values = torch.cat(row_nll)
    result = {
        "energy_score_m": _coordinate_macro(es_values, keys),
        "energy_score_se_m": _coordinate_macro_se(es_values, keys),
        "nll": _coordinate_macro(nll_values, keys),
        "nll_se": _coordinate_macro_se(nll_values, keys),
        "mean_entropy_nats": float(torch.cat(entropy).double().mean()),
        "mean_effective_anchors": float(torch.cat(effective).double().mean()),
        "mean_top1_probability": float(torch.cat(top1).double().mean()),
        "mean_support_size": float(
            torch.cat(support_size).double().mean()
        ),
        "coordinate_groups": int(len(set(keys))),
        "rows": int(len(row_indices)),
    }
    if include_gm:
        gm_error = torch.cat(row_gm_error)
        mean_error = torch.cat(row_mean_error)
        result.update(
            {
                "gm_mae_m": _coordinate_macro(gm_error, keys),
                "gm_mae_se_m": _coordinate_macro_se(gm_error, keys),
                "posterior_mean_mae_m": _coordinate_macro(
                    mean_error,
                    keys,
                ),
            }
        )
    return result


def _build_training_model(
    *,
    data: CanonicalFingerprintData,
    row_indices: np.ndarray,
    rss_all: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict[str, torch.Tensor],
    np.ndarray,
    np.ndarray,
    float,
    torch.Tensor,
    torch.Tensor,
]:
    raw = data.coords_m.astype(np.float32)
    origin = raw[row_indices].mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    centered = (raw - origin[None, :]).astype(np.float32)
    memory, sample_anchor = build_anchor_memory(
        rss_all,
        centered,
        data.building,
        data.floor,
        row_indices,
    )
    scale = training_spatial_scale_m(memory)
    valid_device = data.device[row_indices]
    valid_device = valid_device[valid_device >= 0]
    n_devices = int(valid_device.max()) + 1 if len(valid_device) else 0
    n_buildings = int(memory["building"].max().item()) + 1
    model, _geometry, neighbours, weights = build_v184_model(
        memory=memory,
        n_aps=rss_all.shape[1],
        n_buildings=n_buildings,
        n_devices=n_devices,
        spatial_scale_m=scale,
        device=device,
    )
    model.v184_n_devices = int(n_devices)
    return (
        model,
        memory,
        sample_anchor,
        centered,
        scale,
        neighbours,
        weights,
    )


def _train_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rss_all: torch.Tensor,
    coords_centered: np.ndarray,
    devices: np.ndarray,
    sample_anchor: np.ndarray,
    training_rows: np.ndarray,
    neighbours: torch.Tensor,
    weights: torch.Tensor,
    lambda_ot: float,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(
        len(training_rows),
        generator=generator,
    ).numpy()
    sums: dict[str, float] = {}
    count = 0
    batch_size = int(V184_CONFIG["batch_size"])
    for start in range(0, len(permutation), batch_size):
        local = permutation[start:start + batch_size]
        rows = training_rows[local]
        target_anchor = torch.from_numpy(
            sample_anchor[rows]
        ).long().to(device)
        if bool((target_anchor < 0).any()):
            raise RuntimeError("training row has no fitting anchor")
        rss = rss_all[rows].to(device)
        target = torch.from_numpy(
            coords_centered[rows].astype(np.float32)
        ).to(device)
        device_id = torch.from_numpy(devices[rows]).long().to(device)

        optimizer.zero_grad(set_to_none=True)
        result = v184_training_loss(
            model,
            rss,
            target,
            target_anchor,
            device_id,
            neighbours,
            weights,
            lambda_ot=float(lambda_ot),
        )
        if not bool(torch.isfinite(result["loss"])):
            raise RuntimeError("non-finite V18.4 training loss")
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(V184_CONFIG["gradient_clip"]),
        )
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise RuntimeError("non-finite V18.4 model gradient")
        optimizer.step()

        batch_rows = len(rows)
        count += batch_rows
        for name, value in result.items():
            scalar = float(value.detach())
            sums[name] = sums.get(name, 0.0) + scalar * batch_rows

    return {
        name: value / max(count, 1)
        for name, value in sums.items()
    }


def _fit_one_lambda(
    *,
    lambda_ot: float,
    data: CanonicalFingerprintData,
    rows: Mapping[str, np.ndarray],
    rss_all: torch.Tensor,
    device: torch.device,
) -> tuple[dict, dict]:
    """Train one lambda_OT and choose its base epoch before correction."""

    _set_seed(MODEL_SEED)
    (
        model,
        memory,
        sample_anchor,
        centered,
        spatial_scale,
        neighbours,
        weights,
    ) = _build_training_model(
        data=data,
        row_indices=rows["fit"],
        rss_all=rss_all,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V184_CONFIG["learning_rate"]),
        weight_decay=float(V184_CONFIG["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(MODEL_SEED)

    trajectory = []
    best_energy = float("inf")
    shortlist: list[dict] = []
    maximum_epochs = int(V184_CONFIG["maximum_epochs"])
    minimum_epochs = int(V184_CONFIG["minimum_epochs"])

    for epoch in range(1, maximum_epochs + 1):
        train_metrics = _train_epoch(
            model=model,
            optimizer=optimizer,
            rss_all=rss_all,
            coords_centered=centered,
            devices=data.device,
            sample_anchor=sample_anchor,
            training_rows=rows["fit"],
            neighbours=neighbours,
            weights=weights,
            lambda_ot=float(lambda_ot),
            device=device,
            generator=generator,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
        }
        if epoch >= minimum_epochs:
            development = _evaluate_base(
                model=model,
                rss_all=rss_all,
                coords_centered=centered,
                data=data,
                row_indices=rows["development"],
                device=device,
                include_gm=False,
                batch_size=int(V184_CONFIG["batch_size"]),
                scoring_seed=SCORING_SEED,
            )
            record["development_base"] = development
            best_energy = min(
                best_energy,
                float(development["energy_score_m"]),
            )
            if (
                float(development["energy_score_m"])
                <= best_energy + ENERGY_SHORTLIST_MARGIN_M
            ):
                shortlist.append(
                    {
                        "epoch": epoch,
                        "energy_score_m": float(
                            development["energy_score_m"]
                        ),
                        "nll": float(development["nll"]),
                        "state": _cpu_state(model),
                    }
                )
            shortlist = [
                item
                for item in shortlist
                if item["energy_score_m"]
                <= best_energy + ENERGY_SHORTLIST_MARGIN_M
            ]
        trajectory.append(record)

    if not shortlist:
        raise RuntimeError("V18.4 epoch shortlist is empty")

    final_candidates = []
    for item in shortlist:
        model.load_state_dict(item["state"], strict=True)
        metrics = _evaluate_base(
            model=model,
            rss_all=rss_all,
            coords_centered=centered,
            data=data,
            row_indices=rows["development"],
            device=device,
            include_gm=True,
            batch_size=int(V184_CONFIG["batch_size"]),
            scoring_seed=SCORING_SEED,
        )
        final_candidates.append(
            {
                "epoch": int(item["epoch"]),
                **metrics,
                "state": item["state"],
            }
        )
    selected = min(
        final_candidates,
        key=lambda item: (
            float(item["gm_mae_m"]),
            float(item["nll"]),
            int(item["epoch"]),
        ),
    )
    model.load_state_dict(selected["state"], strict=True)

    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "fit_candidate",
        "lambda_ot": float(lambda_ot),
        "selected_epoch": int(selected["epoch"]),
        "origin": torch.from_numpy(
            data.coords_m[rows["fit"]].mean(
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
        ),
        "spatial_scale_m": float(spatial_scale),
        "n_devices": int(model.v184_n_devices),
        "memory": _cpu_memory(memory),
        "model_state": _cpu_state(model),
        "model_config": _config_json(),
    }
    report = {
        "lambda_ot": float(lambda_ot),
        "selected_epoch": int(selected["epoch"]),
        "selected_base_metrics": {
            key: value
            for key, value in selected.items()
            if key != "state"
        },
        "eligible_epoch_shortlist": [
            {
                key: value
                for key, value in item.items()
                if key != "state"
            }
            for item in final_candidates
        ],
        "trajectory": trajectory,
        "anchors": int(len(memory["coords"])),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "spatial_scale_m": float(spatial_scale),
    }
    return checkpoint, report


def _load_model_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    device: torch.device,
) -> torch.nn.Module:
    memory = checkpoint["memory"]
    model, _geometry, _neighbours, _weights = build_v184_model(
        memory=memory,
        n_aps=int(memory["prototype_rss"].shape[1]),
        n_buildings=int(memory["building"].max().item()) + 1,
        n_devices=int(checkpoint["n_devices"]),
        spatial_scale_m=float(checkpoint["spatial_scale_m"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _fit_reference_for_rows(
    *,
    model: torch.nn.Module,
    rss_all: torch.Tensor,
    row_indices: np.ndarray,
    device: torch.device,
) -> CorrectionReference:
    radio = []
    geometry = []
    probability = []
    masks = []
    offset = []
    scale = []
    batch_size = int(V184_CONFIG["batch_size"])

    model.eval()
    model.zero_grad(set_to_none=True)
    for start in range(0, len(row_indices), batch_size):
        stop = min(start + batch_size, len(row_indices))
        rows = row_indices[start:stop]
        rss = rss_all[rows].to(device)
        with torch.no_grad():
            invariant, embedding = model.retrieval_logits(rss)
            prototype = prototype_retrieval_score(model, rss)
            support = candidate_support(
                model,
                invariant,
                prototype,
            )
            base = base_posterior(
                model,
                rss,
                embedding,
                invariant,
                support,
            )
        raw = raw_correction_energies(
            model,
            rss,
            base["probability"],
            support,
        )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(
                "correction reference profiling touched model gradients"
            )
        radio.append(raw["radio_energy_raw"].cpu())
        geometry.append(raw["geometry_energy_raw"].cpu())
        probability.append(base["probability"].detach().cpu())
        masks.append(base["candidate_mask"].cpu())
        offset.append(raw["style_offset"].cpu())
        scale.append(raw["style_scale"].cpu())

    return fit_correction_reference(
        radio_energy_raw=torch.cat(radio),
        geometry_energy_raw=torch.cat(geometry),
        base_probability=torch.cat(probability),
        candidate_mask=torch.cat(masks),
        style_offset=torch.cat(offset),
        style_scale=torch.cat(scale),
    )


def _reference_payload(reference: CorrectionReference) -> dict[str, torch.Tensor]:
    return {
        "radio_energy_scale": reference.radio_energy_scale.detach().cpu(),
        "geometry_energy_scale": reference.geometry_energy_scale.detach().cpu(),
        "radio_mismatch_reference": (
            reference.radio_mismatch_reference.detach().cpu()
        ),
        "geometry_mismatch_reference": (
            reference.geometry_mismatch_reference.detach().cpu()
        ),
    }


def _reference_from_payload(payload: Mapping[str, torch.Tensor]) -> CorrectionReference:
    return CorrectionReference(
        radio_energy_scale=payload["radio_energy_scale"],
        geometry_energy_scale=payload["geometry_energy_scale"],
        radio_mismatch_reference=payload["radio_mismatch_reference"],
        geometry_mismatch_reference=payload[
            "geometry_mismatch_reference"
        ],
    )


def _evaluate_corrected_grid(
    *,
    model: torch.nn.Module,
    reference: CorrectionReference,
    epsilon_values: Sequence[float],
    rss_all: torch.Tensor,
    coords_centered: np.ndarray,
    data: CanonicalFingerprintData,
    row_indices: np.ndarray,
    device: torch.device,
) -> dict[float, dict[str, float]]:
    """Evaluate all KL budgets with one radio profiling pass per scan."""

    keys = _coordinate_group_ids(data, row_indices)
    randomness = common_student_t_randomness(
        len(row_indices),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V184_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )
    collectors = {
        float(epsilon): {
            "es": [],
            "nll": [],
            "gm": [],
            "mean": [],
            "entropy": [],
            "top1": [],
            "rho": [],
            "kl": [],
            "radio_reliability": [],
            "geometry_reliability": [],
        }
        for epsilon in epsilon_values
    }
    batch_size = int(V184_CONFIG["batch_size"])

    model.eval()
    for start in range(0, len(row_indices), batch_size):
        stop = min(start + batch_size, len(row_indices))
        rows = row_indices[start:stop]
        rss = rss_all[rows].to(device)
        target = torch.from_numpy(
            coords_centered[rows].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            invariant, embedding = model.retrieval_logits(rss)
            prototype = prototype_retrieval_score(model, rss)
            support = candidate_support(model, invariant, prototype)
            base = base_posterior(
                model,
                rss,
                embedding,
                invariant,
                support,
            )
            location, component_scale = model._candidate_components(
                embedding,
                base["candidate_indices"],
            )
        raw = raw_correction_energies(
            model,
            rss,
            base["probability"],
            support,
        )
        batch_randomness = {
            key: (
                value[start:stop].to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in randomness.items()
        }

        for epsilon in epsilon_values:
            eps = float(epsilon)
            corrected = apply_correction(
                base_probability=base["probability"],
                candidate_mask=base["candidate_mask"],
                raw=raw,
                reference=reference,
                epsilon_kl=eps,
            )
            p = corrected["posterior"]
            with torch.no_grad():
                es, _first, _second = continuous_energy_score_mc(
                    p,
                    location,
                    component_scale,
                    target,
                    batch_randomness,
                )
                nll = _row_nll(
                    p,
                    location,
                    component_scale,
                    target,
                    model,
                )
                gm = weighted_component_geometric_median(
                    p,
                    location,
                    mask=base["candidate_mask"],
                )
                mean = torch.sum(p.unsqueeze(2) * location, dim=1)
                ent = _prediction_entropy(p)

            item = collectors[eps]
            item["es"].append(es.cpu())
            item["nll"].append(nll.cpu())
            item["gm"].append(
                torch.linalg.vector_norm(gm - target, dim=1).cpu()
            )
            item["mean"].append(
                torch.linalg.vector_norm(mean - target, dim=1).cpu()
            )
            item["entropy"].append(ent.cpu())
            item["top1"].append(p.max(dim=1).values.cpu())
            item["rho"].append(corrected["rho"].cpu())
            item["kl"].append(corrected["realized_kl"].cpu())
            item["radio_reliability"].append(
                corrected["radio_reliability"].cpu()
            )
            item["geometry_reliability"].append(
                corrected["geometry_reliability"].cpu()
            )

    result = {}
    for epsilon, item in collectors.items():
        es = torch.cat(item["es"])
        nll = torch.cat(item["nll"])
        gm = torch.cat(item["gm"])
        mean = torch.cat(item["mean"])
        entropy = torch.cat(item["entropy"])
        result[float(epsilon)] = {
            "energy_score_m": _coordinate_macro(es, keys),
            "energy_score_se_m": _coordinate_macro_se(es, keys),
            "nll": _coordinate_macro(nll, keys),
            "nll_se": _coordinate_macro_se(nll, keys),
            "gm_mae_m": _coordinate_macro(gm, keys),
            "gm_mae_se_m": _coordinate_macro_se(gm, keys),
            "posterior_mean_mae_m": _coordinate_macro(mean, keys),
            "mean_entropy_nats": float(entropy.double().mean()),
            "mean_effective_anchors": float(entropy.exp().double().mean()),
            "mean_top1_probability": float(
                torch.cat(item["top1"]).double().mean()
            ),
            "mean_rho": float(torch.cat(item["rho"]).double().mean()),
            "mean_realized_kl": float(
                torch.cat(item["kl"]).double().mean()
            ),
            "mean_radio_reliability": float(
                torch.cat(item["radio_reliability"]).double().mean()
            ),
            "mean_geometry_reliability": float(
                torch.cat(item["geometry_reliability"]).double().mean()
            ),
            "rows": int(len(row_indices)),
            "coordinate_groups": int(len(set(keys))),
        }
    return result


def select_method_record(
    records: Sequence[Mapping[str, object]],
) -> dict:
    """Apply the frozen 9-combination lexicographic selection law."""

    if not records:
        raise ValueError("method records cannot be empty")
    best_energy = min(float(record["energy_score_m"]) for record in records)
    shortlist = [
        dict(record)
        for record in records
        if float(record["energy_score_m"])
        <= best_energy + ENERGY_SHORTLIST_MARGIN_M
    ]
    return min(
        shortlist,
        key=lambda record: (
            float(record["gm_mae_m"]),
            float(record["nll"]),
            float(record["lambda_ot"]),
            float(record["epsilon_kl"]),
        ),
    )


def develop(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    result_path: str | Path = DEFAULT_DEVELOPMENT_RESULT,
    model_dir: str | Path = DEFAULT_MODELS,
    device_name: str = "cuda",
) -> dict:
    destination = Path(result_path)
    if destination.exists():
        raise FileExistsError(destination)
    registry = load_registry(registry_path)
    data, rows = _load_registered_data(registry)
    rss_all = torch.from_numpy(data.model_rss()).float()
    device = torch.device(device_name)
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    lambda_reports = []
    method_records = []

    started = time.perf_counter()
    for lambda_ot in map(float, V184_CONFIG["lambda_ot_grid"]):
        checkpoint, report = _fit_one_lambda(
            lambda_ot=lambda_ot,
            data=data,
            rows=rows,
            rss_all=rss_all,
            device=device,
        )
        token = f"{lambda_ot:.2f}".replace(".", "p")
        checkpoint_path = model_root / f"uji_lambda_{token}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)

        model = _load_model_checkpoint(checkpoint, device=device)
        origin = checkpoint["origin"].numpy()
        centered = (
            data.coords_m.astype(np.float32) - origin[None, :]
        ).astype(np.float32)
        reference = _fit_reference_for_rows(
            model=model,
            rss_all=rss_all,
            row_indices=rows["fit"],
            device=device,
        )
        checkpoint["correction_reference"] = _reference_payload(reference)
        _save_torch_new(checkpoint_path, checkpoint)
        checkpoint_sha = _sha256_file(checkpoint_path)

        grid = _evaluate_corrected_grid(
            model=model,
            reference=reference,
            epsilon_values=tuple(
                map(float, V184_CONFIG["epsilon_kl_grid"])
            ),
            rss_all=rss_all,
            coords_centered=centered,
            data=data,
            row_indices=rows["development"],
            device=device,
        )
        report["checkpoint_path"] = str(checkpoint_path.resolve())
        report["checkpoint_sha256"] = checkpoint_sha
        report["correction_reference"] = {
            "radio_energy_scale": float(reference.radio_energy_scale),
            "geometry_energy_scale": float(reference.geometry_energy_scale),
            "radio_mismatch_reference_rows": int(
                reference.radio_mismatch_reference.numel()
            ),
            "geometry_mismatch_reference_rows": int(
                reference.geometry_mismatch_reference.numel()
            ),
        }
        report["corrected_development_grid"] = {
            f"{epsilon:.3f}": metrics
            for epsilon, metrics in sorted(grid.items())
        }
        lambda_reports.append(report)

        for epsilon, metrics in grid.items():
            method_records.append(
                {
                    "lambda_ot": float(lambda_ot),
                    "epsilon_kl": float(epsilon),
                    "selected_epoch": int(checkpoint["selected_epoch"]),
                    "checkpoint_path": _repo_relative(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    **metrics,
                }
            )

    selected = select_method_record(method_records)

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "UJI trainingData development only; no official validation or "
            "external provided test accessed"
        ),
        "registry_sha256": registry["registry_sha256"],
        "official_validation_opened": False,
        "reserved_external_test_opened": False,
        "selection_rule": (
            "min Energy Score; <=0.05m shortlist; min GM MAE; min NLL; "
            "smaller lambda_OT; smaller epsilon_KL"
        ),
        "lambda_reports": lambda_reports,
        "method_grid": method_records,
        "selected": selected,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload["development_result_sha256"] = _canonical_sha(payload)
    _write_json_new(destination, payload)
    return payload


def _train_fixed_epochs(
    *,
    data: CanonicalFingerprintData,
    training_rows: np.ndarray,
    rss_all: torch.Tensor,
    lambda_ot: float,
    epochs: int,
    device: torch.device,
) -> tuple[dict, torch.nn.Module, np.ndarray]:
    _set_seed(MODEL_SEED)
    (
        model,
        memory,
        sample_anchor,
        centered,
        spatial_scale,
        neighbours,
        weights,
    ) = _build_training_model(
        data=data,
        row_indices=training_rows,
        rss_all=rss_all,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(V184_CONFIG["learning_rate"]),
        weight_decay=float(V184_CONFIG["weight_decay"]),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(MODEL_SEED)
    trajectory = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_epoch(
            model=model,
            optimizer=optimizer,
            rss_all=rss_all,
            coords_centered=centered,
            devices=data.device,
            sample_anchor=sample_anchor,
            training_rows=training_rows,
            neighbours=neighbours,
            weights=weights,
            lambda_ot=float(lambda_ot),
            device=device,
            generator=generator,
        )
        trajectory.append({"epoch": epoch, "train": metrics})

    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "selection_refit",
        "lambda_ot": float(lambda_ot),
        "selected_epoch": int(epochs),
        "origin": torch.from_numpy(
            data.coords_m[training_rows].mean(
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
        ),
        "spatial_scale_m": float(spatial_scale),
        "n_devices": int(model.v184_n_devices),
        "memory": _cpu_memory(memory),
        "model_state": _cpu_state(model),
        "model_config": _config_json(),
        "trajectory": trajectory,
    }
    return checkpoint, model, centered


def _cache_corrected_distribution(
    *,
    model: torch.nn.Module,
    reference: CorrectionReference,
    epsilon_kl: float,
    rss_all: torch.Tensor,
    row_indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    probability = []
    mask = []
    location = []
    scale = []
    batch_size = int(V184_CONFIG["batch_size"])
    model.eval()
    for start in range(0, len(row_indices), batch_size):
        stop = min(start + batch_size, len(row_indices))
        rows = row_indices[start:stop]
        distribution = v184_distribution(
            model,
            rss_all[rows].to(device),
            correction_reference=reference,
            epsilon_kl=float(epsilon_kl),
            final_temperature=1.0,
            scale_multiplier=1.0,
        )
        probability.append(
            distribution["corrected_probability"].detach().cpu()
        )
        mask.append(distribution["candidate_mask"].cpu())
        location.append(distribution["component_locations"].detach().cpu())
        scale.append(distribution["component_scales"].detach().cpu())
    return {
        "probability": torch.cat(probability),
        "mask": torch.cat(mask),
        "location": torch.cat(location),
        "scale": torch.cat(scale),
    }


def _calibration_nll(
    *,
    cache: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
    model: torch.nn.Module,
    temperature: float,
    scale_multiplier: float,
    device: torch.device,
) -> float:
    probability = cache["probability"].to(device)
    mask = cache["mask"].to(device)
    calibrated = masked_softmax(
        probability.clamp(min=1e-30).log() / float(temperature),
        mask,
    )
    location = cache["location"].to(device)
    scale = cache["scale"].to(device) * float(scale_multiplier)
    row = _row_nll(
        calibrated,
        location,
        scale,
        target.to(device),
        model,
    ).cpu()
    return _coordinate_macro(row, keys)


def _safe_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    x = first.detach().cpu().double()
    y = second.detach().cpu().double()
    if len(x) < 2 or float(x.std(unbiased=False)) <= 1e-12 or float(
        y.std(unbiased=False)
    ) <= 1e-12:
        return 0.0
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def _selected_calibration_metrics(
    *,
    cache: Mapping[str, torch.Tensor],
    target: torch.Tensor,
    keys: Sequence[tuple[int, int, float, float]],
    model: torch.nn.Module,
    temperature: float,
    scale_multiplier: float,
    device: torch.device,
) -> dict[str, object]:
    probability = cache["probability"].to(device)
    mask = cache["mask"].to(device)
    p = masked_softmax(
        probability.clamp(min=1e-30).log() / float(temperature),
        mask,
    )
    location = cache["location"].to(device)
    scale = cache["scale"].to(device) * float(scale_multiplier)
    target_device = target.to(device)
    row_nll = _row_nll(p, location, scale, target_device, model)

    randomness = common_student_t_randomness(
        len(target),
        draws=PREDICTIVE_DRAWS,
        degrees_freedom=float(V184_CONFIG["student_df"]),
        seed=SCORING_SEED,
    )
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in randomness.items()
    }
    row_es, _first, _second = continuous_energy_score_mc(
        p,
        location,
        scale,
        target_device,
        randomness,
    )
    gm = weighted_component_geometric_median(
        p,
        location,
        mask=mask,
    )
    mean = torch.sum(p.unsqueeze(2) * location, dim=1)
    gm_error = torch.linalg.vector_norm(gm - target_device, dim=1)
    mean_error = torch.linalg.vector_norm(mean - target_device, dim=1)
    entropy = _prediction_entropy(p)
    tail = tail_mass_diagnostics(
        p,
        location,
        target_coords=target_device,
        mask=mask,
    )
    mean_gm_displacement = torch.linalg.vector_norm(mean - gm, dim=1)

    summary: dict[str, object] = {
        "coordinate_macro_nll": _coordinate_macro(row_nll.cpu(), keys),
        "coordinate_macro_energy_score_m": _coordinate_macro(
            row_es.cpu(),
            keys,
        ),
        "coordinate_macro_gm_mae_m": _coordinate_macro(
            gm_error.cpu(),
            keys,
        ),
        "coordinate_macro_posterior_mean_mae_m": _coordinate_macro(
            mean_error.cpu(),
            keys,
        ),
        "mean_entropy_nats": float(entropy.double().mean()),
        "mean_effective_anchors": float(entropy.exp().double().mean()),
        "mean_top1_probability": float(p.max(dim=1).values.double().mean()),
        "mean_gm_displacement_from_mean_m": float(
            mean_gm_displacement.double().mean()
        ),
        "tail_mass": {},
    }
    for radius in (10, 20, 40):
        summary["tail_mass"][f"truth_far_mass_{radius}m"] = float(
            tail[f"truth_far_mass_{radius}m"].double().mean()
        )
        summary["tail_mass"][f"mode_far_mass_{radius}m"] = float(
            tail[f"mode_far_mass_{radius}m"].double().mean()
        )
    summary["mean_gm_spatial_leverage_m"] = float(
        tail["gm_spatial_leverage_m"].double().mean()
    )
    summary["corr_truth_far_mass_20m_mean_error"] = _safe_correlation(
        tail["truth_far_mass_20m"],
        mean_error,
    )
    summary["corr_mode_far_mass_20m_mean_gm_displacement"] = _safe_correlation(
        tail["mode_far_mass_20m"],
        mean_gm_displacement,
    )
    return summary


def calibrate_freeze(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    development_result_path: str | Path = DEFAULT_DEVELOPMENT_RESULT,
    checkpoint_path: str | Path = DEFAULT_REFIT_CHECKPOINT,
    calibration_result_path: str | Path = DEFAULT_CALIBRATION_RESULT,
    freeze_path: str | Path = DEFAULT_FREEZE,
    device_name: str = "cuda",
) -> dict:
    if Path(checkpoint_path).exists():
        raise FileExistsError(checkpoint_path)
    if Path(calibration_result_path).exists():
        raise FileExistsError(calibration_result_path)
    if Path(freeze_path).exists():
        raise FileExistsError(freeze_path)

    registry = load_registry(registry_path)
    development = json.loads(
        Path(development_result_path).read_text(encoding="utf-8")
    )
    if development.get("development_result_sha256") != _canonical_sha(
        development,
        omit="development_result_sha256",
    ):
        raise ValueError("development result self-hash mismatch")
    if development["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("development/registry link mismatch")

    selected = development["selected"]
    lambda_ot = float(selected["lambda_ot"])
    epsilon_kl = float(selected["epsilon_kl"])
    selected_epoch = int(selected["selected_epoch"])

    data, rows = _load_registered_data(registry)
    rss_all = torch.from_numpy(data.model_rss()).float()
    device = torch.device(device_name)

    started = time.perf_counter()
    checkpoint, model, centered = _train_fixed_epochs(
        data=data,
        training_rows=rows["selection_refit"],
        rss_all=rss_all,
        lambda_ot=lambda_ot,
        epochs=selected_epoch,
        device=device,
    )
    reference = _fit_reference_for_rows(
        model=model,
        rss_all=rss_all,
        row_indices=rows["selection_refit"],
        device=device,
    )
    checkpoint["epsilon_kl"] = epsilon_kl
    checkpoint["correction_reference"] = _reference_payload(reference)
    _save_torch_new(checkpoint_path, checkpoint)
    checkpoint_sha = _sha256_file(checkpoint_path)

    cache = _cache_corrected_distribution(
        model=model,
        reference=reference,
        epsilon_kl=epsilon_kl,
        rss_all=rss_all,
        row_indices=rows["calibration"],
        device=device,
    )
    target = torch.from_numpy(
        centered[rows["calibration"]].astype(np.float32)
    )
    keys = _coordinate_group_ids(data, rows["calibration"])

    surface = []
    for temperature in CALIBRATION_TEMPERATURE_GRID:
        for scale_multiplier in CALIBRATION_SCALE_GRID:
            nll = _calibration_nll(
                cache=cache,
                target=target,
                keys=keys,
                model=model,
                temperature=float(temperature),
                scale_multiplier=float(scale_multiplier),
                device=device,
            )
            surface.append(
                {
                    "temperature": float(temperature),
                    "scale_multiplier": float(scale_multiplier),
                    "coordinate_macro_nll": float(nll),
                }
            )

    selected_calibration = min(
        surface,
        key=lambda item: (
            float(item["coordinate_macro_nll"]),
            abs(math.log(float(item["temperature"])))
            + abs(math.log(float(item["scale_multiplier"]))),
            float(item["temperature"]),
            float(item["scale_multiplier"]),
        ),
    )
    selected_metrics = _selected_calibration_metrics(
        cache=cache,
        target=target,
        keys=keys,
        model=model,
        temperature=float(selected_calibration["temperature"]),
        scale_multiplier=float(selected_calibration["scale_multiplier"]),
        device=device,
    )

    calibration = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development[
            "development_result_sha256"
        ],
        "selection_refit_checkpoint_sha256": checkpoint_sha,
        "lambda_ot": lambda_ot,
        "epsilon_kl": epsilon_kl,
        "selected_epoch": selected_epoch,
        "temperature_grid": list(CALIBRATION_TEMPERATURE_GRID),
        "scale_grid": list(CALIBRATION_SCALE_GRID),
        "surface": surface,
        "selected": selected_calibration,
        "selected_metrics": selected_metrics,
        "rows": int(len(rows["calibration"])),
        "coordinate_groups": int(len(set(keys))),
        "official_validation_opened": False,
        "reserved_external_test_opened": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    calibration["calibration_result_sha256"] = _canonical_sha(calibration)
    _write_json_new(calibration_result_path, calibration)

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "V18.4 global method law frozen from UJI trainingData development; "
            "UJI official validation remains pre-exposed and unopened here; "
            "UTS/Tampere/TUJI provided tests remain reserved"
        ),
        "registry_sha256": registry["registry_sha256"],
        "development_result_sha256": development[
            "development_result_sha256"
        ],
        "calibration_result_sha256": calibration[
            "calibration_result_sha256"
        ],
        "selection_refit_checkpoint_sha256": checkpoint_sha,
        "lambda_ot": lambda_ot,
        "epsilon_kl": epsilon_kl,
        "selected_epoch": selected_epoch,
        "posterior_temperature": float(
            selected_calibration["temperature"]
        ),
        "scale_multiplier": float(
            selected_calibration["scale_multiplier"]
        ),
        "decoder": "posterior_weighted_component_center_geometric_median",
        "model_config": _config_json(),
        "official_validation_opened": False,
        "reserved_external_test_opened": False,
    }
    freeze["freeze_sha256"] = _canonical_sha(freeze)
    _write_json_new(freeze_path, freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    prepare_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))

    develop_parser = sub.add_parser("develop")
    develop_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    develop_parser.add_argument(
        "--result",
        default=str(DEFAULT_DEVELOPMENT_RESULT),
    )
    develop_parser.add_argument("--models", default=str(DEFAULT_MODELS))
    develop_parser.add_argument("--device", default="cuda")

    calibrate_parser = sub.add_parser("calibrate-freeze")
    calibrate_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    calibrate_parser.add_argument(
        "--development-result",
        default=str(DEFAULT_DEVELOPMENT_RESULT),
    )
    calibrate_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_REFIT_CHECKPOINT),
    )
    calibrate_parser.add_argument(
        "--calibration-result",
        default=str(DEFAULT_CALIBRATION_RESULT),
    )
    calibrate_parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    calibrate_parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            train_path=args.train,
            registry_path=args.registry,
        )
    elif args.command == "develop":
        result = develop(
            registry_path=args.registry,
            result_path=args.result,
            model_dir=args.models,
            device_name=args.device,
        )
    else:
        result = calibrate_freeze(
            registry_path=args.registry,
            development_result_path=args.development_result,
            checkpoint_path=args.checkpoint,
            calibration_result_path=args.calibration_result,
            freeze_path=args.freeze,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
