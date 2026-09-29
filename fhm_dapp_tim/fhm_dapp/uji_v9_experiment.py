"""Registered nested UJI experiment for the coherent V9 model.

Only ``trainingData.csv`` is in scope.  Complete physical coordinate groups
are split into nested development and outer-test roles.  The outer-test path is
hard-gated by an immutable selection freeze and final refit checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.audit_bplus_sparse50_fem_v5 import (
    _hull_local_indices,
    geometry_support_with_hull,
)
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    build_delaunay_geometry,
    common_student_t_randomness,
    continuous_energy_score_mc,
    geometry_joint_front_loss,
    marked_energy_on_support,
    marked_kl_update,
    maximal_invariant_scan,
    profile_shared_affine_marked_energy,
    select_information_saturated_E,
    simultaneous_paired_upper_bounds,
    spatial_median_from_draws,
    topk_prior_projection,
    uniform_member_anchor_prior,
)
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _decode_key,
    _load_uji,
    _role_record,
    _rows_for_keys,
    _save_torch_new,
    _sha256_file,
    _sha256_indices,
    _sha256_keys,
    _write_json_new,
    support_fit_stop_roles,
)
from fhm_dapp_tim.fhm_dapp.experiment_v71 import (
    fit_hierarchical_marked_memory,
)
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel
from fhm_dapp_tim.fhm_dapp.train_anchor import (
    _ap_building_owner,
    _neighbor_targets,
)


PROTOCOL_VERSION = "uji_v9_coherent_marked_nested_20260809"
GEOMETRY_SEED = 9111
INNER_GEOMETRY_SEED = 9112
EARLY_STOP_SEED = 9113
SCORING_SEED = 9511
PAIRED_INFERENCE_SEED = 9512
MEMBER_SEEDS = tuple(range(9311, 9323))
INITIAL_MEMBER_COUNT = 8
OUTER_REFIT_FRACTION = 0.80
SELECTION_REFIT_FRACTION = 0.80
EARLY_STOP_FRACTION = 0.15
E_GRID = tuple(range(1, INITIAL_MEMBER_COUNT + 1))
K_GRID = (8, 16, 24, 32, 48)
LAMBDA_GRID = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0)
CONCENTRATION_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
STUDENT_DF = 4.0
FLOOR_HEIGHT_M = 4.0
SCORING_DRAWS = 64
BOOTSTRAP_REPETITIONS = 5000
MAE_NONINFERIORITY_MARGIN_M = 0.25
VISIBLE_THRESHOLD = -0.999

COMMON_FRONT = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "max_visible": 64,
    "dropout": 0.10,
    "student_df": STUDENT_DF,
    "min_scale_m": 0.25,
    "batch_size": 128,
    "evaluation_batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "maximum_epochs": 60,
    "minimum_epochs": 12,
    "patience": 8,
    "minimum_improvement": 1e-3,
    "ap_dropout": 0.10,
}
HISTORICAL_FRONT = {
    "max_residual_m": 30.0,
    "neighbor_count": 8,
    "neighbor_temperature_m": 4.0,
    "retrieval_weight": 0.50,
    "hierarchy_weight": 1.0,
    "self_target_weight": 0.25,
    "coordinate_weight": 0.05,
    "rss_offset_aug_db": 2.0,
    "rss_scale_aug": 0.03,
    "train_candidates": 32,
    "inference_neural_scale": 0.75,
    "inference_prototype_weight": 8.0,
    "inference_posterior_temperature": 0.75,
}
AFFINE_PROFILE = {
    "steps": 10,
    "learning_rate": 0.20,
    "offset_max": 0.20,
    "log_scale_max": 0.25,
    "offset_weight": 0.10,
    "log_scale_weight": 0.10,
}
CANDIDATES = ("geometry_invariant", "historical_affine")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "UJIndoorLoc" / "trainingData.csv"
DEFAULT_EVIDENCE = ROOT / "evidence" / "v9_uji"
DEFAULT_DESIGN = DEFAULT_EVIDENCE / "design_v9.md"
DEFAULT_REGISTRY = DEFAULT_EVIDENCE / "registry_v9_uji.json"
DEFAULT_MODELS = ROOT / "models" / "v9_uji_selection"
DEFAULT_FINAL_MODELS = ROOT / "models" / "v9_uji_final"
DEFAULT_FREEZE = DEFAULT_EVIDENCE / "selection_freeze_v9_uji.json"
DEFAULT_RESULT = DEFAULT_EVIDENCE / "outer_test_result_v9_uji.json"

LocationKey = tuple[int, int, float, float]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _registry_without_hash(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("registry_sha256", None)
    return value


def _strata(keys: set[LocationKey]) -> dict[tuple[int, int], set[LocationKey]]:
    result: dict[tuple[int, int], set[LocationKey]] = {}
    for key in keys:
        result.setdefault(key[:2], set()).add(key)
    return result


def construct_coordinate_roles(
    all_keys: set[LocationKey],
) -> tuple[dict[str, set[LocationKey]], dict]:
    """Create the registered nested, geometry-covered coordinate roles."""

    if not all_keys:
        raise ValueError("all_keys must be nonempty")
    outer_refit, outer_details = geometry_support_with_hull(
        all_keys, OUTER_REFIT_FRACTION, seed=GEOMETRY_SEED
    )
    outer_test = set(all_keys) - outer_refit
    selection_refit, selection_details = geometry_support_with_hull(
        outer_refit, SELECTION_REFIT_FRACTION, seed=INNER_GEOMETRY_SEED
    )
    development = outer_refit - selection_refit
    nested, early_details = support_fit_stop_roles(
        selection_refit,
        seed=EARLY_STOP_SEED,
        early_stop_fraction=EARLY_STOP_FRACTION,
    )
    roles = {
        "member_fit": nested["member_fit"],
        "early_stop": nested["early_stop"],
        "selection_refit": selection_refit,
        "development": development,
        "outer_refit": outer_refit,
        "outer_test": outer_test,
    }
    return roles, {
        "outer": outer_details,
        "inner_selection": selection_details,
        "early_stop": early_details,
    }


def _role_diagnostics(role_keys: set[LocationKey], data: dict) -> dict:
    selected = np.asarray([key in role_keys for key in data["keys"]], dtype=bool)
    phone_counts = Counter(map(int, data["phone"][selected]))
    stratum_coordinates = Counter(key[:2] for key in role_keys)
    stratum_rows = Counter(
        (int(building), int(floor))
        for building, floor in zip(
            data["building"][selected], data["floor"][selected]
        )
    )
    return {
        "coordinates": int(len(role_keys)),
        "rows": int(selected.sum()),
        "phones": int(len(phone_counts)),
        "phone_row_counts": {
            str(key): int(value) for key, value in sorted(phone_counts.items())
        },
        "coordinate_counts_by_building_floor": {
            f"{key[0]}:{key[1]}": int(value)
            for key, value in sorted(stratum_coordinates.items())
        },
        "row_counts_by_building_floor": {
            f"{key[0]}:{key[1]}": int(value)
            for key, value in sorted(stratum_rows.items())
        },
    }


def _nearest_same_stratum(
    query: set[LocationKey], reference: set[LocationKey]
) -> dict:
    query_strata = _strata(query)
    reference_strata = _strata(reference)
    values: list[float] = []
    per_stratum = {}
    for stratum, query_keys in sorted(query_strata.items()):
        if stratum not in reference_strata:
            raise RuntimeError(f"reference misses stratum {stratum}")
        query_xy = np.asarray([[key[2], key[3]] for key in query_keys])
        reference_xy = np.asarray(
            [[key[2], key[3]] for key in reference_strata[stratum]]
        )
        distance = np.linalg.norm(
            query_xy[:, None, :] - reference_xy[None, :, :], axis=2
        ).min(axis=1)
        values.extend(distance.tolist())
        per_stratum[f"{stratum[0]}:{stratum[1]}"] = {
            "coordinates": int(len(distance)),
            "mean_m": float(distance.mean()),
            "maximum_m": float(distance.max()),
        }
    vector = np.asarray(values, dtype=np.float64)
    return {
        "coordinates": int(len(vector)),
        "mean_m": float(vector.mean()),
        "median_m": float(np.median(vector)),
        "q90_m": float(np.quantile(vector, 0.90)),
        "maximum_m": float(vector.max()),
        "per_stratum": per_stratum,
    }


def audit_coordinate_roles(
    roles: Mapping[str, set[LocationKey]], data: dict
) -> dict:
    """Fail on coordinate leakage, union errors, hull loss, or stratum loss."""

    all_keys = set(data["keys"])
    identities = {
        "member_fit_early_stop_intersection": len(
            roles["member_fit"] & roles["early_stop"]
        ),
        "selection_refit_development_intersection": len(
            roles["selection_refit"] & roles["development"]
        ),
        "outer_refit_outer_test_intersection": len(
            roles["outer_refit"] & roles["outer_test"]
        ),
    }
    if any(identities.values()):
        raise RuntimeError(f"coordinate leakage detected: {identities}")
    if roles["member_fit"] | roles["early_stop"] != roles["selection_refit"]:
        raise RuntimeError("fit/stop roles do not reconstruct selection_refit")
    if roles["selection_refit"] | roles["development"] != roles["outer_refit"]:
        raise RuntimeError("selection/development do not reconstruct outer_refit")
    if roles["outer_refit"] | roles["outer_test"] != all_keys:
        raise RuntimeError("outer roles do not reconstruct training coordinates")
    all_strata = set(_strata(all_keys))
    for name, keys in roles.items():
        if set(_strata(keys)) != all_strata:
            raise RuntimeError(f"{name} loses a building-floor stratum")

    hull_checks = {}
    for source_name, reference_name in (
        ("all", "outer_refit"),
        ("outer_refit", "selection_refit"),
        ("selection_refit", "member_fit"),
    ):
        source = all_keys if source_name == "all" else roles[source_name]
        reference = roles[reference_name]
        missing_count = 0
        for stratum, keys in _strata(source).items():
            ordered = sorted(keys)
            xy = np.asarray([[key[2], key[3]] for key in ordered], dtype=np.float64)
            hull = {ordered[index] for index in _hull_local_indices(xy)}
            missing_count += len(hull - reference)
        hull_checks[f"{source_name}_hull_missing_from_{reference_name}"] = missing_count
        if missing_count:
            raise RuntimeError(
                f"{reference_name} misses {missing_count} mandatory hull coordinates"
            )

    diagnostics = {
        "grouping": "complete (building,floor,longitude,latitude) groups; all scans and phones stay together",
        "disjoint_intersections": identities,
        "exact_union_identities": True,
        "all_roles_keep_all_building_floor_strata": True,
        "hull_checks": hull_checks,
        "roles": {
            name: _role_diagnostics(set(keys), data)
            for name, keys in roles.items()
        },
        "development_to_selection_refit": _nearest_same_stratum(
            roles["development"], roles["selection_refit"]
        ),
        "outer_test_to_outer_refit": _nearest_same_stratum(
            roles["outer_test"], roles["outer_refit"]
        ),
    }
    return diagnostics


def register_protocol(
    *,
    train_data: str | Path = DEFAULT_TRAIN,
    design_path: str | Path = DEFAULT_DESIGN,
    output: str | Path = DEFAULT_REGISTRY,
) -> dict:
    train_path = Path(train_data).resolve()
    design = Path(design_path).resolve()
    if not design.exists():
        raise FileNotFoundError(design)
    data = _load_uji(train_path)
    roles, split_details = construct_coordinate_roles(set(data["keys"]))
    diagnostics = audit_coordinate_roles(roles, data)
    registry = {
        "protocol_version": PROTOCOL_VERSION,
        "scientific_status": (
            "new leakage-controlled nested split inside trainingData.csv; "
            "not independent external confirmation"
        ),
        "official_validation_file_in_scope": False,
        "design": str(design),
        "design_sha256": _sha256_file(design),
        "train_data": str(train_path),
        "train_data_sha256": _sha256_file(train_path),
        "training_rows": int(len(data["keys"])),
        "training_coordinate_count": int(len(set(data["keys"]))),
        "training_coordinate_sha256": _sha256_keys(set(data["keys"])),
        "n_aps": int(data["stats"]["n_aps"]),
        "phone_ids": sorted(map(int, np.unique(data["phone"]))),
        "roles": {
            name: _role_record(set(keys), data["keys"])
            for name, keys in roles.items()
        },
        "split_details": split_details,
        "split_diagnostics": diagnostics,
        "constants": {
            "geometry_seed": GEOMETRY_SEED,
            "inner_geometry_seed": INNER_GEOMETRY_SEED,
            "early_stop_seed": EARLY_STOP_SEED,
            "scoring_seed": SCORING_SEED,
            "paired_inference_seed": PAIRED_INFERENCE_SEED,
            "member_seeds": list(MEMBER_SEEDS),
            "initial_member_count": INITIAL_MEMBER_COUNT,
            "outer_refit_fraction": OUTER_REFIT_FRACTION,
            "selection_refit_fraction": SELECTION_REFIT_FRACTION,
            "early_stop_fraction": EARLY_STOP_FRACTION,
            "student_df": STUDENT_DF,
            "scoring_draws": SCORING_DRAWS,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "floor_height_m": FLOOR_HEIGHT_M,
            "mae_noninferiority_margin_m": MAE_NONINFERIORITY_MARGIN_M,
        },
        "front": COMMON_FRONT,
        "historical_front": HISTORICAL_FRONT,
        "affine_profile": AFFINE_PROFILE,
        "candidates": list(CANDIDATES),
        "joint_grid": {
            "E": list(E_GRID),
            "K": list(K_GRID),
            "lambda": list(LAMBDA_GRID),
            "member_rule": "preregistered_seed_prefix",
        },
        "radio_memory": {
            "concentration_grid": list(CONCENTRATION_GRID),
            "degrees_freedom_grid": [STUDENT_DF],
            "selection_outcome_access": "none",
        },
        "access_contract": {
            "development_allowed_before_freeze": True,
            "outer_test_allowed_before_freeze": False,
            "outer_test_refit_role": "outer_refit",
            "official_validation_allowed": False,
        },
    }
    registry["registry_sha256"] = _canonical_json_sha256(registry)
    _write_json_new(output, registry)
    return registry


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 registry")
    if registry.get("registry_sha256") != _canonical_json_sha256(
        _registry_without_hash(registry)
    ):
        raise ValueError("V9 registry self-hash failed")
    if _sha256_file(registry["design"]) != registry["design_sha256"]:
        raise ValueError("V9 design changed after registration")
    if _sha256_file(registry["train_data"]) != registry["train_data_sha256"]:
        raise ValueError("V9 training file changed after registration")
    if registry.get("official_validation_file_in_scope") is not False:
        raise ValueError("V9 registry illegally admits the official file")
    return registry


def _registered_role_keys(registry: dict, role: str) -> set[LocationKey]:
    if role not in registry["roles"]:
        raise ValueError(f"unknown V9 role: {role}")
    record = registry["roles"][role]
    keys = {_decode_key(value) for value in record["coordinate_keys"]}
    if _sha256_keys(keys) != record["coordinate_key_sha256"]:
        raise ValueError(f"coordinate hash failed for role {role}")
    return keys


def _registered_rows(registry: dict, role: str, data: dict) -> np.ndarray:
    keys = _registered_role_keys(registry, role)
    rows = _rows_for_keys(data["keys"], keys)
    record = registry["roles"][role]
    if _sha256_indices(rows) != record["row_index_sha256"]:
        raise ValueError(f"row hash failed for role {role}")
    return rows


def _candidate_check(candidate: str) -> str:
    name = str(candidate)
    if name not in CANDIDATES:
        raise ValueError(f"unknown V9 candidate: {name}")
    return name


def selection_checkpoint_path(candidate: str, member_index: int) -> Path:
    name = _candidate_check(candidate)
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("member_index is outside the registered panel")
    return DEFAULT_MODELS / name / f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"


def final_checkpoint_path(candidate: str, member_index: int) -> Path:
    name = _candidate_check(candidate)
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("member_index is outside the registered panel")
    return DEFAULT_FINAL_MODELS / name / f"member_{index:02d}_seed{MEMBER_SEEDS[index - 1]}.pt"


def radio_memory_path(candidate: str, phase: str) -> Path:
    name = _candidate_check(candidate)
    if phase not in {"selection", "final"}:
        raise ValueError("phase must be selection or final")
    return DEFAULT_EVIDENCE / f"radio_memory_{phase}_{name}.pt"


def cache_path(candidate: str, role: str) -> Path:
    name = _candidate_check(candidate)
    if role not in {"development", "outer_test"}:
        raise ValueError("cache role must be development or outer_test")
    return DEFAULT_EVIDENCE / f"{role}_cache_{name}.pt"


def _represented_rss(rss: torch.Tensor, candidate: str) -> torch.Tensor:
    if _candidate_check(candidate) == "geometry_invariant":
        return maximal_invariant_scan(rss)
    return rss


def _build_front_model(
    data: dict,
    rows: np.ndarray,
    *,
    candidate: str,
    device: torch.device,
) -> tuple[AnchorStudentTModel, dict, np.ndarray, torch.Tensor, np.ndarray, dict | None]:
    name = _candidate_check(candidate)
    origin = data["raw_xy"][rows].mean(axis=0).astype(np.float32)
    centered = (data["raw_xy"] - origin[None, :]).astype(np.float32)
    represented = _represented_rss(data["rss"], name)
    memory, sample_anchor = build_anchor_memory(
        represented,
        centered,
        data["building"],
        data["floor"],
        rows,
    )
    if np.any(sample_anchor[rows] < 0):
        raise RuntimeError("training rows escaped the anchor dictionary")
    seen = (memory["prototype_rss"] > VISIBLE_THRESHOLD).any(dim=0)
    owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(data["stats"]["n_buildings"]),
    )
    geometry = None
    if name == "geometry_invariant":
        geometry = build_delaunay_geometry(
            memory["coords"], memory["building"], memory["floor"]
        )
    model = AnchorStudentTModel(
        n_aps=int(data["stats"]["n_aps"]),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen,
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=owner,
        d_model=COMMON_FRONT["d_model"],
        num_heads=COMMON_FRONT["num_heads"],
        num_layers=COMMON_FRONT["num_layers"],
        max_visible=COMMON_FRONT["max_visible"],
        dropout=COMMON_FRONT["dropout"],
        visible_threshold=VISIBLE_THRESHOLD,
        student_df=COMMON_FRONT["student_df"],
        min_scale_m=COMMON_FRONT["min_scale_m"],
        max_residual_m=HISTORICAL_FRONT["max_residual_m"],
        scan_feature_mode=(
            "bounded_maximal_invariant" if name == "geometry_invariant" else "legacy"
        ),
        prototype_metric="none" if name == "geometry_invariant" else "l2",
        local_residual_transform=(
            None if geometry is None else geometry["residual_transform"]
        ),
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))
    return (
        model,
        memory,
        sample_anchor,
        torch.from_numpy(centered).float(),
        origin,
        geometry,
    )


def _training_loader(
    represented: torch.Tensor,
    data: dict,
    centered: torch.Tensor,
    sample_anchor: np.ndarray,
    rows: np.ndarray,
    *,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        represented[rows].float(),
        centered[rows].float(),
        torch.from_numpy(sample_anchor[rows]).long(),
        data["building_t"][rows].long(),
        data["floor_t"][rows].long(),
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=COMMON_FRONT["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _front_logits_embedding(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    candidate: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _candidate_check(candidate) == "historical_affine":
        logits, embedding = model.retrieval_logits(
            rss,
            neural_scale=HISTORICAL_FRONT["inference_neural_scale"],
            prototype_distance_weight=HISTORICAL_FRONT[
                "inference_prototype_weight"
            ],
        )
        return (
            logits / HISTORICAL_FRONT["inference_posterior_temperature"],
            embedding,
        )
    return model.retrieval_logits(rss)


def _train_epoch(
    model: AnchorStudentTModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    candidate: str,
    geometry: dict | None,
    historical_neighbors: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, float]:
    name = _candidate_check(candidate)
    model.train()
    totals: dict[str, float] = {}
    row_count = 0
    for batch in loader:
        rss, xy, anchor, building, floor = [
            value.to(device, non_blocking=True) for value in batch
        ]
        optimizer.zero_grad(set_to_none=True)
        if name == "geometry_invariant":
            if geometry is None:
                raise RuntimeError("geometry candidate misses its target graph")
            values = geometry_joint_front_loss(
                model,
                rss,
                xy,
                anchor,
                geometry["neighbor_indices"].to(device),
                geometry["neighbor_weights"].to(device),
                ap_dropout=COMMON_FRONT["ap_dropout"],
            )
        else:
            if historical_neighbors is None:
                raise RuntimeError("historical candidate misses neighbor targets")
            values = model.training_loss(
                rss,
                xy,
                anchor,
                historical_neighbors[0],
                historical_neighbors[1],
                retrieval_weight=HISTORICAL_FRONT["retrieval_weight"],
                hierarchy_weight=HISTORICAL_FRONT["hierarchy_weight"],
                self_target_weight=HISTORICAL_FRONT["self_target_weight"],
                coordinate_weight=HISTORICAL_FRONT["coordinate_weight"],
                rss_offset_aug_db=HISTORICAL_FRONT["rss_offset_aug_db"],
                rss_scale_aug=HISTORICAL_FRONT["rss_scale_aug"],
                train_candidates=HISTORICAL_FRONT["train_candidates"],
                ap_dropout=COMMON_FRONT["ap_dropout"],
            )
        values["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), COMMON_FRONT["gradient_clip"]
        )
        optimizer.step()
        count = len(rss)
        row_count += count
        for key, value in values.items():
            if value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
    return {key: value / row_count for key, value in totals.items()}


def _coordinate_group_mean(
    row_values: torch.Tensor, keys: Sequence[LocationKey]
) -> tuple[float, float, int]:
    groups: dict[LocationKey, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    values = torch.stack(
        [
            row_values[torch.as_tensor(indices, dtype=torch.long)].mean()
            for indices in groups.values()
        ]
    )
    standard_error = (
        float(values.std(unbiased=True) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return float(values.mean()), standard_error, len(values)


@torch.no_grad()
def evaluate_front_log_score(
    model: AnchorStudentTModel,
    represented: torch.Tensor,
    data: dict,
    centered: torch.Tensor,
    rows: np.ndarray,
    *,
    candidate: str,
    device: torch.device,
) -> dict[str, float | int]:
    """Exact full-dictionary continuous-mixture log score for early stopping."""

    model.eval()
    nll_parts = []
    error_parts = []
    width = COMMON_FRONT["evaluation_batch_size"]
    for start in range(0, len(rows), width):
        selected = rows[start : start + width]
        rss = represented[selected].to(device)
        target = centered[selected].to(device)
        logits, embedding = _front_logits_embedding(model, rss, candidate)
        all_anchor = torch.arange(
            model.n_anchors, device=device, dtype=torch.long
        ).unsqueeze(0).expand(len(rss), -1)
        location, scale = model._components(embedding, all_anchor)
        component_log = model.student_log_prob(target, location, scale)
        log_probability = F.log_softmax(logits, dim=1)
        nll_parts.append(-torch.logsumexp(log_probability + component_log, dim=1).cpu())
        point = (torch.softmax(logits, dim=1).unsqueeze(2) * location).sum(dim=1)
        error_parts.append(torch.linalg.vector_norm(point - target, dim=1).cpu())
    nll = torch.cat(nll_parts)
    error = torch.cat(error_parts)
    keys = [data["keys"][int(row)] for row in rows]
    macro_nll, nll_se, groups = _coordinate_group_mean(nll, keys)
    macro_mae, mae_se, _ = _coordinate_group_mean(error, keys)
    return {
        "coordinate_macro_mixture_nll": macro_nll,
        "coordinate_macro_mixture_nll_se": nll_se,
        "coordinate_macro_front_mae_m": macro_mae,
        "coordinate_macro_front_mae_se_m": mae_se,
        "coordinate_groups": groups,
        "rows": int(len(rows)),
    }


def _prepare_training(
    data: dict,
    rows: np.ndarray,
    *,
    candidate: str,
    seed: int,
    device: torch.device,
) -> tuple[
    AnchorStudentTModel,
    dict,
    torch.Tensor,
    np.ndarray,
    DataLoader,
    dict | None,
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    model, memory, sample_anchor, centered, origin, geometry = _build_front_model(
        data, rows, candidate=candidate, device=device
    )
    represented = _represented_rss(data["rss"], candidate)
    loader = _training_loader(
        represented,
        data,
        centered,
        sample_anchor,
        rows,
        seed=seed,
    )
    historical_neighbors = None
    if candidate == "historical_affine":
        neighbor_index, neighbor_weight = _neighbor_targets(
            memory["coords"],
            memory["building"],
            neighbor_count=HISTORICAL_FRONT["neighbor_count"],
            temperature_m=HISTORICAL_FRONT["neighbor_temperature_m"],
        )
        historical_neighbors = (
            neighbor_index.to(device),
            neighbor_weight.to(device),
        )
    return (
        model,
        memory,
        centered,
        origin,
        loader,
        geometry,
        historical_neighbors,
    )


def _fit_schedule(
    data: dict,
    fit_rows: np.ndarray,
    stop_rows: np.ndarray,
    *,
    candidate: str,
    seed: int,
    device: torch.device,
    log_every: int,
) -> tuple[int, list[dict]]:
    _set_seed(seed)
    (
        model,
        _memory,
        centered,
        _origin,
        loader,
        geometry,
        historical_neighbors,
    ) = _prepare_training(
        data, fit_rows, candidate=candidate, seed=seed, device=device
    )
    represented = _represented_rss(data["rss"], candidate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COMMON_FRONT["learning_rate"],
        weight_decay=COMMON_FRONT["weight_decay"],
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, COMMON_FRONT["maximum_epochs"] + 1):
        train = _train_epoch(
            model,
            loader,
            optimizer,
            candidate=candidate,
            geometry=geometry,
            historical_neighbors=historical_neighbors,
            device=device,
        )
        stop = evaluate_front_log_score(
            model,
            represented,
            data,
            centered,
            stop_rows,
            candidate=candidate,
            device=device,
        )
        score = float(stop["coordinate_macro_mixture_nll"])
        improved = score < best - COMMON_FRONT["minimum_improvement"]
        if improved:
            best = score
            best_epoch = epoch
            stale = 0
        elif epoch >= COMMON_FRONT["minimum_epochs"]:
            stale += 1
        record = {
            "epoch": epoch,
            "train": train,
            "early_stop": stop,
            "best_epoch": best_epoch,
            "best_coordinate_macro_mixture_nll": best,
        }
        history.append(record)
        if epoch == 1 or improved or epoch % int(log_every) == 0:
            print(json.dumps(record, allow_nan=False), flush=True)
        if epoch >= COMMON_FRONT["minimum_epochs"] and stale >= COMMON_FRONT["patience"]:
            break
    if best_epoch <= 0:
        raise RuntimeError("front epoch selection failed")
    return best_epoch, history


def _fit_fixed(
    data: dict,
    rows: np.ndarray,
    *,
    candidate: str,
    seed: int,
    epochs: int,
    device: torch.device,
    log_every: int,
) -> tuple[AnchorStudentTModel, dict, np.ndarray, list[dict], dict | None]:
    _set_seed(seed)
    (
        model,
        memory,
        _centered,
        origin,
        loader,
        geometry,
        historical_neighbors,
    ) = _prepare_training(data, rows, candidate=candidate, seed=seed, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COMMON_FRONT["learning_rate"],
        weight_decay=COMMON_FRONT["weight_decay"],
    )
    history = []
    for epoch in range(1, int(epochs) + 1):
        metrics = _train_epoch(
            model,
            loader,
            optimizer,
            candidate=candidate,
            geometry=geometry,
            historical_neighbors=historical_neighbors,
            device=device,
        )
        history.append({"epoch": epoch, **metrics})
        if epoch == 1 or epoch == int(epochs) or epoch % int(log_every) == 0:
            print(
                json.dumps(
                    {"refit_epoch": epoch, "total_epochs": int(epochs), **metrics},
                    allow_nan=False,
                ),
                flush=True,
            )
    return model, memory, origin, history, geometry


def _checkpoint_payload(
    *,
    registry: dict,
    phase: str,
    refit_role: str,
    candidate: str,
    member_index: int,
    selected_epoch: int,
    model: AnchorStudentTModel,
    memory: dict,
    origin: np.ndarray,
    schedule_history: list[dict],
    refit_history: list[dict],
    geometry: dict | None,
) -> dict:
    role_keys = sorted(_registered_role_keys(registry, refit_role))
    if len(role_keys) != model.n_anchors:
        raise RuntimeError("registered refit role and anchor dictionary disagree")
    index = int(member_index)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "phase": phase,
        "refit_role": refit_role,
        "candidate": _candidate_check(candidate),
        "member_index": index,
        "member_seed": MEMBER_SEEDS[index - 1],
        "selected_epoch": int(selected_epoch),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "coord_origin": torch.as_tensor(origin, dtype=torch.float32),
        "stats": memory.get("stats", {}),
        "anchor_coordinate_keys": tuple(role_keys),
        "schedule_history": schedule_history,
        "refit_history": refit_history,
        "geometry_diagnostics": (
            None if geometry is None else geometry["diagnostics"]
        ),
        "config": {
            **COMMON_FRONT,
            "model_schema": "coherent_marked_front_v9",
            "scan_feature_mode": (
                "bounded_maximal_invariant"
                if candidate == "geometry_invariant"
                else "legacy"
            ),
            "prototype_metric": (
                "none" if candidate == "geometry_invariant" else "l2"
            ),
            "visible_threshold": VISIBLE_THRESHOLD,
            "historical_front": HISTORICAL_FRONT,
        },
    }


def train_selection_member(
    *,
    candidate: str,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path | None = None,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    name = _candidate_check(candidate)
    index = int(member_index)
    if not 1 <= index <= len(MEMBER_SEEDS):
        raise ValueError("member_index is outside the registered pool")
    destination = (
        Path(output) if output is not None else selection_checkpoint_path(name, index)
    )
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device(device_name)
    data = _load_uji(registry["train_data"])
    fit_rows = _registered_rows(registry, "member_fit", data)
    stop_rows = _registered_rows(registry, "early_stop", data)
    refit_rows = _registered_rows(registry, "selection_refit", data)
    seed = MEMBER_SEEDS[index - 1]
    started = time.perf_counter()
    epoch, schedule = _fit_schedule(
        data,
        fit_rows,
        stop_rows,
        candidate=name,
        seed=seed,
        device=device,
        log_every=log_every,
    )
    model, memory, origin, refit, geometry = _fit_fixed(
        data,
        refit_rows,
        candidate=name,
        seed=seed,
        epochs=epoch,
        device=device,
        log_every=log_every,
    )
    memory = dict(memory)
    memory["stats"] = data["stats"]
    payload = _checkpoint_payload(
        registry=registry,
        phase="selection",
        refit_role="selection_refit",
        candidate=name,
        member_index=index,
        selected_epoch=epoch,
        model=model,
        memory=memory,
        origin=origin,
        schedule_history=schedule,
        refit_history=refit,
        geometry=geometry,
    )
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "checkpoint_sha256": _sha256_file(destination),
        "candidate": name,
        "member_index": index,
        "seed": seed,
        "selected_epoch": epoch,
        "anchors": model.n_anchors,
        "training_seconds": payload["training_seconds"],
    }


def load_front_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[AnchorStudentTModel, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    config = payload.get("config", {})
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or config.get("model_schema") != "coherent_marked_front_v9"
    ):
        raise ValueError(f"not a V9 front checkpoint: {path}")
    state = payload["model_state"]
    candidate = _candidate_check(payload["candidate"])
    geometry = None
    if candidate == "geometry_invariant":
        geometry = build_delaunay_geometry(
            state["anchor_coords"],
            state["anchor_buildings"],
            state["anchor_floors"],
        )
    model = AnchorStudentTModel(
        n_aps=int(state["prototype_rss"].shape[1]),
        anchor_coords=state["anchor_coords"],
        anchor_buildings=state["anchor_buildings"],
        anchor_floors=state["anchor_floors"],
        seen_ap_mask=state["encoder.seen_ap_mask"],
        prototype_rss=state["prototype_rss"],
        ap_building_owner=state["ap_building_owner"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_visible=config["max_visible"],
        dropout=config["dropout"],
        visible_threshold=config["visible_threshold"],
        student_df=config["student_df"],
        min_scale_m=config["min_scale_m"],
        max_residual_m=HISTORICAL_FRONT["max_residual_m"],
        scan_feature_mode=config["scan_feature_mode"],
        prototype_metric=config["prototype_metric"],
        local_residual_transform=(
            None if geometry is None else geometry["residual_transform"]
        ),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, payload


def _load_member_panel(
    paths: Sequence[str | Path],
    *,
    registry: dict,
    expected_phase: str,
    expected_candidate: str,
    device: torch.device,
) -> tuple[list[AnchorStudentTModel], list[dict], list[Path]]:
    models = []
    payloads = []
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("at least one V9 member checkpoint is required")
    for expected_index, path in enumerate(resolved, start=1):
        model, payload = load_front_checkpoint(path, device)
        if payload.get("registry_sha256") != registry["registry_sha256"]:
            raise ValueError(f"checkpoint registry mismatch: {path}")
        if payload.get("phase") != expected_phase:
            raise ValueError(f"checkpoint phase mismatch: {path}")
        if payload.get("candidate") != expected_candidate:
            raise ValueError(f"checkpoint candidate mismatch: {path}")
        if payload.get("member_index") != expected_index:
            raise ValueError("members must enter in preregistered prefix order")
        if payload.get("member_seed") != MEMBER_SEEDS[expected_index - 1]:
            raise ValueError("member seed order changed")
        models.append(model)
        payloads.append(payload)
    first_keys = payloads[0]["anchor_coordinate_keys"]
    first_origin = payloads[0]["coord_origin"]
    for payload in payloads[1:]:
        if payload["anchor_coordinate_keys"] != first_keys:
            raise ValueError("member anchor dictionaries are not aligned")
        if not torch.equal(payload["coord_origin"], first_origin):
            raise ValueError("member coordinate origins are not aligned")
    return models, payloads, resolved


def fit_radio_memory(
    *,
    candidate: str,
    phase: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path | None = None,
) -> dict:
    registry = load_registry(registry_path)
    name = _candidate_check(candidate)
    if phase == "selection":
        role = "selection_refit"
    elif phase == "final":
        role = "outer_refit"
        load_selection_freeze(DEFAULT_FREEZE, registry=registry)
    else:
        raise ValueError("phase must be selection or final")
    destination = Path(output) if output is not None else radio_memory_path(name, phase)
    if destination.exists():
        raise FileExistsError(destination)
    data = _load_uji(registry["train_data"])
    rows = _registered_rows(registry, role, data)
    represented = _represented_rss(data["rss"], name)
    anchor_keys = tuple(sorted(_registered_role_keys(registry, role)))
    memory = fit_hierarchical_marked_memory(
        rss=represented,
        phone_ids=data["phone"],
        coordinate_keys=data["keys"],
        training_indices=rows,
        anchor_keys=anchor_keys,
        detection_concentration_grid=CONCENTRATION_GRID,
        center_concentration_grid=CONCENTRATION_GRID,
        scale_concentration_grid=CONCENTRATION_GRID,
        degrees_freedom_grid=(STUDENT_DF,),
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "candidate": name,
        "phase": phase,
        "refit_role": role,
        "role_coordinate_sha256": registry["roles"][role]["coordinate_key_sha256"],
        "memory": memory,
    }
    _save_torch_new(destination, payload)
    return {
        "radio_memory": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": name,
        "phase": phase,
        "anchors": len(anchor_keys),
        "selected_detection_concentration": memory["radio_only_selection"]["detection"]["selected_concentration"],
        "selected_center_concentration": memory["radio_only_selection"]["conditional_mark"]["selected_center_concentration"],
        "selected_scale_concentration": memory["radio_only_selection"]["conditional_mark"]["selected_scale_concentration"],
    }


def _load_radio_memory(
    path: str | Path,
    *,
    registry: dict,
    candidate: str,
    phase: str,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 radio memory")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("radio-memory registry mismatch")
    if payload.get("candidate") != candidate or payload.get("phase") != phase:
        raise ValueError("radio-memory candidate/phase mismatch")
    return payload


def _coordinate_groups(
    keys: Sequence[LocationKey],
) -> tuple[torch.Tensor, tuple[LocationKey, ...]]:
    unique = tuple(sorted(set(keys)))
    mapping = {key: index for index, key in enumerate(unique)}
    return torch.tensor([mapping[key] for key in keys], dtype=torch.long), unique


@torch.no_grad()
def build_cache(
    *,
    candidate: str,
    role: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    member_paths: Sequence[str | Path] | None = None,
    member_count: int | None = None,
    radio_path: str | Path | None = None,
    output: str | Path | None = None,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    name = _candidate_check(candidate)
    if role == "development":
        phase = "selection"
        count = INITIAL_MEMBER_COUNT if member_count is None else int(member_count)
        if not 1 <= count <= len(MEMBER_SEEDS):
            raise ValueError("development member_count is outside the registered panel")
        paths = (
            [selection_checkpoint_path(name, index) for index in range(1, count + 1)]
            if member_paths is None
            else list(member_paths)
        )
    elif role == "outer_test":
        freeze = load_selection_freeze(DEFAULT_FREEZE, registry=registry)
        if name != freeze["selected"]["candidate"]:
            raise ValueError("outer-test cache is allowed only for the frozen candidate")
        phase = "final"
        count = int(freeze["selected"]["E"])
        paths = (
            [final_checkpoint_path(name, index) for index in range(1, count + 1)]
            if member_paths is None
            else list(member_paths)
        )
    else:
        raise ValueError("role must be development or outer_test")
    destination = Path(output) if output is not None else cache_path(name, role)
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device(device_name)
    models, payloads, resolved_paths = _load_member_panel(
        paths,
        registry=registry,
        expected_phase=phase,
        expected_candidate=name,
        device=device,
    )
    radio_file = Path(radio_path) if radio_path is not None else radio_memory_path(name, phase)
    radio = _load_radio_memory(
        radio_file,
        registry=registry,
        candidate=name,
        phase=phase,
    )
    if tuple(radio["memory"]["anchor_keys"]) != tuple(payloads[0]["anchor_coordinate_keys"]):
        raise ValueError("radio memory and member anchor identities are not aligned")
    data = _load_uji(registry["train_data"])
    rows = _registered_rows(registry, role, data)
    represented = _represented_rss(data["rss"], name)
    origin = payloads[0]["coord_origin"].numpy().astype(np.float32)
    target = torch.from_numpy(
        (data["raw_xy"][rows] - origin[None, :]).astype(np.float32)
    )
    anchors = models[0].n_anchors
    member_count = len(models)
    row_count = len(rows)
    logits_cache = torch.empty(member_count, row_count, anchors)
    location_cache = torch.empty(member_count, row_count, anchors, 2)
    scale_cache = torch.empty_like(location_cache)
    width = COMMON_FRONT["evaluation_batch_size"]
    for member_index, model in enumerate(models):
        for start in range(0, row_count, width):
            stop = min(start + width, row_count)
            selected_rows = rows[start:stop]
            query = represented[selected_rows].to(device)
            logits, embedding = _front_logits_embedding(model, query, name)
            all_anchor = torch.arange(
                anchors, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(query), -1)
            location, scale = model._components(embedding, all_anchor)
            logits_cache[member_index, start:stop] = logits.cpu()
            location_cache[member_index, start:stop] = location.cpu()
            scale_cache[member_index, start:stop] = scale.cpu()
    role_keys = [data["keys"][int(row)] for row in rows]
    group_ids, group_keys = _coordinate_groups(role_keys)
    randomness = common_student_t_randomness(
        row_count,
        draws=SCORING_DRAWS,
        degrees_freedom=STUDENT_DF,
        seed=SCORING_SEED,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "candidate": name,
        "phase": phase,
        "role": role,
        "role_coordinate_sha256": registry["roles"][role]["coordinate_key_sha256"],
        "row_indices": torch.from_numpy(rows).long(),
        "row_index_sha256": registry["roles"][role]["row_index_sha256"],
        "coordinate_group_ids": group_ids,
        "coordinate_group_keys": group_keys,
        "target_xy": target,
        "target_building": data["building_t"][rows].long(),
        "target_floor": data["floor_t"][rows].long(),
        "query": represented[rows].float(),
        "member_logits": logits_cache,
        "member_locations": location_cache,
        "member_scales": scale_cache,
        "anchor_coords": models[0].anchor_coords.cpu(),
        "anchor_buildings": models[0].anchor_buildings.cpu(),
        "anchor_floors": models[0].anchor_floors.cpu(),
        "anchor_coordinate_keys": payloads[0]["anchor_coordinate_keys"],
        "coord_origin": payloads[0]["coord_origin"],
        "radio_fields": radio["memory"]["fields"],
        "radio_degrees_freedom": radio["memory"]["degrees_freedom"],
        "radio_memory_path": str(radio_file.resolve()),
        "radio_memory_sha256": _sha256_file(radio_file),
        "member_paths": tuple(str(path.resolve()) for path in resolved_paths),
        "member_sha256": tuple(_sha256_file(path) for path in resolved_paths),
        "randomness": randomness,
    }
    _save_torch_new(destination, payload)
    return {
        "cache": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": name,
        "role": role,
        "rows": row_count,
        "coordinate_groups": len(group_keys),
        "members": member_count,
        "anchors": anchors,
    }


def load_cache(
    path: str | Path,
    *,
    registry: dict,
    expected_role: str,
    expected_candidate: str,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("cache registry mismatch")
    if payload.get("role") != expected_role:
        raise ValueError("cache role mismatch")
    if payload.get("candidate") != expected_candidate:
        raise ValueError("cache candidate mismatch")
    record = registry["roles"][expected_role]
    if payload.get("role_coordinate_sha256") != record["coordinate_key_sha256"]:
        raise ValueError("cache coordinate hash mismatch")
    if payload.get("row_index_sha256") != record["row_index_sha256"]:
        raise ValueError("cache row hash mismatch")
    return payload


def grid_result_path(candidate: str) -> Path:
    return DEFAULT_EVIDENCE / f"development_grid_{_candidate_check(candidate)}.pt"


def grid_summary_path(candidate: str) -> Path:
    return DEFAULT_EVIDENCE / f"development_grid_{_candidate_check(candidate)}.json"


def _group_vector(
    row_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    if row_values.ndim != 1 or group_ids.shape != row_values.shape:
        raise ValueError("row values and group IDs must be aligned vectors")
    total = torch.zeros(group_count, dtype=torch.float64)
    count = torch.zeros(group_count, dtype=torch.float64)
    total.scatter_add_(0, group_ids.long(), row_values.double())
    count.scatter_add_(0, group_ids.long(), torch.ones_like(row_values).double())
    if bool((count <= 0).any()):
        raise RuntimeError("an empty coordinate group entered the metric")
    return (total / count).float()


def _support_components(
    cache: dict,
    anchor_indices: torch.Tensor,
    *,
    members: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather aligned member-specific continuous components on Top-K."""

    count = int(members)
    locations = cache["member_locations"][:count].permute(1, 0, 2, 3)
    scales = cache["member_scales"][:count].permute(1, 0, 2, 3)
    gather = anchor_indices[:, None, :, None].expand(
        -1, count, -1, 2
    )
    return (
        torch.gather(locations, 2, gather),
        torch.gather(scales, 2, gather),
    )


def _marked_energy_for_projection(
    cache: dict,
    projection: Mapping[str, torch.Tensor | int],
    *,
    candidate: str,
    device: torch.device,
    batch_size: int = 96,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one support-specific marked energy before lambda selection."""

    name = _candidate_check(candidate)
    indices = projection["anchor_indices"]
    joint = projection["joint_probability"]
    if not isinstance(indices, torch.Tensor) or not isinstance(joint, torch.Tensor):
        raise TypeError("projection tensors are missing")
    anchor_prior = joint.sum(dim=1)
    fields = {
        key: value.to(device)
        for key, value in cache["radio_fields"].items()
    }
    output = []
    offsets = []
    gains = []
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        query = cache["query"][start:stop].to(device)
        local_indices = indices[start:stop].to(device)
        if name == "geometry_invariant":
            energy = marked_energy_on_support(
                query,
                local_indices,
                fields,
                degrees_freedom=STUDENT_DF,
            )
        else:
            energy, nuisance = profile_shared_affine_marked_energy(
                query,
                local_indices,
                fields,
                anchor_prior[start:stop].to(device),
                degrees_freedom=STUDENT_DF,
                **AFFINE_PROFILE,
            )
            offsets.append(nuisance["style_offset"].cpu())
            gains.append(nuisance["style_scale"].cpu())
        output.append(energy.detach().cpu())
    diagnostics: dict[str, torch.Tensor] = {}
    if offsets:
        diagnostics = {
            "style_offset": torch.cat(offsets),
            "style_scale": torch.cat(gains),
        }
    return torch.cat(output), diagnostics


def _configuration_metrics(
    cache: dict,
    projection: Mapping[str, torch.Tensor | int],
    marked_energy: torch.Tensor,
    *,
    members: int,
    lambda_value: float,
    device: torch.device | None = None,
    evaluation_batch_size: int = 256,
) -> dict:
    """Score one complete `(E,K,lambda)` continuous posterior."""

    prior = projection["joint_probability"]
    indices = projection["anchor_indices"]
    retained = projection["retained_mass"]
    reverse_kl = projection["reverse_kl"]
    if not all(isinstance(value, torch.Tensor) for value in (prior, indices, retained, reverse_kl)):
        raise TypeError("projection tensors are incomplete")
    posterior = marked_kl_update(
        prior,
        marked_energy,
        lambda_value=float(lambda_value),
    )
    locations, scales = _support_components(
        cache, indices, members=int(members)
    )
    compute_device = torch.device("cpu") if device is None else device
    rows = len(posterior)
    energy_parts = []
    action_parts = []
    mean_parts = []
    building_parts = []
    floor_parts = []
    entropy_parts = []
    for start in range(0, rows, int(evaluation_batch_size)):
        stop = min(start + int(evaluation_batch_size), rows)
        probability_cpu = posterior[start:stop].reshape(stop - start, -1)
        probability = probability_cpu.to(compute_device)
        location = locations[start:stop].reshape(
            stop - start, -1, 2
        ).to(compute_device)
        scale = scales[start:stop].reshape(
            stop - start, -1, 2
        ).to(compute_device)
        randomness = {
            key: (
                value[start:stop].to(compute_device)
                if isinstance(value, torch.Tensor) and value.ndim > 0
                else value
            )
            for key, value in cache["randomness"].items()
        }
        row_energy, first, second = continuous_energy_score_mc(
            probability,
            location,
            scale,
            cache["target_xy"][start:stop].to(compute_device),
            randomness,
        )
        action = spatial_median_from_draws(first, second)
        mean = (probability[:, :, None] * location).sum(dim=1)
        local_indices = indices[start:stop]
        anchor_probability = posterior[start:stop].sum(dim=1)
        anchor_building = cache["anchor_buildings"][local_indices]
        anchor_floor = cache["anchor_floors"][local_indices]
        building_count = int(cache["anchor_buildings"].max()) + 1
        floor_count = int(cache["anchor_floors"].max()) + 1
        building_probability = torch.zeros(
            stop - start, building_count, dtype=anchor_probability.dtype
        )
        floor_probability = torch.zeros(
            stop - start, floor_count, dtype=anchor_probability.dtype
        )
        building_probability.scatter_add_(1, anchor_building, anchor_probability)
        floor_probability.scatter_add_(1, anchor_floor, anchor_probability)
        energy_parts.append(row_energy.cpu())
        action_parts.append(action.cpu())
        mean_parts.append(mean.cpu())
        building_parts.append(building_probability.argmax(dim=1))
        floor_parts.append(floor_probability.argmax(dim=1))
        entropy_parts.append(
            -(
                probability_cpu
                * probability_cpu.clamp_min(1e-30).log()
            ).sum(dim=1)
        )
    row_energy = torch.cat(energy_parts)
    action = torch.cat(action_parts)
    mixture_mean = torch.cat(mean_parts)
    predicted_building = torch.cat(building_parts)
    predicted_floor = torch.cat(floor_parts)
    posterior_entropy = torch.cat(entropy_parts)
    row_mae = torch.linalg.vector_norm(action - cache["target_xy"], dim=1)
    row_mean_error = torch.linalg.vector_norm(
        mixture_mean - cache["target_xy"], dim=1
    )
    floor_error = (predicted_floor - cache["target_floor"]).abs().float()
    row_3d = torch.sqrt(row_mae.square() + (FLOOR_HEIGHT_M * floor_error).square())
    building_error = (predicted_building != cache["target_building"]).float()
    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    group_energy = _group_vector(row_energy, group_ids, group_count)
    group_mae = _group_vector(row_mae, group_ids, group_count)
    group_3d = _group_vector(row_3d, group_ids, group_count)
    return {
        "E": int(members),
        "K": int(indices.shape[1]),
        "lambda": float(lambda_value),
        "coordinate_macro_energy_score_2d_m": float(group_energy.mean()),
        "coordinate_macro_mae_2d_m": float(group_mae.mean()),
        "coordinate_macro_mae_3d_m": float(group_3d.mean()),
        "row_energy_score_2d_m": float(row_energy.mean()),
        "row_mae_2d_m": float(row_mae.mean()),
        "row_mixture_mean_error_2d_m": float(row_mean_error.mean()),
        "building_error_rate": float(building_error.mean()),
        "floor_error_rate": float((floor_error > 0).float().mean()),
        "mean_retained_prior_mass": float(retained.mean()),
        "q10_retained_prior_mass": float(torch.quantile(retained, 0.10)),
        "mean_truncation_reverse_kl_nat": float(reverse_kl.mean()),
        "mean_posterior_entropy_nat": float(posterior_entropy.mean()),
        "group_energy_score_2d_m": group_energy,
        "group_mae_2d_m": group_mae,
        "group_mae_3d_m": group_3d,
        "row_energy_score_vector": row_energy,
        "row_mae_vector": row_mae,
    }


def evaluate_development_grid(
    *,
    candidate: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_file: str | Path | None = None,
    base_grid_file: str | Path | None = None,
    output: str | Path | None = None,
    summary_output: str | Path | None = None,
    device_name: str = "cuda",
) -> dict:
    """Evaluate the registered full-factorial grid for one candidate."""

    registry = load_registry(registry_path)
    name = _candidate_check(candidate)
    source = Path(cache_file) if cache_file is not None else cache_path(name, "development")
    destination = Path(output) if output is not None else grid_result_path(name)
    summary_destination = (
        Path(summary_output) if summary_output is not None else grid_summary_path(name)
    )
    if destination.exists() or summary_destination.exists():
        raise FileExistsError("refusing to overwrite a V9 development grid")
    cache = load_cache(
        source,
        registry=registry,
        expected_role="development",
        expected_candidate=name,
    )
    device = torch.device(device_name)
    available_members = int(cache["member_logits"].shape[0])
    configurations = []
    first_new_E = 1
    base_record = None
    if base_grid_file is not None:
        base_path = Path(base_grid_file)
        base = _load_grid(base_path, registry=registry, candidate=name)
        base_cache = load_cache(
            base["cache"],
            registry=registry,
            expected_role="development",
            expected_candidate=name,
        )
        base_members = int(base["available_members"])
        if not base_members < available_members:
            raise ValueError("base grid must contain fewer members than the extension")
        if tuple(base_cache["member_sha256"]) != tuple(
            cache["member_sha256"][:base_members]
        ):
            raise ValueError("extension cache does not preserve the base member prefix")
        for key in (
            "row_index_sha256",
            "radio_memory_sha256",
            "coordinate_group_keys",
        ):
            if base_cache[key] != cache[key]:
                raise ValueError(f"extension cache changed registered field {key}")
        configurations = list(base["configurations"])
        first_new_E = base_members + 1
        base_record = {
            "path": str(base_path.resolve()),
            "sha256": _sha256_file(base_path),
            "available_members": base_members,
        }
    started = time.perf_counter()
    for E in range(first_new_E, available_members + 1):
        full_joint = uniform_member_anchor_prior(
            cache["member_logits"], members=E
        )
        for K in K_GRID:
            if K > full_joint.shape[2]:
                continue
            projection = topk_prior_projection(full_joint, top_k=K)
            marked, nuisance = _marked_energy_for_projection(
                cache,
                projection,
                candidate=name,
                device=device,
            )
            for lambda_value in LAMBDA_GRID:
                metrics = _configuration_metrics(
                    cache,
                    projection,
                    marked,
                    members=E,
                    lambda_value=lambda_value,
                    device=device,
                )
                configurations.append(metrics)
                print(
                    json.dumps(
                        {
                            "candidate": name,
                            "E": E,
                            "K": K,
                            "lambda": lambda_value,
                            "macro_energy_m": metrics[
                                "coordinate_macro_energy_score_2d_m"
                            ],
                            "macro_mae_m": metrics["coordinate_macro_mae_2d_m"],
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
            del marked, projection
            if nuisance and device.type == "cuda":
                torch.cuda.empty_cache()
        del full_joint
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "candidate": name,
        "cache": str(source.resolve()),
        "cache_sha256": _sha256_file(source),
        "available_members": available_members,
        "inherited_base_grid": base_record,
        "coordinate_group_keys": cache["coordinate_group_keys"],
        "configurations": configurations,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _save_torch_new(destination, payload)
    profiled = []
    for E in range(1, available_members + 1):
        selected = min(
            (row for row in configurations if row["E"] == E),
            key=lambda row: row["coordinate_macro_energy_score_2d_m"],
        )
        profiled.append(
            {
                key: value
                for key, value in selected.items()
                if not isinstance(value, torch.Tensor)
            }
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "candidate": name,
        "grid": str(destination.resolve()),
        "grid_sha256": _sha256_file(destination),
        "cache_sha256": payload["cache_sha256"],
        "available_members": available_members,
        "profiled_by_E": profiled,
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    _write_json_new(summary_destination, summary)
    return summary


def _load_grid(
    path: str | Path,
    *,
    registry: dict,
    candidate: str,
) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 development grid")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("grid registry mismatch")
    if payload.get("candidate") != candidate:
        raise ValueError("grid candidate mismatch")
    if _sha256_file(payload["cache"]) != payload["cache_sha256"]:
        raise ValueError("development cache changed after grid evaluation")
    return payload


def _profile_grid(grid: dict) -> dict[int, dict]:
    result = {}
    for E in range(1, int(grid["available_members"]) + 1):
        rows = [row for row in grid["configurations"] if int(row["E"]) == E]
        if not rows:
            raise RuntimeError(f"grid misses E={E}")
        result[E] = min(
            rows,
            key=lambda row: float(row["coordinate_macro_energy_score_2d_m"]),
        )
    return result


def select_and_freeze(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    geometry_grid_path: str | Path | None = None,
    historical_grid_path: str | Path | None = None,
    output: str | Path = DEFAULT_FREEZE,
) -> dict:
    """Compare complete candidates, establish E saturation, and freeze once."""

    registry = load_registry(registry_path)
    geometry_path = (
        Path(geometry_grid_path)
        if geometry_grid_path is not None
        else grid_result_path("geometry_invariant")
    )
    historical_path = (
        Path(historical_grid_path)
        if historical_grid_path is not None
        else grid_result_path("historical_affine")
    )
    grids = {
        "geometry_invariant": _load_grid(
            geometry_path, registry=registry, candidate="geometry_invariant"
        ),
        "historical_affine": _load_grid(
            historical_path, registry=registry, candidate="historical_affine"
        ),
    }
    profiles = {name: _profile_grid(grid) for name, grid in grids.items()}
    common_max_E = min(max(profile) for profile in profiles.values())
    geometry_common = profiles["geometry_invariant"][common_max_E]
    historical_common = profiles["historical_affine"][common_max_E]
    structural_tolerance = 0.01 * float(
        profiles["historical_affine"][1]["coordinate_macro_energy_score_2d_m"]
    )
    energy_comparison = simultaneous_paired_upper_bounds(
        geometry_common["group_energy_score_2d_m"].numpy(),
        [historical_common["group_energy_score_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED,
    )
    mae_comparison = simultaneous_paired_upper_bounds(
        geometry_common["group_mae_2d_m"].numpy(),
        [historical_common["group_mae_2d_m"].numpy()],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 100,
    )
    geometry_mae_eligible = bool(
        mae_comparison["simultaneous_upper_bounds"][0]
        <= MAE_NONINFERIORITY_MARGIN_M
    )
    geometry_energy_equivalent_or_better = bool(
        energy_comparison["mean_differences"][0] <= 0.0
        or energy_comparison["simultaneous_upper_bounds"][0]
        <= structural_tolerance
    )
    selected_candidate = (
        "geometry_invariant"
        if geometry_mae_eligible and geometry_energy_equivalent_or_better
        else "historical_affine"
    )
    selected_profile = profiles[selected_candidate]
    saturation_tolerance = 0.01 * float(
        selected_profile[1]["coordinate_macro_energy_score_2d_m"]
    )
    saturation = select_information_saturated_E(
        {
            E: row["group_energy_score_2d_m"].numpy()
            for E, row in selected_profile.items()
        },
        tolerance=saturation_tolerance,
        minimum_later_values=2,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        seed=PAIRED_INFERENCE_SEED + 200,
    )
    selection_report = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "candidate_comparison_at_common_E": common_max_E,
        "structural_energy_tolerance_m": structural_tolerance,
        "energy_comparison_geometry_minus_historical": energy_comparison,
        "mae_comparison_geometry_minus_historical": mae_comparison,
        "geometry_mae_noninferior": geometry_mae_eligible,
        "geometry_energy_equivalent_or_better": geometry_energy_equivalent_or_better,
        "nominated_candidate": selected_candidate,
        "saturation": saturation,
        "grid_files": {
            "geometry_invariant": {
                "path": str(geometry_path.resolve()),
                "sha256": _sha256_file(geometry_path),
            },
            "historical_affine": {
                "path": str(historical_path.resolve()),
                "sha256": _sha256_file(historical_path),
            },
        },
    }
    selection_basis = "confirmed_information_saturation"
    protocol_amendment = None
    if not saturation["saturated"]:
        max_tested_E = max(selected_profile)
        next_member_indices = list(
            range(
                max_tested_E + 1,
                min(max_tested_E + 2, len(MEMBER_SEEDS)) + 1,
            )
        )
        if next_member_indices:
            report_path = Path(output).with_name(
                f"selection_extension_required_through_E{max_tested_E}_v9_uji.json"
            )
            selection_report["status"] = "extension_required"
            selection_report["next_member_indices"] = next_member_indices
            selection_report["extension_exhausted"] = False
            _write_json_new(report_path, selection_report)
            return selection_report

        # The original protocol specified the two extensions through E=12 but
        # omitted a terminal action when equivalence remained unconfirmed.
        # Outer-test outcomes are still inaccessible here.  The auditable
        # conservative completion is ordinary development-risk minimization;
        # it makes no saturation claim and cannot silently prefer simplicity.
        selected_E = min(
            selected_profile,
            key=lambda E: (
                float(selected_profile[E]["coordinate_macro_energy_score_2d_m"]),
                int(E),
            ),
        )
        selection_basis = "budget_censored_development_risk_minimum"
        protocol_amendment = {
            "id": "V9-A1-terminal-selection",
            "timing": "after_development_before_outer_test",
            "reason": "the original protocol omitted the terminal action when E=12 remained unsaturated",
            "rule": "at the preregistered member-budget ceiling, select the global profiled development proper-risk minimum and make no saturation claim",
            "maximum_registered_E": len(MEMBER_SEEDS),
        }
    else:
        selected_E = int(saturation["selected_E"])
    selected_row = selected_profile[selected_E]
    freeze = {
        **selection_report,
        "status": "selection_frozen",
        "selection_basis": selection_basis,
        "saturation_claim": bool(saturation["saturated"]),
        "protocol_amendment": protocol_amendment,
        "selected": {
            "candidate": selected_candidate,
            "E": selected_E,
            "K": int(selected_row["K"]),
            "lambda": float(selected_row["lambda"]),
            "development_coordinate_macro_energy_score_2d_m": float(
                selected_row["coordinate_macro_energy_score_2d_m"]
            ),
            "development_coordinate_macro_mae_2d_m": float(
                selected_row["coordinate_macro_mae_2d_m"]
            ),
            "member_indices": list(range(1, selected_E + 1)),
            "member_seeds": list(MEMBER_SEEDS[:selected_E]),
        },
        "outer_test_accesses_before_freeze": 0,
    }
    freeze["freeze_sha256"] = _canonical_json_sha256(freeze)
    _write_json_new(output, freeze)
    return freeze


def load_selection_freeze(
    path: str | Path = DEFAULT_FREEZE,
    *,
    registry: dict,
) -> dict:
    freeze_path = Path(path)
    if not freeze_path.exists():
        raise PermissionError(
            "outer-test/final-refit access is forbidden before the V9 selection freeze"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not a V9 selection freeze")
    if freeze.get("status") != "selection_frozen":
        raise ValueError("V9 selection is not frozen")
    if freeze.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("selection-freeze registry mismatch")
    value = copy.deepcopy(freeze)
    stored = value.pop("freeze_sha256", None)
    if stored != _canonical_json_sha256(value):
        raise ValueError("selection-freeze self-hash failed")
    if freeze.get("outer_test_accesses_before_freeze") != 0:
        raise ValueError("outer-test access contract was violated")
    for record in freeze["grid_files"].values():
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("a frozen development grid changed")
    return freeze


def train_final_member(
    *,
    member_index: int,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    output: str | Path | None = None,
    device_name: str = "cuda",
    log_every: int = 5,
) -> dict:
    registry = load_registry(registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry)
    candidate = freeze["selected"]["candidate"]
    index = int(member_index)
    if index not in freeze["selected"]["member_indices"]:
        raise ValueError("member is outside the frozen deployment prefix")
    destination = (
        Path(output) if output is not None else final_checkpoint_path(candidate, index)
    )
    if destination.exists():
        raise FileExistsError(destination)
    selection_path = selection_checkpoint_path(candidate, index)
    _selection_model, selection_payload = load_front_checkpoint(
        selection_path, torch.device("cpu")
    )
    if selection_payload["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("selection member registry mismatch")
    epoch = int(selection_payload["selected_epoch"])
    seed = MEMBER_SEEDS[index - 1]
    data = _load_uji(registry["train_data"])
    rows = _registered_rows(registry, "outer_refit", data)
    device = torch.device(device_name)
    started = time.perf_counter()
    model, memory, origin, history, geometry = _fit_fixed(
        data,
        rows,
        candidate=candidate,
        seed=seed,
        epochs=epoch,
        device=device,
        log_every=log_every,
    )
    memory = dict(memory)
    memory["stats"] = data["stats"]
    payload = _checkpoint_payload(
        registry=registry,
        phase="final",
        refit_role="outer_refit",
        candidate=candidate,
        member_index=index,
        selected_epoch=epoch,
        model=model,
        memory=memory,
        origin=origin,
        schedule_history=selection_payload["schedule_history"],
        refit_history=history,
        geometry=geometry,
    )
    payload["selection_checkpoint"] = str(selection_path.resolve())
    payload["selection_checkpoint_sha256"] = _sha256_file(selection_path)
    payload["freeze_sha256"] = freeze["freeze_sha256"]
    payload["training_seconds"] = float(time.perf_counter() - started)
    _save_torch_new(destination, payload)
    return {
        "checkpoint": str(destination.resolve()),
        "sha256": _sha256_file(destination),
        "candidate": candidate,
        "member_index": index,
        "selected_epoch": epoch,
        "training_seconds": payload["training_seconds"],
    }


def evaluate_outer_test(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    freeze_path: str | Path = DEFAULT_FREEZE,
    cache_file: str | Path | None = None,
    output: str | Path = DEFAULT_RESULT,
) -> dict:
    """Evaluate the frozen final model on outer coordinate groups exactly once."""

    registry = load_registry(registry_path)
    freeze = load_selection_freeze(freeze_path, registry=registry)
    selected = freeze["selected"]
    candidate = selected["candidate"]
    source = Path(cache_file) if cache_file is not None else cache_path(candidate, "outer_test")
    cache = load_cache(
        source,
        registry=registry,
        expected_role="outer_test",
        expected_candidate=candidate,
    )
    if int(cache["member_logits"].shape[0]) != int(selected["E"]):
        raise ValueError("outer cache member count differs from the freeze")
    full_joint = uniform_member_anchor_prior(
        cache["member_logits"], members=int(selected["E"])
    )
    projection = topk_prior_projection(full_joint, top_k=int(selected["K"]))
    marked, nuisance = _marked_energy_for_projection(
        cache,
        projection,
        candidate=candidate,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    metrics = _configuration_metrics(
        cache,
        projection,
        marked,
        members=int(selected["E"]),
        lambda_value=float(selected["lambda"]),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "freeze": str(Path(freeze_path).resolve()),
        "freeze_sha256": freeze["freeze_sha256"],
        "outer_cache": str(source.resolve()),
        "outer_cache_sha256": _sha256_file(source),
        "outer_test_access_number": 1,
        "selected": selected,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, torch.Tensor)
        },
        "coordinate_group_energy_score_2d_m": metrics[
            "group_energy_score_2d_m"
        ].tolist(),
        "coordinate_group_mae_2d_m": metrics["group_mae_2d_m"].tolist(),
        "coordinate_group_mae_3d_m": metrics["group_mae_3d_m"].tolist(),
        "affine_nuisance_summary": (
            None
            if not nuisance
            else {
                "mean_offset": float(nuisance["style_offset"].mean()),
                "mean_scale": float(nuisance["style_scale"].mean()),
            }
        ),
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register")
    register.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    register.add_argument("--design", default=str(DEFAULT_DESIGN))
    register.add_argument("--output", default=str(DEFAULT_REGISTRY))

    train = sub.add_parser("train-member")
    train.add_argument("--candidate", choices=CANDIDATES, required=True)
    train.add_argument("--member-index", type=int, required=True)
    train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    train.add_argument("--output")
    train.add_argument("--device", default="cuda")
    train.add_argument("--log-every", type=int, default=5)

    radio = sub.add_parser("fit-radio")
    radio.add_argument("--candidate", choices=CANDIDATES, required=True)
    radio.add_argument("--phase", choices=("selection", "final"), required=True)
    radio.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    radio.add_argument("--output")

    cache = sub.add_parser("build-cache")
    cache.add_argument("--candidate", choices=CANDIDATES, required=True)
    cache.add_argument("--role", choices=("development", "outer_test"), required=True)
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--radio")
    cache.add_argument("--members", type=int)
    cache.add_argument("--output")
    cache.add_argument("--device", default="cuda")

    grid = sub.add_parser("grid")
    grid.add_argument("--candidate", choices=CANDIDATES, required=True)
    grid.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    grid.add_argument("--cache")
    grid.add_argument("--base-grid")
    grid.add_argument("--output")
    grid.add_argument("--summary-output")
    grid.add_argument("--device", default="cuda")

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    freeze.add_argument("--geometry-grid")
    freeze.add_argument("--historical-grid")
    freeze.add_argument("--output", default=str(DEFAULT_FREEZE))

    final = sub.add_parser("train-final-member")
    final.add_argument("--member-index", type=int, required=True)
    final.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    final.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    final.add_argument("--output")
    final.add_argument("--device", default="cuda")
    final.add_argument("--log-every", type=int, default=5)

    evaluate = sub.add_parser("evaluate-outer")
    evaluate.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evaluate.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    evaluate.add_argument("--cache")
    evaluate.add_argument("--output", default=str(DEFAULT_RESULT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "register":
        result = register_protocol(
            train_data=args.train_data,
            design_path=args.design,
            output=args.output,
        )
    elif args.command == "train-member":
        result = train_selection_member(
            candidate=args.candidate,
            member_index=args.member_index,
            registry_path=args.registry,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "fit-radio":
        result = fit_radio_memory(
            candidate=args.candidate,
            phase=args.phase,
            registry_path=args.registry,
            output=args.output,
        )
    elif args.command == "build-cache":
        result = build_cache(
            candidate=args.candidate,
            role=args.role,
            registry_path=args.registry,
            member_count=args.members,
            radio_path=args.radio,
            output=args.output,
            device_name=args.device,
        )
    elif args.command == "grid":
        result = evaluate_development_grid(
            candidate=args.candidate,
            registry_path=args.registry,
            cache_file=args.cache,
            base_grid_file=args.base_grid,
            output=args.output,
            summary_output=args.summary_output,
            device_name=args.device,
        )
    elif args.command == "freeze":
        result = select_and_freeze(
            registry_path=args.registry,
            geometry_grid_path=args.geometry_grid,
            historical_grid_path=args.historical_grid,
            output=args.output,
        )
    elif args.command == "train-final-member":
        result = train_final_member(
            member_index=args.member_index,
            registry_path=args.registry,
            freeze_path=args.freeze,
            output=args.output,
            device_name=args.device,
            log_every=args.log_every,
        )
    elif args.command == "evaluate-outer":
        result = evaluate_outer_test(
            registry_path=args.registry,
            freeze_path=args.freeze,
            cache_file=args.cache,
            output=args.output,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
