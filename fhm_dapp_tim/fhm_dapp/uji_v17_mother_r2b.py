"""Execute V17-R2B on the full, untruncated atomic member laws."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import (
    common_student_t_randomness,
    continuous_energy_score_mc,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _save_torch_new,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.projected_ensemble import ProjectedEnsemble
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a2 import (
    _installed_adapters,
    _role_record,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r2b_protocol import (
    ANCHOR_COUNT,
    BETA_GRID,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    COMPONENT_BATCH_SIZE,
    DEFAULT_CACHE,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT,
    HISTORICAL_BETA_NEURAL,
    HISTORICAL_BETA_PROTOTYPE,
    HISTORICAL_VARIANT,
    LOG_MULTIPLIERS,
    MAE_NONINFERIORITY_MARGIN_M,
    MEMBER_COUNT,
    PREDICTIVE_DRAWS_PER_STREAM,
    PREDICTIVE_SEED,
    PROTOCOL_VERSION,
    SCORING_BATCH_SIZE,
    STUDENT_DF,
    VARIANTS,
    beta_name,
    load_registry,
)


def _validate_output(registry: Mapping[str, object], field: str, path: Path) -> None:
    allowed = Path(registry["allowed_outputs"][field]).resolve()
    if path.resolve() != allowed:
        raise ValueError(f"{field} path is outside the R2B registry")
    if path.exists():
        raise FileExistsError(path)


def _coordinate_groups(
    keys: Sequence[tuple],
) -> tuple[torch.Tensor, tuple[tuple, ...]]:
    ordered = tuple(sorted(set(map(tuple, keys))))
    mapping = {key: index for index, key in enumerate(ordered)}
    return (
        torch.tensor([mapping[tuple(key)] for key in keys], dtype=torch.long),
        ordered,
    )


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
        raise RuntimeError("an empty coordinate group entered R2B")
    return total / count


def paired_simultaneous_family(
    differences: Mapping[str, np.ndarray],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Max-centered one-sided upper bounds with common group resamples."""

    names = tuple(differences)
    if not names:
        raise ValueError("a simultaneous family cannot be empty")
    arrays = [np.asarray(differences[name], dtype=np.float64) for name in names]
    shape = arrays[0].shape
    if len(shape) != 1 or len(arrays[0]) < 2 or any(x.shape != shape for x in arrays):
        raise ValueError("family differences must be aligned group vectors")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("family differences must be finite")
    matrix = np.stack(arrays, axis=1)
    means = matrix.mean(axis=0)
    draws = np.empty((int(repetitions), len(names)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for start in range(0, int(repetitions), 250):
        stop = min(start + 250, int(repetitions))
        index = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        draws[start:stop] = matrix[index].mean(axis=1)
    maximum_centered = (draws - means[None, :]).max(axis=1)
    critical = float(np.quantile(maximum_centered, float(confidence)))
    comparisons = {}
    for column, name in enumerate(names):
        comparisons[name] = {
            "mean_difference_m": float(means[column]),
            "paired_percentile_ci95_m": [
                float(np.quantile(draws[:, column], 0.025)),
                float(np.quantile(draws[:, column], 0.975)),
            ],
            "simultaneous_upper_bound_m": float(means[column] + critical),
            "bootstrap_probability_difference_below_zero": float(
                np.mean(draws[:, column] < 0.0)
            ),
        }
    return {
        "difference_orientation": "candidate_minus_historical_reference",
        "comparisons": comparisons,
        "critical_value_m": critical,
        "confidence": float(confidence),
        "family_size": len(names),
        "coordinate_groups": int(len(matrix)),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "common_coordinate_group_resamples": True,
    }


def _prototype_distance(model, rss: torch.Tensor) -> torch.Tensor:
    if model.prototype_metric != "l2" or model.prototype_rss.numel() == 0:
        raise ValueError("R2B requires the registered L2 prototype law")
    seen = model.encoder.seen_ap_mask
    query = rss[:, seen].float()
    prototypes = model.prototype_rss[:, seen].float()
    count = int(seen.sum())
    if count <= 0:
        raise ValueError("R2B has no shared visible AP coordinate")
    query_norm = query.square().mean(dim=1, keepdim=True)
    prototype_norm = prototypes.square().mean(dim=1).unsqueeze(0)
    cross = query @ prototypes.t() / count
    return (query_norm + prototype_norm - 2.0 * cross).clamp(min=0.0).sqrt()


@torch.no_grad()
def build_cache(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output: str | Path = DEFAULT_CACHE,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_cache", destination)
    selection_csv = Path(registry["data_contract"]["selection_refit_csv"])
    development_csv = Path(registry["data_contract"]["development_csv"])
    checkpoints = [Path(item["path"]) for item in registry["frozen_checkpoints"]]
    selection_role = _role_record(selection_csv)
    development_role = _role_record(development_csv)
    device = torch.device(device_name)

    with _installed_adapters([selection_role, development_role]):
        ensemble = ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device=device_name,
            dataset="UJI",
        )
        loaded = load_dataset(
            str(development_csv),
            dataset_type="UJI",
            device="cpu",
            reference_stats=ensemble.reference["stats"],
        )
    rss = loaded[0].float().cpu()
    exact = load_uji_v16(development_csv)
    expected_role = registry["data_contract"]["development_role"]
    if len(rss) != expected_role["row_count"] or not torch.equal(rss, exact["rss"]):
        raise ValueError("R2B development RSS or row count changed")
    if not np.array_equal(development_role["building"], exact["building"]):
        raise ValueError("R2B development building labels changed")
    if not np.array_equal(development_role["floor"], exact["floor"]):
        raise ValueError("R2B development floor labels changed")
    if len(ensemble.checkpoints) != MEMBER_COUNT:
        raise ValueError("R2B member count changed")

    models = [model for model, _payload in ensemble.checkpoints]
    reference_model = models[0]
    reference_seen = reference_model.encoder.seen_ap_mask.detach().cpu()
    reference_prototypes = reference_model.prototype_rss.detach().cpu()
    reference_coords = reference_model.anchor_coords.detach().cpu()
    for member_index, model in enumerate(models, start=1):
        if model.use_candidate_reranker:
            raise ValueError(f"member {member_index} unexpectedly uses a reranker")
        if model.residual_experts != 1:
            raise ValueError(f"member {member_index} is not an atomic residual law")
        if not math.isclose(float(model.student_df), STUDENT_DF, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"member {member_index} Student-t law changed")
        if model.n_anchors != ANCHOR_COUNT:
            raise ValueError(f"member {member_index} anchor count changed")
        if not torch.equal(model.encoder.seen_ap_mask.cpu(), reference_seen):
            raise ValueError("member seen-AP masks differ")
        if not torch.equal(model.prototype_rss.cpu(), reference_prototypes):
            raise ValueError("member prototype memories differ")
        if not torch.allclose(model.anchor_coords.cpu(), reference_coords, atol=1e-6):
            raise ValueError("member anchor coordinates differ")

    query = rss.to(device)
    distance_parts = []
    for start in range(0, len(query), COMPONENT_BATCH_SIZE):
        distance_parts.append(
            _prototype_distance(reference_model, query[start:start + COMPONENT_BATCH_SIZE]).cpu()
        )
    prototype_distance = torch.cat(distance_parts)

    neural_members = []
    location_members = []
    scale_members = []
    for member_index, model in enumerate(models, start=1):
        model.eval()
        neural_parts = []
        location_parts = []
        scale_parts = []
        for start in range(0, len(query), COMPONENT_BATCH_SIZE):
            batch = query[start:start + COMPONENT_BATCH_SIZE]
            embedding = model.encoder(batch)
            neural = model.retrieval_logits_from_embedding(embedding)
            indices = torch.arange(
                ANCHOR_COUNT, device=device, dtype=torch.long
            ).unsqueeze(0).expand(len(batch), -1)
            location, scale = model._components(embedding, indices)
            neural_parts.append(neural.cpu())
            location_parts.append(location.cpu())
            scale_parts.append(scale.cpu())
        neural_members.append(torch.cat(neural_parts))
        location_members.append(torch.cat(location_parts))
        scale_members.append(torch.cat(scale_parts))
        print(json.dumps({
            "cached_member": member_index,
            "rows": len(query),
            "anchors": ANCHOR_COUNT,
        }))

    origin = np.asarray(ensemble.reference["coord_origin"], dtype=np.float64)
    target = torch.from_numpy(development_role["raw_xy"] - origin[None, :]).double()
    group_ids, group_keys = _coordinate_groups(exact["keys"])
    expected_keys = tuple(tuple(value) for value in expected_role["coordinate_keys"])
    if group_keys != expected_keys:
        raise ValueError("R2B development coordinate identities changed")
    randomness = common_student_t_randomness(
        len(rss),
        draws=PREDICTIVE_DRAWS_PER_STREAM,
        degrees_freedom=STUDENT_DF,
        seed=PREDICTIVE_SEED,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "role": "development",
        "development_csv_sha256": _sha256_file(development_csv),
        "selection_refit_csv_sha256": _sha256_file(selection_csv),
        "member_paths": tuple(str(path.resolve()) for path in checkpoints),
        "member_sha256": tuple(_sha256_file(path) for path in checkpoints),
        "coordinate_group_ids": group_ids,
        "coordinate_group_keys": group_keys,
        "target_xy": target,
        "prototype_distance": prototype_distance,
        "member_neural_logits": torch.stack(neural_members),
        "member_locations": torch.stack(location_members),
        "member_scales": torch.stack(scale_members),
        "anchor_coordinates": reference_coords,
        "coord_origin": torch.from_numpy(origin.copy()),
        "randomness": randomness,
        "construction_contract": {
            "members": MEMBER_COUNT,
            "rows": int(len(rss)),
            "anchors": ANCHOR_COUNT,
            "full_untruncated_support": True,
            "member_consensus_applied": False,
            "candidate_reranker_applied": False,
            "component_batch_size": COMPONENT_BATCH_SIZE,
            "development_scan_predictions": int(len(rss)),
            "member_laws_per_scan": MEMBER_COUNT,
            "outer_test_predictions": 0,
            "official_validation_predictions": 0,
        },
    }
    _save_torch_new(destination, payload)
    return {
        "cache": str(destination),
        "cache_sha256": _sha256_file(destination),
        "rows": int(len(rss)),
        "coordinate_groups": int(len(group_keys)),
        "members": MEMBER_COUNT,
        "anchors": ANCHOR_COUNT,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def load_cache(
    path: str | Path = DEFAULT_CACHE,
    *,
    registry: Mapping[str, object],
) -> dict:
    source = Path(path).resolve()
    if source != Path(registry["allowed_outputs"]["development_cache"]).resolve():
        raise ValueError("cache path is outside the R2B registry")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("not an R2B cache")
    if payload.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R2B cache and registry differ")
    expected_paths = tuple(item["path"] for item in registry["frozen_checkpoints"])
    expected_hashes = tuple(item["sha256"] for item in registry["frozen_checkpoints"])
    if tuple(payload["member_paths"]) != expected_paths:
        raise ValueError("R2B cache member paths changed")
    if tuple(payload["member_sha256"]) != expected_hashes:
        raise ValueError("R2B cache member hashes changed")
    rows = registry["data_contract"]["development_role"]["row_count"]
    expected = (MEMBER_COUNT, rows, ANCHOR_COUNT)
    if tuple(payload["member_neural_logits"].shape) != expected:
        raise ValueError("R2B neural-logit cache shape changed")
    if tuple(payload["prototype_distance"].shape) != expected[1:]:
        raise ValueError("R2B prototype-distance cache shape changed")
    if tuple(payload["member_locations"].shape) != (*expected, 2):
        raise ValueError("R2B location cache shape changed")
    if tuple(payload["member_scales"].shape) != (*expected, 2):
        raise ValueError("R2B scale cache shape changed")
    if not bool(torch.isfinite(payload["member_neural_logits"]).all()):
        raise ValueError("R2B neural logits are not finite")
    if not bool(torch.isfinite(payload["member_locations"]).all()):
        raise ValueError("R2B locations are not finite")
    if not bool(torch.isfinite(payload["member_scales"]).all()):
        raise ValueError("R2B scales are not finite")
    if bool((payload["member_scales"] <= 0.0).any()):
        raise ValueError("R2B scales are not positive")
    return payload


def _metric_record(
    row_values: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: int,
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


def select_registered_variant(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    energy_family: Mapping[str, object],
    mae_family: Mapping[str, object],
) -> dict[str, object]:
    eligible = [HISTORICAL_VARIANT]
    for name in VARIANTS:
        if name == HISTORICAL_VARIANT:
            continue
        upper = mae_family["comparisons"][name]["simultaneous_upper_bound_m"]
        if float(upper) <= MAE_NONINFERIORITY_MARGIN_M:
            eligible.append(name)

    grid_by_name = {
        beta_name(beta_neural, beta_prototype): (beta_neural, beta_prototype)
        for beta_neural, beta_prototype in BETA_GRID
    }
    order = {name: index for index, name in enumerate(VARIANTS)}

    def tie_key(name: str) -> tuple[float, float, int]:
        beta_neural, beta_prototype = grid_by_name[name]
        distance = (
            math.log(beta_neural / HISTORICAL_BETA_NEURAL) ** 2
            + math.log(beta_prototype / HISTORICAL_BETA_PROTOTYPE) ** 2
        )
        energy = metrics[name]["energy_score_2d_m"]["coordinate_macro_m"]
        return float(energy), float(distance), order[name]

    pointwise_best = min(eligible, key=tie_key)
    evidence_passed = False
    if pointwise_best != HISTORICAL_VARIANT:
        evidence_passed = bool(
            energy_family["comparisons"][pointwise_best][
                "simultaneous_upper_bound_m"
            ] < 0.0
        )
    selected = pointwise_best if evidence_passed else HISTORICAL_VARIANT
    beta_neural, beta_prototype = grid_by_name[selected]
    multipliers = (
        beta_neural / HISTORICAL_BETA_NEURAL,
        beta_prototype / HISTORICAL_BETA_PROTOTYPE,
    )
    return {
        "variant": selected,
        "beta_neural": float(beta_neural),
        "beta_prototype": float(beta_prototype),
        "pointwise_best_eligible": pointwise_best,
        "eligible_variants": eligible,
        "replacement_evidence_passed": evidence_passed,
        "historical_retained_due_to_insufficient_evidence": bool(
            pointwise_best != HISTORICAL_VARIANT and not evidence_passed
        ),
        "selected_on_registered_grid_boundary": bool(
            multipliers[0] in (min(LOG_MULTIPLIERS), max(LOG_MULTIPLIERS))
            or multipliers[1] in (min(LOG_MULTIPLIERS), max(LOG_MULTIPLIERS))
        ),
        "grid_extension_authorized": False,
    }


@torch.no_grad()
def evaluate(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_path: str | Path = DEFAULT_CACHE,
    output: str | Path = DEFAULT_RESULT,
    device_name: str = "cuda",
) -> dict:
    registry = load_registry(registry_path)
    destination = Path(output).resolve()
    _validate_output(registry, "development_result", destination)
    cache = load_cache(cache_path, registry=registry)
    device = torch.device(device_name)
    rows = len(cache["target_xy"])
    group_ids = cache["coordinate_group_ids"]
    group_count = len(cache["coordinate_group_keys"])
    target = cache["target_xy"]
    prototype_distance = cache["prototype_distance"].to(device)
    randomness = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in cache["randomness"].items()
    }
    row_energy = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}
    row_mae = {name: torch.zeros(rows, dtype=torch.float64) for name in VARIANTS}

    for member in range(MEMBER_COUNT):
        neural = cache["member_neural_logits"][member].to(device)
        location = cache["member_locations"][member].to(device)
        scale = cache["member_scales"][member].to(device)
        for beta_neural, beta_prototype in BETA_GRID:
            name = beta_name(beta_neural, beta_prototype)
            probability = torch.softmax(
                float(beta_neural) * neural
                - float(beta_prototype) * prototype_distance,
                dim=1,
            )
            action = torch.sum(probability[:, :, None] * location, dim=1)
            row_mae[name] += (
                torch.linalg.vector_norm(action.cpu().double() - target, dim=1)
                / MEMBER_COUNT
            )
            energy_parts = []
            for start in range(0, rows, SCORING_BATCH_SIZE):
                stop = min(start + SCORING_BATCH_SIZE, rows)
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
                    target[start:stop].to(device),
                    batch_randomness,
                )
                energy_parts.append(energy.cpu().double())
            row_energy[name] += torch.cat(energy_parts) / MEMBER_COUNT
        print(json.dumps({"scored_member": member + 1, "variants": len(VARIANTS)}))

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

    family_names = tuple(name for name in VARIANTS if name != HISTORICAL_VARIANT)
    energy_family = paired_simultaneous_family(
        {
            name: energy_groups[name] - energy_groups[HISTORICAL_VARIANT]
            for name in family_names
        },
        seed=BOOTSTRAP_SEED,
    )
    mae_family = paired_simultaneous_family(
        {
            name: mae_groups[name] - mae_groups[HISTORICAL_VARIANT]
            for name in family_names
        },
        seed=BOOTSTRAP_SEED,
    )
    selection = select_registered_variant(metrics, energy_family, mae_family)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": registry["registry_sha256"],
        "cache_path": str(Path(cache_path).resolve()),
        "cache_sha256": _sha256_file(cache_path),
        "estimand": registry["estimand"],
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
            "energy_score_candidate_minus_historical": energy_family,
            "mae_candidate_minus_historical": mae_family,
        },
        "registered_selection": selection,
        "development_scan_predictions": rows,
        "member_laws_per_scan": MEMBER_COUNT,
        "registered_variants": len(VARIANTS),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }
    result["result_sha256"] = _canonical_json_sha256(result)
    _write_json_new(destination, result)
    return result


def check_result(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    cache_path: str | Path = DEFAULT_CACHE,
    result_path: str | Path = DEFAULT_RESULT,
) -> dict:
    registry = load_registry(registry_path)
    cache = load_cache(cache_path, registry=registry)
    source = Path(result_path).resolve()
    if source != Path(registry["allowed_outputs"]["development_result"]).resolve():
        raise ValueError("result path is outside the R2B registry")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("result_sha256") != _canonical_json_sha256(
        result, omit="result_sha256"
    ):
        raise ValueError("R2B result self-hash failed")
    if result.get("registry_sha256") != registry["registry_sha256"]:
        raise ValueError("R2B result and registry differ")
    if result.get("cache_sha256") != _sha256_file(cache_path):
        raise ValueError("R2B result and cache differ")
    return {
        "registry_sha256": registry["registry_sha256"],
        "cache_sha256": _sha256_file(cache_path),
        "result_sha256": result["result_sha256"],
        "selected": result["registered_selection"],
        "cache_rows": len(cache["target_xy"]),
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    cache.add_argument("--output", default=str(DEFAULT_CACHE))
    cache.add_argument("--device", default="cuda")
    score = sub.add_parser("evaluate")
    score.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    score.add_argument("--cache", default=str(DEFAULT_CACHE))
    score.add_argument("--output", default=str(DEFAULT_RESULT))
    score.add_argument("--device", default="cuda")
    check = sub.add_parser("check-result")
    check.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    check.add_argument("--cache", default=str(DEFAULT_CACHE))
    check.add_argument("--result", default=str(DEFAULT_RESULT))
    args = parser.parse_args(argv)
    if args.command == "build-cache":
        payload = build_cache(
            registry_path=args.registry, output=args.output, device_name=args.device
        )
    elif args.command == "evaluate":
        payload = evaluate(
            registry_path=args.registry,
            cache_path=args.cache,
            output=args.output,
            device_name=args.device,
        )
    else:
        payload = check_result(
            registry_path=args.registry,
            cache_path=args.cache,
            result_path=args.result,
        )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
