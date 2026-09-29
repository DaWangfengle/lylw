"""Post-development diagnosis for the failed V17-R0 reconstruction gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
    _write_json_new,
)
from fhm_dapp_tim.fhm_dapp.uji_v16_protocol import load_uji_v16


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "v17_acrhap_mother"
DEFAULT_RESULT = EVIDENCE / "development_result_v17_r0.json"
DEFAULT_DEVELOPMENT = EVIDENCE / "development_v17_r0.csv"
DEFAULT_OUTPUT = EVIDENCE / "diagnosis_v17_r0_gate.json"


def _summary(prior: np.ndarray, projected: np.ndarray) -> dict:
    if prior.shape != projected.shape or prior.ndim != 1 or len(prior) == 0:
        raise ValueError("paired errors must be non-empty aligned vectors")
    delta = projected - prior
    return {
        "units": int(len(delta)),
        "prior_mae_m": float(prior.mean()),
        "projected_mae_m": float(projected.mean()),
        "projected_minus_prior_mean_m": float(delta.mean()),
        "projected_minus_prior_median_m": float(np.median(delta)),
        "projected_minus_prior_q10_m": float(np.quantile(delta, 0.10)),
        "projected_minus_prior_q90_m": float(np.quantile(delta, 0.90)),
        "fraction_improved": float(np.mean(delta < 0.0)),
        "fraction_unchanged": float(np.mean(delta == 0.0)),
        "fraction_worsened": float(np.mean(delta > 0.0)),
    }


def _coordinate_vectors(
    prior: np.ndarray,
    projected: np.ndarray,
    keys: Sequence[tuple],
) -> tuple[np.ndarray, np.ndarray, list[tuple], np.ndarray]:
    grouped: dict[tuple, list[int]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(tuple(key), []).append(index)
    ordered = sorted(grouped)
    prior_group = []
    projected_group = []
    group_sizes = []
    for key in ordered:
        indices = np.asarray(grouped[key], dtype=np.int64)
        prior_group.append(float(prior[indices].mean()))
        projected_group.append(float(projected[indices].mean()))
        group_sizes.append(int(len(indices)))
    return (
        np.asarray(prior_group, dtype=np.float64),
        np.asarray(projected_group, dtype=np.float64),
        ordered,
        np.asarray(group_sizes, dtype=np.int64),
    )


def _category_summaries(
    prior: np.ndarray,
    projected: np.ndarray,
    categories: Sequence[str],
) -> dict[str, dict]:
    values = np.asarray(categories, dtype=object)
    result = {}
    for category in sorted(set(values.tolist())):
        mask = values == category
        result[str(category)] = _summary(prior[mask], projected[mask])
    return result


def _interval_labels(values: np.ndarray, edges: list[float]) -> list[str]:
    result = []
    for value in values:
        index = int(np.searchsorted(edges, float(value), side="right"))
        lower = "-inf" if index == 0 else f"{edges[index - 1]:.6g}"
        upper = "inf" if index == len(edges) else f"{edges[index]:.6g}"
        result.append(f"[{lower},{upper})")
    return result


def diagnose(
    *,
    result_path: str | Path = DEFAULT_RESULT,
    development_path: str | Path = DEFAULT_DEVELOPMENT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict:
    result_path = Path(result_path).resolve()
    development_path = Path(development_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    wrapper = json.loads(result_path.read_text(encoding="utf-8"))
    if wrapper.get("result_sha256") != _canonical_json_sha256(
        wrapper, omit="result_sha256"
    ):
        raise ValueError("V17 R0 result self-hash failed")
    if wrapper.get("query_role") != "development":
        raise ValueError("diagnosis accepts only the R0 development result")
    if wrapper.get("outer_test_predictions") != 0:
        raise ValueError("outer-test access is outside this diagnosis")
    if wrapper.get("official_validation_predictions") != 0:
        raise ValueError("official validation access is outside this diagnosis")
    if wrapper["reconstruction_gate"].get("passed") is not False:
        raise ValueError("this diagnosis is reserved for the failed R0 gate")

    development = load_uji_v16(development_path)
    legacy = wrapper["legacy_evaluation_payload"]
    prior = np.asarray(
        legacy["errors_m"]["baseline_consensus"], dtype=np.float64
    )
    projected = np.asarray(
        legacy["errors_m"]["projected_consensus"], dtype=np.float64
    )
    if len(prior) != len(development["keys"]):
        raise ValueError("R0 error vector and development rows differ")
    prior_group, projected_group, group_keys, group_sizes = _coordinate_vectors(
        prior, projected, development["keys"]
    )

    row_strata = [
        f"B{int(building)}F{int(floor)}"
        for building, floor in zip(
            development["building"], development["floor"]
        )
    ]
    group_strata = [f"B{int(key[0])}F{int(key[1])}" for key in group_keys]
    phones = [f"phone_{int(value)}" for value in development["phone"]]
    visible = (development["rss"].numpy() > -0.999).sum(axis=1).astype(np.int64)
    visibility_edges = [10.0, 14.0, 17.0, 21.0, 28.0]
    difficulty_edges = np.quantile(prior, [0.25, 0.50, 0.75]).tolist()
    group_size_edges = [2.0, 5.0, 10.0, 20.0, 40.0]
    delta = projected - prior

    strata_coordinate = _category_summaries(
        prior_group, projected_group, group_strata
    )
    phone_row = _category_summaries(prior, projected, phones)
    visibility_row = _category_summaries(
        prior,
        projected,
        _interval_labels(visible.astype(np.float64), visibility_edges),
    )
    difficulty_row = _category_summaries(
        prior,
        projected,
        _interval_labels(prior, difficulty_edges),
    )
    group_size_coordinate = _category_summaries(
        prior_group,
        projected_group,
        _interval_labels(group_sizes.astype(np.float64), group_size_edges),
    )

    payload = {
        "status": "posthoc_v17_r0_gate_diagnosis",
        "interpretation_authority": "descriptive_only_not_model_selection",
        "created_after_development_outcome": True,
        "may_modify_frozen_r0": False,
        "outer_test_predictions": 0,
        "official_validation_predictions": 0,
        "bindings": {
            "result": {
                "path": str(result_path),
                "file_sha256": _sha256_file(result_path),
                "result_sha256": wrapper["result_sha256"],
            },
            "development": {
                "path": str(development_path),
                "sha256": _sha256_file(development_path),
            },
            "diagnostic_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(__file__),
            },
        },
        "overall_row": _summary(prior, projected),
        "overall_coordinate_macro": _summary(prior_group, projected_group),
        "coordinate_group_size": {
            "minimum": int(group_sizes.min()),
            "median": float(np.median(group_sizes)),
            "maximum": int(group_sizes.max()),
        },
        "association_diagnostics": {
            "pearson_delta_vs_prior_error": float(
                np.corrcoef(delta, prior)[0, 1]
            ),
            "pearson_delta_vs_visible_ap_count": float(
                np.corrcoef(delta, visible.astype(np.float64))[0, 1]
            ),
        },
        "by_building_floor_coordinate_macro": strata_coordinate,
        "by_phone_row": phone_row,
        "by_visible_ap_count_row": {
            "edges": visibility_edges,
            "categories": visibility_row,
        },
        "by_prior_error_quartile_row": {
            "edges_m": difficulty_edges,
            "categories": difficulty_row,
        },
        "by_coordinate_scan_count": {
            "edges": group_size_edges,
            "categories": group_size_coordinate,
        },
        "scope_limit": (
            "Error-only diagnostics can localize where projection harms, but "
            "cannot identify affine profiling versus marked likelihood as the "
            "cause without a separately registered matched backend experiment."
        ),
    }
    payload["diagnostic_sha256"] = _canonical_json_sha256(payload)
    _write_json_new(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument("--development", default=str(DEFAULT_DEVELOPMENT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = diagnose(
        result_path=args.result,
        development_path=args.development,
        output_path=args.output,
    )
    concise = {
        "status": result["status"],
        "diagnostic_sha256": result["diagnostic_sha256"],
        "overall_row": result["overall_row"],
        "overall_coordinate_macro": result["overall_coordinate_macro"],
        "association_diagnostics": result["association_diagnostics"],
        "output": str(Path(args.output).resolve()),
    }
    print(json.dumps(concise, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
