"""Complete V17-R0 evaluation under its exact-coordinate contract.

R0 was trained from exact float64 CSV coordinates before centering.  The
historical ``ProjectedEnsemble`` constructor instead reconstructs coordinates
through a normalized float32 round trip.  On UJI's large northing values that
round trip is not identity preserving, so the constructor correctly rejects
the reconstructed radio memory.  This adapter supplies the exact registered
CSV coordinates for the two authorized roles and retains A1's metadata type
normalization.  No model, checkpoint, role, or hyperparameter is changed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from fhm_dapp_tim.fhm_dapp import projected_ensemble as projected_module
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as registered_r0
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
)
from fhm_dapp_tim.fhm_dapp.uji_v17_mother_r0_runtime_a1 import (
    _ORIGINAL_LOADER,
    _load_anchor_checkpoint_numpy_origin,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = (
    ROOT
    / "evidence"
    / "v17_acrhap_mother"
    / "amendment_v17_r0_a2_coordinate_contract.json"
)
ADAPTER_PROTOCOL = "uji_v17_r0_runtime_a2_exact_coordinate_v1_20260821"
_ORIGINAL_RAW_COORDS = projected_module.raw_coords


def _load_amendment(path: str | Path) -> dict:
    amendment_path = Path(path).resolve()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    if payload.get("amendment_sha256") != _canonical_json_sha256(
        payload, omit="amendment_sha256"
    ):
        raise ValueError("V17-R0-A2 amendment self-hash failed")
    if payload.get("adapter_protocol") != ADAPTER_PROTOCOL:
        raise ValueError("V17-R0-A2 adapter protocol changed")
    if payload.get("scientific_change") is not False:
        raise ValueError("runtime amendment must remain scientifically neutral")
    for field in (
        "adapter_source",
        "a1_adapter_source",
        "a1_amendment",
        "registered_execution_source",
        "registered_ensemble_source",
    ):
        record = payload["bindings"][field]
        if _sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"V17-R0-A2 binding changed: {field}")
    adapter_record = payload["bindings"]["adapter_source"]
    if Path(adapter_record["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("V17-R0-A2 adapter path changed")
    registry_record = payload["bindings"]["registry"]
    if _sha256_file(registry_record["path"]) != registry_record["file_sha256"]:
        raise ValueError("V17-R0 registry file hash changed")
    registry = registered_r0.load_registry(registry_record["path"])
    if registry["registry_sha256"] != registry_record["registry_sha256"]:
        raise ValueError("V17-R0 registry self-hash changed")
    return payload


def _role_record(path: str | Path) -> dict:
    data = registered_r0.load_uji_v16(path)
    return {
        "path": str(Path(path).resolve()),
        "raw_xy": np.asarray(data["raw_xy"], dtype=np.float64),
        "building": np.asarray(data["building"], dtype=np.int64),
        "floor": np.asarray(data["floor"], dtype=np.int64),
    }


def _exact_raw_coordinate_adapter(role_records: list[dict]):
    by_length: dict[int, dict] = {}
    for record in role_records:
        count = int(len(record["raw_xy"]))
        if count in by_length:
            raise ValueError("authorized coordinate roles have ambiguous row counts")
        by_length[count] = record

    def exact_raw_coords(coords_norm, building, stats):
        count = int(len(coords_norm))
        if count not in by_length:
            raise ValueError(
                f"raw coordinate request has unauthorized row count: {count}"
            )
        record = by_length[count]
        observed_building = building.detach().cpu().numpy().astype(np.int64)
        if not np.array_equal(observed_building, record["building"]):
            raise ValueError("raw coordinate request does not match registered role")
        reconstructed = _ORIGINAL_RAW_COORDS(coords_norm, building, stats)
        discrepancy = np.linalg.norm(
            reconstructed - record["raw_xy"], axis=1
        )
        if not np.isfinite(discrepancy).all() or float(discrepancy.max()) > 1.0:
            raise ValueError("normalized coordinates do not identify registered role")
        return record["raw_xy"].copy()

    return exact_raw_coords


@contextlib.contextmanager
def _installed_adapters(role_records: list[dict]) -> Iterator[None]:
    current_loader = projected_module.load_anchor_checkpoint
    current_raw_coords = projected_module.raw_coords
    if current_loader is not _ORIGINAL_LOADER:
        raise RuntimeError("ProjectedEnsemble checkpoint loader was already changed")
    if current_raw_coords is not _ORIGINAL_RAW_COORDS:
        raise RuntimeError("ProjectedEnsemble raw_coords was already changed")
    projected_module.load_anchor_checkpoint = _load_anchor_checkpoint_numpy_origin
    projected_module.raw_coords = _exact_raw_coordinate_adapter(role_records)
    try:
        yield
    finally:
        projected_module.raw_coords = current_raw_coords
        projected_module.load_anchor_checkpoint = current_loader


def check_contract(amendment_path: str | Path = DEFAULT_AMENDMENT) -> dict:
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registry"]["path"]
    registry = registered_r0.load_registry(registry_path)
    destination = Path(registry["allowed_outputs"]["development_result"])
    if destination.exists():
        raise FileExistsError("development result already exists")
    checkpoints = [
        Path(value) for value in registry["allowed_outputs"]["r0_checkpoints"]
    ]
    selection_csv = Path(registry["allowed_outputs"]["selection_refit_csv"])
    selection = _role_record(selection_csv)

    # Quantify the serialization issue without accessing development.
    from fhm_dapp_tim.fhm_dapp.data import load_dataset

    loaded = load_dataset(
        selection_csv,
        dataset_type="UJI",
        reference_stats=None,
    )
    reconstructed = _ORIGINAL_RAW_COORDS(loaded[1], loaded[2], loaded[-1])
    roundtrip_error = np.linalg.norm(
        reconstructed - selection["raw_xy"], axis=1
    )

    with _installed_adapters([selection]):
        ensemble = registered_r0.ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device="cpu",
            dataset="UJI",
        )
    anchor_count = int(ensemble.checkpoints[0][0].n_anchors)
    if anchor_count != 595:
        raise ValueError("ProjectedEnsemble anchor count changed")
    if not isinstance(ensemble.reference["coord_origin"], np.ndarray):
        raise TypeError("ProjectedEnsemble retained a non-NumPy coord_origin")
    return {
        "adapter_protocol": ADAPTER_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "checkpoint_count": len(checkpoints),
        "selection_rows": int(len(selection["raw_xy"])),
        "selection_coordinates": anchor_count,
        "float32_roundtrip_max_error_m": float(roundtrip_error.max()),
        "float32_roundtrip_mean_error_m": float(roundtrip_error.mean()),
        "projected_ensemble_constructor_passed": True,
        "exact_coordinate_values_preserved": True,
        "development_result_absent": True,
        "development_predictions_performed": 0,
    }


def evaluate(
    *,
    amendment_path: str | Path = DEFAULT_AMENDMENT,
    device_name: str = "cuda",
) -> dict:
    check_contract(amendment_path)
    amendment = _load_amendment(amendment_path)
    registry_path = amendment["bindings"]["registry"]["path"]
    registry = registered_r0.load_registry(registry_path)
    selection = _role_record(registry["allowed_outputs"]["selection_refit_csv"])
    development = _role_record(registry["allowed_outputs"]["development_csv"])
    with _installed_adapters([selection, development]):
        return registered_r0.evaluate_r0(
            registry_path=registry_path,
            device_name=device_name,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check-contract")
    check_parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--amendment", default=str(DEFAULT_AMENDMENT))
    evaluate_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "check-contract":
        result = check_contract(args.amendment)
    elif args.command == "evaluate":
        result = evaluate(
            amendment_path=args.amendment,
            device_name=args.device,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
