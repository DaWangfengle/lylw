"""Run the V17-R0 development evaluation with a frozen type adapter.

The registered checkpoints are restricted ``torch.save`` payloads.  Their
NumPy ``coord_origin`` values are intentionally converted to CPU tensors by
``_safe_tree`` at serialization time, while ``ProjectedEnsemble`` performs a
NumPy subtraction at construction time.  This module converts only that
in-memory metadata field back to a finite float32 NumPy vector.  It does not
change checkpoint bytes, model state, data roles, predictions, or any
scientific setting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import projected_ensemble as projected_module
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r0 as registered_r0
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256,
    _sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMENDMENT = (
    ROOT
    / "evidence"
    / "v17_acrhap_mother"
    / "amendment_v17_r0_a1_coord_origin.json"
)
ADAPTER_PROTOCOL = "uji_v17_r0_runtime_a1_coord_origin_v1_20260821"
_ORIGINAL_LOADER = projected_module.load_anchor_checkpoint


def _load_anchor_checkpoint_numpy_origin(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Return the registered checkpoint with NumPy coordinate metadata."""

    model, raw_checkpoint = _ORIGINAL_LOADER(path, device)
    checkpoint = dict(raw_checkpoint)
    origin = checkpoint.get("coord_origin")
    if isinstance(origin, torch.Tensor):
        origin = origin.detach().cpu().numpy()
    origin_array = np.asarray(origin, dtype=np.float32)
    if origin_array.shape != (2,) or not np.isfinite(origin_array).all():
        raise ValueError("coord_origin must be one finite two-vector")
    checkpoint["coord_origin"] = origin_array
    return model, checkpoint


@contextlib.contextmanager
def _installed_adapter() -> Iterator[None]:
    current = projected_module.load_anchor_checkpoint
    if current is not _ORIGINAL_LOADER:
        raise RuntimeError("ProjectedEnsemble checkpoint loader was already changed")
    projected_module.load_anchor_checkpoint = _load_anchor_checkpoint_numpy_origin
    try:
        yield
    finally:
        projected_module.load_anchor_checkpoint = current


def _load_amendment(path: str | Path) -> dict:
    amendment_path = Path(path).resolve()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    expected_self_hash = _canonical_json_sha256(
        payload, omit="amendment_sha256"
    )
    if payload.get("amendment_sha256") != expected_self_hash:
        raise ValueError("V17-R0-A1 amendment self-hash failed")
    if payload.get("adapter_protocol") != ADAPTER_PROTOCOL:
        raise ValueError("V17-R0-A1 adapter protocol changed")
    script_record = payload["bindings"]["adapter_source"]
    if Path(script_record["path"]).resolve() != Path(__file__).resolve():
        raise ValueError("V17-R0-A1 adapter path changed")
    if _sha256_file(__file__) != script_record["sha256"]:
        raise ValueError("V17-R0-A1 adapter source hash changed")
    registry_path = Path(payload["bindings"]["registry"]["path"])
    if _sha256_file(registry_path) != payload["bindings"]["registry"]["file_sha256"]:
        raise ValueError("V17-R0 registry file hash changed")
    registry = registered_r0.load_registry(registry_path)
    if registry["registry_sha256"] != payload["bindings"]["registry"]["registry_sha256"]:
        raise ValueError("V17-R0 registry self-hash changed")
    if payload.get("scientific_change") is not False:
        raise ValueError("runtime amendment must remain scientifically neutral")
    return payload


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
    original_types = []
    adapted_types = []
    values_preserved = []
    for checkpoint_path in checkpoints:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_origin = raw["coord_origin"]
        original_types.append(type(raw_origin).__name__)
        _model, adapted = _load_anchor_checkpoint_numpy_origin(
            checkpoint_path, torch.device("cpu")
        )
        adapted_origin = adapted["coord_origin"]
        adapted_types.append(type(adapted_origin).__name__)
        raw_array = (
            raw_origin.detach().cpu().numpy()
            if isinstance(raw_origin, torch.Tensor)
            else np.asarray(raw_origin)
        )
        values_preserved.append(
            bool(np.array_equal(raw_array.astype(np.float32), adapted_origin))
        )

    if original_types != ["Tensor"] * len(checkpoints):
        raise ValueError("registered checkpoint coordinate type changed")
    if adapted_types != ["ndarray"] * len(checkpoints):
        raise ValueError("runtime coordinate adapter failed")
    if not all(values_preserved):
        raise ValueError("runtime coordinate adapter changed numeric values")

    # Exercise the exact failing constructor using selection_refit only.  This
    # is a contract check, not a development prediction.
    selection_csv = Path(registry["allowed_outputs"]["selection_refit_csv"])
    with _installed_adapter():
        ensemble = registered_r0.ProjectedEnsemble(
            checkpoints=checkpoints,
            anchor_data=selection_csv,
            device="cpu",
            dataset="UJI",
        )
    if not isinstance(ensemble.reference["coord_origin"], np.ndarray):
        raise TypeError("ProjectedEnsemble retained a non-NumPy coord_origin")
    if int(ensemble.checkpoints[0][0].n_anchors) != 595:
        raise ValueError("ProjectedEnsemble anchor count changed")
    return {
        "adapter_protocol": ADAPTER_PROTOCOL,
        "amendment_sha256": amendment["amendment_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "checkpoint_count": len(checkpoints),
        "original_coord_origin_types": original_types,
        "adapted_coord_origin_types": adapted_types,
        "numeric_values_preserved": all(values_preserved),
        "projected_ensemble_constructor_passed": True,
        "projected_ensemble_anchor_count": 595,
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
    with _installed_adapter():
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
    else:  # pragma: no cover - argparse prevents this branch.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
