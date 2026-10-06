"""Canonical V18.5 method-contract manifest.

This module is deliberately dependency-light so both the outcome-free
acceptance runner and the formal experiment drivers can compute the exact same
method hash without circular imports.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

METHOD_CONTRACT_PATHS: tuple[str, ...] = (
    # Shared numerical/model/data primitives that V18.5 directly imports.
    "fhm_dapp_tim/fhm_dapp/model.py",
    "fhm_dapp_tim/fhm_dapp/dapp.py",
    "fhm_dapp_tim/fhm_dapp/coherent_marked_v9.py",
    "fhm_dapp_tim/fhm_dapp/v18_single.py",
    "fhm_dapp_tim/fhm_dapp/v18_4_datasets.py",
    "fhm_dapp_tim/fhm_dapp/v18_4_protocol.py",
    # V18.5 method/protocol implementation.
    "fhm_dapp_tim/fhm_dapp/v18_5_contract.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_core.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_datasets.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_radio.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_model.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_protocol.py",
    "fhm_dapp_tim/fhm_dapp/uji_v18_5_development.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_external.py",
    "fhm_dapp_tim/evidence/v18_5/DESIGN_V18_5.md",
    "fhm_dapp_tim/evidence/v18_5/RUNBOOK_UJI_V18_5.md",
    "fhm_dapp_tim/evidence/v18_5/RUNBOOK_EXTERNAL_V18_5.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_manifest(
    paths: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(relative)] = sha256_file(path)
    return result


def method_contract(
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    files = hash_manifest(METHOD_CONTRACT_PATHS, repo_root=repo_root)
    return {
        "files": files,
        "contract_sha256": canonical_sha(files),
    }


__all__ = [
    "METHOD_CONTRACT_PATHS",
    "REPO_ROOT",
    "canonical_sha",
    "hash_manifest",
    "method_contract",
    "sha256_file",
]
