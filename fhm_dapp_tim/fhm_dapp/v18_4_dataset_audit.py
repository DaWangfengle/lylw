"""Read-only dataset audit for the V18.4 multi-dataset protocol.

The audit never rewrites source files.  It records file hashes, CSV shapes and
MATLAB top-level structures so adapters are implemented against the actual
repository inputs rather than assumptions from dataset documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.dtype.names:
        out["field_names"] = list(array.dtype.names)
    if np.issubdtype(array.dtype, np.number) and array.size:
        flat = np.asarray(array).reshape(-1)
        finite = flat[np.isfinite(flat)]
        out["finite_count"] = int(finite.size)
        if finite.size:
            out["minimum"] = float(finite.min())
            out["maximum"] = float(finite.max())
    elif array.dtype.kind in {"U", "S"} and array.size:
        values = np.asarray(array).reshape(-1)
        out["sample_values"] = [str(v) for v in values[:5]]
    elif array.dtype == object and array.size:
        samples = []
        for item in array.reshape(-1)[:5]:
            if isinstance(item, np.ndarray):
                samples.append(
                    {
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    }
                )
            else:
                samples.append(str(type(item).__name__))
        out["object_samples"] = samples
    return out


def audit_mat(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=False, struct_as_record=False)
    variables = {}
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            variables[key] = summarize_array(value)
        else:
            variables[key] = {"python_type": type(value).__name__}
    return {
        "format": "matlab",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "variables": variables,
    }


def _csv_first_nonempty_line(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip():
                return line.rstrip("\r\n")
    return ""


def audit_csv(path: Path) -> dict[str, Any]:
    first = _csv_first_nonempty_line(path)
    first_fields = first.split(",") if first else []
    looks_header = any(
        any(ch.isalpha() for ch in field)
        for field in first_fields
    )
    frame = pd.read_csv(
        path,
        header=0 if looks_header else None,
        low_memory=False,
    )
    report: dict[str, Any] = {
        "format": "csv",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "header_detected": bool(looks_header),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_head": [str(c) for c in frame.columns[:12]],
        "column_tail": [str(c) for c in frame.columns[-12:]],
    }
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    finite = values[np.isfinite(values)]
    report["numeric_fraction"] = float(np.isfinite(values).mean())
    if finite.size:
        report["numeric_minimum"] = float(finite.min())
        report["numeric_maximum"] = float(finite.max())
    if not looks_header and frame.shape[1] <= 8:
        report["first_rows"] = [
            [_scalar(v) for v in row]
            for row in frame.head(5).to_numpy().tolist()
        ]
    return report


def audit_tree(data_root: Path) -> dict[str, Any]:
    datasets = {}
    for directory in sorted(p for p in data_root.iterdir() if p.is_dir()):
        files = {}
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            suffix = path.suffix.lower()
            if suffix == ".csv":
                files[path.name] = audit_csv(path)
            elif suffix == ".mat":
                files[path.name] = audit_mat(path)
            else:
                files[path.name] = {
                    "format": suffix.lstrip(".") or "unknown",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        datasets[directory.name] = {"files": files}
    return {
        "audit_version": "v18_4_multidataset_audit_v1",
        "data_root": str(data_root),
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_tree(Path(args.data_root))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
