"""Canonical execution-environment fingerprint for V18.5 provenance.

The environment is recorded as reproducibility evidence.  Hardware/software
identity is not a model hyperparameter and is not used for method selection.
"""

from __future__ import annotations

import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import torch


def environment_fingerprint(
    *,
    model_seed: int,
    scoring_seed: int,
    predictive_draws: int,
) -> dict[str, object]:
    """Return a deterministic JSON-serializable environment description."""

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, object]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({
                "index": int(index),
                "name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
            })

    sdp: dict[str, bool | None] = {}
    for name, attribute in (
        ("flash_enabled", "flash_sdp_enabled"),
        ("memory_efficient_enabled", "mem_efficient_sdp_enabled"),
        ("math_enabled", "math_sdp_enabled"),
    ):
        function = getattr(torch.backends.cuda, attribute, None)
        sdp[name] = bool(function()) if callable(function) else None

    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "torch": {
            "default_dtype": str(torch.get_default_dtype()),
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "cuda_device_count": (
                int(torch.cuda.device_count()) if cuda_available else 0
            ),
            "cuda_devices": devices,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_available": bool(torch.backends.cudnn.is_available()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "scaled_dot_product_attention": sdp,
        },
        "determinism": {
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "model_seed": int(model_seed),
            "scoring_seed": int(scoring_seed),
            "predictive_draws": int(predictive_draws),
        },
    }


__all__ = ["environment_fingerprint"]
