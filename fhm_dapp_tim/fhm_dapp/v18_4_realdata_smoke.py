"""Real-data, outcome-free V18.4 UJI smoke test.

This script reads only UJI trainingData.csv, constructs the registered V18.4
training roles, builds an anchor model from the fit role, and executes one
forward/backward training batch plus one base-posterior development batch.

It never opens UJI validationData.csv or any reserved external test.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.v18_4_datasets import load_uji
from fhm_dapp_tim.fhm_dapp.v18_4_model import (
    build_v184_model,
    training_spatial_scale_m,
    v184_predict,
    v184_training_loss,
)
from fhm_dapp_tim.fhm_dapp.v18_4_protocol import (
    ROLE_SEEDS,
    construct_roles,
    row_indices_for_roles,
)


SMOKE_SEED = 184499


def _set_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED)
    torch.manual_seed(SMOKE_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)


def main() -> None:
    _set_deterministic()
    data, _reference = load_uji("train")
    roles = construct_roles(
        data,
        seed=ROLE_SEEDS["UJIIndoorLoc"],
    )
    rows = row_indices_for_roles(data, roles)
    fit_idx = rows["fit"]
    dev_idx = rows["development"]

    rss = torch.from_numpy(data.model_rss()).float()
    raw_coords = data.coords_m.astype(np.float32)
    origin = raw_coords[fit_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = (raw_coords - origin[None, :]).astype(np.float32)

    memory, sample_anchor = build_anchor_memory(
        rss,
        centered,
        data.building,
        data.floor,
        fit_idx,
    )
    spatial_scale = training_spatial_scale_m(memory)
    n_devices = (
        int(data.device[data.device >= 0].max()) + 1
        if bool(np.any(data.device >= 0))
        else 0
    )
    n_buildings = int(memory["building"].max().item()) + 1
    model, _geometry, neighbours, weights = build_v184_model(
        memory=memory,
        n_aps=rss.shape[1],
        n_buildings=n_buildings,
        n_devices=n_devices,
        spatial_scale_m=spatial_scale,
        device=torch.device("cpu"),
    )
    model.train()

    train_rows = fit_idx[:8]
    target_anchor = torch.from_numpy(sample_anchor[train_rows]).long()
    if bool((target_anchor < 0).any()):
        raise RuntimeError("fit row has no registered anchor")
    device_id = torch.from_numpy(data.device[train_rows]).long()
    target = torch.from_numpy(centered[train_rows]).float()
    result = v184_training_loss(
        model,
        rss[train_rows],
        target,
        target_anchor,
        device_id,
        neighbours,
        weights,
        lambda_ot=0.25,
    )
    if not bool(torch.isfinite(result["loss"])):
        raise RuntimeError("real-data V18.4 loss is non-finite")
    result["loss"].backward()

    grad_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not grad_tensors or not all(
        bool(torch.isfinite(value).all()) for value in grad_tensors
    ):
        raise RuntimeError("real-data V18.4 backward is not finite")

    model.eval()
    with torch.no_grad():
        prediction = v184_predict(
            model,
            rss[dev_idx[:4]],
        )
    if not bool(torch.isfinite(prediction["coords"]).all()):
        raise RuntimeError("real-data V18.4 prediction is non-finite")

    payload = {
        "status": "success",
        "scientific_status": (
            "outcome-free smoke test using UJI training fit/development roles only"
        ),
        "official_validation_opened": False,
        "reserved_external_test_opened": False,
        "fit_rows": int(len(fit_idx)),
        "development_rows": int(len(dev_idx)),
        "fit_coordinate_groups": int(len(roles.fit)),
        "development_coordinate_groups": int(len(roles.development)),
        "anchors": int(len(memory["coords"])),
        "aps": int(rss.shape[1]),
        "spatial_scale_m": float(spatial_scale),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training_batch_rows": int(len(train_rows)),
        "loss": float(result["loss"].detach()),
        "anchor_nll": float(result["anchor_nll"]),
        "coordinate_nll": float(result["coordinate_nll"]),
        "spatial_ot_loss": float(result["spatial_ot_loss"]),
        "natural_recall_mass": float(result["natural_recall_mass"]),
        "natural_support_size": float(result["natural_support_size"]),
        "forced_support_size": float(result["forced_support_size"]),
        "development_support_sizes": prediction[
            "natural_support_size"
        ].cpu().tolist(),
        "gradients_finite": True,
        "prediction_finite": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
