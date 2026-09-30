"""Read-only V18 verification; the smoke check uses synthetic data only."""
from __future__ import annotations
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from fhm_dapp_tim.fhm_dapp import uji_v18_experiment as exp
from fhm_dapp_tim.fhm_dapp.anchor import build_anchor_memory
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp.v18_single import V18_CONFIG, build_v18_model, v18_training_loss

def finite_tree(value, path="root"):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), path
    elif isinstance(value, float):
        assert math.isfinite(value), path
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")

def smoke():
    exp._set_seed(exp.MODEL_SEED)
    device = torch.device("cuda")
    coords = np.array([[b * 20.0 + x * 2.0, y * 2.0]
                       for b in range(2) for f in range(2)
                       for x in range(4) for y in range(4)], dtype=np.float32)
    building = np.repeat(np.arange(2), 32)
    floor = np.tile(np.repeat(np.arange(2), 16), 2)
    rss = torch.rand(64, 80) * 1.2 - 0.8
    rss[torch.rand(64, 80) < 0.45] = -1.0
    rss[:, 0] = -0.25
    memory, sample_anchor = build_anchor_memory(rss, coords, building, floor, np.arange(64))
    model, geometry, neighbours, weights = build_v18_model(
        memory=memory, n_aps=80, n_buildings=2, n_devices=2,
        device=device, config=V18_CONFIG)
    batch = torch.arange(8)
    terms = v18_training_loss(
        model, rss[batch].to(device), torch.from_numpy(coords[batch]).to(device),
        torch.from_numpy(sample_anchor[batch]).to(device), (batch % 2).to(device),
        neighbours, weights)
    finite_tree(terms)
    terms["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            finite_tree(parameter.grad, name)
    model.eval()
    with torch.no_grad():
        first = model.predict(rss[batch].to(device), top_k=64,
                              target_coords=torch.from_numpy(coords[batch]).to(device),
                              deterministic_building=True)
    finite_tree(first)
    assert torch.allclose(first["candidate_weights"].sum(1), torch.ones(8, device=device), atol=1e-6)
    assert model.residual_experts == 1
    payload = exp._checkpoint_payload(
        registry={"registry_sha256": "synthetic-only"}, phase="selection",
        refit_role="selection_refit", selected_epoch=1, model=model,
        memory={**memory, "stats": {"n_aps": 80}}, origin=np.zeros(2, dtype=np.float32),
        device_mapping={0: 0, 1: 1}, selection_history=[], refit_history=[])
    with tempfile.TemporaryDirectory(prefix="v18-synthetic-") as temporary:
        checkpoint = Path(temporary) / "synthetic.pt"
        exp._save_torch_new(checkpoint, payload)
        restored, restored_payload = exp.load_v18_checkpoint(
            checkpoint, registry={"registry_sha256": "synthetic-only"},
            device=device, expected_phase="selection")
        with torch.no_grad():
            second = restored.predict(rss[batch].to(device), top_k=64,
                                      target_coords=torch.from_numpy(coords[batch]).to(device),
                                      deterministic_building=True)
        for name in first:
            if isinstance(first[name], torch.Tensor):
                assert torch.equal(first[name], second[name]), name
        assert restored_payload["config"] == V18_CONFIG
    return {"synthetic_only": True, "cuda_forward_backward_finite": True,
            "checkpoint_reconstruction_exact": True, "anchors": 64,
            "geometry": geometry["diagnostics"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}

def registry_check():
    registry = exp.load_registry()
    roles = {name: exp._registered_role_keys(registry, name) for name in registry["roles"]}
    assert all(roles.values())
    assert roles["member_fit"].isdisjoint(roles["early_stop"])
    assert roles["member_fit"] | roles["early_stop"] == roles["selection_refit"]
    assert roles["selection_refit"].isdisjoint(roles["development"])
    assert roles["selection_refit"] | roles["development"] == roles["outer_refit"]
    assert roles["outer_refit"].isdisjoint(roles["outer_test"])
    all_keys = roles["outer_refit"] | roles["outer_test"]
    assert len(all_keys) == registry["training_coordinate_count"]
    assert exp._sha256_keys(all_keys) == registry["training_coordinate_sha256"]
    diagnostics = exp.audit_coordinate_roles(roles, {"keys": sorted(all_keys)})
    assert registry["access_contract"]["official_validation_in_scope"] is False
    assert registry["single_member_contract"]["checkpoint_count_at_inference"] == 1
    assert registry["single_member_contract"]["residual_experts"] == 1
    return {"registry_sha256": registry["registry_sha256"], "self_hash_valid": True,
            "training_rows": registry["training_rows"],
            "training_coordinate_count": registry["training_coordinate_count"],
            "roles": {name: {"coordinates": len(keys), "rows": registry["roles"][name]["row_count"]}
                      for name, keys in roles.items()}, "diagnostics": diagnostics}

def checkpoint_check(phase):
    registry = exp.load_registry()
    path = exp.DEFAULT_SELECTION_CHECKPOINT if phase == "selection" else exp.DEFAULT_FINAL_CHECKPOINT
    model, payload = exp.load_v18_checkpoint(path, registry=registry,
                                           device=torch.device("cpu"), expected_phase=phase)
    finite_tree(payload)
    assert payload["single_member_contract"]["checkpoint_count"] == 1
    assert payload["config"] == V18_CONFIG
    assert model.residual_experts == 1
    assert payload["config"]["scan_feature_mode"] == "bounded_maximal_invariant"
    assert payload["config"]["prototype_metric"] == "none"
    assert payload["config"]["candidate_reranker"] is True
    geometry = build_delaunay_geometry(model.anchor_coords, model.anchor_buildings, model.anchor_floors)
    transform = model.local_residual_transform
    assert torch.equal(transform, geometry["residual_transform"])
    assert torch.allclose(transform, transform.transpose(1, 2), atol=1e-5)
    eigenvalues = torch.linalg.eigvalsh(transform)
    assert float(eigenvalues.min()) >= -1e-5
    if phase == "final":
        freeze = exp.load_freeze(registry=registry)
        assert payload["selected_epoch"] == freeze["frozen_selected_epoch"]
        assert payload["freeze_sha256"] == freeze["freeze_sha256"]
    history = payload.get("selection_history") or []
    return {"phase": phase, "checkpoint_sha256": exp._checkpoint_sha(path),
            "selected_epoch": payload["selected_epoch"], "anchors": model.n_anchors,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": payload["training_seconds"],
            "best_early_stop_mixture_nll": min((r["early_stop"]["coordinate_macro_mixture_nll"]
                                               for r in history), default=None),
            "selection_epochs_run": len(history), "refit_epochs": len(payload["refit_history"]),
            "all_checkpoint_tensors_and_history_finite": True,
            "single_member_contract": payload["single_member_contract"], "config": payload["config"],
            "delaunay_transform_reconstruction_exact": True,
            "delaunay_min_eigenvalue": float(eigenvalues.min()),
            "geometry": geometry["diagnostics"],
            "first_refit_loss": payload["refit_history"][0]["loss"],
            "last_refit_loss": payload["refit_history"][-1]["loss"]}

def freeze_check():
    registry = exp.load_registry()
    freeze = exp.load_freeze(registry=registry)
    selection = torch.load(exp.DEFAULT_SELECTION_CHECKPOINT, map_location="cpu", weights_only=True)
    result = json.loads(exp.DEFAULT_DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    assert freeze["frozen_selected_epoch"] == selection["selected_epoch"]
    assert freeze["selection_checkpoint_sha256"] == exp._checkpoint_sha(exp.DEFAULT_SELECTION_CHECKPOINT)
    assert result["result_sha256"] == exp._canonical_json_sha256(result, omit="result_sha256")
    assert freeze["development_result_sha256"] == result["result_sha256"]
    finite_tree(result)
    return {"freeze_sha256": freeze["freeze_sha256"], "self_hash_valid": True,
            "frozen_selected_epoch": freeze["frozen_selected_epoch"],
            "config_matches_source": True, "checkpoint_hash_matches": True,
            "outer_test_access_authorized": freeze["outer_test_access_authorized"],
            "development_metrics": result["development_metrics"]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "registry", "selection", "freeze", "final"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actions = {"smoke": smoke, "registry": registry_check, "freeze": freeze_check,
               "selection": lambda: checkpoint_check("selection"),
               "final": lambda: checkpoint_check("final")}
    report = actions[args.stage]()
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
