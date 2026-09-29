"""Train frozen anchor checkpoints for an added same-loader dataset.

This is intentionally narrow: it produces checkpoints compatible with
ProjectedEnsemble without changing the DAPP operating point selected on UJI.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fhm_dapp_tim.fhm_dapp.anchor import (
    build_anchor_memory,
    raw_coords,
    resolve_path,
)
from fhm_dapp_tim.fhm_dapp.data import load_dataset
from fhm_dapp_tim.fhm_dapp.metrics import coordinate_metrics
from fhm_dapp_tim.fhm_dapp.model import AnchorStudentTModel


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ap_building_owner(
    prototype_rss: torch.Tensor,
    anchor_buildings: torch.Tensor,
    n_buildings: int,
) -> torch.Tensor:
    visible = prototype_rss > -0.999
    counts = []
    for building_id in range(max(int(n_buildings), 1)):
        mask = anchor_buildings == building_id
        if torch.any(mask):
            counts.append(visible[mask].sum(dim=0))
        else:
            counts.append(torch.zeros(prototype_rss.shape[1], dtype=torch.long))
    return torch.stack(counts, dim=0).argmax(dim=0)


def _neighbor_targets(
    anchor_coords: torch.Tensor,
    anchor_buildings: torch.Tensor,
    *,
    neighbor_count: int,
    temperature_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_anchors = int(anchor_coords.shape[0])
    if n_anchors < 2:
        raise ValueError("At least two anchors are required for training")
    k = min(max(int(neighbor_count), 1), n_anchors - 1)
    dist = torch.cdist(anchor_coords.float(), anchor_coords.float())
    same_building = anchor_buildings[:, None] == anchor_buildings[None, :]
    dist = dist.masked_fill(~same_building, float("inf"))
    dist.fill_diagonal_(float("inf"))
    fallback = torch.cdist(anchor_coords.float(), anchor_coords.float())
    fallback.fill_diagonal_(float("inf"))
    missing = ~torch.isfinite(dist).any(dim=1)
    if torch.any(missing):
        dist[missing] = fallback[missing]
    neighbor_dist, neighbor_idx = torch.topk(dist, k=k, largest=False, dim=1)
    logits = -neighbor_dist / max(float(temperature_m), 1e-3)
    weights = torch.softmax(logits, dim=1)
    return neighbor_idx.long(), weights.float()


@torch.no_grad()
def _evaluate_model(
    model: AnchorStudentTModel,
    rss: torch.Tensor,
    target: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    top_k: int,
    neural_scale: float,
    prototype_weight: float,
    posterior_temperature: float,
) -> dict:
    model.eval()
    predictions = []
    for start in range(0, len(rss), batch_size):
        block = rss[start:start + batch_size].to(device)
        result = model.predict(
            block,
            top_k=top_k,
            neural_scale=neural_scale,
            prototype_distance_weight=prototype_weight,
            posterior_temperature=posterior_temperature,
        )
        predictions.append(result["coords"].detach().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    metrics, _ = coordinate_metrics(pred, target.cpu().numpy())
    return metrics


def train_anchor(args: argparse.Namespace) -> dict:
    _set_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))

    data_path = resolve_path(args.train_data)
    rss, coords_norm, building_t, floor_t, _space_t, _user_t, phone_t, _ts_t, stats = (
        load_dataset(data_path, dataset_type=args.dataset)
    )
    raw_xy = raw_coords(coords_norm, building_t, stats)
    coord_origin = raw_xy.mean(axis=0).astype(np.float32)
    centered_xy = (raw_xy - coord_origin).astype(np.float32)

    train_idx = np.arange(len(rss), dtype=np.int64)
    memory, sample_anchor = build_anchor_memory(
        rss,
        centered_xy,
        building_t.numpy(),
        floor_t.numpy(),
        train_idx,
    )
    if np.any(sample_anchor < 0):
        raise ValueError("Some training rows were not assigned to an anchor")

    seen_ap_mask = (memory["prototype_rss"] > -0.999).any(dim=0)
    ap_owner = _ap_building_owner(
        memory["prototype_rss"],
        memory["building"],
        int(stats["n_buildings"]),
    )
    config = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "max_visible": args.max_visible,
        "dropout": args.dropout,
        "student_df": args.student_df,
        "max_residual_m": args.max_residual_m,
        "residual_experts": 1,
        "scan_feature_mode": "legacy",
        "prototype_metric": "l2",
        "candidate_reranker": False,
        "candidate_reranker_mode": "summary",
        "reranker_anchor_decoder": False,
        "anchor_decoder_scale_m": 6.0,
        "n_devices": 0,
        "train_all": True,
        "validation_fraction": 0.0,
        "calibration_fraction": 0.0,
        "split_strategy": "official_training",
    }
    model = AnchorStudentTModel(
        n_aps=int(stats["n_aps"]),
        anchor_coords=memory["coords"],
        anchor_buildings=memory["building"],
        anchor_floors=memory["floor"],
        seen_ap_mask=seen_ap_mask,
        prototype_rss=memory["prototype_rss"],
        ap_building_owner=ap_owner,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_visible=args.max_visible,
        dropout=args.dropout,
        student_df=args.student_df,
        max_residual_m=args.max_residual_m,
        prototype_metric="l2",
    ).to(device)
    model.initialize_anchor_keys(memory["prototype_rss"].to(device))

    neighbor_idx, neighbor_weights = _neighbor_targets(
        memory["coords"],
        memory["building"],
        neighbor_count=args.neighbor_count,
        temperature_m=args.neighbor_temperature_m,
    )
    neighbor_idx = neighbor_idx.to(device)
    neighbor_weights = neighbor_weights.to(device)

    rss_t = rss.float()
    target_t = torch.from_numpy(centered_xy).float()
    anchor_t = torch.from_numpy(sample_anchor).long()
    dataset = TensorDataset(rss_t, target_t, anchor_t, phone_t.long())
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_rss, batch_xy, batch_anchor, _batch_phone in loader:
            batch_rss = batch_rss.to(device)
            batch_xy = batch_xy.to(device)
            batch_anchor = batch_anchor.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model.training_loss(
                batch_rss,
                batch_xy,
                batch_anchor,
                neighbor_idx,
                neighbor_weights,
                retrieval_weight=args.retrieval_weight,
                hierarchy_weight=args.hierarchy_weight,
                self_target_weight=args.self_target_weight,
                coordinate_weight=args.coordinate_weight,
                rss_offset_aug_db=args.rss_offset_aug_db,
                rss_scale_aug=args.rss_scale_aug,
                train_candidates=args.train_candidates,
                ap_dropout=args.ap_dropout,
            )
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate_model(
            model,
            rss_t,
            target_t,
            batch_size=args.eval_batch_size,
            device=device,
            top_k=args.eval_top_k,
            neural_scale=args.eval_neural_scale,
            prototype_weight=args.eval_prototype_weight,
            posterior_temperature=args.eval_posterior_temperature,
        )
        mae = float(metrics["mae"])
        if mae < best_mae - args.min_delta:
            best_mae = mae
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0 or stale == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "train_mae": mae,
                        "best_mae": best_mae,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = _evaluate_model(
        model,
        rss_t,
        target_t,
        batch_size=args.eval_batch_size,
        device=device,
        top_k=args.eval_top_k,
        neural_scale=args.eval_neural_scale,
        prototype_weight=args.eval_prototype_weight,
        posterior_temperature=args.eval_posterior_temperature,
    )
    checkpoint = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": config,
        "stats": stats,
        "coord_origin": coord_origin,
        "conformal_quantiles": {},
        "epoch": int(best_epoch),
        "val_mae": float(best_mae),
        "seed": int(args.seed),
        "train_metrics": final_metrics,
        "training_seconds": float(time.perf_counter() - started),
    }
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    payload = {
        "output": str(output_path),
        "seed": int(args.seed),
        "dataset": args.dataset,
        "rows": int(len(rss)),
        "anchors": int(memory["coords"].shape[0]),
        "features": int(stats["n_aps"]),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_train_mae": float(best_mae),
        "train_metrics": final_metrics,
        "training_seconds": checkpoint["training_seconds"],
    }
    print(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-visible", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--student-df", type=float, default=4.0)
    parser.add_argument("--max-residual-m", type=float, default=30.0)
    parser.add_argument("--neighbor-count", type=int, default=8)
    parser.add_argument("--neighbor-temperature-m", type=float, default=4.0)
    parser.add_argument("--retrieval-weight", type=float, default=0.5)
    parser.add_argument("--hierarchy-weight", type=float, default=1.0)
    parser.add_argument("--self-target-weight", type=float, default=0.25)
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--rss-offset-aug-db", type=float, default=2.0)
    parser.add_argument("--rss-scale-aug", type=float, default=0.03)
    parser.add_argument("--ap-dropout", type=float, default=0.10)
    parser.add_argument("--train-candidates", type=int, default=32)
    parser.add_argument("--eval-top-k", type=int, default=16)
    parser.add_argument("--eval-neural-scale", type=float, default=0.75)
    parser.add_argument("--eval-prototype-weight", type=float, default=8.0)
    parser.add_argument("--eval-posterior-temperature", type=float, default=0.75)
    args = parser.parse_args()
    train_anchor(args)


if __name__ == "__main__":
    main()
