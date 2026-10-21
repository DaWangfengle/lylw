"""Probe: sizes and timings for the GMRF/bandwidth validation."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as s3

torch.set_grad_enabled(False)

ROOT = REPO / "fhm_dapp_tim/evidence/v17_two_stage_freeze_v1"
CKPT = REPO / "fhm_dapp_tim/evidence/v17_spatial_structure_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device, flush=True)

for dataset in ("tuji", "uts", "uji"):
    seed = 17301
    t0 = time.time()
    ctx = s3.reconstruct(dataset, seed, ROOT, device)
    t1 = time.time()
    model = s3.enable_training_arm(ctx["initial"], "D", ctx["graph"])
    path = CKPT / "trained" / f"{dataset}_seed{seed}" / "D" / "best_dev_epoch_21_80.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    n = model.n_anchors
    t2 = time.time()
    fit = s3.make_pack(ctx, model, ctx["cal_fit"])
    t3 = time.time()
    ev = s3.make_pack(ctx, model, ctx["cal_eval"])
    t4 = time.time()
    print(
        f"{dataset}: anchors={n} edges={len(ctx['edges'])} "
        f"fit_rows={len(ctx['cal_fit'])} eval_rows={len(ctx['cal_eval'])} "
        f"reconstruct={t1-t0:.1f}s load={t2-t1:.1f}s fit_pack={t3-t2:.1f}s "
        f"eval_pack={t4-t3:.1f}s",
        flush=True,
    )
    del fit, ev, ctx, model
    torch.cuda.empty_cache()
