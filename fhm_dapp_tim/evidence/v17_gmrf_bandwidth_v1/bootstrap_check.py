"""Seed-balanced bootstrap of the bandwidth NLL gain."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "outputs/v17_gmrf_bandwidth_v1"
runs = [r for r in json.loads((OUT / "hypotheses/HYPOTHESES.json").read_text(encoding="utf-8"))
        if "error" not in r and r["arm"] == "D"]
rng = np.random.default_rng(11)

for dataset in ("tuji", "uts", "uji"):
    per_seed = []
    for r in runs:
        if r["dataset"] != dataset:
            continue
        per_seed.append(np.array(list(r["h3_site_delta_nll"].values()), dtype=float))
    point = statistics.mean(float(a.mean()) for a in per_seed)
    draws = []
    for _ in range(4000):
        draws.append(statistics.mean(float(rng.choice(a, size=len(a), replace=True).mean()) for a in per_seed))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"{dataset}: delta={point:.3f} CI=[{lo:.3f}, {hi:.3f}] seeds={len(per_seed)} "
          f"sites={[len(a) for a in per_seed]}")
