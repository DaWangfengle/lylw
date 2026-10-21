"""Figures for the hypothesis stage."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[3] / "outputs/v17_gmrf_bandwidth_v1"
COLORS = {"tuji": "#1f77b4", "uts": "#d62728", "uji": "#2ca02c"}
LABELS = {"tuji": "TUJI (h=1.19 m)", "uts": "UTS (h=6.06 m)", "uji": "UJI (h=10.26 m)"}
METHODS = [("2A_lse", "2A Cell LSE", "s"), ("2A_mass", "2A mass-preserving", "D"),
           ("2Q4", "2Q4 quadrature", "^"), ("2Q8", "2Q8 quadrature", "v"),
           ("2K", "2K bandwidth", "o")]


def load():
    runs = [r for r in json.loads((OUT / "hypotheses" / "HYPOTHESES.json").read_text(encoding="utf-8"))
            if "error" not in r]
    return runs


def main():
    runs = load()

    # ---- figure: density gain vs point cost ------------------------------ #
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for dataset in ("tuji", "uts", "uji"):
        rows = [r for r in runs if r["arm"] == "D" and r["dataset"] == dataset]
        if not rows:
            continue
        base_nll = statistics.mean(r["h2_cells"]["base"]["nll"] for r in rows)
        base_mae = statistics.mean(r["h2_cells"]["base"]["mae"] for r in rows)
        for method, label, marker in METHODS:
            dnll = base_nll - statistics.mean(r["h2_cells"][method]["nll"] for r in rows)
            dmae = statistics.mean(r["h2_cells"][method]["mae"] for r in rows) - base_mae
            ax.scatter(dmae, dnll, s=46, marker=marker, color=COLORS[dataset],
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=LABELS[dataset] if method == "2A_lse" else None)
            if dataset == "uji":
                ax.annotate(label, (dmae, dnll), textcoords="offset points", xytext=(6, -3),
                            fontsize=6.5, color="0.25")
    ax.axvline(0.0, color="0.55", ls="--", lw=1)
    ax.text(0.004, ax.get_ylim()[1] * 0.55, "exactly MAE-neutral", fontsize=7, color="0.45",
            rotation=90, va="center")
    ax.set_xlabel("ΔMAE vs base (m, positive = worse)")
    ax.set_ylabel("joint NLL gain vs base (nats, positive = better)")
    ax.set_title("Support operators: density gain vs point cost (backbone D)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_operator_tradeoff.pdf")
    fig.savefig(OUT / "fig_operator_tradeoff.png", dpi=180)
    plt.close(fig)

    # ---- figure: interaction --------------------------------------------- #
    main_runs = [r for r in json.loads((OUT / "RESULTS.json").read_text(encoding="utf-8"))
                 if "error" not in r]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    datasets = ["tuji", "uts", "uji"]
    width = 0.26
    for k, arm in enumerate(("C", "D", "E")):
        means, errs = [], []
        for dataset in datasets:
            rows = [r for r in main_runs if r["arm"] == arm and r["dataset"] == dataset]
            values = [r["decomposition"]["1G+2A"]["mae"] - r["decomposition"]["1G"]["mae"]
                      - r["decomposition"]["2A"]["mae"] + r["decomposition"]["base"]["mae"]
                      for r in rows]
            means.append(statistics.mean(values))
            errs.append(statistics.stdev(values) if len(values) > 1 else 0.0)
        ax.bar(np.arange(len(datasets)) + (k - 1) * width, means, width, yerr=errs,
               capsize=2.5, label=f"backbone {arm}")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([LABELS[d] for d in datasets], fontsize=7.5)
    ax.set_ylabel("interaction ΔMAE (m)")
    ax.set_title("1G × 2A interaction: superadditive harm on UJI")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_interaction.pdf")
    fig.savefig(OUT / "fig_interaction.png", dpi=180)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
