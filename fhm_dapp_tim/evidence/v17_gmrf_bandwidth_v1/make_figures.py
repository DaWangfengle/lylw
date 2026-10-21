"""Figures for the GMRF / bandwidth validation."""
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
LABELS = {"tuji": "TUJI (646 anchors, h=1.19 m)",
          "uts": "UTS (1052 anchors, h=6.06 m)",
          "uji": "UJI (672 anchors, h=10.26 m)"}
RHO0 = 1.0 / np.sqrt(12.0)


def load(path):
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r for r in runs if "error" not in r]


def fig_rho(runs):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for dataset in ("tuji", "uts", "uji"):
        rows = [r for r in runs if r["dataset"] == dataset]
        if not rows:
            continue
        rhos = sorted({row["rho"] for r in rows for row in r["rho_sweep"]})
        mean = [statistics.mean(row["eval_nll"] for r in rows for row in r["rho_sweep"]
                                if abs(row["rho"] - x) < 1e-9) for x in rhos]
        sd = [statistics.stdev([row["eval_nll"] for r in rows for row in r["rho_sweep"]
                                if abs(row["rho"] - x) < 1e-9]) if len(rows) > 1 else 0.0
              for x in rhos]
        ax.errorbar(rhos, mean, yerr=sd, marker="o", ms=3.5, lw=1.6, capsize=2.5,
                    color=COLORS[dataset], label=LABELS[dataset])
    ax.axvline(RHO0, color="0.4", ls="--", lw=1)
    ax.text(RHO0 + 0.015, ax.get_ylim()[1] * 0.97, r"$1/\sqrt{12}\approx0.289$",
            color="0.35", va="top", fontsize=8)
    ax.set_xlabel(r"bandwidth $\rho$  ($\kappa_i=\rho\,d_i$)")
    ax.set_ylabel("CAL-eval joint NLL (lower is better)")
    ax.set_title("Discretization bandwidth: one scalar, exact MAE invariance")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rho_nll.pdf")
    fig.savefig(OUT / "fig_rho_nll.png", dpi=180)
    plt.close(fig)


def fig_alpha(runs):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for dataset in ("tuji", "uts", "uji"):
        rows = [r for r in runs if r["dataset"] == dataset]
        if not rows:
            continue
        alphas = sorted({row["a"] for r in rows for row in r["alpha_sweep"]})
        mean = [statistics.mean(row["eval_mae"] for r in rows for row in r["alpha_sweep"]
                                if abs(row["a"] - a) < 1e-9) for a in alphas]
        base = statistics.mean(r["base_eval"]["mae"] for r in rows)
        ax.plot(alphas, mean, marker="o", ms=3.5, lw=1.6, color=COLORS[dataset],
                label=LABELS[dataset])
        ax.axhline(base, color=COLORS[dataset], ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel(r"graph diffusion $a$ in $(I-aW)^{-1}$")
    ax.set_ylabel("CAL-eval MAE (m)")
    ax.set_title("All-order diffusion: helps only where the graph is sparse")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alpha_mae.pdf")
    fig.savefig(OUT / "fig_alpha_mae.png", dpi=180)
    plt.close(fig)


def fig_methods(runs):
    methods = ["base", "2K", "2A", "2A+2K", "1G+2K"]
    labels = ["base", "2K", "2A", "2A+2K", "1G+2K"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    width = 0.16
    for ax, metric, name in ((axes[0], "nll", "joint NLL"), (axes[1], "mae", "MAE (m)")):
        datasets = [d for d in ("tuji", "uts", "uji")
                    if any(r["dataset"] == d for r in runs)]
        x = np.arange(len(datasets))
        for i, (m, lab) in enumerate(zip(methods, labels)):
            vals = []
            for d in datasets:
                rows = [r for r in runs if r["dataset"] == d]
                vals.append(statistics.mean(r["decomposition"][m][metric] for r in rows))
            ax.bar(x + (i - 2) * width, vals, width, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Support refinement decomposition: bandwidth does the work, midpoint cells move the mean", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_decomposition.pdf")
    fig.savefig(OUT / "fig_decomposition.png", dpi=180)
    plt.close(fig)


def main():
    runs = load(OUT / "RESULTS.json")
    fig_rho(runs)
    fig_alpha(runs)
    fig_methods(runs)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
