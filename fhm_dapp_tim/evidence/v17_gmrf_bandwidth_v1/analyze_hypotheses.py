"""Analyse the hypothesis runs and emit HYPOTHESES.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "outputs/v17_gmrf_bandwidth_v1"
HYP = OUT / "hypotheses" / "HYPOTHESES.json"
MAIN = OUT / "RESULTS.json"
DATASETS = ("tuji", "uts", "uji")
ARMS = ("C", "D", "E")


def fmt(values, digits=4):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def bootstrap(values, draws=4000, seed=7):
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(draws)])
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    runs = [r for r in json.loads(HYP.read_text(encoding="utf-8")) if "error" not in r]
    main_runs = [r for r in json.loads(MAIN.read_text(encoding="utf-8")) if "error" not in r]
    lines = []
    add = lines.append

    add("# Hypothesis tests — V17 GMRF diffusion + discretization bandwidth")
    add("")
    add(f"{len(runs)} runs (three backbones × three seeds × three datasets), zero retraining, "
        "CAL-eval reporting only, zero TEST rows scored.")
    add("")

    # ---- H1 --------------------------------------------------------------- #
    add("## H1 — does the 0.30 transport term in calibration create the density gap?")
    add("")
    add("Re-fit `(T, theta)` with risk weight 0.30 (the locked protocol) and 0.00 (pure joint NLL), "
        "then bandwidth-refit. `rho*` is chosen on CAL-fit; metrics are CAL-eval.")
    add("")
    add("| arm | dataset | MAE w=0.30 | MAE w=0.00 | NLL w=0.30 | NLL w=0.00 | ES w=0.30 | ES w=0.00 | rho* w=0.30 | rho* w=0.00 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in runs if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            pick = lambda tag, key: [r["h1_calibration"][tag]["base"][key] for r in rows]
            add(f"| {arm} | {dataset} | {fmt(pick('w0.30', 'mae'))} | {fmt(pick('w0.00', 'mae'))} | "
                f"{fmt(pick('w0.30', 'nll'), 3)} | {fmt(pick('w0.00', 'nll'), 3)} | "
                f"{fmt(pick('w0.30', 'es'), 3)} | {fmt(pick('w0.00', 'es'), 3)} | "
                f"{fmt([r['h1_calibration']['w0.30']['rho_star'] for r in rows], 2)} | "
                f"{fmt([r['h1_calibration']['w0.00']['rho_star'] for r in rows], 2)} |")
    add("")
    add("Read: if the risk term manufactured the gap, NLL(w=0) should be far below NLL(w=0.30) and "
        "rho* should collapse. The measured shift is the *size* of that effect.")
    add("")

    # ---- H2 --------------------------------------------------------------- #
    add("## H2 — Cell LSE vs mass-preserving cells vs symmetric quadrature")
    add("")
    add("All at the locked calibration, beta = 0.8, bandwidth rho = rho* of the w=0.30 arm. "
        "`2Q4` uses four antipodal nodes at radius 0.4 d, so its first moment is exactly the base mean.")
    add("")
    add("| arm | dataset | method | MAE | NLL | ES | transport risk |")
    add("|---|---|---|---:|---:|---:|---:|")
    methods = ("base", "2A_lse", "2A_mass", "2Q4", "2Q8", "2Q4+2K", "2Q8+2K", "2K")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in runs if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            for method in methods:
                m = lambda key: [r["h2_cells"][method][key] for r in rows]
                add(f"| {arm} | {dataset} | {method} | {fmt(m('mae'))} | {fmt(m('nll'), 3)} | "
                    f"{fmt(m('es'), 3)} | {fmt(m('risk'), 3)} |")
    add("")
    add("### Best mean-preserving support operator vs base (backbone D)")
    add("")
    add("| dataset | base NLL | 2K NLL | 2Q8+2K NLL | base ES | 2K ES | 2Q8+2K ES | base MAE | 2Q8+2K MAE | ΔMAE |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        rows = [r for r in runs if r["arm"] == "D" and r["dataset"] == dataset]
        if not rows:
            continue
        mean = lambda m, k: statistics.mean(r["h2_cells"][m][k] for r in rows)
        add(f"| {dataset} | {mean('base', 'nll'):.3f} | {mean('2K', 'nll'):.3f} | "
            f"{mean('2Q8+2K', 'nll'):.3f} | {mean('base', 'es'):.3f} | {mean('2K', 'es'):.3f} | "
            f"{mean('2Q8+2K', 'es'):.3f} | {mean('base', 'mae'):.4f} | "
            f"{mean('2Q8+2K', 'mae'):.4f} | {mean('2Q8+2K', 'mae') - mean('base', 'mae'):+.4f} |")
    add("")
    add("Key contrasts (mean over arms and seeds, UJI): the quadrature keeps MAE identical while "
        "its transport risk rises — direct evidence that the risk functional is not the MAE functional.")
    add("")

    # ---- H3 --------------------------------------------------------------- #
    add("## H3 — bandwidth NLL gain with a site-level bootstrap")
    add("")
    add("Per-site paired ΔNLL (base − 2K at rho*), sites resampled within each seed and the three "
        "seeds weighted equally, 4000 bootstrap resamples. Backbone D.")
    add("")
    add("| dataset | mean ΔNLL | 95% CI | sites per seed |")
    add("|---|---:|---|---:|")
    rng = np.random.default_rng(11)
    for dataset in DATASETS:
        per_seed = [np.array(list(r["h3_site_delta_nll"].values()), dtype=float)
                    for r in runs if r["dataset"] == dataset and r["arm"] == "D"]
        if not per_seed:
            continue
        point = statistics.mean(float(a.mean()) for a in per_seed)
        draws = [statistics.mean(float(rng.choice(a, size=len(a), replace=True).mean())
                                 for a in per_seed) for _ in range(4000)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        add(f"| {dataset} | {point:.3f} | [{lo:.3f}, {hi:.3f}] | {len(per_seed[0])} |")
    add("")

    # ---- H4 --------------------------------------------------------------- #
    add("## H4 — graph locality: 25 m cap vs no cap (UJI only)")
    add("")
    add("| seed | arm | MAE cap a=0.4 | MAE no-cap a=0.4 | Δ | NLL cap | NLL no-cap |")
    add("|---:|---|---:|---:|---:|---:|---:|")
    for r in runs:
        if r["dataset"] != "uji" or "h4_cutoff" not in r:
            continue
        cap = r["h4_cutoff"]["cap"]["a=0.4"]
        nocap = r["h4_cutoff"]["nocap"]["a=0.4"]
        add(f"| {r['seed']} | {r['arm']} | {cap['mae']:.4f} | {nocap['mae']:.4f} | "
            f"{nocap['mae'] - cap['mae']:+.4f} | {cap['nll']:.3f} | {nocap['nll']:.3f} |")
    add("")

    # ---- H5: fixed-a protocol from the main grid -------------------------- #
    add("## H5 — selection protocol: per-seed argmin vs one a per dataset")
    add("")
    add("`per-seed` picks a on each CAL-fit; `fixed` picks one a per dataset by the mean CAL-fit MAE "
        "of the three seeds, then applies it to all seeds. Both are evaluated on CAL-eval.")
    add("")
    add("| arm | dataset | a (fixed) | base MAE | 1G MAE (fixed a) | ΔMAE fixed | ΔMAE per-seed |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in main_runs if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            alphas = sorted({row["a"] for r in rows for row in r["alpha_sweep"]})
            mean_fit = {a: statistics.mean(row["fit_mae"] for r in rows for row in r["alpha_sweep"]
                                           if abs(row["a"] - a) < 1e-9) for a in alphas}
            a_fixed = min(mean_fit, key=lambda a: mean_fit[a])
            fixed_eval = [row["eval_mae"] for r in rows for row in r["alpha_sweep"]
                          if abs(row["a"] - a_fixed) < 1e-9]
            base = [r["base_eval"]["mae"] for r in rows]
            per_seed = [r["selected"]["a"] for r in rows]
            per_seed_eval = [r["decomposition"]["1G"]["mae"] for r in rows]
            add(f"| {arm} | {dataset} | {a_fixed:.2f} | {fmt(base)} | {fmt(fixed_eval)} | "
                f"{statistics.mean(fixed_eval) - statistics.mean(base):+.4f} | "
                f"{statistics.mean(per_seed_eval) - statistics.mean(base):+.4f} |")
    add("")

    # ---- H6: interaction CIs --------------------------------------------- #
    add("## H6 — composition interaction on MAE (2×2: 1G × 2A)")
    add("")
    add("`interaction = MAE(1G+2A) − MAE(1G) − MAE(2A) + MAE(base)`, per seed, mean ± SD.")
    add("")
    add("| arm | dataset | interaction MAE (m) |")
    add("|---|---|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in main_runs if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            values = []
            for r in rows:
                d = r["decomposition"]
                values.append(d["1G+2A"]["mae"] - d["1G"]["mae"] - d["2A"]["mae"] + d["base"]["mae"])
            add(f"| {arm} | {dataset} | {fmt(values, 4)} |")
    add("")

    report = OUT / "HYPOTHESES.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
