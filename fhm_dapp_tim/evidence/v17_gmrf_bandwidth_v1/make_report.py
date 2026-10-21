"""Render the GMRF/bandwidth validation into a markdown report (all backbones)."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "outputs/v17_gmrf_bandwidth_v1"
DATASETS = ("tuji", "uts", "uji")
ARMS = ("C", "D", "E")
METHODS = ("base", "2K", "2A", "2A+2K", "1G", "1G+2K", "1G+2A", "1G+2A+2K")
MIN_GAIN_M = 0.02


def fmt(values, digits=4):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def load_all():
    runs, seen = [], set()
    for path in (OUT / "RESULTS.json", OUT / "CE" / "RESULTS.json"):
        if not path.is_file():
            continue
        for run in json.loads(path.read_text(encoding="utf-8")):
            key = (run.get("dataset"), run.get("seed"), run.get("arm"))
            if key in seen:
                continue
            seen.add(key)
            runs.append(run)
    return [r for r in runs if "error" not in r], [r for r in runs if "error" in r]


def conservative(run):
    base = run["base_fit"]["mae"]
    for row in sorted(run["alpha_sweep"], key=lambda r: r["a"]):
        if row["fit_mae"] <= base - MIN_GAIN_M:
            return row
    return run["alpha_sweep"][0]


def main():
    ok, bad = load_all()
    (OUT / "ALL_RESULTS.json").write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    add = lines.append
    add("# V17 GMRF diffusion + discretization bandwidth — zero-retraining validation")
    add("")
    add("`1G`: `ell <- (I - a W)^-1 ell` on the complete logit (prior and G13 channels together). "
        "`2K`: `sigma <- sqrt(sigma^2 + (rho d)^2)`, `d` = local same-BF anchor spacing. "
        "`2A`: existing midpoint Cell LSE at beta = 0.8. "
        "All selection on CAL-fit; CAL-eval is reporting-only; zero TEST rows scored.")
    add("")
    add(f"Runs: {len(ok)} completed, {len(bad)} failed. Backbones: {', '.join(sorted({r['arm'] for r in ok}))}. "
        f"Seeds: {', '.join(str(s) for s in sorted({r['seed'] for r in ok}))}.")
    add("")

    # ---- reproduction check ---------------------------------------------- #
    add("## Reproduction check")
    add("")
    add("The frozen `joint_calibrated_base` re-fitted here must match the published composition table. "
        "Maximum absolute difference across the nine dataset × backbone cells:")
    add("")
    published = {
        ("C", "tuji"): 1.86585, ("C", "uts"): 3.98481, ("C", "uji"): 7.40799,
        ("D", "tuji"): 1.81460, ("D", "uts"): 4.01216, ("D", "uji"): 7.10655,
        ("E", "tuji"): 1.79444, ("E", "uts"): 3.96847, ("E", "uji"): 7.41298,
    }
    worst = 0.0
    for (arm, dataset), value in published.items():
        rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
        if rows:
            worst = max(worst, abs(statistics.mean(r["base_eval"]["mae"] for r in rows) - value))
    add(f"- max |reference − recomputed| = **{worst:.2e} m** (a=0 identity: {max(r['assertions']['A1_a0_identity_max_abs'] for r in ok):.1e}).")
    add("")

    # ---- graph ------------------------------------------------------------ #
    add("## Graph construction (same-BF, zero diagonal, d_max = 25 m, Metropolis–Hastings doubly stochastic)")
    add("")
    add("| dataset | anchors | edges (was) | p50 m | p95 m | max m | >30 m | dropped by cap | isolated |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    baseline_edges = {"tuji": 2946, "uts": 5000, "uji": 3232}
    seen = set()
    for r in ok:
        if r["dataset"] in seen:
            continue
        seen.add(r["dataset"])
        g = r["graph"]
        add(f"| {r['dataset']} | {g['physical_anchors']} | {g['unique_edges']} ({baseline_edges[r['dataset']]}) | "
            f"{g['edge_p50_m']:.2f} | {g['edge_p95_m']:.2f} | {g['edge_max_m']:.2f} | "
            f"{g['edges_gt_30m']} | {g['dropped_over_cap']} | {g['isolated_anchors']} |")
    add("")

    # ---- NLL / MAE -------------------------------------------------------- #
    add("## Joint NLL, 2×2×2 decomposition (mean over seeds)")
    add("")
    add("| arm | dataset | base | 2K | 2A | 2A+2K | 1G | 1G+2K | 1G+2A | 1G+2A+2K |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            cells = [fmt([r["decomposition"][m]["nll"] for r in rows], 3) for m in METHODS]
            add(f"| {arm} | {dataset} | " + " | ".join(cells) + " |")
    add("")
    add("## MAE (m), same decomposition (mean over seeds)")
    add("")
    add("| arm | dataset | base | 2K | 2A | 2A+2K | 1G | 1G+2K | 1G+2A | 1G+2A+2K |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            cells = [fmt([r["decomposition"][m]["mae"] for r in rows]) for m in METHODS]
            add(f"| {arm} | {dataset} | " + " | ".join(cells) + " |")
    add("")
    add("## Energy Score (m), same decomposition")
    add("")
    add("| arm | dataset | base | 2K | 2A | 2A+2K | 1G | 1G+2K | 1G+2A | 1G+2A+2K |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            cells = [fmt([r["decomposition"][m]["es"] for r in rows], 3) for m in METHODS]
            add(f"| {arm} | {dataset} | " + " | ".join(cells) + " |")
    add("")

    # ---- bandwidth share -------------------------------------------------- #
    add("## Bandwidth vs midpoint cells (a = 0, so the comparison is pure support)")
    add("")
    add("| arm | dataset | NLL base | NLL 2K | NLL 2A | NLL 2A+2K | bandwidth share | MAE base | MAE 2A | ΔMAE 2A |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            mean = lambda m, k: statistics.mean(r["decomposition"][m][k] for r in rows)
            share = []
            for r in rows:
                d = r["decomposition"]["base"]["nll"] - r["decomposition"]["2A"]["nll"]
                if abs(d) > 1e-9:
                    share.append((r["decomposition"]["base"]["nll"] - r["decomposition"]["2K"]["nll"]) / d)
            add(f"| {arm} | {dataset} | {mean('base', 'nll'):.3f} | {mean('2K', 'nll'):.3f} | "
                f"{mean('2A', 'nll'):.3f} | {mean('2A+2K', 'nll'):.3f} | {statistics.mean(share) * 100:.0f}% | "
                f"{mean('base', 'mae'):.4f} | {mean('2A', 'mae'):.4f} | {mean('2A', 'mae') - mean('base', 'mae'):+.4f} |")
    add("")

    # ---- propagation ablation --------------------------------------------- #
    add("## Propagation ablation: does adding the S1 endpoint delta to virtual logits hurt?")
    add("")
    add("`1G+2A` is the published composition rule (virtual logit += mean endpoint S1 displacement). "
        "`1G+2A (clean virtual)` keeps the smoothed parents but leaves the virtual logits at their "
        "clean-calibration values. Same beta, same graph, same seeds.")
    add("")
    add("| arm | dataset | 1G MAE | 2A MAE | 1G+2A MAE (propagated) | 1G+2A MAE (clean virtual) | propagation cost |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            mean = lambda m: statistics.mean(r["decomposition"][m]["mae"] for r in rows)
            prop = mean("1G+2A")
            clean = mean("1G+2A_noprop")
            add(f"| {arm} | {dataset} | {mean('1G'):.4f} | {mean('2A'):.4f} | {prop:.4f} | "
                f"{clean:.4f} | {prop - clean:+.4f} |")
    add("")

    # ---- selection -------------------------------------------------------- #
    add("## Selected parameters and the conservative rule")
    add("")
    add(f"Primary: minimise CAL-fit MAE with a CAL-fit NLL guard. Conservative: require at least "
        f"{MIN_GAIN_M:.2f} m CAL-fit MAE gain before smoothing.")
    add("")
    add("| arm | dataset | seed | a* | range m | rho* | base MAE | 1G MAE | a (cons.) | MAE at a_cons |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            for r in [x for x in ok if x["arm"] == arm and x["dataset"] == dataset]:
                sel = r["selected"]
                rng = "—" if sel["implied_range_m"] is None else f"{sel['implied_range_m']:.1f}"
                cons = conservative(r)
                add(f"| {arm} | {dataset} | {r['seed']} | {sel['a']:.2f} | {rng} | {r['selected_rho']:.3f} | "
                    f"{r['base_eval']['mae']:.4f} | {r['decomposition']['1G']['mae']:.4f} | "
                    f"{cons['a']:.2f} | {cons['eval_mae']:.4f} |")
    add("")

    # ---- rho curve -------------------------------------------------------- #
    add("## NLL vs bandwidth rho (CAL-eval, mean over seeds)")
    add("")
    rhos = sorted({round(row["rho"], 3) for r in ok for row in r["rho_sweep"]})
    for arm in ARMS:
        add(f"**Backbone {arm}**")
        add("")
        add("| dataset | " + " | ".join(f"{rho:g}" for rho in rhos) + " |")
        add("|---|" + "---:|" * len(rhos))
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            cells = [fmt([row["eval_nll"] for r in rows for row in r["rho_sweep"]
                          if abs(row["rho"] - rho) < 1e-9], 3) for rho in rhos]
            add(f"| {dataset} | " + " | ".join(cells) + " |")
        add("")

    # ---- invariants ------------------------------------------------------- #
    add("## Invariants")
    add("")
    add("| arm | dataset | A1 a=0 gap | A2 ΔMAE over rho | A2 ΔBF | A4 composition gap | A6 commutation |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        for dataset in DATASETS:
            rows = [r for r in ok if r["arm"] == arm and r["dataset"] == dataset]
            if not rows:
                continue
            agg = lambda key: max(r["assertions"][key] for r in rows)
            add(f"| {arm} | {dataset} | {agg('A1_a0_identity_max_abs'):.2e} | "
                f"{agg('A2_mae_max_abs_gap_over_rho'):.2e} | {agg('A2_bf_max_abs_gap_over_rho'):.2e} | "
                f"{agg('A4_composition_gap'):.2e} | {agg('A6_commutation_max_abs'):.2e} |")
    add("")

    if bad:
        add("## Failed runs")
        add("")
        for r in bad:
            add(f"- {r['dataset']} seed {r['seed']} arm {r['arm']}: `{str(r['error'])[:300]}`")
        add("")

    report = OUT / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
