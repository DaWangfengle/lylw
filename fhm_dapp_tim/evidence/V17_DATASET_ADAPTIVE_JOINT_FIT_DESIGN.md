# V17 Dataset-Adaptive Spatial Joint Fit — Corrected Iteration Design

## Research rule

Use the same model architecture and mathematical definitions on TUJI, UTS and UJI. Fit the physical scales and calibration values separately from each dataset's own geometry and the current dataset/seed/checkpoint CAL-fit baseline. Treat the datasets as three independent experiments and run them sequentially: TUJI, then UTS, then UJI. Do not mix datasets' meter scales or tune one dataset using another dataset's metrics.

## Existing geometry evidence

These values come from the previous frozen union-kNN graph report; they are not the new k-th-neighbor quantiles.

| Dataset | Anchors | Union-kNN edges | P50 | P95 | Max | Edges >30 m |
|---|---:|---:|---:|---:|---:|---:|
| TUJI | 646 | 2,946 | 1.190 m | 1.787 m | 2.977 m | 0 |
| UTS | 1,052 | 5,000 | 6.061 m | 12.073 m | 21.268 m | 0 |
| UJI | 672 | 3,232 | 12.984 m | 41.445 m | 77.537 m | 384 |

These are the previous graph's edge statistics only. Preflight computes same-building/same-floor k=1,2,4,8 neighbor distances at P0/P1/P5/P10/P25/P50/P75/P90/P95/P99/P100 and reports union-kNN edge lengths separately. Do not infer missing FIT quantiles from the table above.

## Spatial search: no global meter-valued candidate list

Fit these parameters independently from each dataset's FIT geometry:
- S1: \(R_{\max}^{S1}\), \(\tau_{\mathrm{graph}}\)
- S2: \(R_{\max}^{S2}\), \(\tau_{\mathrm{virt}}\)
- Joint calibration: \(T,\theta_v,\theta_s,\alpha_{\mathrm{prior}},\alpha_{\mathrm{obs}},\beta\)
- \(\gamma_{\mathrm{space}}=0\) for this iteration.

Every finite geometry candidate must be derived from the current dataset's FIT k=1/2/4/8 neighbor quantiles and union-kNN edge quantiles. Dimensionless multipliers \((0.50,0.75,1.0,1.25,1.50)\) may refine those observed scales; fixed global meter lists are forbidden. Generate 12–16 deterministic geometry configurations per dataset/seed, include \(\tau_{\mathrm{virt}}=\infty\) controls, and enforce \(R_{\max}^{S2}\le R_{\max}^{S1}\). The old unbounded 8 m setup is a reporting-only baseline, not a selectable spatial candidate. If FIT quantiles cannot produce at least 12 valid configurations, fail rather than reverting to hard-coded meter values.

## Baseline-centered joint calibration grid

For each dataset × seed × checkpoint, first resolve the exact clean baseline \((T_{\mathrm{clean}},\theta_{v,\mathrm{clean}},\theta_{s,\mathrm{clean}})\). Build three local relative-neighborhood values around each exact center, respecting the existing legal bounds and retaining the exact baseline value.

Use:
- \(\alpha_{\mathrm{prior}}\in\{0,0.05,0.10,0.20,0.35,0.65\}\)
- \(\alpha_{\mathrm{obs}}\in\{0,0.10,0.25\}\)
- \(\beta\in\{0,0.05,0.10,0.15,0.20,0.35,0.50,0.65,0.80\}\)

That is 3 × 3 × 3 × 6 × 3 × 9 = 4,374 calibration candidates per geometry, or 69,984 candidates for 16 geometries per run, plus the separate clean-reference row. The actual baseline centers and generated local grids are written to each CAL_FIT_SELECTION.json. No sequential freezing is allowed.

## Beta-independent sparse graph work

For candidates sharing \((T,\theta_v,\theta_s,\alpha_{\mathrm{prior}},\alpha_{\mathrm{obs}})\), parent logits and virtual logits do not depend on \(\beta\). Compute the prior-channel and observation-channel sparse graph multiplications once each, then broadcast all nine beta values over the resulting logits. Regression tests require exactly two sparse.mm calls for a full nine-beta group.

## Selection objective and BF-CE stability

Choose only on CAL-fit and compare \(\lambda\in\{0.02,0.05\}\):

\[
J_\lambda =
\frac{\mathrm{MAE}+0.30W_1}{(\mathrm{MAE}+0.30W_1)_{\mathrm{clean}}}
+0.25\frac{\mathrm{BFCE}-\mathrm{BFCE}_{\mathrm{clean}}}{\max(\mathrm{BFCE}_{\mathrm{clean}},0.05)}
+\lambda\frac{\mathrm{NLL}-\mathrm{NLL}_{\mathrm{clean}}}{\max(|\mathrm{NLL}_{\mathrm{clean}}|,1)}
\]

Keep BF accuracy no worse than the clean CAL-fit baseline. Use the BF-CE guard \(\mathrm{BFCE}\le1.02\,\mathrm{BFCE}_{\mathrm{clean}}+0.005\), and normalize BF-CE with denominator \(\max(\mathrm{BFCE}_{\mathrm{clean}},0.05)\). This avoids magnifying tiny numerical changes when BF-CE is near zero. These constraints do not guarantee simultaneous wins; report the Pareto trade-offs honestly. CAL-eval is reporting-only after candidate selection is locked.

## Run/output

Use a fresh output directory for each iteration so old metrics cannot be mixed with new ones. Preflight writes FIT geometry diagnostics and dataset-specific candidate pools. Run tuji, then uts, then uji; each dataset's full seed × C/D/E set must finish before starting the next. Do not change the code or selection protocol between datasets.

TEST labels and TEST scoring remain untouched. Expected counters are \(\text{internal\_test\_rows\_scored}=0\) and \(\text{official\_test\_rows\_scored}=0\).
