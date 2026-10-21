# V17 Dataset-Adaptive Spatial Joint Fit — Iteration Note

## Question

Does one unified S1+S2 inference architecture improve localization when physical scales and coupled calibration parameters are fitted per dataset, instead of using an unbounded graph with a fixed 8 m temperature and sequentially freezing parameters?

## Existing geometry evidence

These values come from the previous frozen union-kNN graph report; they are not the new k-th-neighbor quantiles.

| Dataset | Anchors | Union-kNN edges | P50 | P95 | Max | Edges >30 m |
|---|---:|---:|---:|---:|---:|---:|
| TUJI | 646 | 2,946 | 1.190 m | 1.787 m | 2.977 m | 0 |
| UTS | 1,052 | 5,000 | 6.061 m | 12.073 m | 21.268 m | 0 |
| UJI | 672 | 3,232 | 12.984 m | 41.445 m | 77.537 m | 384 |

Preflight computes detailed distributions from the anchor geometry: same-building/same-floor k=1,2,4,8 distances at P0/P1/P5/P10/P25/P50/P75/P90/P95/P99/P100, plus union-kNN edge lengths, threshold counts and exp(-d/8) diagnostics. Do not infer missing quantiles from the table above.

## Model/search

Use one model family for all datasets. Fit separate physical scales from each dataset's geometry:
- S1: \(R_{\max}^{S1}\), \(\tau_{\mathrm{graph}}\)
- S2: \(R_{\max}^{S2}\), \(\tau_{\mathrm{virt}}\), with distance-decayed endpoint weights normalized per parent
- Joint calibration: \(T,\theta_v,\theta_s,\alpha_{\mathrm{prior}},\alpha_{\mathrm{obs}},\beta\)
- \(\gamma_{\mathrm{space}}=0\) for this iteration

The spatial search uses a small deterministic outer design over FIT-derived scale pools. Each geometry is evaluated against a joint Cartesian calibration grid; no sequential freezing. The fast first-pass grid has 4 temperatures × 3 visibility weights × 3 strength weights × 4 prior weights × 4 observation weights × 9 beta values = 5,184 candidates per geometry. Beta remains [0, .05, .10, .15, .20, .35, .50, .65, .80].

Choose among CAL-fit candidates using the dimensionless objective:

\[
J_\lambda =
\frac{\mathrm{MAE}+0.30\,W_1}{(\mathrm{MAE}+0.30\,W_1)_{\mathrm{clean}}}
+0.25\frac{\mathrm{BFCE}-\mathrm{BFCE}_{\mathrm{clean}}}{\max(\mathrm{BFCE}_{\mathrm{clean}},10^{-6})}
+\lambda\frac{\mathrm{NLL}-\mathrm{NLL}_{\mathrm{clean}}}{\max(|\mathrm{NLL}_{\mathrm{clean}}|,1)}
\]

Run \(\lambda\in\{0.02,0.05\}\). Keep BF accuracy no worse than the clean CAL-fit baseline and BF cross entropy within 1% of baseline. These constraints do not guarantee simultaneous wins; report trade-offs honestly. Energy Score and tail metrics are reported for selected candidates on CAL-eval, not used to repeatedly retune against CAL-eval.

## Run/output

Use a fresh output directory for each iteration so old metrics cannot be mixed with new ones. Preflight writes the FIT geometry diagnostics and scale search space. Run tuji, then uts, then uji; each run selects on CAL-fit and evaluates once on CAL-eval. TEST labels and TEST scoring are not used.

Key outputs: FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.md/.json, SPATIAL_SCALE_SEARCH_SPACE.json, per-run CAL_FIT_SELECTION.json, CANDIDATE_METRICS.csv, CAL_EVAL_REPORT.json, and aggregate CAL_EVAL_REPORT.md, PARETO_FRONTIER.csv, RUN_REPORT_V17_DATASET_ADAPTIVE_JOINT_FIT.md.
