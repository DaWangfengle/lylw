# V17 Dataset-Adaptive Spatial Scale + Joint Fit

## Research decision

Keep one mathematical model family, but fit physically meaningful hyperparameters separately for TUJI, UTS and UJI. This is not an invitation to select whichever sub-method happens to win each metric after seeing CAL-eval. The protocol must be predeclared, deterministic, and tested on CAL-eval only after fitting is locked. Internal and official TEST must remain untouched.

## Why this experiment is justified

The frozen graph diagnostic in the previous report gives union-kNN edge P50/P95/max:
- TUJI: 1.190 / 1.787 / 2.977 m.
- UTS: 6.061 / 12.073 / 21.268 m.
- UJI: 12.984 / 41.445 / 77.537 m, with 384/3232 (11.88%) edges longer than 30 m.

These are current graph-edge statistics, not yet the requested FIT-only nearest-neighbor distribution. The first deliverable must calculate true FIT-anchor nearest-neighbor and k-th-neighbor distributions from the exact coordinates and split metadata. Do not report the table above as FIT quantiles unless code verifies that the underlying geometry is FIT-only.

## Scientific hypothesis (not a guaranteed result)

The fixed 8 m graph temperature is 6.72× TUJI edge P50, 1.32× UTS P50, and 0.62× UJI P50. Thus the same numeric kernel scale represents very different smoothing regimes. Unbounded S1 and S2 may also allow remote UJI edges to create inappropriate influence or virtual midpoint candidates. These are plausible mechanisms, but the actual normalized operator weights depend on degrees, logits and Sinkhorn; exp(-d/tau) alone is not a complete causal analysis.

The old CAL-fit protocol selected T/theta, then alpha, then beta/gamma conditional on frozen alpha. The new experiment should evaluate all parameter families together, without sequential freezing.

## FIT-only geometry diagnostics

For each dataset, seed, building/floor group and permitted anchor set, report:
- N anchors; duplicate/zero-distance coordinate pairs; k=1,2,4,8 neighbor-distance quantiles P0/P1/P5/P10/P25/P50/P75/P90/P95/P99/P100, mean and standard deviation.
- Current union-kNN edge-distance quantiles and counts/fractions over 5/10/15/20/30/40 m.
- Degree distribution, nodes with no neighbors after cutoff, connected components, cutoff edge retention, Sinkhorn convergence and symmetry.
- Evaluate exp(-d/8) at each quantile as a diagnostic only.
- Verify coordinate units, building/floor semantics, canonical anchor mapping and split provenance. Fail closed on uncertainty rather than infer missing metadata.

## Unified model with dataset-specific spatial scales

Fit per dataset:
- R_max_S1: cutoff for S1 graph edges.
- R_max_S2: cutoff for S2 virtual-edge candidates.
- tau_graph: S1 distance-kernel temperature.
- tau_virt: S2 virtual-edge length-decay temperature.

Use a predeclared grid derived from FIT geometry quantiles (P50/P75/P90/P95) plus interpretable meter-scale candidates. Do not assume S1 and S2 cutoffs are equal. Retain a self-loop W_ii=1 when a node has no retained neighbor; prevent empty rows and NaN/Inf. Keep graph edges symmetric and same-building/same-floor. For S2, apply exp(-d/tau_virt) to virtual-edge mass and document how it combines with the existing degree prior. Do not silently change the probability measure or virtual-cell normalization.

By default, share spatial parameters across C/D/E within a dataset to reduce checkpoint-specific overfitting. Fit calibration parameters T, theta_v, theta_s, alpha_prior, alpha_obs, beta/gamma per dataset × checkpoint, unless the existing model semantics require a different documented unit. Treat E's checkpoint-native trained graph layer separately from an added inference-time S1 layer.

## Joint search

Search the spatial scales and all calibration parameters in the same candidate evaluation loop on CAL-fit:
psi = (R_max_S1, R_max_S2, tau_graph, tau_virt, T, theta_v, theta_s, alpha_prior, alpha_obs, beta, gamma).

Use deterministic coarse-to-fine or successive-halving search with fixed candidate order, seed, budget, tie-break and logged pruning. If exhaustive Cartesian search is too large, reduce candidates by principled screening, not by freezing whole parameter groups sequentially. Beta candidates must include [0, .05, .10, .15, .20, .35, .50, .65, .80]. Report search budget, evaluated candidate count, failed candidates and near-optimal sensitivity.

## Objective and model selection

Do not directly add raw meters, nats and cross entropy as if they had the same scale. Do not guarantee simultaneous wins across MAE, ES, NLL, BF accuracy and tails before running the experiment. Predeclare:
1. Primary decision objective (recommended: coordinate MAE) and an explicit acceptable-degradation tolerance.
2. Dimensionless normalized or constrained secondary objectives for ES, joint NLL, BF loss/accuracy, P95/P99 and worst-site error.
3. Full Pareto frontier and all metrics for every candidate, rather than only a single hand-tuned scalar.
4. Sensitivity analyses for lambda_2d_nll = 0.02 and 0.05 if those weights are used; choose the protocol before CAL-eval.
5. Saturated BF accuracy (e.g. 100% TUJI) must not inflate a composite score; use BF cross entropy/Brier or an explicit constraint.

CAL-fit labels may select parameters. After selection files and hashes are locked, run CAL-eval once for comparison and report. CAL-eval is not TEST, but must not be used to choose the winning configuration. Test score counts must remain zero.

## Required outputs

Write new files only under fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_v1/:
- FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.json and .md
- SPATIAL_SCALE_SEARCH_SPACE.json
- JOINT_FIT_SELECTION.json
- CANDIDATE_METRICS.csv
- PARETO_FRONTIER.csv
- CAL_EVAL_REPORT.md
- INTEGRITY_REVIEW.json
- EXECUTION_MANIFEST.json
- RUN_REPORT_V17_DATASET_ADAPTIVE_JOINT_FIT.md

Keep the prior evidence immutable. Record SHA256 hashes for inputs, checkpoint, code, split and anchor order; record parameters and exact command. Add tests for cutoff symmetry, self-loops, no cross-building/floor edges, monotone kernels, unique virtual edges, normalization, no NaN/Inf, canonical-order stability, deterministic reruns and TEST guards.

The final report must state the actual FIT-only quantiles; selected dataset-specific radii and temperatures; parameter/seed stability; per-metric changes versus clean, S1, S2 and old combo; whether MAE and density metrics actually improve; and confirm that TEST remained unaccessed. If the environment lacks data/checkpoints or a runnable path, document the blocker and do not fabricate diagnostics or results.
