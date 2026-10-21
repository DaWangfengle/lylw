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


## Five locked corrections (2026-10-21)

1. **gamma_space is removed from the search vector and fixed at exactly zero.** This is a 2A-only study. The previous 27-run result is treated as the predeclared reason for fixing it; no gamma candidate or gamma-based model selection is allowed.
2. **S2 uses per-parent-cell measure normalization.** For edge kernel k_e=exp(-d_e/tau_virt), each endpoint allocates w_tilde(i->e)=k_e/sum_{e' incident to i} k_{e'}. The global unnormalized edge mass is beta*(w_tilde(i->e)+w_tilde(j->e))*exp(ell_e)/Z. A non-isolated parent retains physical mass 1-beta; a parent with no retained virtual edge retains physical mass 1. As tau_virt tends to infinity, this reduces exactly to 1/degree(i)+1/degree(j). Multiplying the old degree prior by a distance kernel without endpoint renormalization is forbidden.
3. **Spatial parameters are fit independently for dataset × backbone × seed.** D is the main scientific reference, while E retains its checkpoint-native training-time 8m graph diffusion and receives an independently selected inference-time S1. C/D/E must not share forced spatial parameters.
4. **Use cached outer geometry operators and vectorized inner joint calibration.** For each outer tuple (R_max_S1,tau_graph,R_max_S2,tau_virt), cache the doubly stochastic S1 graph and S2 edge set/endpoint weights. The inner tensor-broadcast grid jointly evaluates T, theta_v, theta_s, alpha_prior, alpha_obs and beta. It must contain alpha_prior=0, alpha_obs=0 and beta=0 boundaries. Beta includes [0,.05,.10,.15,.20,.35,.50,.65,.80]. No sequential freezing.
5. **Selection objective and BF non-degradation guard are predeclared.** The physical primary is Coordinate MAE + 0.30 L_W1. Normalize the primary and joint-NLL delta against the clean CAL-fit baseline, run lambda_2d_nll=0.02 and 0.05 separately, constrain BF cross entropy to no more than 1% degradation from the clean baseline, and require macro BF accuracy to be no lower than that same checkpoint's clean CAL-fit baseline. If no candidate improves while satisfying the constraints, report that result rather than relaxing constraints or using CAL-eval to select.

Implementation entrypoint: fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py.
Test file: tests/test_v17_dataset_adaptive_joint_fit_v1.py.


## Execution-order and geometry audit corrections (2026-10-21)

- The CAL-eval pack is created only after writing and re-reading the immutable selection and lock, then verifying selection/configuration hashes, algorithm code SHA, manifest SHA, historical-input hashes, split/anchor-order hashes, and zero TEST-score counters. Any failed check aborts before CAL-eval pack construction.
- The runner defines its own manifest_path; it must not depend on the preflight function's local variables.
- FIT geometry scale candidates use the same-building/floor k=1/2/4/8 distance distributions plus fixed physical scales. Virtual coordinates are checked for finite [N,2] values, and undirected edge lookups are canonicalized.
- Unit tests cover measure conservation, graph invariants, TEST row guards, and CAL-eval lock ordering. These tests must pass before any real-data run; remote repository edits alone are not evidence that tests executed.
