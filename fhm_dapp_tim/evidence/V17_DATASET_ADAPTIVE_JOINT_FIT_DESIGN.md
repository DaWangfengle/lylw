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

Use separate, predeclared FIT-only candidate pools for R_max_S1, tau_graph, R_max_S2 and tau_virt, drawing from same-building/floor k=1/2/4/8 neighbor distributions, union-kNN edge-length quantiles and fixed meter-scale values. Do not assume S1 and S2 cutoffs are equal. Keep W symmetric and doubly stochastic with W_ii=1 for isolated nodes; construct and balance the identity-plus-edge operator in O(N+E) memory rather than materializing dense N×N matrices. Prevent empty rows and NaN/Inf. For S2, normalize distance-decayed edge weights separately at each parent anchor before global mixture normalization; do not multiply a distance kernel onto the old degree prior without per-parent renormalization.

Fit all four spatial parameters independently for every dataset × backbone × seed. D is the primary scientific reference; C/D/E must not share forced spatial parameters. Fit T, theta_v, theta_s, alpha_prior, alpha_obs and beta jointly for each same run. E's checkpoint-native trained graph diffusion remains in its frozen retrieval logits; the newly selected inference-time S1 operator is an additional, separate factor. gamma_space is fixed at exactly zero and is not a search dimension.

## Joint search

Search the spatial scales and all calibration parameters in the same candidate evaluation loop on CAL-fit:
psi = (R_max_S1, tau_graph, R_max_S2, tau_virt, T, theta_v, theta_s, alpha_prior, alpha_obs, beta), with gamma_space fixed at zero and excluded from the search vector.

Use a deterministic FIT-derived outer geometry design, cache each candidate's balanced S1 graph and S2 edge/endpoint statistics, and evaluate the full inner Cartesian grid by vectorized tensor broadcasting in fixed candidate order. The current locked grid has 5 temperatures × 3 theta_v × 3 theta_s × 5 alpha_prior × 5 alpha_obs × 9 beta = 10,125 inner candidates per outer geometry; keep chunking only as a memory-control mechanism, not as sequential parameter freezing. Beta candidates must include [0, .05, .10, .15, .20, .35, .50, .65, .80]. Report total evaluated candidates, failures, and near-optimal sensitivity.

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

The final report must state the actual FIT-only quantiles; selected dataset × backbone × seed radii and temperatures; parameter/seed stability; per-metric changes versus clean, single S1, single S2 and old combo; whether MAE and density metrics actually improve; and confirm that TEST score counts stayed zero. If the environment lacks data/checkpoints or a runnable path, document the exact blocker and do not fabricate diagnostics or results.


## Five locked corrections (2026-10-21)

1. **gamma_space is removed from the search vector and fixed at exactly zero.** This is a 2A-only study. The previous 27-run result is treated as the predeclared reason for fixing it; no gamma candidate or gamma-based model selection is allowed.
2. **S2 uses per-parent-cell measure normalization.** For edge kernel k_e=exp(-d_e/tau_virt), each endpoint allocates w_tilde(i->e)=k_e/sum_{e' incident to i} k_{e'}. The global unnormalized edge mass is beta*(w_tilde(i->e)+w_tilde(j->e))*exp(ell_e)/Z. A non-isolated parent retains physical mass 1-beta; a parent with no retained virtual edge retains physical mass 1. As tau_virt tends to infinity, this reduces exactly to 1/degree(i)+1/degree(j). Multiplying the old degree prior by a distance kernel without endpoint renormalization is forbidden.
3. **Spatial parameters are fit independently for dataset × backbone × seed.** D is the main scientific reference, while E retains its checkpoint-native training-time 8m graph diffusion and receives an independently selected inference-time S1. C/D/E must not share forced spatial parameters.
4. **Use cached outer geometry operators and vectorized inner joint calibration.** For each outer tuple (R_max_S1,tau_graph,R_max_S2,tau_virt), cache the doubly stochastic S1 graph and S2 edge set/endpoint weights. The inner tensor-broadcast grid jointly evaluates T, theta_v, theta_s, alpha_prior, alpha_obs and beta. It must contain alpha_prior=0, alpha_obs=0 and beta=0 boundaries. Beta includes [0,.05,.10,.15,.20,.35,.50,.65,.80]. No sequential freezing.
## Near-optimal sensitivity (fixed before CAL-eval)

For each lambda, report the feasible CAL-fit candidates whose normalized selection objective is within an absolute 0.01 of the best feasible objective. This tolerance is a descriptive stability window, not a new selection objective. Summarize candidate/geometry counts and finite min/median/max for each spatial and calibration parameter, separately reporting unbounded values. Save the per-run detail in JOINT_FIT_SELECTION.json and aggregate it in NEAR_OPTIMAL_SENSITIVITY.csv. Do not use CAL-eval to change the tolerance or pick among near-optimal candidates.

5. **Selection objective and BF non-degradation guard are predeclared.** The physical primary is Coordinate MAE + 0.30 L_W1. Normalize the primary and joint-NLL delta against the clean CAL-fit baseline, run lambda_2d_nll=0.02 and 0.05 separately, constrain BF cross entropy to no more than 1% degradation from the clean baseline, and require macro BF accuracy to be no lower than that same checkpoint's clean CAL-fit baseline. If no candidate improves while satisfying the constraints, report that result rather than relaxing constraints or using CAL-eval to select.

Implementation entrypoint: fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py.
Test file: tests/test_v17_dataset_adaptive_joint_fit_v1.py.


## Execution-order and geometry audit corrections (2026-10-21)

- The CAL-eval pack is created only after writing and re-reading the immutable selection and lock, then verifying selection/configuration hashes, algorithm code SHA, manifest SHA, historical-input hashes, split/anchor-order hashes, and zero TEST-score counters. Any failed check aborts before CAL-eval pack construction.
- The runner defines its own manifest_path; it must not depend on the preflight function's local variables.
- FIT geometry scale candidates use the same-building/floor k=1/2/4/8 distance distributions plus fixed physical scales. Virtual coordinates are checked for finite [N,2] values, and undirected edge lookups are canonicalized.
- Unit tests cover measure conservation, graph invariants, TEST row guards, and CAL-eval lock ordering. These tests must pass before any real-data run; remote repository edits alone are not evidence that tests executed.


## Baseline comparability and resume protection

- A historical clean baseline or old-combo result is reusable only if its frozen checkpoint SHA exactly matches the checkpoint under evaluation. Otherwise, recompute the clean baseline on the current checkpoint's CAL-fit rows and omit the mismatched old-combo comparison.
- Resuming an existing CAL-eval report requires re-verifying the selection lock, code/source manifest, historical input hashes, and the actual checkpoint file hash. A report's mere existence is not sufficient to skip a run.
- The final JOINT_FIT_SELECTION status mirrors the aggregate integrity result and cannot remain PASS when runs or locks are incomplete.


## Empty-S2 boundary and sequential dataset execution

- If an S2 cutoff retains zero virtual edges, every candidate with beta > 0 is invalid, not a valid single-S2 result that happens to equal Clean. Such rows must be excluded from feasible selection and Pareto claims; beta=0 remains a valid boundary.
- The mathematical family is shared across datasets, but FIT-derived scale pools and selected parameters are dataset-specific. Within each dataset, this version allows the spatial regularization scales to depend on the frozen backbone/seed because their interaction with the predictor can differ; this is explicitly reported as a finer-grained calibration choice, not evidence that the datasets share scales.
- Execute the three datasets as separate, ordered runs (tuji, then uts, then uji) with `run --dataset ...`, using the same output root so run artifacts remain in distinct `runs/{dataset}_seed{seed}/{C,D,E}` directories. The aggregate status is expected to remain INCOMPLETE after the first two datasets and must become PASS only after all 27 checkpoint runs and their locks are verified.
