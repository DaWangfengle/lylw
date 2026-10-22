# V17 Dataset-Adaptive Spatial Joint Fit — Iter3 Fix Design

## Research scope and evidence boundary

Use one architecture and one set of mathematical definitions on TUJI, UTS and UJI. Derive meter-valued candidates from each dataset/seed's own FIT anchor geometry; resolve temperature/theta calibration centers from the current checkpoint's CAL-fit baseline. Run dataset experiments sequentially, never tune a dataset from another dataset's geometry or metrics.

This is a frozen-backbone inference-time calibration experiment. It does not train the backbone, does not take optimizer steps, and must not read or score internal TEST or official TEST. CAL-fit is used for candidate selection; CAL-eval is reporting-only. The previous CAL-eval has already been seen during development, so it is development evidence rather than a fresh blind test.

## Confirmed issues fixed in iter3

1. The prior candidate builder mixed nearest-neighbor and union-kNN edge quantiles, multiplied them by generic dimensionless factors, then used the mixed pool for both edge cutoffs and kernel temperatures. This created cutoffs smaller than every base edge and cutoffs larger than the true maximum edge.
2. The prior selection objective used MAE + 0.30 * W1 as one primary term and gave a negative objective contribution when BF cross-entropy improved. This allowed BF-CE/NLL improvements to compensate for coordinate-MAE deterioration.
3. The prior code did not make coordinate-MAE non-degradation a feasibility condition and did not runtime-assert that the exact clean calibration anchor remained feasible.

These are implementation/design defects in the search protocol. They do not, by themselves, prove the causal origin of every CAL-eval regression or establish that a new objective will improve held-out accuracy.

## Separated spatial scale pools

Cutoff radii and kernel temperatures have different operator semantics and are now derived independently.

- R_max_S1: actual FIT union-kNN edge-distance quantiles P75, P90, P95, P100.
- R_max_S2: actual FIT union-kNN edge-distance quantiles P50, P75, P90, P95, P100, subject to R_max_S2 <= R_max_S1.
- tau_graph and finite tau_virt: the current dataset's FIT k=1/2/4/8 nearest-neighbor P50 values together with union-kNN edge P50/P75/P90 values.
- tau_virt=+infinity: explicit no-distance-decay control.

No generic multiplier is allowed to cross from the kernel-temperature pool into a cutoff-radius pool. P100 is the actual maximum edge distance, not an extrapolation such as 1.5 × P99.

The deterministic design produces 12 configurations per dataset/seed (four S1 quantile levels, three S2/temperature settings). The preflight and run-time validators recompute retained support from the actual edge list. Every geometry must retain at least one S1 edge and at least ceil(0.45 * base_union_knn_edges) S2 edges; all S2 radii must be observed edge quantiles and satisfy R_max_S2 <= R_max_S1. The 45% threshold is a declared guard for this iteration, not a claim that sparser spatial graphs are universally invalid. The search policy itself starts S2 at P50 to avoid spending this iteration on empty or extremely sparse supports.

## Baseline-centered calibration grid

For each dataset × seed × checkpoint, resolve the clean baseline (T_clean, theta_v_clean, theta_s_clean). Use the existing three-point relative neighborhoods for temperature and theta, retaining each exact center. The joint grid remains:

- alpha_prior ∈ {0, 0.05, 0.10, 0.20, 0.35, 0.65}
- alpha_obs ∈ {0, 0.10, 0.25}
- beta ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50, 0.65, 0.80}

This is 4,374 calibration candidates per geometry, or 52,488 grid candidates per run with 12 geometries, plus a separate clean-reference row. No sequential freezing is allowed.

Every geometry must contain an exact clean calibration anchor at (T_clean, theta_v_clean, theta_s_clean, alpha_prior=0, alpha_obs=0, beta=0). The runtime assertion checks one anchor per geometry and stops if any anchor fails the MAE/BF feasibility guards. Do not weaken the guard to force a run through if the exact anchor is not feasible; investigate a real metric-path mismatch instead.

## CAL-fit selection objective

Coordinate MAE is the sole primary term. W1 is a secondary term. BF-CE improvements are not rewarded; only BF-CE deterioration is penalized. The exact objective is:

\[
J_\lambda =
\frac{\mathrm{MAE}}{\max(\mathrm{MAE}_{clean},10^{-9})}
+0.10\frac{W_1-W_{1,clean}}{\max(W_{1,clean},10^{-9})}
+0.25\frac{\max(0,\mathrm{BFCE}-\mathrm{BFCE}_{clean})}{\max(\mathrm{BFCE}_{clean},0.05)}
+\lambda\frac{\mathrm{NLL}-\mathrm{NLL}_{clean}}{\max(|\mathrm{NLL}_{clean}|,1)}
\]

where lambda ∈ {0.02, 0.05}.

Feasibility additionally requires all three conditions on CAL-fit:

- coordinate_mae_m <= clean_coordinate_mae_m + 1e-6 m;
- macro BF accuracy is no worse than the clean CAL-fit baseline;
- BFCE <= 1.02 * BFCE_clean + 0.005.

The MAE guard is a selection constraint on CAL-fit, not a guarantee about CAL-eval or external generalization. Report held-out deltas honestly; never use CAL-eval to relax the guard or retune the objective.

## Run and audit requirements

Use the fresh output directory fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3. Before scoring candidates:

1. Compile the module and run tests/test_v17_dataset_adaptive_joint_fit_v1.py.
2. Run preflight; inspect all nine dataset/seed geometry sets.
3. Verify exactly 12 configurations per set; cutoff radii equal the specified edge quantiles; every S1 graph retains edges; every S2 graph meets the support threshold; R_max_S2 <= R_max_S1; and the infinity control is present.
4. Run TUJI first. Do not begin UTS or UJI until TUJI has been reviewed.
5. Inspect selected parameters, candidate metrics, support counts, and CAL-eval deltas. Internal and official TEST score counters must both remain zero.

The prior geometry summary is historical context only; current FIT diagnostics are authoritative for the iter3 candidate pool.

## Historical edge geometry (context only)

| Dataset | Anchors | Union-kNN edges | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| TUJI | 646 | 2,946 | 1.190 m | 1.787 m | 2.977 m |
| UTS | 1,052 | 5,000 | 6.061 m | 12.073 m | 21.268 m |
| UJI | 672 | 3,232 | 12.984 m | 41.445 m | 77.537 m |

These historical summary values do not replace fresh dataset/seed-specific FIT quantiles.
