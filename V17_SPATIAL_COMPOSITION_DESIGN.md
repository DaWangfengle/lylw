# V17 Spatial Composition — Locked Zero-Retraining Protocol

## Objective

Evaluate whether the three already implemented structures compose beneficially when applied to the existing C/D/E training checkpoints. This is an inference/calibration-only study: no optimizer, training loop, checkpoint rewrite, or TEST scoring is permitted.

## Frozen inputs

- Source/split root: fhm_dapp_tim/evidence/v17_two_stage_freeze_v1/
- Checkpoint root: fhm_dapp_tim/evidence/v17_spatial_structure_v1/
- Checkpoints: trained/<dataset>_seed<seed>/{C,D,E}/best_dev_epoch_21_80.pt
- Datasets: TUJI, UTS, UJI.
- Seeds: 17301, 17302, 17303.
- Evaluation roles: CAL-fit for all parameter selection; CAL-eval for the final locked comparison only.
- Internal TEST and official TEST score counts must remain zero.

The original source, training checkpoints, Phase-F/Phase-T evidence, and existing operators are read-only.

## Four progressive comparison groups

For every dataset × seed × C/D/E checkpoint:

1. S3 baseline: independently fit clean joint calibration (T, theta_visibility, theta_strength) on CAL-fit; evaluate joint_calibrated_base.
2. S3 + S1: evaluate 1A_global (prior-only diffusion) and 1D_global (independent prior/observation diffusion). Select alpha parameters on CAL-fit using the existing site-equal joint BF-XY NLL + 0.30 transport objective.
3. S3 + S2: evaluate the existing compact N+E midpoint-cell mixture. Select (beta, gamma_space) on CAL-fit. 2A uses the beta selected by 2B with gamma_space=0; also retain the best gamma-zero beta-only control.
4. S3 + S1 + S2: freeze the corresponding 1A_global or 1D_global alpha selected in group 2, then select (beta, gamma_space) conditionally on those diffused parent logits using CAL-fit only. Evaluate combo_1A_2A, combo_1A_2B, combo_1A_2A_beta_only_control, and the analogous three 1D methods.

This is a deliberately progressive composition protocol: the S1 alpha is fixed from its own CAL-fit selection, and S2 is then selected conditionally on that fixed S1. Do not jointly re-optimize alpha against CAL-eval. This makes the incremental contribution of the continuous field interpretable and keeps the selection protocol computationally practical.

## Exact composition semantics

- Build one model from the exact C/D/E checkpoint. E retains its own learned training-time graph layer and alpha; any post-hoc S1 operator is an additional inference-time structure and must be identified as such in the report.
- Compute physical parent logits as the existing complete law: scores / T + theta_visibility*z_visibility + theta_strength*z_strength.
- For S1, diffuse the prior channel and G13 observation channel independently with the existing symmetric doubly-stochastic same-building/same-floor graph. 1A fixes alpha_obs=0; 1D uses separately selected alpha_prior and alpha_obs.
- For S2, use the existing virtual retrieval score computed from the interpolated neural key and FIT-only RSS prototype with the original distance law; use the existing interpolated G13 features, residual-interpolated continuous centers, interpolated scales, and compact unique-edge Cell LSE.
- For S1+S2, apply S1 to the physical parent logits first. Then feed those parent logits and the normal S2 virtual logits into the existing Cell LSE. Do not replace virtual retrieval scores by endpoint-logit averages; virtual candidates are not graph vertices in this protocol.
- The expanded component order is exactly N physical parents followed by E unique undirected midpoint edges, with no padding. BF representative IDs for edges are valid only because each edge is same-building/same-floor.
- Use global softmax normalization for the requested 1A_global/1D_global comparisons. Do not silently mix BF-mass-preserving results into these arms.

## Calibration and evaluation isolation

- Fit clean (T, theta) once per checkpoint on CAL-fit with the existing locked coarse-to-fine grid and deterministic tie-break.
- Select S1 alpha values on CAL-fit using the same clean calibration parameters.
- Select standalone S2 beta/gamma on CAL-fit.
- For each fixed S1 arm, select combo beta/gamma on CAL-fit with the same objective and grids.
- Write CAL_FIT_SELECTION.json and CAL_EVAL.lock before constructing any CAL-eval prediction pack. The lock must bind selection SHA256, checkpoint SHA256, CAL-fit key hash, CAL-eval key hash, and zero TEST-score counts.
- CAL-eval is never used for parameter selection, method ranking, or early stopping.
- Use identical CAL-eval rows and the same ES Monte Carlo seed (seed + 900) across methods within each run.

## Required outputs

Under fhm_dapp_tim/evidence/v17_spatial_composition_v1/:

- EXECUTION_MANIFEST.json
- runs/<dataset>_seed<seed>/<C|D|E>/RUN.lock
- runs/<dataset>_seed<seed>/<C|D|E>/CAL_FIT_SELECTION.json
- runs/<dataset>_seed<seed>/<C|D|E>/CAL_EVAL.lock
- runs/<dataset>_seed<seed>/<C|D|E>/RESULT.json
- runs/<dataset>_seed<seed>/<C|D|E>/SITE_DIAGNOSTICS.json
- RUN_LEVEL_METRICS.csv, AGGREGATE_SUMMARY.csv, PAIRED_DELTAS.csv, SITE_DELTAS_ALL_RUNS.csv
- INTEGRITY_REVIEW.json, RUN_REPORT_V17_SPATIAL_COMPOSITION.md

Report MAE, Energy Score, joint NLL, BF accuracy, P95/P99 and worst-site MAE; include paired per-site deltas and seed mean/sample SD. Preserve every negative result. The run is complete only when all 27 checkpoint runs and all integrity checks pass.
