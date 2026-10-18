# V17 Spatial Identifiability — Corrected Locked Protocol

Branch: `v17-spatial-identifiability-full-v1`

This document supersedes the initial design draft. It incorporates the five required corrections: compact zero-mass-safe mixtures; restoration of the G13 continuous field and gamma_space; clean CAL-fit calibration with a true-BF-masked joint objective; raw-score-only BF mass preservation; and an exact O→A→C→D→E loss replacement/addition chain.

## Scientific motivation

The frozen-calibration report supports two findings. Same-building-and-floor neighbor targets remove contradictory supervision: wrong building-floor neighbor mass was 93.61% for UTS and 79.01% for UJI under the old neighbor target, versus zero for the same-floor target; the paired primary-checkpoint MAE deltas were about -0.419 m and -0.346 m. By contrast, post-hoc temperature/G13 calibration did not yield consistent three-domain MAE gains. The previous Stage-2 calibration objective used an unmasked two-dimensional XY mixture NLL and could reward coordinate density contributed by a component on the wrong building-floor.

The next experiment therefore distinguishes: (i) spatial smoothing of the complete posterior logit, (ii) continuous RSS/G13 support between physical reference anchors, and (iii) joint training losses aligned to building-floor-coordinate risk.

No internal TEST role or official test labels may be loaded, scored, used for selection, or used for debugging.

## Frozen input and split requirements

Use the primary checkpoints named `best_dev_epoch_21_80`, both old `raw_neighbors` and `same_floor_neighbors` backbones, for TUJI, UTS, and UJI and paired seeds 17301, 17302, and 17303. Checkpoint binaries and reference bundles are local artifacts; verify each required file, manifest, source hash, checkpoint hash, split hash, and CAL-fit/CAL-eval site disjointness before any inference. Missing/mismatched inputs block the zero-retraining Phase F. Never silently retrain a missing frozen checkpoint.

All graph operations use canonical anchor-ID order. The historical `candidate_evidence` helper returns candidates in per-row rank order; restore canonical anchor IDs before multiplying by a graph matrix. CAL-fit selects parameters. CAL-eval is evaluated once after all choices are frozen.

## Shared calibration objective for Phase F

For each frozen backbone, fit a new clean calibration base `joint_calibrated_base` on CAL-fit. Do not initialize or inherit the old Stage-2 theta values as a fixed base. Fit `T` and both G13 coefficients `theta=[theta_visibility, theta_strength]` under the bounded domain `T in [0.25, 3.0]`, `theta_k in [0, 1.5]`, minimizing the site-equal objective

`L_CAL = mean_over_sites(mean_over_rows(joint_bf_xy_nll + 0.30 * component_center_transport_risk))`.

For each row, `q=softmax(scores/T + sum_k theta_k*z_k)` is normalized over every physical anchor. The joint NLL numerator includes only Student-t components whose building AND floor match the true row's building-floor; its posterior normalizer remains the full all-anchor posterior. The transport term is `sum_i q_i * ||mu_i-y||_2` over all anchors. Average row losses within each CAL-fit physical site first, then average sites equally. Do not replace this objective with unmasked XY NLL or with distance from the posterior mean.

Use a deterministic bounded coarse-to-fine grid (fixed in code and written to results): coarse `T` on 25 log-spaced points from 0.25 to 3.0, and each theta on `{0, .25, .5, .75, 1.0, 1.25, 1.5}`; refine around the best tuple with a predeclared local grid. Resolve ties by lower T, then lower theta_visibility, then lower theta_strength. No CAL-eval selection.

Keep the original Stage-2 parameter records as immutable historical controls: raw, temperature_only, g13_fixed_t06, and temperature_plus_g13. Report their old parameters and re-score them on the same CAL-eval metrics, but do not use them as the new spatial arms' parameter base. This explicitly quantifies the effect of fixing the calibration objective separately from the effect of adding spatial structure.

## Phase F, Structure 1: complete-logit graph diffusion

Build a symmetric doubly-stochastic same-building-and-floor graph W. Let `P(alpha)=(1-alpha)I+alpha W`, alpha in [0,1]. Complete physical-anchor logits are

`ell_i(r) = s_i_retr(r)/T* + theta_visibility* z_visibility_i(r) + theta_strength* z_strength_i(r)`.

Use the clean fitted `T*, theta*` fixed for all Structure-1 comparisons. Search only the predeclared alpha grids on CAL-fit under the same site-equal masked joint objective. Run:
- 1A: prior-only `P(alpha_prior) (scores/T*)`, observation channel unchanged;
- 1B: observation-only `P(alpha_obs) (theta*z)`, prior channel unchanged;
- 1C: one shared alpha on both channels, equivalent to diffusing the complete logit `P(alpha) ell`;
- 1D: independent alpha_prior and alpha_obs.

Use alpha grid `{0, .05, .1, .2, .35, .5, .65, .8, 1.0}`; 1D uses its Cartesian product. Report both global softmax and BF-mass-preserving normalization for every arm.

**BF-mass-preserving rule is locked:** the first argument to `layer_mass_preserving_softmax` must be the unscaled original backbone scores `scores` (T=1, theta=0), never `scores/T*`, never old Stage-2 logits, and never the clean calibrated logits. Preserve each BF class mass from `softmax(scores)`; only its within-BF conditional distribution comes from the diffused complete logits. Test this when T>1 so cross-floor temperature cannot contaminate the preserved BF masses.

For 1C, record `logsumexp(P(alpha)ell)-logsumexp(ell)` per row and verify it is nonpositive within numerical tolerance. Jensen's bound is not a guarantee of lower MAE.

## Phase F, Structure 2: G13 continuous field + measure-preserving Cell LSE

For every unique undirected same-BF graph edge `e=(i,j)`, create exactly one virtual midpoint. Its retrieval score must use the same scoring law as a physical anchor: interpolate the parent anchor Key and FIT-only RSS prototype, then calculate the neural and prototype-distance terms with the original model constants. Do not replace this score with a heuristic average of endpoint complete logits. The neural inner product is linear in Key, but G13 observation evidence is not.

### G13 midpoint statistics and evidence

Interpolate the FIT-only observation memory at the midpoint:
- `pi_vl = 0.5 * (pi_il + pi_jl)`;
- `mu_vl = 0.5 * (mu_il + mu_jl)`;
- `var_vl = 0.5 * (var_il + var_jl) + gamma_space * (1 - 0.5^2 - 0.5^2) = 0.5*(var_il+var_jl) + 0.5*gamma_space`.

`gamma_space` is nonnegative and measured in normalized RSS-squared units. Do not upper-clip the variance after adding gamma. Use the same G13 Bernoulli detection likelihood, conditional Student-t RSS likelihood (df=4), detected-AP count denominator, and per-query stabilization convention as the physical anchors.

To preserve exact beta=0 identity, calculate the visibility/strength stabilization mean and RMS from the physical-anchor candidates only, then apply those same physical-reference normalization constants to virtual raw likelihoods. This leaves every physical `z` exactly unchanged when virtual points are appended. Do not standardize virtual candidates separately and do not re-standardize parents using the expanded N+E set: either would change the beta=0 baseline. Full virtual logits are

`ell_v(r) = s_v_retr(r)/T* + theta_visibility*z_visibility_v(r) + theta_strength*z_strength_v(r)`.

The parent and virtual evidence tensors must share the same physical-reference standardization, temperature, and theta. Use the clean `joint_calibrated_base` parameters, fixed.

### Cell LSE and compact mixture

Each physical parent cell has unit base measure. For cell i with degree d_i>0:

`ell_cell_i = log((1-beta_virt)*exp(ell_i) + (beta_virt/d_i)*sum_{e incident to i} exp(ell_e))`.

For d_i=0, `ell_cell_i=ell_i`. Each shared midpoint is included in both endpoint cells. Its single global mixture weight is the sum of its two cell allocations:
`q_virtual_e = q_cell_i*w_internal(i,e) + q_cell_j*w_internal(j,e)`.
Physical weight is `q_physical_i=q_cell_i*w_internal(i,endpoint)`.

The returned mixture MUST be compact with exactly `[batch, N+E]` weights, centers, scales, and IDs: N unique physical components followed by E unique undirected midpoint components. Never return a padded `[batch, N*(1+max_degree)]` mixture. All scales must be finite and strictly positive; there are no zero-scale padding slots. Each edge joins anchors with the same BF, so either endpoint ID is a valid representative ID for BF masking.

At `beta_virt=0`, cell logits, physical weights, physical centers, and physical scales must exactly equal the corresponding no-virtual parent baseline; every virtual weight must be exactly zero. The zero-weight virtual components must be strictly masked out of joint NLL using `valid_mask = true_BF_mask & (q > 0)`, with all q<=0 terms set to `-inf`. Unit tests must compare expanded-vs-parent joint NLL and Energy Score with `torch.allclose`.

For both 2A and 2B, the primary virtual continuous center is residual interpolation:
`y_v = 0.5*(y_i+y_j)`,
`mu_v = y_v + 0.5*((mu_i-y_i)+(mu_j-y_j))`.
This equals `0.5*(mu_i+mu_j)` algebraically. Linearly interpolate continuous Student-t component scales. Pure geometric midpoint centers are recorded only as a diagnostic; they are not the primary output.

### Locked 2A / 2B comparison

- 2A: linear G13 field interpolation, `gamma_space=0`.
- 2B: spatial interpolation variance calibration, `gamma_space>=0`.

Select `(beta_virt, gamma_space)` jointly on CAL-fit for 2B using the same site-equal masked joint objective as Phase F. Fixed beta grid: `{0, .1, .2, .35, .5, .65, .8}`; fixed gamma grid in normalized RSS-squared units: `{0, .005, .01, .025, .05, .1, .2}`. Primary 2A comparison uses the same selected beta with gamma fixed at zero, isolating the variance-calibration contribution. Also report the best gamma-zero beta-only CAL-fit control as a secondary diagnostic. Evaluate all selected arms once on CAL-eval. Preserve physical and virtual component weights/centers/scales for NLL and Energy Score; do not collapse the expanded mixture to cell centers for probabilistic metrics.

For both 2A and 2B, also report the pure-geometric-center result as a diagnostic column only. It may not replace the residual-interpolated primary result or be selected from CAL-eval.

## Phase T: paired training chain O → A → C → D → E

Use the same FIT/DEV/CAL/CAL-fit/CAL-eval site split, initialization per dataset/seed, optimizer, augmentation, batch size, and full 80-epoch budget for each new training arm. Paired seeds are 17301/17302/17303. The primary checkpoint is minimum DEV MAE over epochs 21–80, with no early stopping. CAL-eval is reporting-only. Never use internal TEST or official test.

Let `L_A` be the exact existing A loss with original same-floor neighbor targets and original term weights:
`L_A = L_nll_unmasked + L_retrieval + L_hierarchy + 0.25*L_coord + all existing configured auxiliary terms`.
Use the actual existing retrieval/hierarchy/auxiliary coefficients from the run manifest unchanged; the coordinate term in this locked protocol remains `0.25*L_coord`.

- O: original neighbor targets and original loss.
- A: same-building-and-floor neighbor targets, otherwise original loss.
- C: replace, do not add to, the old unmasked NLL. Exact formula: `L_C = L_A - L_nll_unmasked + 1.0*L_joint_BF_XY`. The joint NLL denominator uses q normalized over all anchors; only the numerator density sum is masked to the true building AND floor. All other loss terms and weights, including `0.25*L_coord`, remain unchanged. Do not accidentally double NLL weight to 2.0.
- D: exact formula `L_D = L_C + 0.30*L_transport`, where `L_transport = sum_i q_i*||mu_i-y||_2` over all anchor components. This is a fixed increment; it does NOT replace or rescale `0.25*L_coord`.
- E: exact formula `L_E = L_D` evaluated with a differentiable same-BF doubly-stochastic graph diffusion layer inside the full retrieval-logit path. Use `P(alpha)` with alpha parameterized by sigmoid, initialized at alpha=0.1, and included in the optimizer. Apply to the complete retrieval scores during training and DEV/CAL inference, not to G13 alone. Save alpha with the checkpoint.

C/D/E must reuse logits and embedding from the original training forward pass; do not run a second independently augmented/dropout forward pass for the added losses. Obtain all-anchor q and continuous components from that same embedding for the joint NLL/transport terms. For the joint NLL, use true-BF-masked numerator and all-anchor posterior denominator; do not condition the inference posterior on the true class.

Reuse existing O/A checkpoints only if their source/config/split hashes and selected epochs match exactly. Otherwise train O/A in the same runner so all chain arms are paired. Keep this separate from Phase F's fail-closed zero-retraining rule.

## Metrics, artifacts, and audit

For every arm report site-macro full MAE, predicted-BF-conditioned MAE, Energy Score, masked joint BF-XY NLL, macro/row BF accuracy, per-site deltas, P95/P99/worst-site MAE, selected epoch, T/theta/alpha/beta/gamma, and all calibration/normalization mode labels. Report means and sample SD across three seeds descriptively; do not treat three seeds as independent site-level significance tests.

Write per-run RESULT.json and SITE_DIAGNOSTICS.json; trained EPOCH_HISTORY.csv and CHECKPOINT_SELECTION.csv; RUN_LEVEL_METRICS.csv, AGGREGATE_SUMMARY.csv, PAIRED_DELTAS.csv, SITE_DELTAS_ALL_RUNS.csv; INTEGRITY_REVIEW.json; and RUN_REPORT_V17_SPATIAL_STRUCTURE.md. Hash sources, data, split role keys, input checkpoints, and results. Record explicitly: internal TEST rows scored = 0; official test rows scored = 0.

Run unit/invariant tests before any expensive run. Preserve all negative results and site harms. Do not choose parameters on CAL-eval. Do not run Git commands, commit, push, or upload anything; repository publication is user-managed.
