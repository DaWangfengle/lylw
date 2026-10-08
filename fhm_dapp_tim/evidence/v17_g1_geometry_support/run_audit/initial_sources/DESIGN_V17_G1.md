# V17-G1 | Prospective geometry-support debiasing pilot

**Status: preregistered development-only mechanism experiment. Not a continuation of V9/V17 reproduction and not an outer-test confirmation.**

## Why this experiment exists

The historical V17 R3 system (25 independently trained members, E16/L192/K192/alpha1 balanced cyclic estimand) reproduced 5.725985 m coordinate-macro MAE and 4.380591 m coordinate-macro energy score. These are historical metrics on its original outer-test, not reusable tuning results. The V9 reproduction returned 6.355559 m coordinate-macro MAE; its 5.915939 m row-level MAE does not override the gate. The V9 and V17 outer-test coordinate keys differ. Historical next_stage_authorized=false must never be changed.

The scientific question is whether nonuniform physical density of training anchor support creates an implicit bias in the atomic V17 posterior against sparsely sampled coordinates when localization risk is averaged equally over coordinates. This is a new G1 hypothesis motivated by the geometry/protocol analyses; it is NOT a direct transfer of V9's maximal-affine-invariant RSS representation. A full V9/V17 fusion is expressly out of scope.

This is only a first, falsifiable paper-method candidate. JCR Q1 suitability would require real novelty relative to the literature, an explicit method and theoretical justification, multiple data domains/datasets, external independently held-out evaluation, uncertainty/calibration and robust ablations. Nothing in this pilot alone establishes that level of contribution.

## Exact model and estimand

- Original V17 R0 FROZEN_TRAINING architecture, optimizer, augmentation, loss, data transform, and epoch counts.
- Seeds 17301..17325, per-member selected epochs [65,69,56,67,62,60,52,74,79,62,65,57,64,61,50,74,59,73,60,68,52,58,53,61,56]. These were selected on historical member_fit/early_stop, not any development or outer-test metric. No fresh epoch tuning.
- New 25-member training using original R0 _fit_refit on V17 selection_refit only: 12717 rows, 595 coordinate anchors. Do NOT load historical 25 outer_refit models because they trained on development coordinates and would leak.
- V17 R3 E16/L192/K192/alpha1, beta neural=1, beta prototype=32/3, Student t df4, 64 common predictive draws, 25 fixed balanced cyclic subsets; 25 outcomes are overlapping realization estimates, not independent model repetitions.
- Baseline exactly gamma=0; corrected_probability returns the original atomic probability object, preserving the original V17 R3 numerical path.
- Let training anchors j in building b and floor f have coordinate x_j. Fixed support count n_j = 1 + number of OTHER anchors k on same building/floor with Euclidean distance <= 10m. This uses only training anchor coordinates, never prediction targets or query-label corrections.
- Correct full atomic prior by p_gamma(j|r) = p_0(j|r) * n_j^(-gamma) / sum_k p_0(k|r) * n_k^(-gamma). This is a post-hoc posterior modification, not a new neural training method and not the V9 maximal invariant.
- Gamma in {0, 0.35, 0.70}; radius fixed 10m without search. Negative control: shuffle the log-density assignments within each training building/floor using seed 18112. If the shuffled correction improves as much, no credible geometry-specific signal exists.

## Blinded evaluation protocol

1. Source file SHA256 = 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168. Reproduce exact original V17 nested geometry coordinate roles and verify against uploaded historical V17 registry coordinate-key and row-index hashes.
2. Split the original V17 development coordinates (151 distinct physical locations) into fixed calibration vs confirmation strata by building/floor and deterministic hash seed 18111. Never split rows of one coordinate across folds.
3. prepare hashes new split roles, raw input, scientific source code, design, gamma grid and allowed output paths before any G1 outcome computation. The new output directory is disjoint from historical assets.
4. Train 25 selection_refit members; verify seed/epoch/phase/registry/anchor count. No outer-test row may be passed to the model or exposed as an evaluation target.
5. calibrate evaluates only calibration rows for gamma 0/0.35/0.70. Choose the smallest coordinate-macro exact mixture mean MAE among candidates with coordinate-macro energy score <= baseline +0.02m. Ties use the smaller gamma. Freeze the calibration result and chosen gamma (self hashes and file hashes) in a new immutable record.
6. confirm (one shot) evaluates ONLY confirmation rows for exact gamma-0 baseline, frozen candidate and its density-permutation placebo, using shared V17 subsets and predictive randomness. No candidate re-search or adaptive retry after confirmation.
7. Paired resample physical coordinate groups, not individual RSS rows. Compute familywise studentized max-T 95% simultaneous upper bounds for exact_mixture_mean_mae_m and energy_score_2d_m, 5000 bootstrap repetitions, seed 18113.
8. A GO decision requires ALL: candidate − baseline macro MAE <= −0.15m, macro MAE simultaneous upper bound <0, energy simultaneous upper bound <=0 and candidate macro MAE gain strictly exceeds shuffled-density placebo gain. If any fails, STOP; do not tune against confirmation.

The bootstrap inference describes conditional statistical uncertainty on the heldout confirmation coordinate sample; it does not turn a development split into an independent cross-dataset result. The E16 reported risk is mean loss over fixed 25 balanced 16-member subsets, not a frozen deployment checkpoint.

## Guardrails

- No V9, V18, V19, original V17, or historical-reproduction science files may be edited; implement only new uji_v17_g1 modules, tests/test_uji_v17_g1.py and new G1 evidence.
- trainingData.csv only. validationData.csv forbidden. No outer_test/official/external predictions or labels used in training, calibration or confirmation.
- No access to historical V17 outer results for any tuning choice; historical frozen epoch records may be reused because they were obtained from non-outer early stopping.
- The GitHub repo includes historical result metadata, but local full model weights are not required for this stage: all development-safe weights are freshly trained.
- Never interpret next_stage_authorized=false as a missing/waived historical gate. This separately user-authorized exploratory G1 design does not alter the old historical verdict.
- CUDA deterministic-algorithm warnings should be reported without silently changing original optimizer or precision. Freeze/code hash deviations require an auditable repair report, never retroactively rewrite the preregistration.

## Required outputs

registry_v17_g1.json, 25 model weights and hash audit, selection_refit_v17_g1.csv, calibration_v17_g1.json, freeze_v17_g1.json, confirmation_v17_g1.json, RUN_REPORT_V17_G1.md. Report physical coordinate/row counts by fold, all epochs/checkpoint SHA256, exact gamma0 risk, candidate and placebo risk, paired 95% simultaneous bounds and every gate subcondition; keep failures too. Do not synthesize expected MAE values or claim external generalization.
