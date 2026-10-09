# V17-G5-A | Tampere TRAIN-only Bayes action alignment pilot

Protocol date 2026-10-09; created before any G5 numeric outcome.
This is a small falsifiable mechanism experiment, NOT a Q1-level novelty claim.

## Motivation

V17 continuous mixture outputs a predictive law but its point action is the
posterior mean, which minimizes squared Euclidean loss. For Euclidean distance
loss the Bayes action is the distributional spatial median. Test whether this
loss-action mismatch helps explain current MAE limitations using the current
V17/R3 continuous mixture.

Three prespecified point actions with one unchanged distribution:
1. V17 exact mixture mean, equal to sum(component weights * component centres).
2. SECONDARY weighted geometric median of component centres; deliberately
   ignores scale of continuous Student-t components.
3. PRIMARY approximate geometric median of the full Student-t mixture, using
   two independent 128 draws/scan (256 total) and the pre-existing V9
   spatial_median_from_draws/Weiszfeld helper (64 iterations, tolerance 1e-5m).

The V9 helper already implements this standard method. It is NOT a new
theoretical invention, and a positive test is not a publishable algorithm by
itself. Student-t degrees of freedom fixed at 4.

Energy Score is identical for all three point actions BY CONSTRUCTION,
because none alters the predictive distribution. Compute baseline ES once as
a descriptive distribution metric; do not demand improvement in ES in the
action-only gate, and do not claim ES improved.

## One new, train-only data split

Tampere Training_rss.csv and Training_coordinates.csv ONLY, via existing
V18.4 canonical adapter. XY are metric local coordinates; z height maps to
a floor label, NOT a third Euclidean target dimension. Unknown handset IDs
are consistently treated as one device category by original R0 mapping.

Site split seeded SHA256 independently in each floor/building stratum:
75% physically distinct sites FIT, remaining 25% CONFIRM. All scans at one
building/floor/x/y key (4 decimals) must stay together; at least 2 sites
per role/stratum, and >=192 FIT anchors overall. R0 mean-centering/float32
coordinate collisions must fail before any training.

Freeze complete manifest with source SHA256s, full role coordinate lists and
numerical design before training. FIXED seeds 17301–17305 and epochs
65, 69, 56, 67, 62 copied from original historical R0. Train exactly five
separate original V17 R0 models on FIT scans ONLY with same augmentation,
optimizer, retrieval, residual mixture and frozen configuration. No model
tuning/epoch selection or validation predictions on confirmation.

The official Tampere Test_rss/Test_coordinates and TUJI test, UJI
official/outer and UTS_test.csv are forbidden. G4's already observed UTS
internal calibration/confirmation cannot act as independent G5 evidence.
Five-member E5 baseline is NOT directly comparable with original E16/25
full V17 absolute MAE. Tampere dataset is used for internal confirmation,
not externally blind testing yet.

## Fixed predictive law

Same five member models, same training anchors and exact R3 E5/L192/K192/
alpha=1 mixture (neural beta=1, prototype beta=32/3, Student-t df=4).
The only modified object is the 2D point action selected from that law.
No gamma, G1 geometric correction or q25 gating at all in this experiment.

Numerical stability controls:
- frozen spatial_median_from_draws from V9, two 128-draw streams,
  ACTION seed=19503, max64 Weiszfeld steps, 1e-5m tolerance;
- independent predictive Euclidean risk verification with two different
  128-draw streams and seed=19504, using no target labels;
- weighted-centres Weiszfeld secondary median with exactly the same iteration
  budget and tolerance;
- one common V17 original Energy Score calculation seed=19505, 64 draws/
  stream, identical across actions;
- precompute all action randomness before using confirmation labels;
  report baseline, secondary and primary per-coordinate losses and shifts.

## Frozen one-shot statistical gate

5 checkpoints plus hashes and manifest are frozen before first confirmation
score. Write immutable internal_confirmation.lock before label scoring. If
scoring fails after lock, STOP; preserve all records without silent rescore.

Two predeclared coordinate-macro paired differences (candidate − baseline):
D1 = continuous Student-t spatial median − exact mixture mean (PRIMARY)
D2 = component-centre median − exact mixture mean (SECONDARY).
Studentized maxT, 95% one-sided simultaneous upper, 5000 paired coordinate
group resamples, seed 19502, one 2-contrast family.

PROMISING_ACTION_ALIGNMENT iff BOTH: primary D1 <= -0.15m AND familywise
upper(D1) < 0. Otherwise NO_INTERNAL_PROMOTION.

Record descriptive per-floor changes, paired win/loss/tails, predictive
expected distance, action displacement, one common ES. Do not tune any
threshold, iterations, draw count or seed using confirmation outcomes.
Physical coordinate groups are spatially correlated, so simple coordinate
bootstrap may overstate effective sample size.

After G5, if FAIL, consider distribution calibration or controlled density
interventions instead of further median tuning. If PASS, plan a new
independent full-panel/fresh-dataset verification. No Q1 claim from this
small training-only five-member pilot.

No Git commands in Codex instructions: user manages all Git actions.
Keep checkpoints locally; compact evidence/logs suffice for GitHub review.
