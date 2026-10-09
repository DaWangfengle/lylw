# G6 | TUJI1 TRAIN-variable-only two-level V17 predictive calibration

**Pre-outcome design freeze: 2026-10-09.** G1 global support weight,
G3 UTS transfer, G4 UTS low-support gate, and G5 Tampere Bayes-action
experiments retain their original final gate decisions. G6 does NOT revise
or re-examine their confirmation/official test cases to choose a method.

## Research question

G5's independent predictive-risk average exceeded realized coordinate
MAE strongly, but that is **only a possible sign** of overdispersion:
it does NOT prove it. Student-t scale and categorical/multimember mode
mass are two different drivers of predictive spread. A fixed, small
intervention grid tests separately:

A. **Mixture-mode probability spread**: preserve all 960 component
centres/scales and transform the final R3 joint component probability
q_j(T) ∝ p_j^(1/T), with T in {1,0.8,0.6}. This is a pure mixture-weight
temperature applied AFTER original R3 top L and consensus top K. T<1
sharpens: mass concentrates on relatively dominant components. This
can change the exact mixture-mean point action (MAE) and Energy Score.
It does NOT remove spatial sample density bias by itself.

B. **Conditional spatial scale spread**: preserve all positions and
mixture weights; multiply *both coordinate-axis Student-t scales*
by s in {1,0.75,0.5}. This leaves the exact mixture-mean action
mathematically unchanged, so same T, different s MUST share essentially
the same 2D MAE; scale can alter ES and predictive credible coverage.

The cross {T,s} yields exactly 9 fixed variants, including exact bitwise
baseline T=1,s=1. No learned neural gate, no novel Student-t likelihood
claim, no floor-specific manual thresholds. These are established generic
distributional-calibration tools. A positive pilot would motivate
new domain-invariant calibration methodology, not establish Q1 novelty.

**Primary practical target:** better true 2D MAE while improving (not
sacrificing) proper probabilistic Energy Score.

## New untouched internal data

TUJI1 is the chosen new development cohort. Five physical MAT files
are shared containers of results_train and results_test. The CURRENT
V18.4 load_tuji1("train") does NOT restrict scipy.loadmat variable
decoding; G6 must NOT use it. Instead request EXACTLY
variable_names=("results_train",) for each original MAT file and reject
payloads that contain results_test. The physical MAT files are necessarily
opened/hashed; do **not** assert file-byte-level test isolation. No
results_test variable, coordinates or scores are requested or decoded.
No UJI official/outer, UTS_test, Tampere Test_rss/Test_coordinates,
G3/G4/G5 confirmation labels or any external benchmark outcome is
used in G6 model, calibration selection, or confirmation.

Canonical AP IDs, RSS dBm clipping, device mapping, 2D XY labels, one
building/floor use the unchanged V18.4 schema semantics. The G6
adapter is new and isolated: no historical loader edit. It must
match canonical loader's results_train rows/properties while ensuring
test variable is not decoded. Five TUJI device names are frozen by
TUJI_DEVICE_FILES constant from V18.4.

Deterministically partition distinct physical coordinate keys
(building,floor,x,y rounded 4 decimals) by seeded SHA256, seed19601:
70% FIT, 15% CAL, remainder CONFIRM, within each building/floor stratum.
All scans at the same physical site belong to one role; device scans
at same site must NOT leak across roles. Verify original-vs-float32-
centred key mapping injectivity, >=192 FIT anchors, nonempty CAL+CONFIRM,
all devices represented descriptively in each role; fail for unexpected
data/incompatible schema rather than relaxing contracts. Log role site/
row/device distributions before any training. Coordinates can be
spatially dependent even if unique.

## Model training and scoring

Five ORIGINAL V17 R0 members only: seeds17301–17305, fixed epochs
65,69,56,67,62. FIT scans only. Original R0 AdamW, loss, augmentation,
embedding, memory, no new early-stop. Frozen base R3:
E5/L192/K192/alpha1, df4, beta_neural1, beta_prototype32/3.
Candidates use identical member embeddings, prototype distances,
candidate centres/scales, same 64 draws x 2 streams per query,
same MC seed19602, same holdout query and single label. T=1,s=1
must exactly reproduce R3 baseline. Temperature changes weight of
the complete joint mixture of members × anchors, NOT the training
anchor density feature from G1/G4.

## CAL-only frozen selection

All 9 variants scored on CAL *physical-coordinate macro* MAE and ES.
Admissible candidate must simultaneously:
- Improve CAL baseline MAE by **at least 0.10m**.
- Improve CAL baseline Energy Score by **at least 0.02m**.
Of eligible variants, select the lowest CAL coordinate-macro MAE,
break tie by ES then name. If none eligible, frozen **baseline**.
A scale-only candidate has identical MAE and should fail the MAE
gain filter by design. NO confirmation-driven candidate selection.

Also record predicted mean radial spread and nominal 50/80/90% XY
credible-ball coverage vs empirical true location on CAL, as an
interpretability diagnostic. Coverage is marginal over the entire
mixture around its mean, estimated with fixed MC samples. It is NOT
conformal coverage, and use of true coordinates is confined to the
CAL labels; it does not feed into decisions.

## One-shot CONFIRM + fail-safe protocol

After CAL, freeze candidate name,T,s and the 5 checkpoint SHA256s.
Write immutable confirmation_v17_g6.lock BEFORE any CONFIRM scoring.
One post-lock failure is STOP, preserve lock and logs, never delete/
rerun to hunt a favorable result. Score baseline, single selected
candidate, and one-factor ablations (T,1) and (1,s) from the frozen
selected pair. Ablations are DESCRIPTIVE only; can't become new
winner after confirmation.

For candidate minus baseline, use 5000x one-sided studentized maxT
simultaneous 95% upper bounds, family of TWO confirm outcomes
coordinate-macro paired MAE and Energy Score. Exact gate:
- Selected policy nonbaseline.
- MAE delta ≤ -0.15m.
- Simultaneous upper(MAE) < 0.
- Simultaneous upper(ES) ≤ 0.
Else **NO_INTERNAL_PROMOTION**. Baseline fallback always fails.

Also preserve full coordinate paired MAE/ES loss vectors, raw
coordinate keys, descriptive win/loss, p90 error, candidate/anchor
mode vs scale-only ablations, and nominal 50/80/90% coverage.
No changes to protocol after CONFIRM. Normal coordinate bootstrap
can be optimistic under spatial dependence; discuss caveat.

## Q1 positioning and next decision

This is a cheap controlled decomposition of a possible root failure,
not a purported new Q1 method. Even a positive internal pilot is not
cross-dataset proof: 5-member TUJI-specific T/s selection could be
dataset-dependent. If no candidate works, stop global post-hoc T/s
calibration and shift priority to representation/geometry/likelihood
training changes on new development folds. If positive, next derive
a scale- and mode-calibration law using TRAIN-only reproducible signals
and check on fresh internally disjoint folds **before** consuming
any protected provided test. Prioritize absolute MAE competitiveness
and ES/tail non-degradation; do not chase gates mechanically.

All Git operations, commits, branch selection, and uploads remain with
the user. Codex runs scientific Python commands, not Git.
