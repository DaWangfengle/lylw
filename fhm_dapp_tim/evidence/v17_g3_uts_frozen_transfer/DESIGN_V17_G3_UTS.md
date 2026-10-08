# V17-G3: fixed-geometry transfer to UTS provided test (2026-10-08)

## Hypothesis (fixed before any G3 UTS test score)

With V17's unchanged R0 training and R3 E16/L192/K192/alpha1 inference, the UJI-derived, frozen G1 anchor-density correction gamma=0.70 / 10m radius may improve coordinate-macro MAE and Energy Score on UTS provided external test. Same 25 UTS-trained members produce three predictions:
1. Frozen baseline gamma=0.
2. Frozen G1 gamma=0.70, p_j -> p_j*n_j^-0.70 / sum_k(p_k*n_k^-0.70), n_j=1+same-building/floor other anchors within 10m.
3. Within-building/floor density-shuffled placebo, seed 18112.

All settings and scoring methods are frozen, including the original G1 gamma/10m. No UTS test-based selection.

## Blindness provenance and proper limits

The user specifically attests that UTS, Tampere and TUJI provided test labels and performance outcomes were NEVER viewed or used in research design, despite earlier examination of *file format/schema* for adapters. Format compatibility alone does not constitute performance leakage. We therefore call this a **performance-blind, user-attested external test**; not the stronger claim that all file bytes were never opened by anyone. Earlier V18.4/V18.5 "pre-exposed" statements described adapter/schema exposure, not necessarily label/outcome exposure.

Only UTS_training.csv is loaded in preflight, prepare, training and freeze; the provided UTS_test.csv is first read during locked evaluation. Hash and content access of UTS_test.csv are forbidden in pre-test stages, despite being outcome-free operations. The irreversible external invocation lock is persisted before first test file access. Any failure after lock requires review, not silent rerun.

Historical G1 UJI decision is STOP, never retroactively overwritten. The external dataset is independent of UJI's G1 development cohort but the scientific method was selected on UJI. No UJI outer/official evaluation, Tampere or TUJI test scoring is authorized in G3.

## Experimental design

- UTS model data: existing V18.4 canonical UTS train-reference loader, with original V17 model_rss [-1,1] normalizer and original V17 R0 training/scoring. Legacy -110 dBm / missing=-1 representational collision is counted from TRAINING data and disclosed, not silently modified.
- Train all UTS_training rows: no new training-only epoch search. Same V17 25 seeds 17301..17325 and original UJI-frozen epoch counts (65,69,56,67,62,60,52,74,79,62,65,57,64,61,50,74,59,73,60,68,52,58,53,61,56). These epochs are *not* presented as UTS optimal.
- Physical geometry: x/y assumed in metres. Validate building/floor schema, train-defined AP categories, per-building/floor 10m anchors, train-only legacy float32 centered anchor identity without spatial collisions. Note that train/test may share a physical site without sharing an RSS row; this is a disclosed benchmark property rather than hidden role overlap.
- Inference: unchanged R3 E16/L192/K192/alpha1, beta_neural=1, beta_prototype=32/3, 25 overlapping balanced cyclic member subsets, 64 shared Student-t samples, frozen seed 18114 for score draws. Subsets share panel models and are not 25 statistically independent replicates.
- Primary: per-physical-test-coordinate macro 2D MAE. Secondary: coordinate-macro 2D Energy Score. Also report row-MAE/ES, placebo, per-coordinate vectors and number of physical sites. Do not equate row and macro metrics.
- 5000 coordinate-group paired bootstrap replications of ORIGINAL G1 studentized maxT family (MAE and ES), seed 18113, 95% simultaneous upper bounds.
- Frozen transfer criterion: G1 minus baseline mean MAE <= -0.15m, familywise MAE upper <0, familywise ES upper <=0, placebo minus baseline MAE >0. Both criteria and placebo are exactly specified before test and may not change in response to UTS metrics.
- Absolute performance is to be reported, even if G1 "wins"; report tradeoffs, regressions and effect concentration. Do not treat one favorable external dataset as Q1 proof.

## Workflow, irreversibility, evidence

1. preflight: only training schema/data and G1/G2 source integrity checks.
2. prepare: train-only, immutable source/recipe manifest.
3. train-member: train original V17 R0 with fixed seed/epochs for 1..25; only train rows, resumable before freeze.
4. check-members and freeze: bind 25 checkpoint hashes BEFORE accessing provided test.
5. evaluate-test ONCE: durable lock first; load provided UTS_test.csv for the first time; score baseline, fixed G1, shuffled-density placebo with exact same ensemble and predictive randomness; save paired group vectors and inference.
6. check: verify result and report, source/manifest/checkpoint hash chain, G1 STOP preserved; preserve failure logs and do not silently repeat.

Outputs strictly under fhm_dapp_tim/evidence/v17_g3_uts_frozen_transfer and fhm_dapp_tim/models/v17_g3_uts_frozen_transfer. Existing V9/V17/G1/G2 evidence and all source datasets untouched. Codex must not do Git operations; user handles all Git separately.

## Post-result interpretation / further Q1 work

- A UTS positive result motivates separately frozen Tampere/TUJI replication and mechanistic ablations, NOT optimizing gamma on UTS_test.
- If MAE fails but ES improves, disclose the tradeoff and seek a *new* development cohort before changing method.
- If UTS transfer fails, universal fixed gamma=0.70 is not supported: a new method requires fresh training-only model selection and independent external evaluation.
- Historical UJI V17 5.726m outer score uses a different reference cohort and must never be directly ranked against this UTS outcome.
