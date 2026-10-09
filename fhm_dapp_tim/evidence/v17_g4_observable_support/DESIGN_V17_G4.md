# V17-G4 — UTS-training-only observable posterior support pilot

**Freeze date:** 2026-10-09. **Status:** designed before G4 calibration/confirmation outcomes.

## Motivation, competing explanations, and hypothesis

Historical G1 UJI confirmation was **STOP** despite small coordinate-macro improvement. G2 found retrospective label-coordinate support associations; these cannot be used as an online input and are not causal. G3 showed frozen global gamma=0.70 / 10m on provided UTS test gave MAE +0.07148m, Energy Score +0.15803m (candidate-minus-baseline), both undesirable; G3 external STOP remains immutable. The G3 UTS provided test has ALREADY been opened. Its records MUST NOT become a new selection/validation cohort for G4.

**G4 scientific hypothesis:** the **query-observable posterior-weighted train-anchor log-density** may be negatively associated with fixed G1's per-site MAE gain; a low-support selective policy calibrated entirely on an internal development fold might avoid the harm of applying gamma=0.70 uniformly. This is NOT yet a claim of causality or external transfer. Important competing explanation: expected support merely proxies posterior entropy or baseline error; the association and any selected policy must survive a separate internal confirmation.

Observable, prior to seeing query true coordinates:

\[
s(x) = \sum_j \bar p_0(j\mid x)\,\log n_j,\quad
n_j=1+\#\{k\ne j:\text{same building/floor and }\|\mu_k-\mu_j\|\le10m\}
\]

Here \(\bar p_0\) is the **mean of five baseline member atomic priors**. A **low** value is an inferred low-support location. Unlike G2's density measured around the true query coordinate, this uses only RSS, trained member logits and TRAINING-anchor geometry.

Hypothesis H1 (association): low \(s(x)\) indicates greater gain of G1 relative to baseline, measured per physically grouped calibration and then confirmation coordinate. Record Spearman separately; no p-value interpreted as independent given spatial dependence.
Hypothesis H2 (policy): apply original frozen gamma=0.70 correction **only when** \(s(x)\le t\) where threshold \(t\) is selected from three predeclared development quantiles; elsewhere exactly identity gamma=0. Compare with baseline, global gamma=.70 and within-stratum shuffled-density placebo. Calibrate on one group set; one-shot confirmation on another.
H1 can fail while H2 happens to pass due to noise; report both separately. H2 is not used as proof of mechanism. Relative absolute point MAE priority retained.

## Roles / bias constraints

ONLY the **UTS_training.csv** is allowed. Split by **physical coordinate key** (building/floor/x/y at 4 decimals, using the frozen V17 loader and canonical UTS AP mapping). Within every building/floor stratum, sort physical keys by seeded SHA256 and assign about 65% of sites to FIT, half the rest to CALIBRATION, and remainder to CONFIRMATION. Every scan of one physical site belongs to exactly one role; no row leakage. Record role key lists and source hashes before fitting any member.

No official UTS_test, Tampere/TUJI provided tests, UJI official/outer, or G3 coordinate outcome artifacts are inputs to G4 model selection. G3's already-known negative conclusion motivates the research question, but G4 cannot produce a fresh independent UTS external validation. Remaining other-dataset provided tests are preserved for a subsequent independently frozen full-panel stage.

## Fixed methods, budget, and candidate set

- Retain original V17 R0 model, augmentations, optimizer, loss, 10m G1 geometry law and R3 continuous Student-t mixture scorer. NO network architecture modifications.
- **5** original R0 seeds 17301..17305 and fixed historical epochs (65,69,56,67,62). Train on FIT scans only, not calibration/confirmation; no tuning epochs on confirmation.
- Screening prediction is **E=5, L=192, K=192, alpha=1.0**, same R3 2D joint scoring formulation; it is intentionally NOT equivalent to the original 25-member E16 full V17. Compare only treatments WITHIN this five-member panel.
- Atomic prior \(p_0\) = softmax(neural_logit - 32/3 * prototype_L2). Use baseline, global gamma=.7, within-stratum density shuffle as negative control, and gated policy variants using three quantile cutoffs q=0.25/0.50/0.75 on **CALIBRATION coordinate-group mean** \(s\).
- All policies use identical model embeddings, radial member weights, Student-t random draws and query data. Different schemes cannot alter the training models.
- Threshold \(t\) calculation is based only on observed \(s(x)\), not labels. Select one candidate using **calibration labels**: lowest coordinate-macro MAE among candidates with at least 0.05m improvement over baseline AND Energy Score no more than +0.02m worse than baseline. Global gamma and three gated candidates eligible; shuffled placebo is NEVER eligible to win. If none eligible, freeze baseline. Once selected, threshold/variant are permanently frozen before confirmation.
- Confirmation evaluates baseline, global gamma, shuffled placebo, and the **single** frozen candidate. Selected-baseline MAE threshold >=0.15m gain, studentized 5000x coordinate-macro paired resampling one-sided familywise MAE upper<0 and ES upper<=0. Selected baseline = automatic **NO_INTERNAL_PROMOTION**. No tuning post-confirm.
- Report calibration and confirmation independent per-coordinate effect vectors; include all floors, positive/negative group counts, tail gains, mean/median MAE/ES, role overlap zero, descriptive Spearman for \(s\) vs fixed gamma gain. Note spatial correlation reduces bootstrap effective sample size.
- A negative internal result falsifies this simple observable support gate as presently specified; a positive result licenses only further *new-development* full-panel and new independent dataset verification, not Q1 publication claims or reuse of UTS_test.

## Interpretation / anticipated limitations

This pilot makes an initial **testable, cheap online-control hypothesis** from G2/G3, not full causal reference-density intervention. A true causal density study would require matched training-anchor ablations with model retraining and equal scan/count/coverage controls; that is a **separate** future G4B phase, not asserted here.

UJI-selected epoch counts may be suboptimal on UTS; they are held fixed intentionally to avoid confounding methods with training selection. Floor-ID numeric mapping is lexicographic in existing adapter: never interpret code 14 as physical floor 14. Overlapping 5-member predictions are not 5 independent datasets, and coordinate sites on same floors are spatially dependent. No claim of G1 historical authorization.

**Scientific priority over bureaucracy:** no expansive audit system, no new huge model, no testing of more gamma values, no post-outcome thresholds, no git commands in Codex instruction. The only required outputs are split manifest, five checkpoints (kept locally), calibration+selection, one-shot confirmation, one report, and short execution log.
