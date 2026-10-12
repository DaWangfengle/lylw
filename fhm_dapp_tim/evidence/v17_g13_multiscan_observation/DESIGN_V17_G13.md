# G13 — FIT multi-scan AP visibility and conditional RSS evidence

**Preregistered before any G13 outcomes, 2026-10-12.**

## Scientific goal and dataset scope

G8–G9 established that original V17 E5 error reduction mainly comes from shared anchor probability sharpening, while correct-neighbourhood anchors are typically retained. G10's four-parameter generic evidence correction yielded modest gains. G11 showed raw RSS contains useful but not uniformly better signals, and G12 showed that changing point action can help TUJI yet harm UTS. G13 tests a *physically interpretable observation model*: independently quantify AP detection and RSS intensity conditional on AP detection using **repeated FIT scans at each physical anchor**. No new E5 neural model training and no global T grid.

G13 evaluates **TUJI and UTS only**. Tampere is excluded from new experiments, all decision gates, and reported primary tables because its one-scan-per-reference acquisition does not match this study's multi-scan aim. Do not delete historical evidence. UJIIndoorLoc remains part of the final target family but needs its own compatible FIT-trained model and unviewed evaluation protocol; we shall NOT fabricate a UJI E5 checkpoint or access official UJI test data in G13.

The original TUJI G6 and UTS G4 five-member checkpoints, physical FIT/CAL/CONFIRM splits and model-seen AP masking are immutable; original checkpoints must match their historic SHA256. All previously analysed CAL/CONFIRM labels are **retrospective**, not independent blind validation.

## FIT-only observation statistics

Map each FIT RSS row to the exact original E5 anchor ID using G11's verified building/floor/XY bijection. For anchor a and AP l with FIT scan count n_a and detected count d_al:

Detection probability pi_al = (d_al + kappa * pi_global_l)/(n_a + kappa).

pi_global_l is FIT-wide Beta(0.5,0.5)-smoothed detection frequency; kappa=4 pseudo scans. Detect only normalized model RSS > -0.999, the original G11 rule. A model RSS exactly -1 cannot distinguish true missing from clipped -110 dBm, so report this limitation honestly. Do not treat absence as an observed weak RSS value.

For detected RSS only, derive global conditional AP means/variances from FIT; estimate anchor-specific means and conditional variances with kappa=4 global pseudo observations. Minimum conditional normalized RSS standard deviation is 0.06. The conditional strength uses fixed Student-t degrees of freedom 4. This is a robust **composite** likelihood under simplifying AP independence, not a proven physical joint RSS density. Report actual FIT scans per anchor, especially sites with only one observation; do not assume all site counts are large or scans independent.

For each original R3 Top-K=192 selected anchor ID compute two label-free query scores: averaged log Bernoulli AP detection/absence evidence and averaged log Student-t RSS evidence for detected APs. Each score is centred across K candidates with variance stabilization sqrt(candidate_variance + 0.2 squared), so absence of real discrimination is not artificially magnified. Never use query true XY/building/floor or a post-hoc correct-neighbourhood indicator as an inference feature.

## One tiny trainable calibration, no grid search

Preserve R3 E5/L192/K192/alpha1 component centres, Student-t df4 scales and original member probabilities. Strong original anchor-only T=0.6 is the fixed comparator. Apply two nonnegative scalar evidence coefficients in [0,1.5], initialized exactly zero:

qtheta(a|r) = softmax(log(original_consensus_c_a)/0.6 + theta_v * z_v + theta_s * z_s).

Fit **once** using ORIGINAL TUJI G6 CAL and UTS G4 CAL physical locations, equal sites within domain and equal domain contribution; Adam lr 0.05, 24 epochs, batch24, seed20261012, no early stopping or hyperparameter selection. The loss is G10's pre-existing Dirac-mixture Energy Score at member-averaged centres + 0.3 Euclidean point MAE, divided by 10m, plus 0.1 KL(qtheta || anchor T=.6). It is a **surrogate**, NOT the full continuous Student-t ES used at assessment. Freeze both coefficients before extracting or scoring any CONFIRM rows.

At historical CONFIRM compare exactly five policies: base original R3; strong anchor_t06; visibility-only fitted coefficient; strength-only fitted coefficient; joint. Both one-feature policies use the coefficients from the **same** joint fit; no re-fit. Score full continuous Student-t Energy Score with G8 common-random-number stream seed19602, 64+64 draws, physical-coordinate macro MAE, ES, P90, P95, worst location and paired site harm counts. Single primary descriptive contrast is joint minus strong anchor_t06 on TUJI and UTS. Require at least 0.15m MAE gain and ES nonworsening in **each** domain to register a descriptive encouraging signal. This screen is NOT a 95% test, prospectively independent promotion or publication claim; all evaluated labels were previously explored.

## Scientific safeguards

No model training on original FIT for the new coefficients beyond FIT-only likelihood statistics; no CONFIRM leakage into fitting; no provided/official test, no Tampere. Numeric identities check the original R3 member-anchor law, candidate ID alignment and zero-evidence equivalence to original Anchor T=.6. One run lock is created before learning coefficients; immutable coefficient record written before CONFIRM inference; no deleting lock/retrying after failure. Correct implementation bugs only prior to plan freeze, never modify old G4–G12 scientific files or experiments. User handles Git commands; Codex never runs them.

If G13 does not consistently beat Anchor T=.6, stop this simple conditional-independence evidence model rather than inflate complexity. If useful, build a physically disjoint future UJIIndoorLoc experiment with E5/E16/E25 competitive absolute MAE, full proper ES and adverse tail evaluation; only there investigate AP correlations, session/device effects and principled density-ratio or contrastive learning. JCR Q1 novelty must come from a well-supported multi-scan observation-model contribution, not rebranding common statistical components.
