# V17-G4 | UTS training-only observable support pilot

UTS_training.csv **only**. UTS_test.csv, Tampere/TUJI provided tests and UJI outer/official are forbidden for this pilot.

**Design:** same 5 frozen V17 members, 3 physically disjoint coordinate roles; query-observable posterior expected log anchor support; calibration-selected threshold/identity; internal one-shot confirmation.

Calibration selection: **gate_q25**; threshold=2.180776695410411
Calibration fixed G1 support-vs-gain Spearman: -0.16290706983785672
Confirmation fixed G1 support-vs-gain Spearman: -0.22239981740605774
Confirmation entropy-vs-gain Spearman (competing proxy): -0.1451727386188826
Confirmation visible-AP-count-vs-gain Spearman (competing proxy): -0.052492309037198974

| Confirmation variant | Coordinate macro MAE (m) | Energy Score (m) |
|---|---:|---:|
| baseline | 6.819208 | 4.788740 |
| g1 | 6.818034 | 4.848047 |
| gate_q25 | 6.688934 | 4.741960 |
| placebo | 6.905431 | 4.852990 |

Selected policy minus baseline MAE: -0.130274m
Selected policy minus baseline ES: -0.046780m
One-shot internal gate: False; **NO_INTERNAL_PROMOTION**

All thresholds selected only on CALIBRATION coordinates, not confirmation labels. Rank correlation is descriptive and not causal proof.
Five members are a screening pilot, not directly comparable to 25-member G3.
This study does not rehabilitate G1's original STOP or reverse G3 UTS external STOP.
A positive internal pilot requires new independent verification; no further UTS_test tuning or scoring is authorized.

Result self hash: f0bd0416fa6d2c7c8665b760a4e8eb687085bdb9c478af4fcb22338ae433a765
