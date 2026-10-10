# G7 | Frozen G6 calibration transfer stress test

### Scientific status

No new training. G4 UTS and G5 Tampere previously seen internal role labels;
**this is NOT a blind external transfer claim**.
TUJI-derived T=0.6,s=0.5 was fixed before all G7 target scoring.
G7 target labels NEVER select a new T or s. E5/L192/K192/alpha1 frozen.

| Domain / historic role | Action | Macro MAE (m) | Energy Score (m) |
|---|---|---:|---:|
| uts / calibration | T1_S1 | 7.155118 | 5.024129 |
| uts / calibration | T0.6_S0.5 | 5.380405 | 4.166231 |
| uts / calibration | T0.6_S1 | 5.380405 | 4.140842 |
| uts / calibration | T1_S0.5 | 7.155118 | 5.031197 |
| uts / confirmation | T1_S1 | 6.819208 | 4.786196 |
| uts / confirmation | T0.6_S0.5 | 5.107270 | 3.983257 |
| uts / confirmation | T0.6_S1 | 5.107270 | 3.954783 |
| uts / confirmation | T1_S0.5 | 6.819208 | 4.794379 |
| tampere / confirmation | T1_S1 | 7.202666 | 5.212627 |
| tampere / confirmation | T0.6_S0.5 | 5.986884 | 5.019885 |
| tampere / confirmation | T0.6_S1 | 5.986884 | 4.968090 |
| tampere / confirmation | T1_S0.5 | 7.202666 | 5.236285 |

### Fixed transfer decision

- uts: candidate-baseline MAE=-1.711938m, ES=-0.802939m; MAE threshold=True, ES threshold=True
- tampere: candidate-baseline MAE=-1.215782m, ES=-0.192742m; MAE threshold=True, ES threshold=True

**TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL**

T-only/s-only are predeclared ablations, never replacements.
Historical labels were exposed in G4/G5, so even a positive
G7 gate is useful directional evidence, NOT independent validation.
No access to UTS_test, Tampere Test or official UJI/TUJI tests.
Do not claim JCR Q1 originality for fixed mixture temperature.
Immutable result: 689c62a942ff85fcab71c8ea307a90895bc3ea2985041233ea7d614ce893a267
