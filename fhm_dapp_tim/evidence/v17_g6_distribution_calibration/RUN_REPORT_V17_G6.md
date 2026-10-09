# G6 | TUJI1 TRAIN-variable-only two-axis distribution calibration

5-member frozen original V17; FIT/CAL/CONFIRM split by physical coordinates.
TUJI MAT files include results_test but only results_train is parsed. No external benchmark used.

Frozen CAL-selected policy: T0.6_S0.5

| CONFIRM variant | Macro MAE (m) | Energy Score (m) |
|---|---:|---:|
| T0.6_S0.5 | 2.657763 | 1.866291 |
| T0.6_S1 | 2.657763 | 1.867301 |
| T1_S0.5 | 2.969113 | 2.100576 |
| T1_S1 | 2.969113 | 2.102097 |

Selected-baseline MAE: -0.311351 m
Selected-baseline ES: -0.235806 m
95% familywise MAE upper: -0.239045 m
95% familywise ES upper: -0.198813 m
Gate passed: True — **PROMISING_JOINT_CALIBRATION**

The two one-factor variants are mechanistic ablations, NOT alternative selected policies.
Baseline vs selected nominal coverage and predicted/observed radius are descriptive; Student-t sample size is finite.
Joint weight temperature is NOT a demonstrated physical sample-density prior correction.
G6 is a non-novel, exploratory calibration pilot only; do not compare absolute MAE with E16/25 historical benchmarks.
No new independent external provided-test result. No G1/G3/G4/G5 result overturned.
Result hash: e068b40149201b7c9d32577e0771a1c1cf27c271a2170da555cee0dd78c292b4
