# G2 mechanism audit | post-outcome, descriptive only

**This is not a new confirmatory test and cannot authorize a model change.**

Post-outcome exploratory diagnosis of already opened G1 calibration and confirmation. NO fresh holdout, confirmatory p-values, new PASS, model selection, deployment gate, or proof of causality.

## Frozen G1 provenance

- Source G1 gamma: 0.7
- G1 formal decision: STOP_GEOMETRY_DENSITY_MECHANISM
- G1 formal MAE familywise upper: 0.012353 m
- Training anchor groups: 595
- No model weight, checkpoint, gradient, hyperparameter, outer-test, official or external data was used to produce new predictions.

## Reconstructed paired risk and influence

| Diagnostic | Calibration | Confirmation |
|---|---:|---:|
| groups | 72 | 79 |
| rows | 1529 | 1625 |
| macro_baseline_mae_m | 6.194178 | 5.495018 |
| macro_g1_mae_m | 6.072322 | 5.351249 |
| macro_gain_m | 0.121857 | 0.143770 |
| macro_es_gain_m | 0.009652 | 0.123079 |
| median_gain_m | -0.095240 | -0.000546 |
| improved_groups | 29 | 39 |
| worsened_groups | 43 | 40 |
| top_five_gain_sum_m | 19.457425 | 13.173827 |
| net_group_gain_sum_m | 8.773673 | 11.357811 |
| net_gain_concentration_top_five | 2.217706 | 1.159891 |

## Support-association diagnostics

All associations are exploratory and outcome-dependent. Partial rank correlations control for baseline error and building/floor, but are **not causal effects**.

| Geometry/proxy | Calibration Spearman | Confirmation Spearman | Cal partial rank | Conf partial rank |
|---|---:|---:|---:|---:|
| training_anchors_within_10m | -0.4313 | -0.3958 | -0.3880 | -0.3521 |
| nearest_training_anchor_m | +0.1095 | +0.1591 | +0.0030 | +0.0363 |
| training_anchor_d3_m | +0.3797 | +0.3089 | +0.2439 | +0.2276 |
| training_anchor_d5_m | +0.4205 | +0.3700 | +0.3179 | +0.2719 |
| query_rss_row_count | -0.0800 | -0.0255 | +0.1522 | -0.0054 |
| training_anchor_count_same_stratum | -0.2379 | -0.1619 | +0.0002 | +0.0000 |

## Cross-split geometry sign checks (descriptive)

| Proxy | Direction consistent with simple hypothesis on BOTH opened splits? |
|---|---|
| training_anchors_within_10m | True |
| nearest_training_anchor_m | True |
| training_anchor_d3_m | True |
| query_rss_row_count | None |

## Failure-sensitive concentration

| Removed top improving coordinates | Cal remaining gain (m) | Conf remaining gain (m) |
|---|---:|---:|
| 0 | +0.12186 | +0.14377 |
| 1 | -0.05976 | +0.08282 |
| 2 | -0.09425 | +0.04421 |
| 3 | -0.11864 | +0.02031 |
| 5 | -0.15946 | -0.02454 |
| 10 | -0.22027 | -0.09472 |

## Paper-readiness and next-step boundary

- G1 is not a generally beneficial correction: report gains and losses by physical coordinate.
- If support-density association lacks consistent direction or vanishes after risk controls, **do not claim geometric sampling density caused the gain**.
- Even consistent association is observational, potentially confounded and mathematically coupled to baseline error; robust causal evidence is still missing.
- The true query coordinate was used only after prediction, so its local anchor density is NOT an online gating feature.
- New adaptive gates or gamma changes cannot be validated on the already-opened G1 confirmation. Any new hypothesis must be fixed before an independent, compatible evaluation cohort is used.
- Independent benchmark evaluation and detailed prior-art assessment remain necessary before Q1 submission claims.

Evidence SHA256: 2063b8d3402d789e03965d2435f72d8661eeab3dd26bf7f5a6c77246050d0bfb
