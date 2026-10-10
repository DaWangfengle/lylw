# G12 | Frozen posterior: ranking vs MAE point decision

**Retrospective G8 anchor-only T=.6 across original five TRAIN-internal roles.**
G5 already used geometric medians on untempered Tampere;
G12 does not claim a novel median algorithm.

| Domain / role | Mean MAE m | Discrete centre median m | Continuous t-median m | ES (all decisions) m |
|---|---:|---:|---:|---:|
| tuji/calibration | 2.390619 | 2.229682 | 2.245287 | 1.713443 |
| tuji/confirmation | 2.657774 | 2.502682 | 2.521668 | 1.867255 |
| uts/calibration | 5.380266 | 5.457483 | 5.446014 | 4.123892 |
| uts/confirmation | 5.107529 | 5.206837 | 5.166874 | 3.937965 |
| tampere/confirmation | 5.987721 | 7.015508 | 6.945454 | 4.996576 |

## Ranking stage: coordinate-site macro Top1@10m

- tuji/calibration: prototype_only=0.9124, neural_only=0.9903, fused_equal_members=0.9942, fused_radial_members=0.9942, topL_radial_members=0.9942, R3_consensus_K192=0.9942
  Decision ΔMAE vs mean: component -0.160937m; continuous -0.145332m
  Continuous median model-risk Δ -0.032696m; risk better / real MAE worse on 36 sites
- tuji/confirmation: prototype_only=0.9228, neural_only=0.9863, fused_equal_members=0.9847, fused_radial_members=0.9847, topL_radial_members=0.9847, R3_consensus_K192=0.9847
  Decision ΔMAE vs mean: component -0.155092m; continuous -0.136105m
  Continuous median model-risk Δ -0.031145m; risk better / real MAE worse on 33 sites
- uts/calibration: prototype_only=0.8180, neural_only=0.7690, fused_equal_members=0.7940, fused_radial_members=0.8095, topL_radial_members=0.8095, R3_consensus_K192=0.8095
  Decision ΔMAE vs mean: component +0.077217m; continuous +0.065748m
  Continuous median model-risk Δ -0.370583m; risk better / real MAE worse on 129 sites
- uts/confirmation: prototype_only=0.8285, neural_only=0.7532, fused_equal_members=0.8130, fused_radial_members=0.8054, topL_radial_members=0.8041, R3_consensus_K192=0.8041
  Decision ΔMAE vs mean: component +0.099308m; continuous +0.059345m
  Continuous median model-risk Δ -0.365299m; risk better / real MAE worse on 119 sites
- tampere/confirmation: prototype_only=0.9196, neural_only=0.8393, fused_equal_members=0.9196, fused_radial_members=0.9107, topL_radial_members=0.9107, R3_consensus_K192=0.9107
  Decision ΔMAE vs mean: component +1.027786m; continuous +0.957733m
  Continuous median model-risk Δ -0.490126m; risk better / real MAE worse on 77 sites

## Scientific interpretation boundaries

- Point actions share EXACTLY one frozen 960-component posterior.
- The training-free continuous median is approximate: 2x128 fixed
  Student-t draws, Weiszfeld 64 iterations, independent MC risk
  sample and the original common-random-numbers ES stream.
- Predicted model risk is NOT the realized Euclidean position MAE.
- Median may lower its own posterior expected risk while worsening
  actual positional error if the posterior is misspecified.
- Prototype/neural-only stage probabilities are analytical
  counterfactuals, not stages of a deployed cascaded predictor.
- Near10m/true-floor labels are only posthoc scientific diagnosis.
- Historical CAL/CONFIRM outcomes have ALREADY been observed;
  no new blind testing, significance-based promotion or Q1 claim.

Self hash: 30e661aa6784889d9315b318fa5f40020553d8b75891123f5b2f6f71b5a85ffc
