# V19-R0 Accuracy Recovery

## Scientific purpose

V19-R0 is an accuracy-first architecture reset after V18.5.

The experiment does not claim that V18.5 probabilistic calibration was
incorrect. It asks a more basic question: did the V18 single-model direction
discard localization capacity that was present in V9/V17?

R0 isolates two historical capabilities:

1. independent-member epistemic diversity with wide anchor consensus;
2. a full-predictive spatial-median point action aligned with Euclidean error.

No radio projection or calibration is allowed in R0.

## Frozen historical evidence motivating R0

The V18.5 UJI development selected lambda_spatial=0 and epoch 27. Its formal
calibration improved point MAE by only about 0.22 m through radio projection,
which is too small to explain the gap to historical 5-6 m systems.

V17-R3 registered an information-saturated configuration with:
- E=16 independent members;
- member support L=192;
- consensus support K=192;
- alpha=1;
- member-specific Student-t residual components.

The coherent V9 family evaluated Euclidean point error using a predictive
spatial median computed from full Student-t mixture draws.

## R0 model contract

Member 1 is the exact frozen V18.5 lambda=0, epoch-27 FIT checkpoint.

Members 2-16:
- exact V18.5 architecture and training loss;
- exact V18.5 FIT rows;
- independent fixed seeds;
- lambda_spatial=0;
- exactly 27 epochs;
- no member-specific model selection.

Inference for every member:
- evaluate the learned candidate reranker over the complete anchor dictionary;
- retain Top-L=192 member posterior mass;
- compute V17 radial predictive scale;
- member reliability proportional to inverse radial variance.

For E in {1,2,4,8,16}:
- use the first E registered members;
- alpha=1 anchor consensus;
- K=192;
- preserve each member's component location and Student-t scale;
- form the member-anchor joint mixture.

Point actions reported for every E:
- posterior mixture mean;
- full-predictive spatial median from common Student-t draws.

Probabilistic scores reported:
- coordinate-macro Energy Score;
- coordinate-macro mixture NLL.

R0 does not select an E. The E-profile is diagnostic evidence for the next
design decision.

## Mandatory controls

The experiment must also report:

1. frozen V18.5 member 1 under its original <=128 support law;
2. the same frozen member evaluated densely over all anchors;
3. E=1 Top-192 consensus;
4. paired original -> dense and dense -> Top-192 effects;
5. paired E={2,4,8,16} minus E=1 coordinate-group differences;
6. Top-192 retained posterior mass;
7. member-weight entropy and maximum member weight;
8. individual-member dense MAE and panel disagreement diagnostics.

These controls separate:
- dense/wider support effects;
- point-decision effects;
- true multi-member effects.

## Data boundary

Allowed:
- UJI trainingData.csv only;
- exact V18.5 FIT and DEVELOPMENT roles.

Forbidden:
- UJI validationData.csv;
- V18.5 calibration role for V19 member tuning;
- all external test outcomes;
- radio/temperature/scale tuning.

The separate V18.5 predictive-decision probe may read the already-consumed
V18.5 calibration role because it is explicitly post-formal and read-only. It
cannot change V18.5 or V19.

## Interpretation after R0

R0 supports continuing to V19-R1 only if the evidence shows material accuracy
recovery from the registered mechanisms.

Particularly important patterns:

- a large V18.5 mean -> predictive spatial-median improvement means the point
  decision was materially misaligned with Euclidean MAE;
- E=1 -> E=4/8/16 improvement means member epistemic diversity is important;
- a monotone or still-improving curve at E=16 means member information has not
  saturated;
- little change with E means the historical accuracy advantage came from a
  different training/retrieval protocol and R0 should stop.

Only after R0 demonstrates a strong accuracy base should V18.5 radio trust and
joint calibration be reintroduced in a later R1 experiment.
