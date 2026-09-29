# V18 — Single-Member Geometry-Radio Anchor Posterior

## Scientific goal

V18 is a **single-checkpoint** probabilistic WiFi localization model. Its
definition must remain meaningful at E=1 because there is no ensemble axis at
all. V18 absorbs mechanisms from V9 and V17 but does not consume their trained
checkpoints, selected ensemble widths, consensus parameters, or test outcomes.

## Model law

For scan x and anchor j,

p(y | x) = sum_j p(j | x) StudentT_nu(y; mu_j(x), Sigma_j(x)).

One SparseAPSetEncoder produces one scan embedding. The anchor posterior is
multimodal even though there is only one trained model.

### Representation

The neural retrieval encoder uses V9's bounded positive-affine maximal
invariant representation. Visibility is preserved explicitly. Raw absolute RSS
is intentionally not used by the neural prototype-distance path
(`prototype_metric="none"`).

### Radio evidence

Raw RSS is not discarded. For retrieved candidates, the V17-style learned
candidate reranker receives deterministic radio-memory statistics:

- anchor AP detection probability;
- robust detection-conditioned RSS median;
- robust RSS MAD;
- anchor sample count;
- visible/missing consistency and affine-residual summaries.

Thus device/style-sensitive information is used as candidate evidence rather
than contaminating the invariant neural representation.

### Geometry

Within each (building, floor) stratum, the registered anchor dictionary defines
a deterministic Delaunay one-ring. It has two roles:

1. soft retrieval targets during training;
2. a local residual transform that constrains continuous coordinate correction
   to the empirical geometry around each anchor.

No prediction label or validation outcome enters this geometry.

### Predictive residual

There is exactly one residual head and one Student-t family with nu=4.
`residual_experts` is frozen to 1. Multiple residual experts would reintroduce
an internal mixture-of-experts interpretation and are outside the V18 main
model.

## Training objective

The primary term is the continuous anchor-mixture Student-t negative
log-likelihood already implemented by `AnchorStudentTModel.training_loss`.
Auxiliary terms regularize retrieval, hierarchy, coordinate error, radio
reranking, and device robustness.

The Delaunay target replaces historical distance-kNN retrieval targets.
RSS offset/scale augmentation and a weak device-adversarial head are retained
from the V17 base learner.

## Explicitly absent

The V18 main model contains none of the following:

- multiple random-seed members at inference;
- member reliability weighting;
- alpha-consensus;
- member Top-L support;
- ensemble E;
- projected-ensemble coordinate fusion;
- residual expert ensembles.

Top-K in V18 is only the support size of **one model's anchor posterior**.

## Experimental protocol

The experiment uses only `trainingData.csv` and groups all scans at the same
physical coordinate together.

Nested roles:

member_fit + early_stop = selection_refit

selection_refit + development = outer_refit

outer_refit + outer_test = all coordinates in trainingData.csv

Mandatory convex-hull coordinates are retained in every inner reference set.
Every role retains every building-floor stratum.

1. `member_fit -> early_stop` selects only the epoch count by coordinate-macro
   continuous-mixture NLL.
2. Exactly one model is refit on `selection_refit`.
3. Development is reported once. It does not select a V18 hyperparameter.
4. A self-hashed freeze authorizes outer-test access and freezes the epoch
   count and full V18 configuration.
5. Exactly one final model is refit on `outer_refit`.
6. `outer_test` is evaluated once after freeze.
7. UJI `validationData.csv` is outside the V18 main experiment.

## Primary reported metrics

- coordinate-macro 2D MAE;
- coordinate-macro continuous-mixture NLL;
- coordinate-macro Monte-Carlo Energy Score;
- building error;
- floor error;
- anchor posterior entropy;
- predictive radial scale.

## Files

- `fhm_dapp_tim/fhm_dapp/v18_single.py`: model definition/configuration.
- `fhm_dapp_tim/fhm_dapp/uji_v18_experiment.py`: registered training,
  freezing, refit and evaluation pipeline.
- `tests/test_uji_v18_experiment.py`: single-member and leakage-control tests.
