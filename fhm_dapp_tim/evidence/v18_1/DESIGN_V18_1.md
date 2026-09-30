# V18.1 — Single-Member Leave-Anchor-Out Geometry-Radio Posterior

## Status

V18.1 is a new registered experiment line. It does not reuse the V18.0 outer
split as a blind test because that split has already been inspected. All rows in
trainingData.csv are now treated as development material. The previously
unopened official validation file is reserved as the only one-shot external
confirmation set.

## Why V18.1 exists

V18.0 established that a one-checkpoint probabilistic anchor model is viable,
but its training trajectory exposed a mismatch:

- training NLL kept improving;
- held-out-coordinate MAE and Energy Score generally improved later;
- held-out-coordinate NLL worsened as posterior entropy and radial scale shrank;
- NLL-only epoch selection therefore froze epoch 1 even though localization
  quality continued to improve.

The main mechanism was that the V18.0 mixture likelihood could exploit the true
training anchor because that anchor was forcibly present in the candidate set.
That is not the task faced at an unseen coordinate.

V18.1 changes the training law rather than merely retuning learning rate or K.

## Single-member contract

The deployed model has:

- exactly one checkpoint;
- exactly one residual expert;
- no member axis;
- no ensemble weighting;
- no alpha-consensus;
- no ProjectedEnsemble;
- no seed selection at inference.

The predictive distribution remains multimodal over anchors:

p(y|x) = sum_j p(j|x) StudentT_nu(y; mu_j(x), Sigma_j(x)).

Multimodality is part of one probabilistic model and is not an ensemble.

## 1. Invariant neural representation

The Sparse AP Set Transformer uses the bounded positive-affine maximal
invariant representation. Visibility is retained explicitly.

Raw RSS is not used through the old prototype L2 retrieval path:

prototype_metric = none

This keeps device-style positive-affine nuisance out of the primary learned
location representation.

## 2. Strict leave-self-out Delaunay geometry

For each observed training anchor i, V18.1 builds a same-building/floor
Delaunay graph and removes i itself from its geometry target.

If q_ij is the normalized non-self Delaunay target weight, the primary loss is

L_joint =
  - sum_j q_ij [
      log p_theta(j | x_i)
      + log p_theta(y_i | x_i, j)
    ].

This is the V9-style joint anchor/coordinate log score.

It differs fundamentally from the V18.0 training mixture

-log sum_j p(j|x) p(y|x,j),

which could be dominated by a sharply fitted true self-anchor.

V18.1 fails closed if any building-floor stratum would require a positive
self-target fallback.

## 3. Leave-anchor-out learned radio reranker

The learned candidate reranker is also trained without the true anchor.

Natural neural Top-K retrieval is measured before target forcing. The training
candidate set may then force non-self Delaunay neighbours only so the supervised
reranker target remains defined.

The run reports natural Delaunay target recall mass so a weak coarse retriever
cannot be hidden by forced training support.

## 4. Explicit robust marked-radio posterior

At inference the candidate posterior is not only an MLP output.

For neural/reranked candidate logits z_j, V18.1 computes a deterministic marked
radio energy E_radio(x,j) using:

- AP detection probability;
- robust detection-conditioned RSS center;
- robust RSS scale;
- present/missing consistency;
- Student-t marked energy;
- one scan-level profiled affine nuisance (offset and positive gain).

The posterior correction is

z'_j = z_j - E_radio(x,j) / lambda_radio

with lambda_radio fixed in the registered model configuration.

The affine nuisance is shared across all candidate anchors for a scan. Only the
two nuisance variables are optimized during profiling; model parameters and the
candidate prior are detached. This is deterministic inference, not a hidden
second learner.

## 5. Epoch selection uses localization quality

Epochs before minimum_epochs are ineligible for final selection.

The primary selection statistic is coordinate-macro Monte-Carlo Energy Score.
After training stops, the selected epoch is chosen as:

1. find the minimum eligible Energy Score;
2. keep epochs within the registered 0.05 m Energy Score margin;
3. among them choose the lowest coordinate-macro MAE;
4. then lower Energy Score and earlier epoch break remaining ties.

This prevents probability-density calibration alone from terminating location
learning at epoch 1.

## 6. Calibration is separate from representation learning

A physically separate calibration coordinate role is never used for gradient
training or epoch selection.

After the selection checkpoint is fixed, V18.1 fits only two scalar inference
parameters on calibration coordinates:

- posterior temperature T;
- Student-t scale multiplier s.

The registered grids are finite and frozen before results are observed.

Calibration minimizes coordinate-macro mixture NLL. The entire grid surface and
whether the selected value lies on a grid boundary are recorded.

The backbone, reranker, radio lambda, K, geometry, loss weights, seed and epoch
count are not tuned on calibration outcomes.

The selected T and s are transferred unchanged to the final model refit on all
trainingData rows.

## 7. Physical role isolation

The prepare stage reads trainingData.csv once and materializes only four
verified role shards:

- member_fit;
- early_stop;
- selection_refit;
- calibration.

No full_train shard exists before freeze.

train-selection loads only member_fit, early_stop and selection_refit shards.

calibrate-freeze loads only the calibration shard plus the frozen selection
checkpoint.

After freeze, train-final reconstructs the full development dataset in memory
by combining the verified selection_refit and calibration shards. It does not
reopen the original CSV.

## 8. Sealed external confirmation

The official validation file is not opened or hashed by prepare,
train-selection, calibrate-freeze or train-final.

The external command requires:

- a valid registry;
- a valid freeze;
- a final checkpoint linked to that freeze.

Immediately before the first external file read, the command creates an
exclusive consumption lock.

If evaluation crashes after the lock is written, the external set is considered
consumed and must not be silently rerun.

Only after the lock exists does the program load and hash the external file.

## 9. Deterministic runtime

V18.1 sets a strict deterministic training runtime:

- CUBLAS_WORKSPACE_CONFIG=:4096:8;
- flash SDPA disabled;
- memory-efficient SDPA disabled;
- math SDPA enabled;
- torch deterministic algorithms enabled with warn_only=False;
- cuDNN benchmark disabled;
- cuDNN deterministic enabled.

If a remaining nondeterministic CUDA operation is encountered, the formal run
must fail rather than silently downgrade the reproducibility contract.

## 10. Diagnostics

In addition to coordinate-macro MAE, Energy Score and mixture NLL, V18.1
reports:

- building and floor error;
- anchor entropy;
- effective anchor count;
- top-1 posterior mass;
- predictive radial scale;
- within-component uncertainty;
- between-anchor uncertainty;
- nearest same-stratum anchor recall@8/16/32/64;
- mean posterior radio energy;
- profiled style offset/gain;
- error versus predicted radial-scale correlation;
- empirical radial coverage at 50/80/90/95%;
- corresponding mean predictive radii.

These diagnostics distinguish retrieval failure, posterior overconfidence,
local residual uncertainty and calibration failure.

## Files

- fhm_dapp_tim/fhm_dapp/v18_1_single.py
  - V18.1 model law, leave-anchor-out objective and explicit radio correction.
- fhm_dapp_tim/fhm_dapp/uji_v18_1_experiment.py
  - role preparation, training, calibration, freeze, final refit and one-shot
    external evaluation.
- tests/test_uji_v18_1_experiment.py
  - scientific-contract unit tests.
- V18_1_CODEX_RUN.txt
  - exact execution instructions for the formal run.

## 11. Independent E=1 stability panel

Single-member does not mean one lucky initialization should define the method.

After freeze, V18.1 optionally runs five pre-registered seeds:

18311, 18312, 18313, 18314, 18315.

Each seed trains an independent E=1 model on member_fit for the frozen epoch
count and evaluates only the internal early_stop role using the already frozen
calibration scalars.

All five runs are reported. No best seed is selected, no predictions are
averaged, and no ensemble is constructed.

The stability report gives mean, sample standard deviation, minimum and maximum
for MAE, Energy Score, mixture NLL, recall@64, entropy and radial scale.

The stability panel is descriptive only and is forbidden from changing the
formal frozen model or the external evaluation.
