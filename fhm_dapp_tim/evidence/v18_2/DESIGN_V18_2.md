# V18.2 — Single-Member Adaptive Multi-Evidence Anchor Posterior

## 1. Scientific objective

V18.2 is designed as a journal-method model, not a deployment compression
exercise.

It therefore obeys three strict rules:

1. one trainable checkpoint at inference;
2. no V9/V17 teacher checkpoint or distilled target;
3. no multi-seed ensemble, member axis, consensus across models, or best-seed
   selection.

The goal is to retain the strongest *principles* of V9 and V17 inside one
probabilistic model.

## 2. Why V18.1 was not enough

V18.1 fixed the V18.0 self-anchor likelihood shortcut and established stable
single-member learning.  Its external diagnostics, however, showed a remaining
retrieval-ranking bottleneck:

- final nearest-anchor recall@64 was very high;
- recall@8 was materially lower;
- between-anchor uncertainty exceeded within-component uncertainty.

This means that the correct spatial region was usually present in the broad
support, but often not ranked strongly enough.

V18.1 also used a one-way inference sequence:

invariant neural Top-K -> learned reranker -> marked-radio correction.

A candidate omitted by the initial neural Top-K could never be recovered by the
radio evidence, no matter how strong the radio-map match.

V18.2 removes that asymmetry.

## 3. One model, six evidence opinions

For scan x and anchor a, V18.2 creates six normalized opinions.

### 3.1 Invariant neural opinion

A bounded positive-affine maximal-invariant Sparse AP Transformer produces

p_inv(a | x).

This retains the V9 representation principle and protects the primary learned
representation from global RSS offset/gain style.

### 3.2 Prototype radio opinion

A separate raw-RSS prototype distance produces

p_proto(a | x).

This restores the useful V17 base-learner principle that absolute radio-map
similarity participates in *candidate discovery*, not only post-retrieval
correction.

Importantly, it remains a separate evidence source.  Raw RSS is not inserted
back into the invariant Transformer representation.

### 3.3 Learned candidate-evidence opinion

The existing candidate reranker uses visibility overlap, robust anchor RSS
statistics, correlation/affine residual, standardized residual, sample count,
and neural evidence to produce

p_learn(a | x).

This is trained on leave-anchor-out geometry targets.

### 3.4 Robust marked-radio opinion

On the candidate union, V18.2 evaluates the V9-style marked radio likelihood
using:

- AP detection probability;
- robust detection-conditioned RSS center;
- robust RSS scale;
- AP reliability;
- present/missing consistency;
- Student-t marked energy;
- one shared scan-level positive-affine nuisance profile.

The result is

p_radio(a | x) proportional to exp(-E_radio(x,a)/lambda).

The nuisance profile is shared across all candidate anchors for a scan; it is
not an anchor-specific free fit.

### 3.5 Soft floor opinion

The neural floor marginal is converted to an anchor opinion

p_floor(a | x).

A floor's probability mass is divided across the candidates from that floor,
so a floor cannot gain probability merely because more of its anchors happen
to be present in the current support.

This is a soft prior, never hard floor filtering.

### 3.6 Geometry-manifold opinion

A geometry support energy is calculated from the preliminary candidate
posterior and anchor locations.  Anchors compatible with the current local
support manifold receive greater probability:

p_geo(a | x).

This brings V9 spatial structure into posterior ranking, not only the residual
transform and training target.

## 4. Multi-source candidate union

V18.2 does not obtain all candidates from a single neural Top-K.

The registered candidate discovery path is:

- invariant neural Top-32;
- prototype-radio Top-32;
- floor-aware Top-16;
- Delaunay expansion of the top eight invariant and top eight prototype seeds.

These supports are unioned and deterministically reduced to

L = 96 candidate anchors.

A fallback coarse score guarantees 96 candidates when the source lists overlap
heavily.

The wider L is then reduced by the final evidence posterior to

K = 64 predictive anchors.

This transfers the V17 principle of "broad information support before final
posterior contraction" into a single-model setting.  There is no member
support union.

## 5. Adaptive evidence reliability

Different scans do not make every evidence source equally reliable.

V18.2 therefore learns one per-scan six-way evidence gate from label-free
diagnostics:

- visible AP fraction;
- normalized entropy of each of the six source opinions;
- invariant/prototype candidate-support overlap;
- absolute profiled RSS style offset;
- absolute profiled log gain.

The gate outputs non-negative weights that sum to one.

It is initialized to equal weights, so no source is privileged before
leave-anchor-out training.

## 6. Evidence consensus

V17's final selected consensus used alpha=1.

V18.2 retains that mathematical principle, but changes the objects being
combined.

V17:
multiple model opinions -> alpha=1 member consensus.

V18.2:
multiple evidence opinions from one model -> alpha=1 evidence consensus.

For source probabilities p_s and per-scan weights w_s,

p_cons(a | x) = sum_s w_s(x) p_s(a | x).

This arithmetic opinion pool is deliberately chosen rather than a product of
experts because an anchor strongly supported by radio evidence must be able to
rescue an anchor under-ranked by the invariant neural source.

The six opinions all belong to one checkpoint or deterministic radio/geometry
memory.  This is not an ensemble.

## 7. Leave-anchor-out training law

The V18.1 correction is retained.

For a training anchor i, the positive support is the non-self Delaunay
neighborhood N(i), with fixed geometry weights q_ij.

The primary loss is

L_joint =
  - sum_j q_ij [
      log p_cons(j | x_i)
      + log p(y_i | x_i, j)
    ].

The observed self anchor is not a positive geometry component.

This prevents a training scan from winning the likelihood by collapsing onto
its own radio-map anchor.

## 8. Ranking-specific supervision

V18.1 diagnostics indicated a Top-8 ranking problem rather than a Top-64
coverage problem.

V18.2 therefore adds a small hard-negative margin term:

positive fused target score
  > strongest non-target candidate score + margin.

The positive score is the Delaunay-weighted consensus log probability.
The hard negative is the strongest candidate outside the geometry target.

This term optimizes ordering rather than increasing K.

## 9. Auxiliary losses

The full registered objective is:

L =
  L_joint
  + lambda_inv L_invariant
  + lambda_rerank L_reranker
  + lambda_h L_hierarchy
  + lambda_coord L_coordinate
  + lambda_rank L_ranking
  + lambda_domain L_domain.

The joint evidence/coordinate likelihood has coefficient one.

The invariant retrieval and learned reranker auxiliaries keep the trainable
sources individually identifiable even if the adaptive gate temporarily
downweights them.

## 10. Continuous predictive distribution

After the evidence consensus, the top K=64 anchors define one continuous
Student-t mixture:

p(y | x)
  = sum_j p_cons(j | x)
      StudentT_nu(y; mu_j(x), Sigma_j(x)).

There is exactly one residual head.

The V9 Delaunay local residual transform remains active.

## 11. What V18.2 absorbs from V9

V18.2 retains:

- bounded maximal-invariant RSS representation;
- sparse AP-set Transformer;
- Delaunay same-building/floor geometry;
- leave-anchor-out geometry joint anchor/coordinate training;
- Delaunay local residual transform;
- Student-t local predictive distribution;
- marked radio memory;
- AP detection probability;
- robust RSS center and scale;
- Student-t hurdle energy;
- shared affine nuisance profiling;
- geometry support-manifold reasoning.

It does not copy V9's multi-member prior.

## 12. What V18.2 absorbs from V17

V18.2 retains or abstracts:

- neural and absolute radio-map retrieval as complementary opinions;
- wider information support before final posterior contraction;
- alpha=1 arithmetic opinion consensus;
- AP reliability;
- robust radio-map evidence;
- probabilistic candidate correction.

It deliberately removes:

- E stochastic members;
- member reliability weights;
- member Top-L;
- member-support union;
- alpha consensus across checkpoints;
- ProjectedEnsemble;
- any teacher or distillation dependency.

The consensus principle remains; the ensemble implementation does not.

## 13. Experimental protocol

The completed V18.1 study already opened the official UJI validation set.
Therefore V18.2 cannot honestly claim it as a new blind external test.

V18.2 registers a fresh nested coordinate split inside trainingData:

all training coordinates
  = outer_refit + confirmation

outer_refit
  = selection_refit + calibration

selection_refit
  = member_fit + early_stop.

Every split is by complete physical coordinate groups and preserves every
building-floor stratum and mandatory hull support.

The confirmation coordinates are evaluated only after:

1. epoch selection;
2. selection refit;
3. calibration;
4. configuration freeze;
5. a fixed-epoch refit on outer_refit.

A consumption lock is written before the confirmation shard is loaded.

After one-shot confirmation, one final checkpoint may be refit on all
trainingData.

The official validation set may then be evaluated only as a pre-exposed
descriptive benchmark.  The result file explicitly states:

blind_confirmation_claim = false.

## 14. Calibration

As in V18.1, representation learning and probability calibration are separated.

Only two scalar inference parameters are calibrated:

- posterior temperature;
- Student-t scale multiplier.

They are selected from fixed registered grids by coordinate-macro mixture NLL.

No evidence-source weight, K, L, radio lambda, loss coefficient, model
architecture, or epoch count is selected on calibration outcomes.

## 15. Required diagnostics

Every evaluation reports:

- coordinate-macro MAE;
- Energy Score;
- continuous-mixture NLL;
- building and floor error;
- final nearest-anchor recall@8/16/32/64;
- invariant retrieval recall;
- prototype retrieval recall;
- floor-aware retrieval recall;
- 96-anchor union recall;
- six mean evidence-gate weights;
- six source entropies;
- posterior entropy;
- effective anchor count;
- top-1 mass;
- within-component uncertainty;
- between-anchor uncertainty;
- marked-radio energy;
- profiled style offset and gain;
- 50/80/90/95% predictive radial coverage.

These metrics allow a journal ablation/interpretation section to determine
whether any gain comes from candidate recovery, ranking, adaptive source
weighting, hierarchy, or uncertainty calibration.

## 16. Single-member stability

Five pre-registered independent E=1 seeds are reported on the internal
fit/early-stop roles after freeze.

All seeds are reported.

No best seed is selected and no prediction averaging is allowed.

## 17. Files

- fhm_dapp_tim/fhm_dapp/v18_2_single.py
  - adaptive multi-evidence model and objective.
- fhm_dapp_tim/fhm_dapp/uji_v18_2_experiment.py
  - nested protocol, calibration, confirmation, final refit and descriptive
    benchmark.
- tests/test_uji_v18_2_experiment.py
  - model-law and leakage-contract tests.
- V18_2_CODEX_RUN.txt
  - exact formal execution instructions.
