# V18.4 — Spatially Regularized Trust-Region Posterior Localization

## 1. Scientific status

V18.4 is a new single-checkpoint method generation motivated by two observed
failure modes:

- V18.2: an unconstrained learned evidence gate collapsed onto one learned
  source;
- V18.3: hard positive evidence-family mass floors prevented collapse but
  produced a spatially diffuse posterior.

A preregistered V18.4 decoder probe on already-open V18.3 development roles
then established:

- component-center weighted geometric median materially improves Euclidean
  point-location MAE;
- Top-8 contraction also materially improves MAE.

Therefore V18.4 treats loss/decision mismatch and posterior-tail dilution as
coexisting mechanisms.

V18.4 does not continue or rescue V18.3. V18.3 confirmation remains
unconsumed.

## 2. Method thesis

V18.4 is not an evidence-voting model.

It is organized as

Support Sufficiency
    -> Spatial Posterior Learning
    -> Reliability-Bounded Trust-Region Correction
    -> Loss-Aligned Spatial Decision.

The principal posterior is a single learned discriminative anchor posterior.
Radio-map and geometry information are structured correction energies, not
peer probability voters.

## 3. Deployment contract

The deployed model has:

- one checkpoint;
- one bounded maximal-invariant Sparse-AP Transformer;
- one learned anchor posterior;
- one Student-t residual head;
- residual_experts = 1;
- no teacher;
- no distillation;
- no stochastic member axis;
- no multi-seed prediction averaging;
- no learned evidence gate;
- no mandatory evidence-family probability floor.

Multiple training seeds are stability diagnostics only.

### 3.1 Frozen learned-backbone configuration

The registered learned backbone is:

- d_model = 128;
- attention heads = 4;
- Transformer layers = 2;
- max_visible = 64;
- dropout = 0.10;
- bounded maximal-invariant scan representation;
- one summary-mode learned candidate reranker;
- no prototype metric hidden inside neural retrieval;
- Student-t degrees of freedom = 4;
- minimum residual scale = 0.25 m;
- local anisotropic Delaunay residual transform, globally capped at 30 m.

Each deterministic residual transform is multiplied by min(1, 30/sigma_max)
before use. Its axes and singular-value ratio are preserved. The residual-head
direction is constrained to the unit Euclidean ball, so every component-center
displacement has Euclidean norm <=30 m (apart from tiny floating-point roundoff).
This cap is specific to V18.4; earlier model generations retain their semantics.

Training uses:

- batch size = 64;
- AdamW learning rate = 3e-4;
- weight decay = 1e-4;
- gradient clip = 5;
- maximum epochs = 60;
- epochs 1--11 ineligible for selection;
- RSS offset augmentation = 4 dB standard deviation;
- RSS gain augmentation standard deviation = 0.08;
- AP dropout = 0.10.

These values are frozen before V18.4 UJI development outcomes are observed.

## 4. Multi-dataset scientific protocol

### 4.1 Canonical data layer

V18.4 uses the new canonical adapters in v18_4_datasets.py.

Every evaluation split is projected into a feature dictionary learned only from
its corresponding training data. Evaluation data never expand the AP
dictionary or categorical mappings.

The legacy data.py remains unchanged so earlier experiments retain their
historical preprocessing.

### 4.2 Training-data roles

For each dataset training partition:

training source = selection_refit + calibration

selection_refit = fit + development

Registered fractions:

- development: 15% of physical-coordinate groups;
- calibration: 15%;
- fit: remainder.

Roles are complete coordinate groups, stratified by building-floor.

All same-building/floor convex-hull support coordinates remain in fit.

### 4.2.1 Formal artifact/source binding

The UJI formal development chain `prepare -> develop -> calibrate-freeze` is
executed on one unchanged Git commit. The prepare registry records the exact
40-character source commit SHA and the SHA256 of `trainingData.csv` plus each
role's row-index hash. Later stages fail closed if the current source commit
changes.

Generated evidence/model artifacts may remain untracked during the run, but no
tracked source modification, pull, commit, cherry-pick, or branch change is
allowed between prepare and method freeze.

### 4.3 Dataset roles

UJIIndoorLoc is the global method-development benchmark. Its official
validation file has already been exposed by earlier versions and is descriptive
only.

UTSIndoorLoc, Tampere, and TUJI1 provided test splits were pre-exposed during
early V18.4 adapter/schema compatibility validation: their RSS/coordinate
content was canonicalized and summarized by CI before the formal method freeze.

Therefore they are NOT blind confirmation sets and V18.4 must never describe
their later evaluation as first semantic access.  The protected claim is
narrower and auditable: their performance outcomes are excluded from V18.4
method/hyperparameter selection, and formal performance scoring is performed
only after the UJI method law is frozen.

After this audit correction, default dataset CI is training-only and no longer
re-opens those evaluation splits.

### 4.4 TUJI1 device shift

In addition to pooled five-device evaluation, V18.4 registers five
leave-one-device-out evaluations.

For each held-out device:

- its training scans are excluded from model fitting;
- the AP dictionary is built from the other four devices only;
- evaluation-only APs do not expand that dictionary;
- no held-out-device-specific method hyperparameter tuning is allowed.

## 5. Support sufficiency

### 5.1 Proposal sources

The natural candidate support is the union of:

- invariant-neural Top-32;
- raw-radio prototype Top-32;
- hierarchy/floor-aware Top-16;
- Delaunay-neighbor expansion from the strongest invariant/prototype seeds.

The first three sources form a registered primary proposal quota of 80 before
deduplication.

### 5.2 Dynamic support

V18.4 does not define a statistical fixed K posterior support.

Instead every scan has a natural support U(x), represented in batches by

- candidate indices;
- a candidate-validity mask;
- padding to a computational ceiling.

Registered computational ceiling:

L_max = 128.

L_max is a fail-closed capacity ceiling, not a fixed statistical support size.
Within each mini-batch, tensors are padded only to that batch's maximum valid
support width, which can be substantially below 128. When variable-width
batches must be aggregated for fitting references or calibration, they are
right-padded to the maximum width of that aggregation and the validity mask is
preserved exactly.

Padding positions always have zero posterior mass and cannot contribute to
losses, energies, calibration, or diagnostics.

### 5.3 Training-positive preservation

Training additionally forms

U_plus(x_i) = U(x_i) union N_plus(i),

where N_plus(i) is the complete positive non-self Delaunay target.

Only strictly positive target weights count toward the capacity contract.
Zero-weight padding never counts as a positive target.

### 5.4 Pretraining capacity audit

Before model implementation, repository data were statically audited.

With the conservative 80 primary proposal quota before deduplication, the
uploaded repository data were audited on the actual registered roles.

Full-training maxima are:

- UJI: 80 + 18 = 98;
- UTS: 80 + 19 = 99;
- Tampere: 80 + 12 = 92;
- pooled TUJI1: 80 + 15 = 95.

Across the roles that are actually optimized:

- every fit role, including all five TUJI1 LODO fits, requires at most 95;
- every selection_refit role requires at most 97;
- the worst full-training audit is UTS at 99.

Therefore L_max=128 leaves at least:

- 33 slots on every fit role;
- 31 slots on every selection_refit role;
- 29 slots over every audited full-training geometry.

The five TUJI1 LODO audits use their registered held-out-device-specific role
seeds and audit fit, selection_refit, and full-training separately.

If future data violate this contract, the run fails closed before training.

### 5.5 Natural versus forced recall

Every training/evaluation report must distinguish:

- natural proposal recall;
- forced training-support recall.

Forced-positive support may never be used to claim retrieval quality.

## 6. Base posterior q_theta

The V18.4 learned posterior is

q_theta(a | x, U).

Candidate proposal and posterior scoring are distinct operations.

The posterior scorer is the single learned discriminative candidate scorer
conditioned on the invariant scan representation and candidate evidence.

There is no mixture over six source posteriors and no learned source gate.

The posterior is normalized only over valid candidates in U.

## 7. Leave-self-out spatial target

V18.4 retains the validated V18.1/V18.2 leave-self-out Delaunay target.

For training coordinate i:

q_i(j) >= 0,
sum_j q_i(j) = 1,

and the observed/self anchor has zero target weight.

Geometry is built independently within each building-floor stratum.

The same target is used for anchor posterior supervision and spatial energy distance; no new
target-temperature hyperparameter is introduced.

## 8. Spatial posterior learning

### 8.1 Physical cost

For anchors a_j and a_k in the same registered support:

C_jk = ||a_j - a_k||_2 / s_C.

The scale s_C is not tuned.

For each training geometry it is the median positive same-stratum Delaunay edge
length computed from the fitting anchor dictionary.

This makes the cost dimensionless and adapts automatically to dataset spatial
scale.

### 8.2 Exact spatial energy-distance posterior supervision

For predicted candidate-anchor coordinates x_i, leave-self-out Delaunay target
coordinates y_j, and normalized probabilities q_i and t_j, define

d(x,y) = ||x-y||_2 / s_C.

The training-only spatial divergence is the exact discrete half energy distance:

D_spatial(q,t)
= sum_ij q_i t_j d(x_i,y_j)
- 0.5 sum_ik q_i q_k d(x_i,x_k)
- 0.5 sum_jl t_j t_l d(y_j,y_l).

All three expectations are evaluated exactly on the supplied finite supports,
with float64 accumulation. Masks remove padding before normalization and spatial
cost computation. The loss is differentiable with respect to predicted logits.
Identical distributions on identical coordinates yield zero up to roundoff.

There is no iterative transport solver, entropic regularization coefficient,
convergence-depth hyperparameter, or nonnegativity clamp hiding a negative loss.
The same fitting-derived spatial_scale_m and leave-self-out target are retained.

### 8.3 Training objective

The base model objective is

L =
    L_anchor
  + L_coordinate
  + lambda_spatial D_spatial
  + 0.25 L_invariant
  + 0.50 L_hierarchy
  + 0.05 L_domain,

where:

- L_anchor is cross entropy from q_i to q_theta over forced-positive support;
- L_coordinate is the leave-self-out expected Student-t coordinate NLL;
- D_spatial is the exact dimensionless half energy distance defined above;
- L_invariant supervises full-dictionary invariant retrieval;
- L_hierarchy supervises building/floor where meaningful;
- L_domain is used only when at least two training device classes exist.

There is no hard-negative ranking loss.

There is no posterior-mean coordinate loss.

There is no separate reranker auxiliary loss because the learned candidate
posterior is itself the primary anchor posterior.

### 8.4 Spatial-supervision coefficient development grid

The only registered spatial-supervision strength candidates are

lambda_spatial in {0.10, 0.25, 0.50}.

These values are selected on UJI V18.4 development coordinates only.

No wider/adaptive sweep is allowed within V18.4.

## 9. Radio correction energy

V18.4 retains the robust marked-radio model:

- AP detection probability;
- robust RSS center;
- robust RSS scale;
- AP reliability;
- Student-t marked energy;
- scan-level shared affine offset/gain profiling.

For candidate a:

E_r_raw(a,x)

is the profiled marked-radio energy.

The radio energy is not converted into an independent posterior.

The registered affine nuisance profiler uses:

- 6 optimization steps;
- nuisance learning rate = 0.20;
- absolute normalized RSS offset bound = 0.20;
- absolute log-gain bound = 0.25;
- offset regularization weight = 0.10;
- log-gain regularization weight = 0.10.

Only the temporary per-scan offset/gain nuisance tensors receive gradients in
this profiler.  Model parameters and anchor radio-memory fields are detached.

## 10. Geometry correction energy

Geometry produces a local support-consistency energy

E_g_raw(a,x)

from the temperature-calibrated base posterior q_T and candidate coordinates.
During training-reference fitting and global method development, T=1.

All geometry remains same-building/floor.

Geometry bandwidth and margin are expressed in units of the training median
Delaunay edge scale rather than absolute meters.

Registered dimensionless geometry law:

- sigma_geometry = 1.5 * s_C;
- margin_geometry = 0.5 * s_C.

These are not externally tuned.

## 11. Energy normalization

Energy values do not need to be intrinsically nonnegative.

For each source, V18.4 estimates a label-free robust scale from fitting scans:

s_source =
median_i MAD_j(E_source(i,j)).

At inference:

E_tilde_source(i,j)
=
(E_source(i,j) - min_k E_source(i,k))
/ max(s_source, delta).

Subtracting the scan-wise minimum is posterior-invariant under exponential
tilting and supplies a nonnegative representation without nonlinear clipping.

No softplus is applied to the energy.

All posterior correction is computed in log space.

## 12. Label-free reliability

V18.4 does not learn a reliability MLP.

For each fitting scan, define source mismatch:

Radio:
m_r =
0.5 * [1 - exp(-E_q_T[E_r_tilde])]
+ 0.25 * normalized_abs_style_offset
+ 0.25 * normalized_abs_log_gain.

Geometry:
m_g = E_q_T[E_g_tilde].

The empirical mismatch distributions and robust energy scales are fitted on
fitting scans with the uncalibrated training base q (T=1), then frozen. At
deployment, nuisance profiling, geometry energy and mismatch expectations are
computed under q_T and compared with those fixed training references. Neither
calibration nor evaluation refits any reliability reference.

For source s, reliability is the smoothed empirical survival probability

r_s(x)
=
1 - F_hat_s(m_s(x)),

implemented from the training empirical CDF with finite-sample smoothing.

Thus high mismatch yields low reliability.

Reliability values are deterministic, label-free, and constrained to [0,1].

Evaluation data never refit these distributions.

## 13. Reliability-bounded trust-region posterior correction

Separate correction direction from allowed magnitude. For empirical source
reliabilities r_r,r_g in [0,1], the directional weights are

w_r = r_r / (r_r+r_g), w_g = r_g / (r_r+r_g).

When the sum is numerically zero (fixed numerical tolerance 1e-12), both
weights and the effective budget are zero, and correction is identity.
Otherwise:

E_dir = w_r E_r_tilde + w_g E_g_tilde,
R(x) = max(r_r(x),r_g(x)),
epsilon_eff(x) = R(x) * epsilon_KL.

For rho in [0,1]:

p_rho(a|x) proportional to q_T(a|x) exp[-rho E_dir(a,x)].

The applied correction uses the maximum rho satisfying

KL(p_rho || q_T) <= epsilon_eff(x).

KL is monotone nondecreasing in rho for a fixed direction, so deterministic
row-wise bisection finds the maximum feasible rho. Bisection uses float64;
realized KL is recomputed for the actual returned posterior. Scalar KL budgets
remain supported by the mathematical helper for compatibility.

Uniformly lowering both reliabilities leaves the direction fixed and lowers
the maximum budget. Increasing rho therefore cannot cancel reliability. Zero
reliability returns q_T bitwise, and masked positions retain exactly zero mass.
There are no learned reliability parameters or labels in this law.

Diagnostics include the registered maximum epsilon_KL, row-wise epsilon_eff,
rho, realized and full-correction KL, both reliabilities, and normalized source
weights. The KL reference is q_T, including at non-unit calibration temperatures.

## 14. Trust-region development grid

The registered global KL budgets are

epsilon_KL in {0.025, 0.050, 0.100}.

The registered maximum budget is a global scalar. Its row-wise effective
budget is max(r_r,r_g) times that maximum, never larger; this is a fixed
empirical law rather than a learned input-dependent parameter.

It is selected jointly with lambda_spatial on UJI V18.4 development coordinates.

Therefore the full registered method grid contains only

3 lambda_spatial values x 3 epsilon_KL values = 9 method combinations.

The base no-spatial-supervision/no-correction system is reported as an ablation but is not an
additional searched method combination.

## 15. Global method selection rule

Selection is deliberately staged so posterior learning and posterior
correction remain causally separable.

For each registered lambda_spatial value:

1. train the base posterior on the fitting role;
2. select that run's epoch using the uncorrected base posterior only;
3. freeze the selected checkpoint;
4. fit radio/geometry energy scales and empirical reliability references using
   fitting-role scans only;
5. evaluate the three registered epsilon_KL values on development coordinates.

The resulting 3 lambda_spatial x 3 epsilon_KL = 9 corrected method combinations are
then ranked on the UJI V18.4 development role using the fixed lexicographic
rule:

1. minimum coordinate-macro Energy Score;
2. shortlist configurations within 0.05 m of the best Energy Score;
3. within the shortlist, minimum component-center GM coordinate-macro MAE;
4. then minimum coordinate-macro mixture NLL;
5. then smaller lambda_spatial;
6. then smaller epsilon_KL.

Trust-region correction never participates in epoch selection.  This preserves
a direct causal interpretation of the spatial-supervision training effect and avoids repeatedly
profiling radio nuisance parameters at every epoch.

External test outcomes never affect this selection.

## 16. Epoch selection

For each lambda_spatial-trained base model:

- epochs before 12 are ineligible;
- correction is disabled for epoch selection;
- primary development statistic is coordinate-macro Energy Score;
- shortlist epochs within 0.05 m of best eligible Energy Score;
- select lowest GM coordinate-macro MAE in the shortlist;
- then lower NLL;
- then earlier epoch.

The global architecture and method hyperparameters remain fixed across
datasets.

Dataset-specific epoch selection is allowed using that dataset's training-only
development role.

## 17. Predictive calibration

After model and method selection, refit on selection_refit using frozen
lambda_spatial and the selected epoch, and fit/freeze correction references
on selection_refit at T=1. Only the registered training-derived calibration
role may then choose scalar predictive calibration parameters:

- learned-base posterior temperature T from
  {0.50,0.75,1.00,1.25,1.50,2.00,2.50,3.00};
- Student-t scale multiplier from {0.50,0.75,1.00,1.25,1.50,2.00}.

The deployed order is:

base q -> temperature q_T -> reliability-bounded trust correction -> final posterior.

q_T(a|x) proportional to q(a|x)^(1/T).

Radio profiling, geometry energy and reliability mismatches use q_T. The final
structured posterior satisfies KL(p_final || q_T) <= epsilon_eff(x). There is
no second posterior-temperature map after correction.

Each candidate temperature gets its own corrected-distribution cache, obtained
by executing this entire order; a cache/temperature mismatch fails closed.
The six scale values reuse that temperature's final probabilities and multiply
only Student-t component scales. Scale calibration is probability-neutral and
leaves anchor weights, component centers and trust diagnostics unchanged.

Calibration minimizes coordinate-macro mixture NLL over the same fixed 8 x 6
surface on every dataset. These dataset-specific predictive calibration values
cannot modify lambda_spatial, epsilon_KL maximum, reliability law, architecture,
proposal quotas or GM decoder. Reference fitting is never repeated on the
calibration or evaluation role.

## 18. Loss-aligned point decision

The preregistered decoder probe fixed the V18.4 point decision as the
posterior-weighted component-center geometric median:

y_hat_GM
=
argmin_z sum_j p_j ||z - mu_j||_2.

This is explicitly described as the geometric median of component centers,
weighted by posterior anchor mass.

It is not claimed to be the exact spatial median of the full Student-t mixture.

The full predictive Student-t mixture remains the object used for:

- NLL;
- Energy Score;
- predictive coverage;
- uncertainty.

Top-K mean, MAP, trimmed means, alternative medians, or new decoder searches
cannot replace GM inside V18.4 based on later outcomes.

## 19. Tail-dilution diagnostics

On development/calibration roles V18.4 reports, for R in {10,20,40} m:

Truth-relative far mass:

M_R_truth
=
sum_j p_j I(||mu_j-y|| > R).

Mode-relative far mass:

M_R_mode
=
sum_j p_j I(||mu_j-mu_MAP|| > R).

Spatial leverage:

L_GM
=
sum_j p_j ||mu_j-y_hat_GM||.

Also report:

- entropy;
- effective anchor count;
- top-1 mass;
- mean-GM displacement;
- correlation between M_20_truth and posterior-mean error;
- correlation between M_20_mode and mean-GM displacement.

These diagnostics are never used on reserved tests to retune the method.

## 20. Primary ablations

The required causal ladder is:

A0. Base posterior + posterior-mean decision
A1. Base posterior + registered GM decision
A2. Base + spatial energy distance + GM
A3. Base + spatial energy distance + trust-region correction + posterior mean
A4. Full: Base + spatial energy distance + trust-region correction + GM

Additional source ablations:

- Full minus radio correction;
- Full minus geometry correction;
- fixed reliability r_r=r_g=1 versus empirical reliability;
- trust-region disabled rho=1 versus KL-bounded rho_star.

No ablation outcome may redefine the frozen full model after reserved test
evaluation.

## 21. External evaluation

After the UJI global method law is frozen:

- train/evaluate UTSIndoorLoc using the same global method hyperparameters;
- train/evaluate Tampere using the same global method hyperparameters;
- train/evaluate pooled TUJI1;
- run five TUJI1 leave-one-device-out evaluations.

The global lambda_spatial, epsilon_KL, support law, reliability law, spatial loss definition,
trust-region direction, architecture, and GM decoder cannot change after the
UJI method freeze.

### 21.1 Two-gate external protocol

Every external protocol uses three commands:

1. prepare;
2. fit-freeze;
3. evaluate-test.

Prepare and fit-freeze access only the dataset training source. They may:

- construct training-only fit/development/calibration coordinate-group roles;
- select a dataset-specific epoch from the uncorrected base posterior on that
  dataset's training-only development role;
- refit on selection_refit;
- fit correction references on selection_refit;
- choose only the registered scalar posterior-temperature and Student-t scale
  calibration values on the training-only calibration role.

They may not alter any global method hyperparameter.

The evaluate-test command is separate. Because provided tests are already
pre-exposed for adapter compatibility, the lock does not represent first
semantic access. It represents the start of the first formal frozen-model
performance evaluation. If that lock already exists, rerun is forbidden.

Before the lock is written, the implementation verifies:

- required test files exist;
- requested CUDA device is available;
- checkpoint and manifest hashes match;
- the frozen model can be reconstructed;
- the correction reference can be reconstructed;
- one complete final-posterior inference path is finite on a training-derived
  prototype scan.

Thus ordinary device/checkpoint failures cannot unnecessarily consume the
one-shot formal performance evaluation.

### 21.2 External source compatibility

External runs must use the same V18.4 method-source tree as the UJI global
method freeze. Evidence-only commits after UJI freeze are allowed, but changes
under the method implementation or DESIGN_V18_4.md invalidate external
evaluation.

For each individual external protocol, prepare -> fit-freeze -> evaluate-test
must run on one unchanged execution commit. Generated untracked artifacts do
not count as source changes.

### 21.3 TUJI1 container nuance

Each TUJI1 MAT file physically contains both results_train and results_test.
Therefore reading results_train necessarily reads the MAT container bytes.
V18.4 does not claim that the TUJI1 test files are byte-level untouched.

The protected boundary is formal performance use, not historical semantic
access:

- the TUJI results_test field was already inspected during adapter-validation
  CI before formal method freeze;
- formal training still selects and uses only results_train;
- results_test is never used for model, epoch, calibration, or global
  hyperparameter selection;
- formal frozen-model scoring of results_test occurs only after the
  irreversible evaluate-test performance lock.

### 21.4 TUJI1 leave-one-device-out roles

The five registered held-out-device role seeds are:

- A12: 184451;
- POCO: 184452;
- S20: 184453;
- S7: 184454;
- tabS7: 184455.

For each LODO protocol, the AP dictionary and every learned/training-derived
quantity are built from the other four devices' training scans only. The
held-out device's provided test split is projected into that frozen dictionary;
evaluation-only APs do not expand it.

Provided tests are evaluated once per frozen formal model. No post-test
retuning is permitted.

UJI official validation is reported separately as a pre-exposed descriptive
benchmark only.

## 22. Required metrics

Point localization:
- coordinate-macro GM MAE;
- row MAE;
- median / P75 / P90 error.

Probabilistic:
- Energy Score;
- mixture NLL;
- 50/80/90/95% coverage;
- radial predictive scale.

Hierarchy:
- building error where applicable;
- floor error where applicable.

Support:
- natural proposal recall;
- invariant/prototype/hierarchy source recall;
- forced-positive retention;
- support-size distribution.

Posterior:
- entropy;
- effective anchors;
- top-1 mass;
- between-anchor uncertainty;
- far-mass diagnostics.

Trust region:
- radio reliability;
- geometry reliability;
- rho_star;
- realized KL;
- fraction rho_star=1;
- radio/geometry correction energy;
- style offset/gain diagnostics.

Compute:
- parameters;
- training/evaluation time;
- peak GPU memory;
- support-size compute statistics.

## 23. Claims V18.4 must not make

Do not claim:

- exponential tilting itself is novel;
- geometric median was numerically the best decoder in the V18.3 probe;
- UJI official validation is blind;
- provided UTS/Tampere/TUJI tests are blind or were never semantically
  inspected before method freeze;
- early-stop and calibration from V18.3 were independent external validations;
- no backward operation occurred in the V18.4 decoder probe.

Correct wording for the probe is:

"no model-parameter training or update."

## 24. Implementation order

V18.4 implementation proceeds in this fixed order:

1. canonical adapters and dataset audit;
2. grouped training-role protocol;
3. pretraining support-capacity audit;
4. mathematical core tests for spatial energy distance, reliability, trust region and GM;
5. base single-checkpoint posterior model;
6. UJI nine-combination method development;
7. selection refit and scalar predictive calibration;
8. frozen multi-dataset training;
9. one-shot formal performance evaluations on pre-exposed provided tests;
10. causal ablations and final reporting without changing the frozen full model.

Steps 1-3 are complete and CI-verified before the main model is implemented.

## 25. Pre-run mathematical revision boundary (2026-10-04)

Before formal UJI nine-combination development, revise and verify the exact
spatial loss, reliability-scaled trust budget, calibration order and residual
cap. Run the registered regression suite, v18_4_realdata_smoke, and the
outcome-free v18_4_revision_audit. The audit uses only UJI training-derived
fit/development roles and synthetic constructions, with a small CUDA smoke
when available. It reports finite losses/gradients, KL and residual bounds,
small-sample natural-support truncation and forced-positive preservation.

The method still has exactly 9 global combinations. Active code, configuration,
and new serialized records use lambda_spatial; epsilon_OT is absent. The UJI
and external protocol metadata version advances to v2_20261004 so the revised
law is distinct from the initial implementation. There were no formal V18.4
registry/development/calibration/freeze artifacts to migrate at revision start.

After PRE_RUN_REVISION_AUDIT_V18_4.json/.md and tests succeed, stop for review.
Do not execute formal UJI prepare/develop/calibrate-freeze or external formal
training/evaluation in this pre-run revision task. Numerical or protocol
failures must be reported without weakening checks or silently retrying.
