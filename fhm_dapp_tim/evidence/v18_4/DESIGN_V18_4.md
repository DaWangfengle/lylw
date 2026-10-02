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

### 4.3 Dataset roles

UJIIndoorLoc is the global method-development benchmark. Its official
validation file has already been exposed by earlier versions and is descriptive
only.

UTSIndoorLoc, Tampere, and TUJI1 provided test splits are reserved from
performance evaluation and model selection until the V18.4 method law and
global method hyperparameters are frozen.

Their schemas have already been inspected; they are therefore described as
reserved outcome test sets, not as completely untouched/blind files.

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

Padding positions always have zero posterior mass.

### 5.3 Training-positive preservation

Training additionally forms

U_plus(x_i) = U(x_i) union N_plus(i),

where N_plus(i) is the complete positive non-self Delaunay target.

Only strictly positive target weights count toward the capacity contract.
Zero-weight padding never counts as a positive target.

### 5.4 Pretraining capacity audit

Before model implementation, repository data were statically audited.

With the 80 primary proposal quota, the largest observed required capacities
are:

- UJI full training: 80 + 18 = 98;
- UTS full training: 80 + 19 = 99;
- Tampere full training: 80 + 12 = 92;
- TUJI1 full training: 80 + 15 = 95;
- all five TUJI1 leave-one-device-out pools: at most 95.

Thus L_max=128 leaves at least 29 slots in every registered geometry audited
before outcome evaluation.

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

The same target is used for anchor posterior supervision and spatial OT; no new
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

### 8.2 Debiased entropic transport

V18.4 uses a log-domain, balanced, debiased entropic transport objective
between the learned anchor posterior and the leave-self-out Delaunay target.

The registered entropic regularization is fixed at

epsilon_OT = 0.10

in dimensionless cost units.

It is not selected on external datasets.

Sensitivity at epsilon_OT in {0.05, 0.20} may be reported on UJI development
after the primary method law is frozen, but those values cannot replace the
registered method by outcome shopping.

### 8.3 Training objective

The base model objective is

L =
    L_anchor
  + L_coordinate
  + lambda_OT L_OT
  + 0.25 L_invariant
  + 0.50 L_hierarchy
  + 0.05 L_domain,

where:

- L_anchor is cross entropy from q_i to q_theta over forced-positive support;
- L_coordinate is the leave-self-out expected Student-t coordinate NLL;
- L_OT is debiased entropic spatial transport;
- L_invariant supervises full-dictionary invariant retrieval;
- L_hierarchy supervises building/floor where meaningful;
- L_domain is used only when at least two training device classes exist.

There is no hard-negative ranking loss.

There is no posterior-mean coordinate loss.

There is no separate reranker auxiliary loss because the learned candidate
posterior is itself the primary anchor posterior.

### 8.4 OT coefficient development grid

The only registered OT strength candidates are

lambda_OT in {0.10, 0.25, 0.50}.

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

## 10. Geometry correction energy

Geometry produces a local support-consistency energy

E_g_raw(a,x)

from the current base posterior and candidate coordinates.

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
0.5 * [1 - exp(-E_q[E_r_tilde])]
+ 0.25 * normalized_abs_style_offset
+ 0.25 * normalized_abs_log_gain.

Geometry:
m_g = E_q[E_g_tilde].

The empirical mismatch distributions are fitted using fitting data only.

For source s, reliability is the smoothed empirical survival probability

r_s(x)
=
1 - F_hat_s(m_s(x)),

implemented from the training empirical CDF with finite-sample smoothing.

Thus high mismatch yields low reliability.

Reliability values are deterministic, label-free, and constrained to [0,1].

Evaluation data never refit these distributions.

## 13. Trust-region posterior correction

Define

E_c(a,x)
=
r_r(x) E_r_tilde(a,x)
+
r_g(x) E_g_tilde(a,x).

For rho in [0,1]:

p_rho(a|x)
proportional to
q_theta(a|x) exp[-rho E_c(a,x)].

The applied correction is

rho_star
=
max {
  rho in [0,1] :
  KL(p_rho || q_theta) <= epsilon_KL
}.

Because KL(p_rho || q_theta) is monotone nondecreasing in rho >= 0, rho_star
is solved deterministically by scalar bisection.

This KL direction is intentionally conservative: structured evidence can
correct the learned posterior but cannot freely replace it.

## 14. Trust-region development grid

The registered global KL budgets are

epsilon_KL in {0.025, 0.050, 0.100}.

The budget is a global scalar, not an input-dependent learned function.

It is selected jointly with lambda_OT on UJI V18.4 development coordinates.

Therefore the full registered method grid contains only

3 lambda_OT values x 3 epsilon_KL values = 9 method combinations.

The base no-OT/no-correction system is reported as an ablation but is not an
additional searched method combination.

## 15. Global method selection rule

Selection is deliberately staged so posterior learning and posterior
correction remain causally separable.

For each registered lambda_OT value:

1. train the base posterior on the fitting role;
2. select that run's epoch using the uncorrected base posterior only;
3. freeze the selected checkpoint;
4. fit radio/geometry energy scales and empirical reliability references using
   fitting-role scans only;
5. evaluate the three registered epsilon_KL values on development coordinates.

The resulting 3 lambda_OT x 3 epsilon_KL = 9 corrected method combinations are
then ranked on the UJI V18.4 development role using the fixed lexicographic
rule:

1. minimum coordinate-macro Energy Score;
2. shortlist configurations within 0.05 m of the best Energy Score;
3. within the shortlist, minimum component-center GM coordinate-macro MAE;
4. then minimum coordinate-macro mixture NLL;
5. then smaller lambda_OT;
6. then smaller epsilon_KL.

Trust-region correction never participates in epoch selection.  This preserves
a direct causal interpretation of the OT training effect and avoids repeatedly
profiling radio nuisance parameters at every epoch.

External test outcomes never affect this selection.

## 16. Epoch selection

For each lambda_OT-trained base model:

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

After model and method selection, the model is refit on selection_refit using
the frozen lambda_OT and selected epoch.

Only then may its training-only calibration role choose scalar predictive
calibration parameters:

- final corrected-posterior temperature from
  {0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00};
- Student-t scale multiplier from
  {0.50, 0.75, 1.00, 1.25, 1.50, 2.00}.

The temperature is applied after the frozen trust-region correction:

p_cal(a|x) proportional to p_corrected(a|x)^(1/T).

Therefore the registered KL bound describes the structured correction before
this separate probability-calibration map.  Calibration is not part of the
trust-region optimization.

Calibration minimizes coordinate-macro mixture NLL over the fixed 8 x 6
surface.  The same registered grids are used for every dataset.

These are dataset-specific probability calibration parameters, not method
hyperparameters.

They cannot modify lambda_OT, epsilon_KL, reliability law, architecture,
candidate quotas, or decoder.

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
A2. Base + spatial OT + GM
A3. Base + spatial OT + trust-region correction + posterior mean
A4. Full: Base + spatial OT + trust-region correction + GM

Additional source ablations:

- Full minus radio correction;
- Full minus geometry correction;
- fixed reliability r_r=r_g=1 versus empirical reliability;
- trust-region disabled rho=1 versus KL-bounded rho_star.

No ablation outcome may redefine the frozen full model after reserved test
evaluation.

## 21. External evaluation

After the UJI global method law is frozen:

- train/evaluate UTSIndoorLoc using the same method hyperparameters;
- train/evaluate Tampere using the same method hyperparameters;
- train/evaluate pooled TUJI1;
- run five TUJI1 leave-one-device-out evaluations.

Provided tests are evaluated once per frozen formal model.

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
- provided UTS/Tampere/TUJI tests were never inspected at schema level;
- early-stop and calibration from V18.3 were independent external validations;
- no backward operation occurred in the V18.4 decoder probe.

Correct wording for the probe is:

"no model-parameter training or update."

## 24. Implementation order

V18.4 implementation proceeds in this fixed order:

1. canonical adapters and dataset audit;
2. grouped training-role protocol;
3. pretraining support-capacity audit;
4. mathematical core tests for OT, reliability, trust region and GM;
5. base single-checkpoint posterior model;
6. UJI nine-combination method development;
7. selection refit and scalar predictive calibration;
8. frozen multi-dataset training;
9. one-shot reserved test evaluations;
10. causal ablations and final reporting without changing the frozen full model.

Steps 1-3 are complete and CI-verified before the main model is implemented.
