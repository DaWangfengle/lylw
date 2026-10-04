# V18.5 Design Freeze — Jointly Calibrated Radio-Trust Spatial Posterior

## Status

V18.5 is the first implementation generation after the completed V18.4
mechanism program.  This document freezes the method law **before** any formal
V18.5 UJI development outcome is consumed.

Working name: **JRT-SP — Jointly Calibrated Radio-Trust Spatial Posterior**.

Scientific parent evidence: V18.4 final coupling/stratum audit at
`317a81faa07c297cf986af7c0674153c09486211`.

## 1. Evidence chain

The design is not an arbitrary architecture search.

- V18.2 showed that an unconstrained learned evidence gate could collapse onto
  one source.
- V18.3 prevented collapse with structured evidence families but produced a
  severely diffuse anchor posterior; effective-anchor count was about 42 and
  posterior-mean point error was badly affected by tail mass.
- V18.4 replaced peer evidence voting by one learned posterior plus exact
  spatial energy supervision.  The posterior became substantially more
  concentrated and posterior mean became competitive with robust decoders.
- V18.4 decoder probes showed that component-center geometric median was no
  longer supported as the default point rule.
- V18.4 correction probes showed that radio evidence is intrinsically
  informative, but its posterior effect changes sign with base-posterior
  temperature.
- The final 312-cell V18.4 coupling audit found broad common beneficial
  temperature regions for radio-only correction.  Therefore temperature and
  correction strength are not scientifically separable.
- The same audit diagnosed two structural geometry-correction failures:
  conditional within-stratum normalization discards stratum posterior mass,
  and anchor-coordinate geometry is mismatched with residual-shifted
  predictive components.

V18.5 therefore keeps the successful single spatial posterior, removes
inference-time geometry correction, and reformulates radio use as a safe
KL-constrained projection jointly calibrated with temperature.

## 2. Learned spatial model

One checkpoint contains:

- bounded maximal-invariant Sparse-AP Transformer;
- one learned candidate posterior;
- one bivariate Student-t residual component per candidate;
- one residual head;
- dynamic support;
- deterministic Delaunay geometry used only for support expansion, spatial
  supervision and local residual transforms.

Registered backbone:
`d_model=128, heads=4, layers=2, max_visible=64, dropout=.10`.

Student-t:
`df=4, min scale=.25m`.

Local residual transform:
maximum singular value `30m`.

No ensemble, teacher, distillation, learned evidence gate or member voting is
part of V18.5.

## 3. Explicit AP observation event

V18.5 does not infer missingness from an RSS sentinel.

For every scan the model receives

`(rss_value, observed_mask)`.

`observed_mask` is authoritative.  In particular an observed RSS clipped at
`-110 dBm` maps numerically to `-1` but remains observed.  A missing AP may
also use `-1` as an inert numeric fill without changing semantics.

This contract applies to the encoder, augmentation, prototype proposal,
candidate evidence and radio likelihood.

## 4. Count-aware marked radio memory

For anchor `a`, AP `k`:

1. use local MAD when at least two local observations exist and MAD >= .05;
2. otherwise use AP-pooled training MAD when estimable and >= .05;
3. otherwise use a training-global robust RSS scale, floored at .05.

No evaluation row enters these statistics.

The fallback is deterministic and has no tuned shrinkage coefficient.  The
purpose is to make the marked likelihood well-defined for dense repeated
anchors and singleton-heavy datasets such as Tampere.

## 5. Spatial posterior supervision

The exact half energy distance from V18.4 is retained.

Registered grid:

`lambda_spatial in {0, .10, .25, .50}`.

`lambda=0` is a first-class model, not an ablation implemented after the
fact.

Training objective:

`L_anchor + L_coordinate + lambda*D_spatial + .25L_invariant + .50L_hierarchy + .05L_domain`.

## 6. Base-posterior development

Development does not use radio correction.

For each lambda:

- train on FIT;
- epochs 1..11 are ineligible;
- evaluate eligible epochs with `T=1, epsilon=0, scale=1`;
- primary metric is coordinate-macro Energy Score;
- keep epochs within .05m of best Energy Score;
- choose minimum mixture NLL;
- then minimum posterior-mean MAE;
- then earlier epoch.

Across the four selected lambda models use the same priority and finally the
smaller lambda.

Geometric median does not participate in training, epoch selection, global
lambda selection, calibration selection or deployment.

## 7. Base temperature

For learned posterior `q`:

`q_T(j) proportional to q(j)^(1/T)`.

Registered temperatures:

`{.50,.75,1,1.25,1.50,2,2.50,3}`.

Temperature is applied **before** radio profiling and posterior projection.
There is no post-correction re-temperature.

## 8. Temperature-matched radio compatibility

For every registered temperature, the selection-refit role produces a
separate label-free compatibility reference.

For a scan, the radio nuisance profiler optimizes one bounded scan offset and
one bounded positive gain under `q_T` for exactly six Adam steps.

The mismatch statistic is the final profiled objective itself:

`m_T(x) = E_{j~q_T}[E_radio(j|x, nuisance_hat)] + .10 offset_hat^2 + .10 log(gain_hat)^2`.

The training reference is the sorted collection of `m_T` values at the same
temperature.

Compatibility:

`r_T(x) = empirical_survival(m_T(x); reference_T)`.

Interpretation: `r_T` is a **training-compatibility / safety score**.  It is
not claimed to predict whether correction will improve localization.

## 9. Exact radio trust projection

For calibration parameter `epsilon >= 0` define row budget

`b(x) = r_T(x) * epsilon`.

The final anchor posterior solves

`min_p <p,E_radio>`

subject to

`p in simplex, KL(p || q_T) <= b`.

For finite dual `beta`:

`p_beta(j) proportional to q_T(j) exp(-beta E_j)`.

There is no scientific cap `beta<=1`.

If the budget reaches the limiting KL, the exact optimizer is `q_T`
conditioned on the complete minimum-energy tie set.

### Proposition P1 — identity

`epsilon=0 => p*=q_T`.

### Proposition P2 — additive energy invariance

`E' = E+c` gives the same posterior.

### Proposition P3 — positive-scale energy invariance

For `E'=aE, a>0`, the posterior is unchanged and finite dual transforms as
`beta'=beta/a`.

### Proposition P4 — budget monotonicity

Increasing the KL budget cannot worsen the optimum expected radio energy.

### Proposition P5 — descent

`<p*,E> <= <q_T,E>`.

### Proposition P6 — active finite constraint

Before the limiting solution, an active optimum satisfies
`KL(p*||q_T)=b` up to numerical root tolerance.

### Proposition P7 — minimum-energy limit

When `b >= -log(sum_{j in M}q_T(j))`, where `M` is the minimum-energy set,
the optimizer is `q_T(.|M)`.

### Proposition P8 — padding invariance

Masked padding cannot alter the solution.

The implementation uses float64 bracketing and at least 64 bisection steps.
Failure to bracket is an error; beta is never silently clipped.

## 10. Joint calibration

After lambda and epoch freeze, refit once on `selection_refit`, build the
temperature-specific compatibility references, then evaluate

- 8 temperatures;
- 4 epsilon values `{0,.025,.05,.10}`;
- 6 Student-t scale multipliers `{.5,.75,1,1.25,1.5,2}`.

Total: **192 cells**.

`epsilon=0` is permanently eligible and means no radio posterior movement.

Student-t scale changes only component scale; it cannot change anchor
probabilities.

## 11. Paired one-standard-error selector

Raw minimum NLL is not selected directly.

1. Find the cell with minimum coordinate-macro NLL.
2. For every cell compute per-coordinate-group paired NLL differences to this
   reference and retain it when mean difference <= paired SE.
3. Inside that NLL one-SE set, repeat the paired one-SE procedure using Energy
   Score.
4. In the doubly retained set choose:
   - minimum posterior-mean MAE;
   - smaller epsilon;
   - temperature closest to 1 in `abs(log T)`;
   - scale closest to 1 in `abs(log s)`;
   - lower T;
   - lower scale.

This law is frozen before formal outcomes.

## 12. Registered point output

The V18.5 point summary is

`y_hat = sum_j p*_j mu_j`.

This is the posterior mean of predictive component centers.

The paper must not claim that this is the Bayes action for Euclidean absolute
distance.  The full Student-t mixture is the primary probabilistic output; the
mean is the registered efficient point summary supported by the V18.4
mechanism evidence.

## 13. Geometry boundary

Geometry is permitted for:

- Delaunay leave-self-out training target;
- support expansion;
- local residual transform;
- spatial-energy supervision.

Geometry is forbidden as an inference posterior-correction energy in V18.5.

The repaired mass-preserving component-space geometry discovered by the V18.4
probe is retained as future ablation evidence, not silently promoted into the
main method.

## 14. V18.5 grouped roles

V18.5 preregisters fresh grouped-coordinate seeds:

- UJIIndoorLoc: 185501
- UTSIndoorLoc: 185502
- Tampere: 185503
- TUJI1: 185504

The hull-preserving split algorithm and 15% development / 15% calibration
fractions remain outcome-free.

This is not described as restoring benchmark blindness; it reduces exact
reuse of V18.4's heavily inspected role partition.

## 15. External protocol

After UJI freezes the method structure, each external dataset may fit its own
training-only model and calibrate its own scalar `T, epsilon, scale` using the
same frozen grids and selector.

Provided test performance never enters method selection.

TUJI leave-one-device-out training excludes the held-out device from model
fitting, radio reference fitting and scalar calibration.

## 16. Explicitly excluded mechanisms

V18.5 does not contain:

- inference-time geometry correction;
- peer evidence voting;
- learned evidence gate;
- GM deployment or selection;
- Top-K point contraction;
- MAP as the registered point output;
- post-correction temperature;
- ensemble inference;
- teacher/student distillation.

## 17. Future paper ablations

These are planned experiments, not current evidence:

- lambda=0 vs positive spatial supervision;
- epsilon=0 vs exact radio projection;
- temperature-matched vs mismatched compatibility reference;
- exact unbounded dual vs V18.4 beta/rho<=1;
- compatibility-scaled vs constant KL budget;
- explicit mask vs sentinel collision;
- count-aware scale fallback vs local fixed floor;
- separate T/epsilon selection vs joint calibration;
- repaired mass-preserving component-space geometry as a non-main ablation.

No ablation result may alter the frozen V18.5 main protocol after formal
development starts.
