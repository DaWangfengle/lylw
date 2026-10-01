# V18.3 — Single-Member Hierarchical Robust Evidence Consensus

## 1. Motivation

V18.2 established that multi-source candidate discovery was useful: the wide
candidate union retained almost every nearest spatial anchor even under the
pre-exposed official benchmark.  Its adaptive six-way learned gate, however,
collapsed onto the learned candidate reranker.  At the selected V18.2 epoch,
more than 99% of evidence mass was assigned to one source while the invariant,
prototype, marked-radio, floor, and geometry opinions were effectively ignored.

That result identifies the next problem precisely:

> candidate discovery was not the bottleneck; unrestricted learned evidence
> selection was.

V18.3 therefore keeps the successful multi-source support construction and
replaces the free mixture-of-experts gate with a structured, bounded,
interpretable consensus law.

## 2. Scientific contract

V18.3 is a journal-method model, not an ensemble or compression pipeline.

It has:

- one inference checkpoint;
- one Student-t residual head;
- no teacher model;
- no knowledge distillation;
- no member axis;
- no multi-seed prediction averaging;
- no trainable evidence gate.

The model must remain scientifically meaningful at E=1.

## 3. The same six evidence opinions

V18.3 retains the six V18.2 opinions because V18.2 did not show that the
opinions themselves were useless; it showed that the fusion law was unstable.

For candidate anchor a and scan x:

1. invariant neural opinion p_inv(a|x);
2. raw prototype opinion p_proto(a|x);
3. learned candidate-evidence opinion p_learn(a|x);
4. robust marked-radio opinion p_radio(a|x);
5. soft floor opinion p_floor(a|x);
6. geometry-manifold opinion p_geo(a|x).

The invariant path remains the bounded positive-affine maximal-invariant Sparse
AP Transformer.  The prototype and marked-radio paths remain explicit radio-map
opinions and are not hidden inside the invariant representation.

## 4. Three semantic evidence families

The six opinions are grouped before global fusion:

Representation:
- invariant neural;
- learned reranker.

Radio:
- raw RSS prototype;
- robust marked-radio likelihood.

Structure:
- soft floor;
- geometry manifold.

This prevents one fine-grained source from directly suppressing all five other
sources.

## 5. Bounded within-family consensus

For the two source probabilities in each family, normalized entropy is used as
an interpretable confidence statistic.

Let H(p) in [0,1] be normalized entropy and c=1-H(p).

A soft reliability score is formed from the two confidences, but the final
within-family source weights are bounded:

w_source >= 0.30

and, because every family has exactly two members,

w_source <= 0.70.

Thus no source can disappear and no single source can exceed 70% of its family.

The registered family probabilities are:

p_rep   = w_inv p_inv + w_learn p_learn
p_radio = w_proto p_proto + w_marked p_radio
p_str   = w_floor p_floor + w_geo p_geo.

All are normalized alpha=1 arithmetic opinion pools.

## 6. Reliability without a learned gate

V18.3 deliberately has no trainable gate MLP.

For each family f, reliability is determined by two label-free quantities:

1. family uncertainty: normalized entropy H_f;
2. family disagreement: mean pairwise Jensen-Shannon divergence D_f from the
   other evidence families.

The radio family additionally receives a radio mismatch penalty M_radio based
on:

- posterior-weighted robust marked-radio energy;
- normalized absolute affine style offset;
- normalized absolute log gain.

The registered reliability score is

r_f = - H_f - D_f

for Representation and Structure, and

r_radio = - H_radio - D_radio - M_radio.

These coefficients are frozen at registration.  They are not trained by
location labels.

## 7. Positive family-weight floor

Raw family reliability scores are converted through a softmax, but V18.3 then
applies an explicit convex floor:

w_f = m + (1 - 3m) softmax(r / tau)_f

with

m = 0.15.

Therefore every family has at least 15% inference mass and at most 70%.

This makes V18.2-style family collapse mathematically impossible.

## 8. Evidence-family dropout

A fixed training-only probability of 0.30 is registered.

For a scan selected for evidence dropout:

- exactly one of the three evidence families is removed;
- the other two family weights are renormalized;
- at inference no family is dropped.

At most one family is ever removed.

This is not ordinary hidden-unit dropout.  It is an explicit anti-shortcut
training law: the model cannot solve every training example by depending on a
single family.

The primary leave-anchor-out objective is still optimized under the resulting
two- or three-family consensus.

## 9. Disagreement-aware posterior temperature

V18.2 showed that a fixed sharpened calibration temperature could become
over-confident under domain shift.

V18.3 adds a model-internal temperature driven only by evidence disagreement.

Let D_global be the mean of the three pairwise normalized Jensen-Shannon
divergences among Representation, Radio, and Structure.

The internal temperature is

T_dis(x) = clip(
    1 + lambda_D D_global(x),
    1,
    T_max
)

with registered values

lambda_D = 1
T_max = 2.

The hierarchical consensus is softened by T_dis before final Top-K contraction.

When the three evidence families agree, T_dis approaches 1.
When they conflict, the posterior automatically broadens.

## 10. Calibration may soften but never sharpen

V18.3 still performs scalar probability calibration on a physically separate
calibration coordinate role.

However the posterior-temperature grid is restricted to

T >= 1.

The calibration grid is:

1.00, 1.25, 1.50, 2.00, 2.50, 3.00.

A calibration result is therefore allowed to preserve or soften the
disagreement-aware posterior, but it cannot make it sharper.

The Student-t scale multiplier remains separately calibrated on the registered
finite grid.

## 11. Candidate discovery remains V18.2's successful component

The registered support construction is unchanged:

- invariant Top-32;
- prototype Top-32;
- floor-aware Top-16;
- Delaunay expansion around the top invariant/prototype seeds;
- deterministic source-quota preservation;
- L=96 wide candidate union;
- K=64 final predictive support.

This is important scientifically: V18.3 changes the component that failed
(fusion), not the component that was empirically successful (candidate
discovery).

## 12. Leave-anchor-out geometry objective

V18.3 retains the V18.1/V18.2 leave-self-out Delaunay target.

For training coordinate i with non-self Delaunay neighbors j and weights q_ij:

L_joint =
 - sum_j q_ij [
     log p_cons(j|x_i)
     + log p(y_i|x_i,j)
   ].

The observed training anchor is not a positive target component.

This preserves the central V18.1 correction that aligned training likelihood
with unseen-coordinate localization.

## 13. Ranking loss is reduced, not strengthened

V18.2 made the posterior sharper but did not improve cross-domain benchmark
localization.

V18.3 therefore reduces the hard-negative ranking coefficient from 0.10 to
0.05.

The purpose is to retain ordering pressure without allowing sharpness to
dominate robust evidence consensus.

## 14. Robust marked-radio mismatch

The radio reliability penalty is dimensionless.

The marked-radio energy contribution is mapped to [0,1) by

1 - exp(-E_radio).

The affine style terms are normalized by the same registered maximum offset
and log-gain bounds used by the profiler.

The radio mismatch score combines likelihood mismatch and style correction.
A scan requiring large radio correction therefore automatically reduces, but
never eliminates, the Radio family.

## 15. Diagnostics required by the method

Every evaluation must report:

Localization:
- coordinate-macro MAE;
- Energy Score;
- mixture NLL;
- building/floor error.

Retrieval:
- final nearest-anchor recall@8/16/32/64;
- invariant recall;
- prototype recall;
- floor-aware recall;
- L=96 union recall.

Source diagnostics:
- six source entropies;
- six within-family source weights.

Family diagnostics:
- Representation / Radio / Structure weights;
- three family entropies;
- three family disagreements;
- global evidence disagreement;
- disagreement-aware temperature;
- radio mismatch;
- error/disagreement correlation;
- radio-mismatch/radio-weight correlation.

Predictive uncertainty:
- posterior entropy;
- effective anchor count;
- top-1 mass;
- within-component uncertainty;
- between-anchor uncertainty;
- 50/80/90/95% radial coverage.

These diagnostics directly test whether V18.3 solves the specific V18.2
failure mechanism.

## 16. Sequential confirmation status

V18.3 is a new experiment, but it is not claimed to be independent of all prior
model development.

V18.2 confirmation and the pre-exposed official validation benchmark informed
the V18.3 design.

Therefore V18.3 registers a new coordinate holdout using new split seeds and
labels it a one-shot *sequential internal confirmation*.

It is not described as a new independent external test.

The official UJI validation file remains pre-exposed and descriptive only.

## 17. One-shot confirmation and immutable freeze

Before confirmation:

- epoch selection is completed;
- calibration is completed;
- all V18.3 configuration values are frozen;
- a fixed-epoch confirmation model is trained on outer_refit.

The confirmation command writes an irreversible consumption lock before loading
the confirmation shard.

After the lock exists, confirmation cannot be repeated.

Any outcome-driven model change belongs to V18.4.

## 18. Single-member stability

Five pre-registered E=1 seeds are trained independently.

All five runs are reported.

No best seed may be selected and no prediction averaging is allowed.

The stability panel is descriptive and cannot modify the frozen formal model.

## 19. What V18.3 keeps from V9 and V17

From V9:
- maximal-invariant RSS representation;
- Delaunay geometry;
- leave-anchor-out geometry supervision;
- local residual transform;
- Student-t predictive distribution;
- marked-radio statistics and robust likelihood;
- shared affine nuisance profiling;
- geometry-manifold reasoning.

From V17:
- neural/radio complementary retrieval;
- wide support before contraction;
- alpha=1 arithmetic opinion-pool principle;
- explicit AP/radio-map evidence;
- consensus as a probabilistic operation.

What V18.3 deliberately does not inherit:
- stochastic member axis;
- member weighting;
- ensemble support union;
- checkpoint-level alpha consensus;
- teacher outputs;
- knowledge distillation.

## 20. Files

- fhm_dapp_tim/fhm_dapp/v18_3_single.py
  - hierarchical robust evidence consensus.
- fhm_dapp_tim/fhm_dapp/uji_v18_3_experiment.py
  - nested protocol, calibration, sequential confirmation, final refit, and
    descriptive benchmark.
- tests/test_uji_v18_3_experiment.py
  - anti-collapse and protocol-contract tests.
- V18_3_CODEX_RUN.txt
  - exact formal execution instructions.
