# V18.4 Design Gate — Trust-Region Posterior Localization

## Status

V18.4 is intentionally split into two steps.

Step A is a read-only mechanism diagnostic on the already-open V18.3
early-stop and calibration roles.

Step B is the formal V18.4 model implementation.

Step B must not be finalized until Step A has been executed once and its
preregistered decision rule has been applied.

This prevents the V18.4 point decoder from being chosen by post-hoc metric
shopping.

## Why a diagnostic gate is necessary

V18.3 produced a striking pattern before confirmation was consumed:

- high broad-support recall;
- high Top-8 recall;
- very poor coordinate MAE;
- very diffuse posterior mass and large between-anchor uncertainty.

This pattern can arise from at least two distinct mechanisms:

1. the posterior contains useful local mass but the posterior mean is a poor
   Bayes decision for Euclidean-distance MAE;
2. the posterior itself is spatially wrong or too diffuse, so changing the
   point decoder cannot solve the main error.

Those mechanisms imply different V18.4 architectures and must be separated
before implementing the final model.

## Frozen decoder diagnostic

The diagnostic compares four deterministic decisions from exactly the same
frozen V18.3 posterior:

1. posterior mean;
2. MAP component location;
3. Top-8 renormalized posterior mean;
4. weighted geometric median over the full predictive support.

No network parameter changes.

No retraining.

No posterior reweighting.

No confirmation access.

No official validation access.

The full probabilistic distribution remains unchanged, so this diagnostic does
not create alternative NLL or Energy-Score models.

## Why geometric median is tested

For squared Euclidean loss, the posterior mean is the Bayes decision.

For Euclidean absolute-distance loss,

E[ ||Y-z||_2 ],

the Bayes decision is a spatial/geometric median rather than the posterior
mean.

The project's main coordinate metric is Euclidean MAE, so a mismatch between
probabilistic training and point-decision decoding is scientifically plausible.

The probe therefore tests this mechanism directly rather than assuming it.

## Why Top-8 mean is tested

V18.3 showed high nearest-anchor recall among the highest-ranked anchors while
the full posterior was diffuse.

Top-8 renormalized mean is included as a fixed diagnostic for posterior-tail
dilution.

It is not automatically promoted to the V18.4 method even if numerically best.
Its role is to distinguish a far-tail posterior problem from a Bayes-decision
problem.

## Preregistered mechanism rule

The same rule is applied independently to both V18.3 early-stop and
calibration roles.

A decoder mechanism is considered materially supported only if:

- coordinate-macro MAE improves by at least 10% relative to posterior mean on
  BOTH roles; and
- the candidate decoder does not worsen either role by more than 0.25 m.

Decision:

### A. Geometric median passes

Conclusion:

loss_aligned_geometric_median_supported

V18.4 will use weighted geometric median as its registered point decoder.

The full posterior remains unchanged for NLL, Energy Score, calibration and
uncertainty.

### B. Geometric median fails, Top-8 mean passes

Conclusion:

posterior_tail_dilution_supported

The primary V18.4 problem is posterior contraction/transport.

Geometric median will not be claimed as a primary contribution from this
evidence alone.

### C. Neither passes

Conclusion:

posterior_learning_dominates

V18.4 will prioritize trust-region posterior inference and spatially aware
posterior supervision.

Decoder redesign will remain secondary.

## Planned V18.4 model after the gate

The architectural direction is fixed independently of the exact decoder
outcome:

### 1. Dynamic multi-source proposal support

Keep the successful candidate-recall principle:

- maximal-invariant neural retrieval;
- raw-radio retrieval;
- hierarchy-aware proposal;
- Delaunay positive-target inclusion.

Unlike V18.3, training support will not rely on a brittle L=96 equality.

A computational ceiling, tentatively L_max=128, will be padding/masking
capacity rather than a statistical fixed-support requirement.

Every positive Delaunay target must be included.

### 2. One discriminative base posterior

V18.4 will no longer treat all evidence sources as equal probability voters.

A learned discriminative anchor posterior q_theta(a|x) is the primary
posterior.

Radio and geometry become correction energies rather than forced mixture
components.

### 3. Trust-region posterior correction

The correction has the form

p_rho(a|x) proportional to
q_theta(a|x)
exp[-rho (lambda_r E_radio + lambda_g E_geometry)].

The correction strength rho is bounded by a posterior trust region such as

KL(p_rho || q_theta) <= epsilon(x).

Unreliable external evidence therefore approaches "no correction" instead of
receiving a mandatory probability floor.

This is designed to avoid both observed failure modes:

- V18.2 source collapse;
- V18.3 posterior dilution.

### 4. Spatially aware posterior supervision

Anchor posterior training should distinguish a 2 m mistake from an 80 m
mistake.

The planned mechanism is spatial optimal transport / Sinkhorn supervision over
anchor coordinates, combined with the continuous Student-t predictive
likelihood.

The exact OT formulation and coefficient must be preregistered before a formal
V18.4 run.

### 5. Point decision

If the decoder probe passes the geometric-median rule, V18.4 will use weighted
geometric median for coordinate MAE output.

If it does not, geometric median will not be promoted merely because it is a
plausible theory.

## Probe inputs

The probe requires local artifacts from the interrupted V18.3 run:

- V18.3 registry;
- V18.3 selection checkpoint;
- V18.3 freeze;
- V18.3 early-stop role shard;
- V18.3 calibration role shard.

The checkpoint and role shards are not committed to GitHub, so the full
diagnostic must run in the user's local/Codex environment.

## Probe outputs

The one-shot diagnostic writes:

- fhm_dapp_tim/evidence/v18_4/decoder_probe_v18_3.json
- fhm_dapp_tim/evidence/v18_4/decoder_probe_v18_3.md

The output includes:

- coordinate-macro MAE;
- row MAE and quantiles;
- paired coordinate-level improvement vs posterior mean;
- fraction of coordinates improved;
- posterior entropy/effective support diagnostics;
- geometric-median displacement from posterior mean;
- preregistered mechanism conclusion;
- SHA links to the V18.3 registry/freeze/checkpoint/shards.

## Scientific boundary

This diagnostic does not rescue or continue V18.3.

V18.3 confirmation remains unconsumed.

The diagnostic only informs the new V18.4 model-development branch.
