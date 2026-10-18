# V17 Spatial Identifiability Full-Strength Protocol

Branch: `v17-spatial-identifiability-full-v1`

## Why this experiment follows the frozen-calibration closeout

The frozen-calibration report supports two separate findings:

- The same-building-and-floor neighbor target removes substantial contradictory supervision: wrong building-floor neighbor mass was 93.61% on UTS and 79.01% on UJI, while the same-floor graph has zero such mass. The paired primary-checkpoint MAE changes were -0.419 m and -0.346 m, respectively.
- Post-hoc temperature/evidence calibration is not a universal localization repair. The backbone, component centers, scales, and candidate support remain frozen; only posterior weights change. TUJI's same-floor target is a mathematical no-op because it has one building-floor stratum, so its small Stage-1 difference is not a causal same-floor effect.

The next phase must therefore separate (i) graph structure on the complete frozen logit, (ii) continuous RSS support between physical anchors, and (iii) training with a joint building-floor-coordinate objective.

## Phase F: frozen, zero-retraining experiments

Use the existing primary checkpoints `best_dev_epoch_21_80` for both `raw_neighbors` and `same_floor_neighbors`, all three datasets, and seeds 17301/17302/17303. Rebuild canonical anchor-order tensors; never diffuse tensors in per-row Top-K rank order. The frozen stage must not call an optimizer on model weights or retrain a model.

### Structure 1: complete-logit / dual-channel graph diffusion

Construct a same-building-and-floor symmetric kNN graph. Add self-loops and symmetrically Sinkhorn-scale its nonnegative adjacency to a symmetric doubly-stochastic matrix W. Use
`P(alpha) = (1-alpha) I + alpha W`, with alpha in [0,1].

The complete logits are
`ell = scores/T + theta_v*z_visibility + theta_s*z_strength`.
Apply diffusion to channels before summing:
`ell_tilde = P(alpha_prior)(scores/T) + P(alpha_obs)(theta_v*z_visibility + theta_s*z_strength)`.

Run 1A (prior-only diffusion), 1B (observation-only diffusion), 1C (one shared alpha, so P acts on the complete total logit), and 1D (two independently fitted alphas). Fit alphas on CAL-fit only, with a fixed, predeclared grid, and evaluate once on site-disjoint CAL-eval. Keep the previous stage's CAL-fit-fitted T/theta fixed within each relevant arm so the graph effect is not silently mixed with a new T/theta optimizer.

For every arm, compare:
- Global softmax over all anchors.
- BF-mass-preserving normalization: retain each BF class's undiffused total probability and only redistribute mass within that BF.

For 1C, record `logsumexp(P ell) - logsumexp(ell)` and verify it is nonpositive within numerical tolerance. This is a partition-function theorem, not a guarantee of lower MAE.

### Structure 2: measure-preserving cell LSE and virtual midpoints

Generate one midpoint for every undirected same-BF graph edge. The midpoint belongs symmetrically to both incident cells. Fix every physical cell's base measure to one. Within cell i, split its measure into endpoint mass (1-beta) and total virtual mass beta, divided uniformly across its n_i incident midpoint children:
`ell_cell_i = log((1-beta) exp(ell_i) + beta/n_i * sum_v exp(ell_v))`.

The virtual RSS score must be computed from an interpolated FIT-only RSS prototype/anchor key and the same retrieval scoring law as a physical anchor. Do not merely average the two endpoint logits: a genuine interpolated RSS field can score above both endpoint scores.

Run:
- 2A: the same cell LSE and virtual RSS scores, with virtual component centers at the pure geometric midpoint (diagnostic control).
- 2B: the same cell LSE and scores, with virtual component centers `mu_v = y_v + sum_k omega_k * residual_k`, using linear parent-residual interpolation. This is the primary implementation and must exactly reduce to the frozen parent component center at beta=0.

Fit beta on CAL-fit from the predeclared grid [0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8] using 2B, then use that same beta for 2A and 2B comparisons. Keep `m_i = 1`. Interpolate component scales linearly and report pure-coordinate versus residual-interpolated centers separately.

Mathematical boundary: cell measure is exactly conserved for a locally constant score field (all parent and virtual scores equal). Because a shared midpoint has one score but belongs to two cells, it cannot simultaneously equal two unequal parent scores; do not claim per-cell identity for arbitrary heterogeneous parent logits. This is not a reason to remove symmetric ownership; it is a condition that must be stated in the report.

## Phase T: paired training chain O -> A -> C -> D -> E

Use the same FIT/DEV/CAL physical-site split, initial model state, seed, optimizer, augmentation, batch size, and 80-epoch budget for every newly trained variant. Select the primary checkpoint by minimum DEV MAE across epochs 21-80, with no early stopping. Do not score either internal TEST or official test.

- O: original baseline neighbor targets and original loss.
- A: same-building-and-floor neighbor targets and original loss.
- C: A + full joint BF-XY NLL, where the numerator sums Student-t component density only over the true BF, while the posterior normalizer still includes every anchor.
- D: C + closed-form center transport risk `sum_i q_i * ||mu_i-y||_2`, the W1 cost from the discrete component-center measure to a Dirac target. Fixed weight: 0.30.
- E: D + a trainable graph-diffusion logit layer P(alpha) inside retrieval logits during training and evaluation. Alpha starts at 0.1 and is constrained to [0,1]. Use the same W graph; do not apply diffusion only to G13.

Use the existing O/A checkpoints only if the exact local files and their recorded hashes verify. If they are missing or mismatched, fail closed and report that fact; do not silently claim a zero-retraining frozen result. Train C/D/E from the same initial state for each dataset/seed.

## Required metrics and artifacts

For each frozen and trained arm, report site-macro full MAE, predicted-BF-conditioned MAE, Energy Score, joint BF-XY NLL, BF accuracy, per-site deltas, P95/P99/worst-site MAE, selected alpha/beta, and relevant parameters. Report means and sample SD over three seeds descriptively; these are not independent-site significance tests.

Save:
- `RESULT.json` per dataset/seed and phase;
- `RUN_LEVEL_METRICS.csv`, `AGGREGATE_SUMMARY.csv`, `PAIRED_DELTAS.csv`, `SITE_DELTAS_ALL_RUNS.csv`;
- `INTEGRITY_REVIEW.json` and a Markdown report;
- source/data/split/checkpoint hashes and explicit counts showing that no TEST role was scored.

Do not modify the frozen-calibration source, prior evidence, data splits, or existing result artifacts. Keep all checkpoint binaries local/ignored as in the prior protocol; summaries and auditable metadata may be written to the experiment evidence directory.
