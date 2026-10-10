# G12 — Frozen Anchor-T=.6 posterior: MAE action vs retrieval-stage ranking
**Pre-outcome, single retrospective mechanism check, 2026-10-10.**

## Why this question?

G8 anchor-only T=.6 became the strong common E5/L192/K192/alpha1 comparator (TUJI 2.657774m, UTS 5.107529m, Tampere 5.987721m on historical CONFIRM). G9–G11 found (i) wrong dominant spatial modes can yield catastrophic point errors even with near-site candidates present, (ii) raw RSS/learned ranking has complementary corrections, and (iii) Tampere RSS 5NN's 0.224m aggregate gain is entirely due to two disastrous V17 decisions. Before training a new network, isolate *ranking* versus *decision* effects.

**Prior art in this same repo:** G5 already tested weighted component-centre geometric median and sampled continuous Student-t Weiszfeld median on the **untempered Tampere** E5 posterior, 7.202666m mean versus 6.680664m component median and 6.820973m continuous median, with NO internal promotion. The V9 sampler/Weiszfeld implementation is reused exactly. Do not claim this is new or algorithmically novel. G12's nonredundant contribution is the **stronger G8 sharpened anchor posterior on three original data domains** plus stage-by-stage ranking diagnosis, with a pair of independent posterior risk / realized MAE measurements.

## Fixed experiment, no training

Use the exact same old 15 frozen E5 checkpoints, the original G8 R3 member Top-L192, alpha=1 K192 consensus, df4 Student-t component centres and scales, shared Anchor-only T=.6 law. No E16/E25 inference, no official or external test file, no recalibration, no hyperparameter search, no new models, no confirmation fitting, no candidate pruning. All old CAL/CONFIRM data have already been observed many times, so no prospective scientific claim or p-value-based promotion is legitimate.

Action intervention holds *identical* q(m,k), original 960 centres and per-coordinate scales. For each query compare ONLY:

1. Original exact mixture mean (G8/G11 baseline). Floating-point summation remains original E5*K order.
2. Existing G5 64-iteration weighted component-centre geometric median: Dirac-centre surrogate which deliberately ignores Student-t component spread.
3. Existing V9 Weiszfeld geometric median from **two independent Student-t sampling streams**, 128 draws per stream, seed 19801. Same Student-t 4df full 960-component distribution, original centres/scales and weights. Approximation not analytic Bayes action.

Fixed Weiszfeld tolerance 1e-5m, 64 iterations. The *model-implied* expected Euclidean loss for these three fixed actions is estimated with a **separate independent** two-stream 128+128 Student-t sample seed 19802. Do NOT use realization target Y to choose which action, change number of draws, or stop algorithm early on score. The model-implied risk is NOT the ground truth realized error. All three actions have identical continuous energy score (the predictive law is unchanged); compute it ONCE from original G8 shared MC seed 19602, 64+64 per scan, and verify numeric match with G8 historic original posterior later. Do NOT count shared ES as evidence of one action beating another.

This separates:
(a) decision improvement consistent with posterior loss minimization;
(b) posterior misspecification when risk improvement fails to translate to measured error;
(c) finite-draw approximation limitations (reported, not tuned).

## Ranking attribution: same original full FIT dictionary

With NO labels passed into ranking, calculate six well-defined [rows, all FIT anchors] probability maps, independently of the new action:

- prototype_only: softmax(-beta_prototype * original FIT prototype RSS distance).
- neural_only: mean over E5 individual softmax(beta_neural * original neural retrieval logits).
- fused_equal_members: arithmetic mean of *actual original* neural+prototype fused full per-member priors, before member Top-L and radial weights.
- fused_radial_members: original full fused priors, combined using inverse-square of R3 member radial spread estimated from original Top-L.
- topL_radial_members: original Top-L normalized member probabilities, scattered to all FIT anchors using original radial member weights, prior to alpha consensus epsilon.
- R3_consensus_K192: original true alpha1 K192 sparse/dense consensus, zero outside the selected K anchors. Anchor T=.6 strictly preserves its Top1 identity.

These maps are explanatory counterfactual scoring operators. In particular prototype_only and neural_only are **alternative inputs**, not literal serial transformations. The fused_equal_members to fused_radial_members step isolates actual member confidence reweighting, while radial to Top-L and Top-L to consensus compare truncation/support effects. Because Top-L is renormalized and consensus adds epsilon and Top-K truncation, these stages need not yield identical rankings even at alpha1. Do not present the above maps as independently calibrated posteriors.

Using the query true XY/building/floor only AFTER all predictions, compute 10m same-building+floor Top1 hit, probability mass and first eligible anchor rank for every map; undefined rank if no physically near FIT anchor. Report site-macro summaries and site vectors. Report label-facing consecutive transition counts (right-to-wrong, wrong-to-right). This is not a deployable gate; do not build a rule selecting the best stage by comparing to true site.

## Data, reporting and interpretation

Five original TRAIN-internal historical roles:
TUJI CAL+CONFIRM, UTS CAL+CONFIRM, Tampere CONFIRM. No original FIT-label-driven new training. Compute row actions with fixed draws and then average row errors at each physical coordinate, then average sites. Save each physical-site key and complete paired vectors for mean, component median, continuous median, model-implied risk, action displacement and six-stage ranking. Report macro MAE, median/p90/p95/max and severe tail (>=10m/20m). ES for shared posterior only. Include matched comparison to the G8 exact mean MAE/ES across all five roles **after G12 lock/result freeze**, never in G12 decision code.

Decision for subsequent stage:
- If continuous median markedly improves both absolute MAE and severe tails across domains, its fixed-Bayes-action benefit merits *independent prospective* confirmation; it is not a novel theoretical contribution and cannot independently substantiate Q1.
- If model-implied risk falls but physical error does not, evidence shifts toward posterior *misidentification*, not point-action optimization.
- If prototype is correct/neural wrong and fused loses correct anchors in UTS hard sites, prioritise a learned spatial discriminative likelihood/representation, not another T grid. If neural-only helps compensate raw RSS catastrophic mistakes, preserve its benefit.
- If the learned consensus is often correct top1 but the mixture mean is far, distinguish wrong spatial mode concentration from mismatched point decision.
- If neither probe materially changes future design, stop the cheap G-series retrospectives: next training step needs a genuinely fresh, physically isolated development split and equal-budget E5/E16/E25, full continuous proper scores.

**JCR Q1 boundary:** The posterior, action method and ranking diagnostics here are existing tools. This iteration exists to falsify a modelling assumption, not to present an incremental algorithm as publishable invention.
