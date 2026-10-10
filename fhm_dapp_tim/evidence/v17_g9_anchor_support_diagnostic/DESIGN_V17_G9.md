# G9: Frozen anchor-support, retrieval-survival and probability-mass diagnostic

**Pre-outcome scientific question (2026-10-10).** G6–G8 show that sharpening shared anchor consensus changes MAE far more than changing member weights, but that observation does not identify whether failures come from candidate **coverage**, candidate **selection**, candidate **probability assignment**, or point-decision choice. G9 uses the original 5-member E5 frozen networks from G6 TUJI, G4 UTS and G5 Tampere. All analyzed TRAIN-internal CAL/CONFIRM labels have already been seen: this is **purely retrospective mechanistic diagnosis**, no new external or blind validation, no statistical promotion or method selection.

## The four explicitly different support concepts

For each RSS observation with true site (building, floor, XY) and original FIT anchor dictionary:

1. **FIT geometric coverage**: Does *any original training anchor* with exactly the same building and floor lie within a fixed 2D radius of the true coordinate? This is a *discrete anchor geometry indicator*, **not an impossibility theorem for continuous localization**.
2. **Original E5 member retrieval**: Does the *union of five member Top-L=192* contain a correct-neighbourhood anchor? Anchor IDs are global and duplicates are counted once.
3. **Original R3 consensus**: Does its final Top-K=192 anchor dictionary contain such an anchor? Dense alpha=1 consensus includes epsilon-smoothing and can occasionally select an ID outside member Top-L union; count these explicitly rather than assume nested sets.
4. **Probability allocation**: How much base shared consensus mass (c_k) lies on selected anchors in the true-neighbourhood? Also record fixed anchor-only T=0.6 mass on the **same** selected anchors. This metric uses the true label **only for post-hoc diagnosis**, and may never be fed into a deployable model, action rule or gate.

Separately compute nearest **residual-adjusted Student-t component centre**, with the floor/building identity inherited from its anchor ID. It is not the same as anchor-dictionary proximity: learned residuals can bridge geometric gaps. Report all distances and binary neighborhood coverage separately. Do not confuse 2D closeness on another floor with correct 3D/semantic support. The original published MAE remains macro-2D; these new support metrics deliberately require same floor/building.

## Fixed radii, no tuning

Use **5 m and 10 m**, chosen as interpretable physical spatial neighborhoods, and report **both** even when one behaves worse. No dataset-specific radius fitting or target-label-driven filtering. Each role is grouped by original physical site: means within repeated-device/site rows, then macro mean across sites. Save physical-site vectors for follow-up assessment. A site's hit fraction can be between zero and one because repeated RSS observations may have different Top-L/Top-K supports. Report mean mass, geometric/union/consensus hit fractions, conditional recall given FIT geometry, and exact baseline vs **anchor-only T=.6** point-MAE sanity check (no new stochastic Energy Score computation needed for this diagnostic).

These radius thresholds are **descriptive**, and there is no fixed criterion that alone establishes probability-weighting failure. In particular, zero true-neighborhood mass cannot prove correct continuous residual modes are absent; anchor geometry coverage alone is no oracle benchmark. Likewise, positive true-neighborhood probability does not guarantee accurate point estimation. An advanced learned mode model is justified only after examining these distinctions with evidence.

## Scientific implementation

Reuse original frozen model loading, original FIT group roles, original RSS canonical TRAIN adapters, R3 retrieval, original `_member_top_l`, original `_dense_joint_law`, and an independent invocation of `dense_alpha_consensus_prior`. Numerically assert that the consensus probabilities coincide with the member-summed R3 joint probabilities, and the gathered residual component positions coincide with the dense joint locations. Treat mismatches as scientific STOP, never silently change the reference operator. No .pt retraining, no changes to G1–G8 files, no original outcome JSONs used to score or select anything, no official test files.

## How to interpret without falling into a circular oracle

Classify failures conditionally, **not** by post-hoc optimizing a new detector:

- If FIT has no nearby same-floor anchor: sparse reference geometry, but learned continuous residual can still locate accurately; investigate representation and region support, not just probabilities.
- If FIT contains a near anchor but original five-member Top-L misses it: **retrieval candidate suppression**; a reweighting rule restricted to Top-L cannot recover that anchor directly.
- If Top-L contains near anchors but Top-K loses all of them: **consensus selection/ranking loss**, with epsilon edge cases explicitly counted.
- If Top-K contains nearby anchors yet assigns them little mass: **distribution/mode weighting may be implicated**, though optimality is not identifiable from one sample.
- If Top-K and probability are reasonable yet mean error remains large: investigate multimodal point-decision mismatch or residual-centre bias, rather than assuming probability calibration is the only cause.

The 5/10-m results and error examples inform only the *next prospective experiment design*, not G9 model promotions. Compare actual absolute MAE to original G8 baseline/Anchor-only if desired **after G9 final freeze and execution**, not as a source for any G9 parameter.

A JCR Q1 paper still requires a novel, principled solution on unviewed data, competitive absolute full-budget MAE, calibrated Energy Score, hard-location tails, and independent multi-dataset replication. Keep the diagnostic minimal.
