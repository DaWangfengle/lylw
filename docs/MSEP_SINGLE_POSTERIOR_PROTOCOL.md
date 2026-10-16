# MSEP-Loc — Single-member anchor posterior research protocol (v1)

## Aim, not a claim

Evaluate whether correcting one V17 member's anchor posterior can substantially improve previously unseen physical-site localization before changing ensemble architecture. This is an **experimental branch**, not an already-validated replacement for V17-R3 or G13. Historical G13 TUJI/UTS E5 results do not license claims about new single-member runs.

## Data contract

Use canonical existing `v18_4_datasets.py` adapters for `TUJI1`, `UTSIndoorLoc`, `UJIndoorLoc`. Build four partitions by physical site `(building,floor,round(raw_x,4),round(raw_y,4))`. Never distribute scans from a single site across FIT/DEV/CAL/TEST. FIT only: V17 neural training, anchor dictionary and multi-scan AP statistics. DEV: early stopping only. CAL: temperature, G13 scalars and *frozen-backbone* dynamic gate fitting. TEST: once, after all choices frozen. UJI's provided official validation and UTS's provided test remain distinct/pre-exposed and are **not** consumed in this study. Avoid comparing these internal TEST estimates to differently partitioned historical outer tests.

All fits are dataset-local. The number of APs is not the number of reference anchors. Use only APs visible in FIT. Exact coordinates can be near-duplicates after 4-decimal rounding; data audit must record site counts and fail if FIT anchor grouping does not match held sites. If only 1–2 sites are available, abort rather than silently break role isolation. All stats are built **only from FIT** and each query RSS at inference.

## Controlled comparisons

1. `v17_single_raw`: original V17-R3 atomic evidence of a **single member**, with fixed neural scale 1, prototype weight 32/3. Not the older V17-R0 `top_k=24`, scale .75 configuration.
2. `v17_single_temp_t06`: exact R3/G13 strong anchor-only temperature 0.6 on that same single member.
3. `v17_single_temp_fit`: scalar T calibrated on CAL; keep this as a stronger calibration baseline.
4. `v17_single_g13`: T=.6 plus original G13 Bernoulli visibility and conditional RSS Student-t df4 scores, with two nonnegative bounded coefficients calibrated on CAL. The FIT observation memory, score stabilizer and candidate normalization are imported from the **original G13** to avoid accidental formula drift.
5. `v17_single_unified_frozen`: same base model/Student-t components and same calibrated G13 global coefficients; learn a 4-feature, shared low-dimensional reliability gate on CAL. Initialization is exactly G13; no anchor-ID-specific trainable table. This is **CAL-trained correction**, not end-to-end learning.
6. `unified_joint`: start with identical pretrained V17 base, optimize encoder/retrieval keys, continuous Student-t head and reliability-aware posterior **together**, using FIT queries with their own FIT anchor masked. Original G13 memory is frozen; theta and gate are trained on FIT. Do not claim this is fully disjoint episodic meta-learning, since pretrained backbone has seen all FIT sites. `unified_joint_no_evidence` is a matched control using the same jointly-updated backbone, but with evidence switched off during *evaluation* (not a separately trained no-evidence backbone).

## Constraints and information boundaries

- The G13 features are a composite score of the SAME RSS used by V17, not an independent sensor modality. Low-dimensional gating tests residual usefulness, does not establish causality or novelty.
- Avoid geometry/oracle leakage: query true XY/building/floor may only enter loss on FIT/CAL, DEV stopping or post-hoc TEST metric computation; they cannot enter any inference feature.
- The returned coordinate is a posterior mixture **mean**, appropriate to squared Euclidean risk; it is not theoretically optimal for Euclidean MAE. Report both MAE and 2D full Student-t energy score to avoid conflating decision and posterior quality.
- `correct_neighborhood_mass`, `coverage`, `correct_neighborhood_top1_rate`, `correct_neighborhood_top10_rate`, `nearest_component_center_error_m_oracle`, `true_floor_probability`, and `true_building_probability` are diagnostic with ground-truth labels; they **must never be used for gating or during deployment**. True class probability is NOT floor classification accuracy; also report argmax building-floor joint accuracy.
- No hard true-building or true-floor conditioning. Site-macro MAE/ES is the main score: each physical site contributes equally regardless of scan count. Report row metrics for context.
- Dynamic gate has [scan count shrinkage, query AP detection coverage, reference visibility confidence, reference RSS variance-derived confidence]. These are hypotheses, not proved reliability measures. Coefficients bounded in [0,1.5], gate output always positive multiplier; zero gate reproduces G13 exactly.
- Study uses all FIT anchors by default (`--support all`) to avoid incorrectly attributing Top-K exclusions to posterior errors; `--support topk --k 192` is a separate explicit sensitivity run and must have its own output directory.
- Model training baseline retains original V17 same-building neighbor loss; cross-floor neighbor-soft-target concern remains an explicitly identified baseline limitation. Main training uses domain adversarial weight zero to make non-UJI device categories robust; this differs from the exact historical V17 configuration, and therefore these are new controlled baselines, not exact V17-R3 replica checkpoints.
- Compared fitted CAL models have different free parameter counts. Independent TEST, gate penalty, and correct comparison against stronger `temp_fit` are mandatory; a gain on CAL alone is not evidence.
- Continuous Student-t ES is Monte-Carlo approximate, using common seeded uniforms/t draws for paired models at each batch. It must not be equated with the G13 CAL discrete surrogate objective.

## Required gates / analysis

Before GPU: compile changed Python, run `pytest -q tests/test_msep_single_posterior.py`, verify loaders and AP/site counts, perform one short `--smoke` experiment (new output). The smoke mode **must evaluate DEV only**, never inspect TEST labels/predictions or expose test site keys in checkpoints. With GPU run three datasets in separate directories. Do not rerun on the same TEST while altering the design. If changed after TEST, allocate a fresh scientifically defensible holdout or describe repetitions as exploratory.

Compute paired site differences between `v17_single_raw`, `v17_single_temp_fit`, `v17_single_temp_t06`, `v17_single_g13`, `v17_single_unified_frozen`, `unified_joint_no_evidence`, and `unified_joint`. MAE gain with ES deterioration is not unqualified success. Check P90/P95, building-floor joint accuracy, correct neighborhood mass and coverage, and fit support counts. Never assert that a fitted latent anchor posterior is a calibrated true anchor distribution solely on coordinates; multiple anchor mixtures can explain the same continuous coordinate.

The final member-count/ensemble decision is **outside** this branch. This stage answers only whether single-member posterior correction deserves to become the core MSEP-Loc mechanism.

### Audit-only geometry diagnostic

`nearest_component_center_error_m_oracle` is a hindsight minimum over the actual candidate **continuous Student-t component centers** using the true query XY. It is **not a bound** on the error achievable by an arbitrary convex mixture of component centers, and is NOT accessible to a deployed model. The entropy and Top-1/Top-10 correct-neighborhood rate distinguish confidence/relative ranking from support recall. All such diagnostics must be interpreted only after the score/policy is locked.