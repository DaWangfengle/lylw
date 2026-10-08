# V17-G2: geometry-support / long-tail risk mechanism audit

**Status:** Post-outcome *exploratory diagnostics only*. Freeze 2026-10-08. This document was designed AFTER the V17-G1 confirmation had been opened, so neither the present hypotheses nor results are preregistered independent confirmation. No parameter selection, new prediction, refit, or deployment authorization follows from G2.

## Motivation / evidence that may not be rewritten

- Frozen G1 is formally STOP. Its 0.143769762 m coordinate-macro MAE gain missed BOTH the 0.15m magnitude and familywise one-sided bound conditions (upper +0.012352838 m). The energy-score familywise upper was -0.018468910 m and its fixed shuffled-density placebo was weaker than real geometry.
- G1 gains are heterogeneous: 39/79 confirmation coordinates benefit and 40/79 worsen. The five most improved coordinates contribute more than the total net MAE reduction. This is a hypothesis-generating observation, not repeatable confirmatory evidence.
- Historical V17 outer 5.725985 m is not directly comparable to the G1 595-anchor development model. Old historical next_stage_authorized=false remains unchanged.
- All G1 calibration and confirmation metrics are already disclosed. The same data may support careful retrospective causal *hypothesis generation*, but cannot certify a new mechanism.

## Concrete research question

**Does low spatial sampling support explain the unusually large improvements from G1, even after describing baseline model error, floor/building effects and query measurement count?**

Competing explanations which need to be distinguished:
1. Local geometry/sampling density: sparse support or greater distance to reference anchors predicts a larger G1 gain (baseline_mae - G1_mae > 0).
2. Generic correction of difficult points: baseline_error alone explains gains, with little geometry-specific residual association.
3. Accidental high-leverage coordinates: aggregate gain disappears after excluding 1–5 benefit outliers.
4. Sample count or floor/building heterogeneity: results may be driven by the number of RSS measurements per coordinate or by particular buildings/floors.
5. A post-hoc query *label*-geometry association is not the same as a label-free online detector; no new gating operator should be justified without identifying observable test-time inputs.

## Read-only computation

Inputs (all from immutable G1 stage):
- Original trainingData.csv SHA256 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168.
- G1 registry self SHA256 bfd44edf4f5f05f08dea2bc2c9972138b361fff529467c166c3e7ab333ebccf6.
- G1 calibration self SHA256 0729b512f74de0d607afb4f4717919a2f139c1110f62e798cd5c0c443045f86d.
- G1 freeze self SHA256 2bf7e709997f27ad83898e57f06cd63370dab508b622ea2577930b94050e75fe.
- G1 confirmation self SHA256 a54a0a8ff452adf3245bd5145afa52b56f9a2c8eedec650e3e29923b57a9b76c.

Recreate the exact V17/G1 raw float64 physical-coordinate roles using the original role constructor, and validate SHA256 checksums. V17 selection_refit has 595 disjoint physical training coordinate anchors. G1 calibration and confirmation consist of 72 and 79 coordinate groups respectively, 1529 and 1625 scans. The JSON loss vectors are indexed by sorted (building_id, floor, longitude, latitude) exact keys; never assume CSV row order aligns with vectors.

**Physical support diagnostics (query truth is used retrospectively ONLY):** For each heldout physical key, within its building/floor stratum only, report number of distinct training anchors within 10m; distance to closest, third-closest and fifth-closest training anchor; training-stratum anchor count; number of raw RSS rows at that physical key. These are evaluative features derived from known query labels, unavailable for an online unlabeled scan.

**Risk diagnostics:** Baseline vs frozen gamma0.70 MAE and ES per group; signed gain baseline-minus-corrected; one-sided gain/loss counts; median and mean; best 1/2/3/5/10 leverage sensitivity (explicitly not substitute MAEs); quartiles by support and baseline difficulty; concentration of positive gains. All fixed in code before G2 output.

**Association / anti-confounding checks:** Pearson and tied-average-rank Spearman of each support proxy with gain. Partial *rank residual correlations* projecting both ranks on (intercept, ranked baseline MAE, building/floor dummy variables). These are descriptive, not causal effect estimates; baseline MAE participates in the gain definition so mathematical coupling and regression-to-mean can create association. Repeat identical checks on calibration and confirmation and report both, without promoting agreement to independent replication.

**Directional falsifiers (interpretation only, never automated PASS):**
- Neighbor count should correlate negatively with gains on both old splits if the simple "sparser helps more" story is plausible.
- Distance to nearest/third anchor should correlate positively with gains on both old splits.
- If these signs disagree, or baseline-error/building-floor adjustments collapse the associations, the simple local geometric-density explanation loses support.
- If gain is dominated by a few coordinates, describe a tail-repair mechanism instead of broad MAE uplift.
- Positive correlations are still insufficient to claim causal identification or Q1 novelty.

## Methodological guardrails

- **Never** train/retrain, import model weights, load a G1 .pt checkpoint, run inference, or compute new prediction on any outer, official, external, calibration, or confirmation row.
- **Never** search gamma, support radius, top_k, threshold, gated architecture, or reclassify the G1 STOP. No new confidence intervals or significance assertions after disclosure.
- Existing 25 overlapping E16 subsets are not 25 independent replications. Report raw G1 risk unchanged, and do not present a new "improved" model.
- G2 code generates diagnostic output only. Diagnostics can reveal structural failures worth pursuing, but any redesigned inference needs an explicit prospective protocol and a genuinely unexposed *independent* cohort. UJI trainingData labels already opened in this project; official validation and Tampere/TUJI1/UTS must be checked for domain compatibility and prior use before claiming independent validation.
- This analysis alone cannot prove geometric causality or JCR Q1 novelty. Reserve literature screening and independent transfer evaluation for a later registered stage. In particular, adaptive tail gating based on the true coordinate is invalid because the query coordinate is what the predictor must estimate.

## Outputs

New files only, under evidence/v17_g2_mechanism:
- G2_MECHANISM_AUDIT.json: hashed immutable original provenance, per-coordinate physical diagnostics/risk, both role reports, descriptive sign-checks, explicit non-confirmatory labels.
- RUN_REPORT_V17_G2.md: reproducible human-readable interpretation.
- Execution logs/tests from Codex under evidence/v17_g2_mechanism/run_audit/.

No G1 registry, calibration, freeze, confirmation, model weights or historical evidence may be edited. One-shot G2 report: no rewrite to cherry-pick favorable geometry proxies. A negative mechanism result is a valuable answer; it should redirect paper research rather than trigger unrestricted experimentation.
