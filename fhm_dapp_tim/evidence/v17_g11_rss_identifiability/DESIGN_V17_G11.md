# G11 — RSS identifiability versus frozen V17 ranking

**2026-10-10 — one retrospective, no-training science iteration; NOT a publication promotion.**

## The falsifiable question

G6–G9 showed that sharpening the anchor marginal helps mean Euclidean MAE when dominant spatial modes are plausible, but may catastrophically amplify wrong modes. G10's 4-coefficient evidence recycler improves two source-domain means but generally does not fix initially wrong Top-1 modes. G11 asks: **Does frozen FIT RSS memory contain an accessible spatial discrimination signal that V17 R3 ranking fails to use, or do elementary raw RSS comparisons also confuse distant physical sites?**

A failed RSS nearest-neighbor does NOT imply fundamental information-theoretic nonidentifiability. A good 1NN does NOT establish an oracle attainable error or unique learned model failure.

## Four fixed decision baselines

All use exact original frozen E5 networks, original FIT coordinate and RSS memory, and original TRAIN-internal disjoint CAL/CONFIRM sites.

1. Frozen V17 Anchor T=.6: original R3 Top-L192, Top-K192, alpha1 and Student-t df4, point mixture mean. Recompute the original mean; no old outcome JSON read during scoring.
2. Raw RSS L2 1NN: the EXACT pre-existing R3 upstream FIT prototype RSS distance computed on model-seen APs. Not novel independent sensor information.
3. Raw RSS L2 5NN: original five nearest FIT anchors, weights 1/(d+0.001)^2, all buildings/floors allowed. Predetermined count and smoothing, no search.
4. AP-presence 1NN: from original FIT scans only, estimate model-seen AP detection Bernoulli frequency for each frozen reference anchor, Beta(.5,.5) smoothing. AP detected only when normalized model RSS > -0.999 (matching original FIT anchor memory). Query/reference Brier distance is the mean squared difference in observed AP bit and FIT detection frequency. No label-dependent candidates. Caveat: observed signal clipped to -110 dBm may also be exactly -1 in normalized RSS, so this observation channel cannot distinguish clipping from truly missing.

Predictions are fully **label free**, with no true building/floor preselection; only original FIT reference geometries may be known. Mapping of raw FIT keys to frozen anchor IDs must be verified as a full one-to-one bijection by building, floor and physical XY tolerance <=0.03m before computing AP frequency prototypes.

## Frozen retrospective diagnostic rules

Use true building/floor/XY AFTER prediction, for diagnosis only:
- Top-1 in true same-floor/building 10m neighborhood, comparing raw, AP, and original V17.
- Define NEAR: same building/floor and distance <=10m. Clearly WRONG: any different building/floor, or distance >=20m. For each raw distance score calculate best clearly-wrong minus best near score; positive margin means raw nearest correct-region anchor beats nearest geographically distant impostor in that distance metric. Missing nearby FIT anchors or clearly-wrong anchors means undefined margin, serialized as null.
- Original Top-K rank of first selected near anchor (null if none); original shared mass on true near anchors.
- Physical-site macro averaging means average rows per site, then sites equally. Site hit rate is fractional where repeated RSS scans vary.
- Paired quadrant analysis raw 1NN vs V17 Top-1 (both correct, raw only, V17 only, both wrong), which uses labels as diagnostic and **cannot be used as a model gating feature**. Site category is determined by at least half of its repeated query scans yielding Top-1 near (<0.5 means far), not by averaging any oracle prediction. Also report AP detection Top-1 near when both raw L2 and frozen V17 Top-1 are far.

Do not move the 10m/20m thresholds after looking at results.

## Evaluation and scientific boundary

Five historical TRAIN-internal roles: TUJI CAL+CONFIRM, UTS CAL+CONFIRM, Tampere CONFIRM. Original FIT rows only for reference memory. No new model training, no threshold/model selection, no optimizer, no provided official test, no actual blind confirmation. Original checkpoint identities G4/G5/G6 must match and remain byte-identical.

Report coordinate-macro Euclidean 2D MAE, P90/P95/max physical-site error and complete paired physical-site score/margin/hit vectors for four decision baselines. Raw 1NN/5NN are **point predictors**, not continuous Student-t mixtures; do NOT compare them using purported continuous Energy Score. Verify original V17 Anchor T=.6 score matches G8 after run; do not use G8 outcome in G11 scoring.

Scientific next-step decision: if simple raw RSS or AP-detection can identify regions missed by the learned consensus, test a new spatially contrastive RSS representation/likelihood on **fresh physically isolated development data**. If all fail, inspect scanner variability or new sensing evidence; do not conclude Bayes impossibility. If V17 outperforms all raw point predictors, avoid replacing V17 with 1NN and scrutinize spatial-mode decision separately.

**JCR Q1 threshold** is stronger than the present retrospective benchmark: new independent splits, equal-budget full V17 E5/E16/E25 comparisons, full proper continuous predictive scoring and hard-tail robustness.
