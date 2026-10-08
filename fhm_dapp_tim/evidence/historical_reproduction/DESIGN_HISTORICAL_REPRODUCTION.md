# Historical V9/V17 Reproduction Stage

## Purpose

This stage answers one question before any further model design: can the repository still reproduce the strong historical V9 and V17 localization systems at the expected sub-6 m coordinate-macro MAE level?

This stage is **not** V18 or V19 continuation. It does not reproduce V18/V19, does not tune a new architecture, and does not perform final-model fusion. V9 and V17 are the only predictive systems executed here.

## Gate

- V9 reproduced outer coordinate-macro 2D MAE must be finite and < 6.0 m.
- V17 reproduced selected R3 coordinate-macro 2D MAE must be finite and < 6.0 m.
- Final-model redesign is forbidden until both gates pass.
- No official UJI validation prediction and no external-dataset prediction are authorized in this stage.

## V9

V9 is executed through uji_v9_experiment.py without changing its model, training, scoring, selection, extension, freeze, final-refit, or outer-test logic. Only output paths are isolated under evidence/historical_reproduction and models/historical_reproduction.

The replay preserves seeds 9311..9322, initial E=8, both historical candidate fronts, the historical K/lambda grids, the information-saturation rule, and the two-member extensions through at most E=12.

## V17

The complete historical V17 R0/R1/R2 research chain references design artifacts that are no longer present in the repository. Recreating those documents would not be a faithful historical replay. Therefore this stage performs a frozen-model reproduction of the already-selected V17 R3 system.

For seeds 17301..17325 it re-runs the exact V17 R0 member-fit/early-stop epoch selection, reinitializes each member with the same seed, refits on the historical outer-refit role for the selected epoch count, and evaluates the 25-member panel with the original R3 inference/scoring helpers.

The two frozen configurations are immutable:
- selected: E=16, L=192, K=192, alpha=1;
- historical comparator: E=5, L=24, K=24, alpha=0.25.

No favorable seed subset is selected.

## After reproduction

If and only if both gates pass, the next stage may design a new final model using V9/V17 as the accuracy anchor. Mechanisms learned from V18/V19 may then be reintroduced one at a time under strict validation-only ablation. This stage itself implements no new model.
