# V18.5 UJI Formal Runbook

This runbook is registered before any formal V18.5 UJI result is produced.

## 1. Boundary

Only `data/UJIndoorLoc/trainingData.csv` is allowed during
`prepare -> develop -> calibrate-freeze`.

Do not open `validationData.csv`.

Do not score UTS/Tampere/TUJI test outcomes.

## 2. Prepare

Command:

`python -m fhm_dapp_tim.fhm_dapp.uji_v18_5_development prepare`

Expected artifact:

`fhm_dapp_tim/evidence/v18_5/uji_registry_v18_5.json`

The registry freezes:

- source SHA;
- training source hash;
- fresh V18.5 grouped-coordinate roles;
- lambda grid;
- temperature grid;
- epsilon grid;
- scale grid.

## 3. Develop

Command:

`python -m fhm_dapp_tim.fhm_dapp.uji_v18_5_development develop --device cuda`

Train exactly four base models:

- lambda=0
- lambda=.10
- lambda=.25
- lambda=.50

For every lambda:

- train FIT only;
- evaluate eligible epochs 12..60 on DEVELOPMENT;
- use T=1, epsilon=0, scale=1;
- no radio correction;
- no GM.

Epoch law:

1. minimum coordinate-macro Energy Score;
2. <= .05m Energy Score shortlist;
3. minimum mixture NLL;
4. minimum posterior-mean MAE;
5. earlier epoch.

Global lambda law:

1. minimum selected-base Energy Score;
2. <= .05m shortlist;
3. minimum NLL;
4. minimum posterior-mean MAE;
5. smaller lambda.

Expected checkpoints:

- `models/v18_5/uji_lambda_0p00.pt`
- `models/v18_5/uji_lambda_0p10.pt`
- `models/v18_5/uji_lambda_0p25.pt`
- `models/v18_5/uji_lambda_0p50.pt`

Expected result:

`evidence/v18_5/uji_development_v18_5.json`

## 4. Selection refit

`calibrate-freeze` rebuilds the selected lambda model from scratch on
`selection_refit = fit + development` for exactly the selected epoch count.

It rebuilds:

- explicit-mask anchor memory;
- count-aware radio scales;
- model state;
- one radio compatibility reference for every registered temperature.

Calibration labels do not fit the radio references.

## 5. Temperature-specific references

For each T in

`{.5,.75,1,1.25,1.5,2,2.5,3}`

on selection_refit:

- compute q;
- compute q_T;
- profile the radio affine nuisance under q_T;
- collect the final profiled objective;
- freeze its empirical reference.

A reference fitted at one T cannot be used at another T.

## 6. Joint calibration

Command:

`python -m fhm_dapp_tim.fhm_dapp.uji_v18_5_development calibrate-freeze --device cuda`

Calibration grid:

- 8 T values;
- epsilon in `{0,.025,.05,.1}`;
- scale in `{.5,.75,1,1.25,1.5,2}`.

Total = 192 cells.

For each T:

- q_T computed once per batch;
- radio profile computed once per batch;
- compatibility computed once;
- four epsilon posteriors computed;
- six Student-t scale variants scored per posterior.

Use common Student-t randomness for all 192 cells.

## 7. Calibration selector

First paired NLL one-SE set.

Second paired Energy Score one-SE set inside it.

Final deterministic choice:

1. minimum posterior-mean MAE;
2. smaller epsilon;
3. T closest to 1 in abs(log T);
4. scale closest to 1 in abs(log scale);
5. lower T;
6. lower scale.

`epsilon=0` remains fully eligible.

## 8. Final freeze

Expected artifacts:

- `models/v18_5/uji_selection_refit.pt`
- `evidence/v18_5/uji_calibration_v18_5.json`
- `evidence/v18_5/uji_method_freeze_v18_5.json`

The freeze records:

- lambda;
- epoch;
- T;
- epsilon;
- scale;
- radio-only exact KL projection;
- posterior-mean decoder;
- no geometry correction.

## 9. Read-only formal artifact audit

After `calibrate-freeze` succeeds, validate the complete provenance and
numerical graph before interpreting results:

`python -m fhm_dapp_tim.fhm_dapp.v18_5_artifact_audit`

The validator must confirm:

- registry self-hash, source SHA and trainingData hash;
- exact FIT / DEVELOPMENT / SELECTION_REFIT / CALIBRATION role algebra;
- exactly four lambda checkpoints with matching hashes;
- checkpoint explicit-mask memory and finite model tensors;
- exactly 192 unique calibration cells covering the registered Cartesian grid;
- epsilon=0 identity cells;
- returned-dtype KL budget compliance;
- one sorted selection-refit radio reference per registered temperature;
- registry -> development -> refit checkpoint -> calibration -> freeze hashes;
- exact agreement between selected calibration cell and frozen T/epsilon/scale.

This validator is read-only and must not access UJI validation or any external
test split.

## 10. Hard stop

After method freeze and artifact-audit PASS:

STOP.

Do not open UJI official validation.

Do not run external provided tests until the frozen UJI artifacts and the
scientific behavior of the selected method have been reviewed.
