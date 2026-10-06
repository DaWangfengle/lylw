# V19-R0 UJI Runbook

This runbook executes an accuracy-recovery development experiment only.

## 0. Pre-check

Use branch:

v19-r0-accuracy-recovery

Compile:

python -m py_compile fhm_dapp_tim/fhm_dapp/v19_r0_core.py
python -m py_compile fhm_dapp_tim/fhm_dapp/v18_5_predictive_decision_probe.py
python -m py_compile fhm_dapp_tim/fhm_dapp/uji_v19_r0_accuracy_recovery.py

Run:

python -m pytest -q \
  tests/test_v18_5_core.py \
  tests/test_v18_5_model.py \
  tests/test_v18_5_radio.py \
  tests/test_v19_r0_core.py \
  tests/test_v18_5_predictive_decision_probe.py \
  tests/test_uji_v19_r0_accuracy_recovery.py

Any failure stops the experiment.

## 1. Read-only V18.5 decision probe

Run:

python -m fhm_dapp_tim.fhm_dapp.v18_5_predictive_decision_probe --device cuda

This writes:

fhm_dapp_tim/evidence/v19_r0/V18_5_PREDICTIVE_DECISION_PROBE.json
fhm_dapp_tim/evidence/v19_r0/V18_5_PREDICTIVE_DECISION_PROBE.md

Do not modify V18.5 based on the probe during this run.

## 2. Register V19-R0

Run:

python -m fhm_dapp_tim.fhm_dapp.uji_v19_r0_accuracy_recovery prepare

This binds:
- exact V18.5 FIT/DEVELOPMENT rows;
- exact inherited lambda=0 epoch-27 member;
- 16-member seed panel;
- E={1,2,4,8,16};
- L=192;
- K=192;
- alpha=1.

## 3. Train members 2-16

Member 1 is inherited and MUST NOT be retrained.

For each member index 2 through 16 run:

python -m fhm_dapp_tim.fhm_dapp.uji_v19_r0_accuracy_recovery train-member --member-index N --device cuda

All 15 new members must finish exactly 27 epochs.

Do not early-stop or select member-specific epochs.

## 4. Cache all 16 development members

For each member index 1 through 16 run:

python -m fhm_dapp_tim.fhm_dapp.uji_v19_r0_accuracy_recovery cache-member --member-index N --device cuda

Each cache evaluates the complete anchor dictionary and records:
- dense member probability;
- member-specific component locations/scales;
- Top-192 support;
- radial predictive scale;
- retained Top-192 mass.

## 5. Evaluate member prefixes

Run:

python -m fhm_dapp_tim.fhm_dapp.uji_v19_r0_accuracy_recovery evaluate --device cuda

The output is:

fhm_dapp_tim/evidence/v19_r0/development_result_v19_r0.json

Required E profiles:

1
2
4
8
16

For each E report:
- posterior-mean coordinate-macro MAE +/- SE;
- predictive-spatial-median coordinate-macro MAE +/- SE;
- Energy Score +/- SE;
- NLL +/- SE;
- mean member-weight entropy;
- mean maximum member weight;
- consensus Top-1;
- mean member Top-192 retained mass.

Also report:
- inherited V18.5 member dense-all-anchor result;
- E=1 Top-192 minus dense-all-anchor;
- paired E={2,4,8,16} minus E=1 effects.

## Hard boundaries

Do not:
- open UJI validationData.csv;
- use V18.5 calibration rows to train/select V19;
- access external test outcomes;
- change E/L/K/alpha after seeing results;
- tune member seeds;
- change 27 epochs;
- change lambda_spatial=0;
- add radio correction;
- add temperature or Student-t scale calibration;
- select a final member count;
- implement V19-R1.

Stop after the V19-R0 development result is written.
