# V18.4 Reserved External Evaluation Runbook

Pre-run revision gate: the revised code, regression suite and
`PRE_RUN_REVISION_AUDIT_V18_4.json/.md` must be reviewed before any formal
commands below are executed. The pre-run revision task stops after tests and
outcome-free diagnostics; it does not prepare, develop, calibrate-freeze or
formally evaluate any dataset.

The spatial objective is exact dimensionless half energy distance with
lambda_spatial in {0.10,0.25,0.50}; there is no iterative Sinkhorn solver or
entropic coefficient. Maximum epsilon_KL stays {0.025,0.050,0.100}, with
row budget epsilon_eff=max(r_radio,r_geometry)*epsilon_KL. Directional source
weights are reliabilities normalized by their sum; zero sum gives identity.
The 9 global method combinations are unchanged.

Calibration/deployment order is `q -> temperature q_T -> trust correction ->
final posterior`. Each of the 8 candidate temperatures requires a distinct
corrected-distribution cache; there is no post-correction temperature map.
Radio/geometry energies, reliability mismatches and realized KL use q_T.
References remain fitted on training q at T=1 and frozen thereafter. The 6
Student-t scale values multiply only component scales and leave probabilities
unchanged. Local anisotropic Delaunay residuals are globally capped at 30 m.

Do not run this file until
`fhm_dapp_tim/evidence/v18_4/uji_method_freeze_v18_4.json` exists and was
produced by the formal UJI V18.4 run.

The external protocols are:

- `uts`
- `tampere`
- `tuji-pooled`
- `tuji-lodo-A12`
- `tuji-lodo-POCO`
- `tuji-lodo-S20`
- `tuji-lodo-S7`
- `tuji-lodo-tabS7`

Each protocol is independent and must complete:

`prepare -> fit-freeze -> evaluate-test`

on one unchanged execution Git commit.

## Safety rule

The UJI freeze fixes all global method hyperparameters. External training may
select only:

- dataset-specific epoch on training-only development coordinates;
- dataset-specific posterior temperature on the registered 8-value grid;
- dataset-specific Student-t scale multiplier on the registered 6-value grid.

External test results cannot change the method.

## Preflight

From the repository root:

```powershell
cd D:\liuyanlunwen
git switch v18-4-trust-region-posterior
git pull --ff-only
git status --short
git rev-parse HEAD

python -m pytest -q tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py tests/test_v18_4_external.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Tracked files must be clean.

## One protocol

Replace `<PROTOCOL>` with one of the eight names above.

### 1. Training-only prepare

```powershell
python -m fhm_dapp_tim.fhm_dapp.v18_4_external --protocol <PROTOCOL> prepare
```

This creates:

`fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_training_manifest.json`

No provided test performance is evaluated or used for selection in this stage.
The provided test split is historically pre-exposed for adapter compatibility.

### 2. Training-only fit and freeze

```powershell
python -m fhm_dapp_tim.fhm_dapp.v18_4_external --protocol <PROTOCOL> fit-freeze --device cuda
```

This:

- trains with the UJI-frozen lambda_spatial;
- selects epoch on this dataset's training-only development role;
- refits on selection_refit;
- fits radio/geometry correction references on selection_refit;
- uses the UJI-frozen epsilon_KL;
- calibrates only base-posterior temperature before trust correction and Student-t scale on the
  training-only calibration role.

Outputs:

- `fhm_dapp_tim/models/v18_4/external/<PROTOCOL>_selection_refit.pt`
- `fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_training_freeze.json`

At this point the provided test performance is still unused by the formal
V18.4 model-selection/calibration process.

### 3. One-shot formal performance evaluation

Read the training freeze before continuing. If it is correct:

```powershell
python -m fhm_dapp_tim.fhm_dapp.v18_4_external --protocol <PROTOCOL> evaluate-test --device cuda
```

Before semantic test access, the command performs all possible device,
checkpoint, correction-reference and full-posterior checks. It then writes:

`fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_test_consumption_lock.json`

Only after that lock is written does it formally score the already pre-exposed
provided test benchmark.

Successful evaluation writes:

`fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_test_result.json`

If the lock exists, rerunning `evaluate-test` is forbidden even if a previous
attempt crashed.

## TUJI1 note

The TUJI1 MAT containers contain both `results_train` and `results_test`.
Earlier adapter-validation CI already inspected `results_test`, so V18.4
does not claim semantic or byte-level blindness. Formal training nevertheless
uses only `results_train`, and the held-out test performance is never used
for model/epoch/calibration/global-hyperparameter selection.

For a LODO protocol, the held-out device is excluded from:

- training scans;
- AP dictionary construction;
- device mapping;
- fit/development/calibration roles.

Its provided test split is projected into the four-device training dictionary.

## Recommended execution order

Use a fixed order and do not use earlier external results to alter later runs:

1. uts
2. tampere
3. tuji-pooled
4. tuji-lodo-A12
5. tuji-lodo-POCO
6. tuji-lodo-S20
7. tuji-lodo-S7
8. tuji-lodo-tabS7

Evidence-only commits between completed protocols are allowed. Method-source
changes are not.

## After each successful protocol

Stop after verifying the manifest, training freeze, checkpoint hash,
performance-evaluation lock, and result hash. Git staging/commit/push is outside
this runbook and is handled separately by the user.
