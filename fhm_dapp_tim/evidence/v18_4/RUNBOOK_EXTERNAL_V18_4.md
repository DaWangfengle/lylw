# V18.4 Reserved External Evaluation Runbook

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

No provided test outcome is semantically accessed.

### 2. Training-only fit and freeze

```powershell
python -m fhm_dapp_tim.fhm_dapp.v18_4_external --protocol <PROTOCOL> fit-freeze --device cuda
```

This:

- trains with the UJI-frozen lambda_OT;
- selects epoch on this dataset's training-only development role;
- refits on selection_refit;
- fits radio/geometry correction references on selection_refit;
- uses the UJI-frozen epsilon_KL;
- calibrates only posterior temperature and Student-t scale on the
  training-only calibration role.

Outputs:

- `fhm_dapp_tim/models/v18_4/external/<PROTOCOL>_selection_refit.pt`
- `fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_training_freeze.json`

At this point the provided test outcome is still unused.

### 3. One-shot provided test

Read the training freeze before continuing. If it is correct:

```powershell
python -m fhm_dapp_tim.fhm_dapp.v18_4_external --protocol <PROTOCOL> evaluate-test --device cuda
```

Before semantic test access, the command performs all possible device,
checkpoint, correction-reference and full-posterior checks. It then writes:

`fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_test_consumption_lock.json`

Only after that lock is written does it load/score the provided test split.

Successful evaluation writes:

`fhm_dapp_tim/evidence/v18_4/external/<PROTOCOL>_test_result.json`

If the lock exists, rerunning `evaluate-test` is forbidden even if a previous
attempt crashed.

## TUJI1 note

The TUJI1 MAT containers contain both `results_train` and `results_test`.
Training therefore reads the container bytes. Before the test lock, however,
the code selects and uses only `results_train`; `results_test` is not
canonicalized or scored. This is a semantic outcome-access boundary, not a
claim of byte-level blindness.

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

Commit the manifest, training freeze, checkpoint, test lock, and test result
without modifying method code. If checkpoint storage policy prevents committing
the .pt file, preserve it separately and do not alter its SHA256 recorded in
the training freeze.
