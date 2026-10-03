# V18.4 UJI Formal Development Runbook

This runbook executes the registered UJI method-development protocol for
V18.4. It uses only `data/UJIndoorLoc/trainingData.csv`. The already exposed
`validationData.csv` is not opened by prepare, develop, or calibrate-freeze.

## Non-negotiable run law

Run all three stages on one unchanged Git commit:

1. prepare
2. develop
3. calibrate-freeze

Do not pull, commit, cherry-pick, edit tracked files, or change branches between
those stages. The registry records the exact source commit and later stages
fail closed if HEAD changes.

Generated result/model files may remain untracked until the complete freeze
succeeds. Only after freeze should they be committed as evidence.

The formal method grid is fixed:

- lambda_OT: 0.10, 0.25, 0.50
- epsilon_KL: 0.025, 0.050, 0.100
- 9 global UJI method combinations total
- epsilon_OT = 0.10
- point decision = posterior-weighted component-center geometric median
- calibration temperature grid:
  0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00
- calibration Student-t scale grid:
  0.50, 0.75, 1.00, 1.25, 1.50, 2.00

No extra sweep is allowed after outcomes are seen.

## Windows PowerShell commands

From the repository root (historically `D:\liuyanlunwen`):

```powershell
cd D:\liuyanlunwen

git switch v18-4-trust-region-posterior
git pull --ff-only

git status --short
git rev-parse HEAD

python --version
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
nvidia-smi
```

The tracked working tree must be clean before prepare. Existing untracked
formal V18.4 output files from an earlier incomplete attempt should be moved
out of the repository rather than overwritten.

### 1. Verification gate

```powershell
python -m pytest -q tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py
python -m fhm_dapp_tim.fhm_dapp.v18_4_realdata_smoke
```

Required properties:

- all tests pass;
- smoke status is success;
- `uji_official_validation_used_for_v184_selection=false`;
- `external_provided_tests_preexposed_for_adapter_validation=true`;
- `external_test_performance_used_for_v184_selection=false`;
- finite gradients and predictions.

Do not continue if any condition fails.

### 2. Freeze UJI training roles

```powershell
python -m fhm_dapp_tim.fhm_dapp.uji_v18_4_development prepare
```

Expected output file:

`fhm_dapp_tim/evidence/v18_4/uji_registry_v18_4.json`

The registry must contain:

- trainingData SHA256;
- exact source commit SHA;
- fit/development/selection_refit/calibration row hashes;
- 933 total physical coordinate groups;
- 520 APs;
- `uji_official_validation_used_for_v184_selection=false`.

After this command, do not change Git HEAD.

### 3. Run registered 3 x 3 UJI development

```powershell
python -m fhm_dapp_tim.fhm_dapp.uji_v18_4_development develop --device cuda
```

This trains exactly one base model for each registered lambda_OT value.
For each lambda_OT:

- training uses fit only;
- epoch is selected using the uncorrected base posterior on development only;
- correction reference is fitted on fit only after epoch freeze;
- three registered epsilon_KL values are evaluated on development.

The 9 combinations are selected by:

1. minimum coordinate-macro Energy Score;
2. retain combinations within 0.05 m of the best ES;
3. minimum GM coordinate-macro MAE;
4. minimum coordinate-macro mixture NLL;
5. smaller lambda_OT;
6. smaller epsilon_KL.

Expected outputs include:

- `fhm_dapp_tim/models/v18_4/uji_lambda_0p10.pt`
- `fhm_dapp_tim/models/v18_4/uji_lambda_0p25.pt`
- `fhm_dapp_tim/models/v18_4/uji_lambda_0p50.pt`
- `fhm_dapp_tim/evidence/v18_4/uji_development_v18_4.json`

Do not use UJI validationData or any UTS/Tampere/TUJI test result to alter the
selection.

### 4. Refit, calibrate, and freeze global V18.4 method law

```powershell
python -m fhm_dapp_tim.fhm_dapp.uji_v18_4_development calibrate-freeze --device cuda
```

This stage:

- refits the selected lambda_OT model on selection_refit;
- uses the already selected epoch count;
- fits correction references on selection_refit;
- uses calibration groups only for the fixed 8 x 6 scalar calibration surface;
- records tail-dilution diagnostics for the selected calibration;
- writes the global method freeze.

Expected outputs:

- `fhm_dapp_tim/models/v18_4/uji_selection_refit.pt`
- `fhm_dapp_tim/evidence/v18_4/uji_calibration_v18_4.json`
- `fhm_dapp_tim/evidence/v18_4/uji_method_freeze_v18_4.json`

A valid freeze must retain:

- the same source commit SHA as the registry;
- linked registry/development/calibration/checkpoint SHA256 values;
- `uji_official_validation_used_for_v184_selection=false`;
- `external_test_performance_used_for_v184_selection=false`.

## After a successful freeze

Stop the experiment task after verifying the generated registry, development,
calibration, freeze, and checkpoint hashes. Git staging/commit/push is outside
this runbook and is handled separately by the user.
