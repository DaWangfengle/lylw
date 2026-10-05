# V18.5 Reserved External Evaluation Runbook

This runbook is registered before formal V18.5 UJI development or any V18.5
external performance result.

## 1. Preconditions

Do not start an external protocol until the formal UJI artifact

`fhm_dapp_tim/evidence/v18_5/uji_method_freeze_v18_5.json`

exists, self-validates, and has been scientifically reviewed.

The UJI freeze fixes:

- the V18.5 architecture and explicit-mask input law;
- radio-only exact-KL posterior projection;
- the spatial objective and globally selected `lambda_spatial`;
- posterior-mean point output;
- the registered T / epsilon / Student-t scale grids;
- the paired NLL one-SE -> Energy Score one-SE selector.

The UJI-selected T, epsilon and scale are **not imposed on external datasets**.
Each external dataset recalibrates those three scalars using its own
training-only calibration role and the same frozen 192-cell selector.

## 2. External protocols

Exactly eight protocols are registered:

1. `uts`
2. `tampere`
3. `tuji-pooled`
4. `tuji-lodo-A12`
5. `tuji-lodo-POCO`
6. `tuji-lodo-S20`
7. `tuji-lodo-S7`
8. `tuji-lodo-tabS7`

Recommended execution order is the order above.  Earlier external outcomes
must not alter later protocols.

## 3. Historical exposure statement

UTS, Tampere and TUJI provided evaluation sources were inspected in earlier
adapter/schema validation.  V18.5 therefore does not claim first semantic or
byte-level access.

The protected claim is narrower and auditable:

- external test **performance** does not enter method design;
- it does not enter epoch selection;
- it does not enter T / epsilon / scale calibration;
- formal scoring is one-shot after a training-only freeze.

TUJI MAT files contain both `results_train` and `results_test` in the same
container.  File-level existence/hash knowledge is therefore not equivalent
to semantic test performance use.

## 4. Per-protocol state machine

Every protocol must run:

`prepare -> fit-freeze -> evaluate-test`

The first two stages use training records only.

### 4.1 prepare

Example:

`python -m fhm_dapp_tim.fhm_dapp.v18_5_external --protocol uts prepare`

Creates:

`evidence/v18_5/external/<protocol>_training_manifest.json`

The manifest freezes:

- execution source commit;
- UJI method-freeze hash;
- UJI-frozen lambda;
- dataset training-file hashes;
- training-derived AP/category reference;
- fresh V18.5 grouped-coordinate roles;
- the registered 8 x 4 x 6 = 192 external calibration law.

No provided test loader is called.

### 4.2 fit-freeze

Example:

`python -m fhm_dapp_tim.fhm_dapp.v18_5_external --protocol uts fit-freeze --device cuda`

Training-only procedure:

1. train one model with the UJI-frozen lambda on FIT;
2. select epoch on DEVELOPMENT with the uncorrected T=1, epsilon=0, scale=1
   base posterior;
3. refit from scratch on SELECTION_REFIT for exactly that epoch count;
4. fit one label-free radio-compatibility reference per registered
   temperature on SELECTION_REFIT;
5. evaluate exactly 192 T x epsilon x scale cells on CALIBRATION;
6. select with the same paired NLL one-SE -> paired ES one-SE law;
7. freeze the dataset-specific T, epsilon and scale.

Outputs:

- `models/v18_5/external/<protocol>_selection_refit.pt`
- `evidence/v18_5/external/<protocol>_training_freeze.json`

The freeze contains the complete 192-cell surface and correction diagnostics.

## 5. TUJI leave-one-device-out isolation

For `tuji-lodo-<DEVICE>`, the held-out device is excluded before any
training-derived object is constructed.

It cannot enter:

- AP dictionary construction;
- device mapping;
- grouped-coordinate roles;
- FIT / DEVELOPMENT / SELECTION_REFIT / CALIBRATION rows;
- anchor memory;
- model fitting;
- radio compatibility references;
- T / epsilon / scale calibration.

Only after freeze may `results_test` for that held-out device be scored.

## 6. Tampere singleton anchors

Tampere training coordinates are predominantly/entirely singleton anchors.
V18.5 therefore never trusts a one-sample local RSS MAD.

The frozen count-aware scale law is:

1. local MAD only with >=2 local observations and informative MAD;
2. otherwise AP-pooled training MAD;
3. otherwise global training robust scale;
4. final lower floor .05 normalized RSS units.

No external result may alter this law.

## 7. evaluate-test irreversible gate

Example:

`python -m fhm_dapp_tim.fhm_dapp.v18_5_external --protocol uts evaluate-test --device cuda`

Before test semantic access the command must validate:

- UJI method freeze;
- external training freeze;
- checkpoint SHA;
- checkpoint/manifest/UJI links;
- source-method compatibility;
- selected T / epsilon / scale;
- temperature-reference deserialization;
- one complete training-derived V18.5 posterior preflight;
- KL-budget compliance;
- requested device availability.

Only if all preflight checks pass does it write:

`evidence/v18_5/external/<protocol>_test_consumption_lock.json`

After the lock is written, formal test loading/scoring begins.

If the lock exists, rerunning `evaluate-test` is forbidden even when a prior
scoring attempt failed.

Successful scoring creates:

`evidence/v18_5/external/<protocol>_test_result.json`

## 8. Registered external metrics

The one-shot result records at least:

- posterior-mean coordinate-macro MAE and SE;
- row MAE, median, P75 and P90 error;
- coordinate-macro Energy Score and SE;
- coordinate-macro mixture NLL and SE;
- entropy, effective anchors and top-1 probability;
- support size and support-ceiling truncation rate;
- predictive coverage/radius at 50/80/90/95%;
- building/floor error where labels are representable;
- component-center dispersion and target-tail mass;
- radio compatibility, effective KL budget, realized KL and dual diagnostics
  when epsilon > 0.

## 9. Hard scientific boundary

After any external test result is produced:

- no method-source change is permitted;
- no grid change is permitted;
- no selector change is permitted;
- no decoder change is permitted;
- no post-test retuning is permitted.

External results are evaluation evidence, not development feedback for V18.5.
