# V18.4 pre-run revision acceptance retry

Status: **PASS**. The narrow acceptance retry is complete; execution stops for scientific review.

Source revision: `1e8db29eb80442bb885edf955aab2dc9dca57d45` on `v18-4-trust-region-posterior`.
Interpreter: `D:\miniminiconda\envs\liuyan\python.exe`.
PyTorch: `2.10.0+cu128`.

The production implementation is byte-for-byte unchanged from the reviewed revision. Only the three specified tests were repaired. The prior 64-pass/5-fail execution records and original raw log were preserved byte-for-byte.

## Exact repairs

1. `tests/test_v18_4_core.py`: construct the rotation angle and sine/cosine in float64 before constructing the matrix. The `1e-12` comparison, bitwise unchanged normal transform, and singular-value bound assertions are unchanged.
2. `tests/test_v18_4_model.py`: replace the one-argument zero-reliability side effect with `lambda mismatch, reference: torch.zeros_like(mismatch)`. Bitwise q_T identity and exact-zero effective budget, realized KL and rho assertions are unchanged.
3. `tests/test_v18_4_external.py`: import the external module and call its patched `_current_head` from the synthetic freeze helper. The production clean-tree/source-freeze checks and dedicated dirty/source-changed rejection tests remain active and passed.

## Executed acceptance gate

| Stage | Result |
|---|---|
| py_compile | PASS, all 10 prescribed implementation/test modules |
| Prescribed pytest suite | PASS, 69 passed / 0 failed; 1 invocation; no skip or xfail |
| Standalone CPU smoke | PASS |
| Standalone CUDA smoke | PASS |
| Revision audit | PASS on CPU and CUDA |

Exact commands, in execution order:

```text
D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/v18_4_core.py fhm_dapp_tim/fhm_dapp/v18_4_model.py fhm_dapp_tim/fhm_dapp/uji_v18_4_development.py fhm_dapp_tim/fhm_dapp/v18_4_external.py fhm_dapp_tim/fhm_dapp/v18_4_realdata_smoke.py fhm_dapp_tim/fhm_dapp/v18_4_revision_audit.py tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py tests/test_v18_4_external.py
```
Exit code: 0; wall time: 0.144 s.

```text
D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py tests/test_v18_4_external.py tests/test_v18_4_protocol.py tests/test_v18_4_datasets.py tests/test_v18_4_capacity_audit.py
```
Exit code: 0; wall time: 71.466 s.

```text
D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.v18_4_realdata_smoke --device cpu
```
Exit code: 0; wall time: 8.905 s.

```text
D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.v18_4_realdata_smoke --device cuda
```
Exit code: 0; wall time: 10.001 s.

```text
D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.v18_4_revision_audit
```
Exit code: 0; wall time: 13.438 s.

The complete stdout/stderr output of every command is embedded in the acceptance JSON. Separate raw logs are preserved locally under `run_audit/` and remain ignored by the repository policy.

## Standalone real-data smoke metrics

| Metric | CPU | CUDA |
|---|---:|---:|
| fit_rows | 14126 | 14126 |
| development_rows | 2936 | 2936 |
| fit_coordinate_groups | 655 | 655 |
| development_coordinate_groups | 139 | 139 |
| anchors | 655 | 655 |
| aps | 520 | 520 |
| spatial_scale_m | 10.492310524 | 10.492310524 |
| training_batch_rows | 8 | 8 |
| parameters | 614553 | 614553 |
| loss | 15.592408942 | 15.5280616233 |
| anchor_nll | 3.96330833435 | 3.93143725395 |
| coordinate_nll | 8.62640285492 | 8.62640285492 |
| spatial_energy_loss | 2.75937033405 | 2.78577856205 |
| gradient_tensor_count | 63 | 63 |
| gradient_norm | 11.6759294284 | 11.6512646448 |
| optimizer_steps | 0 | 0 |

On both devices, all loss terms were finite, all computed model gradients were finite, the aggregate gradient norm was nonzero, the base posterior and prediction were finite, padding posterior mass was exactly zero, and model optimizer steps were zero.

All standalone loss terms:

| Term | CPU | CUDA |
|---|---:|---:|
| anchor_nll | 3.96330833435 | 3.93143725395 |
| coordinate_nll | 8.62640285492 | 8.62640285492 |
| spatial_energy_loss | 2.75937033405 | 2.78577856205 |
| invariant_aux_loss | 5.06356906891 | 5.01333236694 |
| hierarchy_loss | 1.81670558453 | 1.76425528526 |
| domain_loss | 2.77220201492 | 2.76631784439 |
| loss | 15.592408942 | 15.5280616233 |

CPU/CUDA diagnostics are both successful; they are not a claim of cross-device bitwise equality of the training batch.

## Revision audit evidence

JSON: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.json`.
Markdown: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.md`.

Registered KL numerical tolerance: `1e-07`.
Registered residual numerical tolerance: `1e-05 m`.

| Check | CPU | CUDA |
|---|---:|---:|
| Identical spatial loss | 0 | 0 |
| Near spatial loss | 0.08 | 0.08 |
| Far spatial loss | 19.28 | 19.28 |
| Spatial logit gradient norm | 0.163659200856 | 0.163659200856 |
| Real effective transform maximum singular value (m) | 30.0000016061 | 30.0000016061 |
| Real maximum component displacement (m) | 0 | 0 |
| Synthetic oversized-transform maximum displacement (m) | 29.8184517106 | 29.8184517106 |
| Sample support truncation fraction | 0 | 0 |

CPU maximum realized-KL minus effective-budget rounding difference: `2.25926184011e-09`, within the original `1e-7` tolerance.

CUDA maximum realized-KL minus effective-budget rounding difference: `3.88043569499e-10`, within the original `1e-7` tolerance.

Synthetic realized KL decreases from 0.1 to 0.05 to 0.01 to exactly zero under the registered same-direction reliability construction. Zero reliability gives bitwise identity and exact-zero padding. Real-data KL is measured against q_T. The correction profiler leaves all network gradients absent and model state bitwise unchanged.

The real float32 capped transform has maximum singular value approximately 30.0000016061 m, an excess of approximately 0.0000016061 m within the original 0.00001 m numerical tolerance. No tolerance was changed.

Natural support sizes on the 16 fit/development audit scans: `[49, 55, 105, 51, 58, 67, 64, 48, 62, 64, 73, 54, 49, 44, 75, 44]`. No sample hit the 128-candidate ceiling. Every positive forced target was retained within the existing float32 mass tolerance; positive self target mass was exactly zero.

The audit file-open guard recorded only `data/UJIndoorLoc/trainingData.csv` (three opens: two device runs and the training-file hash). No official validation or external test-performance file was used by the smoke/audit. Existing dataset-adapter regression tests retain their original file-reading scope.

## Scientific calibration note

> dataset-specific scalar base-posterior calibration before the frozen reliability-bounded correction law.

T changes q_T before correction. Consequently it may change nuisance profiling, geometry coherence energies, mismatches, empirical reliabilities, the effective row KL budget, and the final correction. It is not merely a post-hoc uncertainty-only scale. This is the reviewed V18.4 definition, with no new hyperparameter. Student-t scale calibration remains separate and probability-neutral.

## Limits and observed warnings

No acceptance-blocking scientific defect remains in the prescribed checks. These are small, outcome-free numerical/design diagnostics on an untrained model: they do not establish localization accuracy, optimization convergence, or full-dataset support sufficiency. The real-data residual head is initially zero, so the real sample displacement is zero; nonzero oversized residual behavior is exercised by the synthetic tests/audit.

The audit emitted one nonfatal PyTorch warning when converting a requires-grad scalar loss to a Python number after backward for a diagnostic assertion. It is preserved in full in the execution JSON/raw log. The finite, nonzero gradient check had already run; no source change, check bypass, or tolerance adjustment was applied.

## Preserved boundaries and files

Formal UJI prepare/develop/calibrate-freeze: **NOT RUN**.
External formal UTS/Tampere/TUJI evaluation: **NOT RUN**.
No formal registry, development, calibration, freeze, external formal artifact or V18.4 checkpoint was created. The global method grid is unchanged at 9 combinations. The scientific audit files were created once and not overwritten. No commit or push was performed in this acceptance task.

Changed/new files (three test files and four diagnostic records):

- `D:\liuyanlunwen\tests\test_v18_4_core.py`
- `D:\liuyanlunwen\tests\test_v18_4_model.py`
- `D:\liuyanlunwen\tests\test_v18_4_external.py`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.json`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.md`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\pre_run_revision_acceptance_v18_4.json`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\PRE_RUN_REVISION_ACCEPTANCE_V18_4.md`

Preserved historical records and SHA-256 digests:

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\PRE_RUN_REVISION_EXECUTION_V18_4.md`: `7b9a24c8093695471cc73064a9780b735df2dfaade4873420e6fb5a2233efbc9`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\pre_run_revision_execution_v18_4.json`: `dbf34ff40456d4bedee0a9b5130a5a263ff74e33cced6bb3b08338fbbaa9cd48`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\pre_run_revision_pytest.log`: `eaa7a690f39810c281ceddc02c21b763c80cc7db7a5f64368f00be9276737a21`

## Test-only patch

```diff
diff --git a/tests/test_v18_4_core.py b/tests/test_v18_4_core.py
index 0659d5a..0d49481 100644
--- a/tests/test_v18_4_core.py
+++ b/tests/test_v18_4_core.py
@@ -380,9 +380,11 @@ def test_reliable_source_dominates_direction_without_probability_floor() -> None


 def test_anisotropic_cap_preserves_axes_and_ratio() -> None:
-    angle = .37
-    rotation = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
-                             [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]], dtype=torch.float64)
+    angle = torch.tensor(.37, dtype=torch.float64)
+    rotation = torch.stack([
+        torch.stack([torch.cos(angle), -torch.sin(angle)]),
+        torch.stack([torch.sin(angle), torch.cos(angle)]),
+    ])
     oversized = rotation @ torch.diag(torch.tensor([300., 90.], dtype=torch.float64)) @ rotation.T
     normal = torch.tensor([[9., 1.], [1., 3.]], dtype=torch.float64)
     transforms = torch.stack([oversized, normal, torch.zeros_like(normal)])
diff --git a/tests/test_v18_4_external.py b/tests/test_v18_4_external.py
index 9b9e841..dd23ea3 100644
--- a/tests/test_v18_4_external.py
+++ b/tests/test_v18_4_external.py
@@ -11,6 +11,7 @@ import torch
 import pytest

 from fhm_dapp_tim.fhm_dapp.uji_v18_4_development import _canonical_sha
+from fhm_dapp_tim.fhm_dapp import v18_4_external as external_module
 from fhm_dapp_tim.fhm_dapp.v18_4_external import (
     TUJI_DEVICES,
     TUJI_LODO_ROLE_SEEDS,
@@ -38,7 +39,7 @@ def clean_execution_fixture():
 def _fake_uji_freeze(path: Path) -> Path:
     payload = {
         "protocol_version": "test",
-        "source_commit_sha": _current_head(),
+        "source_commit_sha": external_module._current_head(),
         "lambda_spatial": 0.25,
         "epsilon_kl": 0.05,
         "uji_official_validation_preexposed": True,
diff --git a/tests/test_v18_4_model.py b/tests/test_v18_4_model.py
index b6f5b93..4d305f3 100644
--- a/tests/test_v18_4_model.py
+++ b/tests/test_v18_4_model.py
@@ -409,7 +409,8 @@ def test_v184_zero_reliability_returns_q_T_bitwise() -> None:
     model.eval()
     with patch('fhm_dapp_tim.fhm_dapp.v18_4_model.base_posterior', side_effect=_controlled_base), patch(
         'fhm_dapp_tim.fhm_dapp.v18_4_model.raw_correction_energies', side_effect=_controlled_raw
-    ), patch('fhm_dapp_tim.fhm_dapp.v18_4_model.empirical_survival_reliability', side_effect=torch.zeros_like):
+    ), patch('fhm_dapp_tim.fhm_dapp.v18_4_model.empirical_survival_reliability',
+             side_effect=lambda mismatch, reference: torch.zeros_like(mismatch)):
         result = v184_distribution(model, memory['prototype_rss'][:2],
                  correction_reference=_fixed_reference(), epsilon_kl=.1, final_temperature=1.5)
     assert torch.equal(result['candidate_weights'], result['calibrated_base_probability'])
```
