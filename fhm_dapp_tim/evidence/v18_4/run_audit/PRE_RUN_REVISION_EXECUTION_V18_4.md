# V18.4 pre-run revision execution record

Status: **FAIL_CLOSED_TESTS**. The revision is implemented in the working tree,
but has not passed the full verification gate. This is an execution/failure
record, not the requested scientific revision audit and not a method freeze.

Repository: `DaWangfengle/lylw`.
Working directory: `D:\liuyanlunwen`.
Branch: `v18-4-trust-region-posterior`.
Unchanged HEAD: `b4e471868fffb2e3623a4f8471f0a95b4549543b`.
Interpreter: `D:\miniminiconda\envs\liuyan\python.exe`.

Instruction attachment:
`C:\Users\LiuYan\.codex\attachments\bb5570a8-7bb9-4c57-9060-6e5d5613eb64\Pasted text.txt`.

The instruction's section 12 says:

> STOP and report instead of hiding/fixing around the issue if:
> any existing protocol/data-role regression test breaks.

Three existing external training-role tests failed. Consequently, execution
stopped after the first prescribed regression-suite invocation. No source or
test fixes were applied after that failed invocation, and no second pytest
invocation, real-data smoke, or revision audit was launched. The failure log
and the full command output remain preserved.

## Mathematical changes implemented before the failure

1. **Spatial supervision:** V18.4 now uses exact discrete half energy distance
   with Euclidean distance divided by the registered training spatial scale.
   It computes the cross expectation minus half of each self expectation;
   there is no iterative Sinkhorn solver or active `epsilon_ot`. The active
   coefficient is `lambda_spatial` with the original three-value grid.
   Masked padding is sanitized before normalization and distance computation.

2. **Reliability and trust budget:** radio/geometry reliabilities determine
   normalized direction weights, while
   `epsilon_eff = max(r_radio, r_geometry) * epsilon_KL` determines each row's
   KL budget. The core correction accepts scalar or row budgets and reports
   maximum/effective budgets, rho, realized/full KL, reliabilities and weights.
   A numerically zero reliability sum gives an identity correction.

3. **Calibration order:** the implemented order is
   `q -> temperature q_T -> structured correction -> final posterior`.
   Energies, empirical mismatches and the KL reference use `q_T`; there is no
   second temperature map. Each candidate calibration temperature gets a
   separate corrected-distribution cache in both UJI and external code.
   References are fitted on training base probabilities at `T=1` and frozen.
   The Student-t scale multiplier changes component scales only.

4. **Residual bound:** the V18.4 model caps each anisotropic Delaunay residual
   transform by uniformly scaling the matrix when its largest singular value
   exceeds 30 m. This preserves local axes and anisotropy and leaves the
   existing differentiable unit-ball prediction in place. The generic model
   used by earlier experiments was not changed.

The global method grid remains exactly 3 spatial coefficients times 3 maximum
KL budgets, i.e. 9 combinations. No ensemble, evidence router, new ranking
loss or posterior-mean coordinate loss was introduced.

## Executed checks

`py_compile` compiled all 10 edited/new Python modules successfully (exit 0):

```text
D:\miniminiconda\envs\liuyan\python.exe -m py_compile
  fhm_dapp_tim/fhm_dapp/v18_4_core.py
  fhm_dapp_tim/fhm_dapp/v18_4_model.py
  fhm_dapp_tim/fhm_dapp/uji_v18_4_development.py
  fhm_dapp_tim/fhm_dapp/v18_4_external.py
  fhm_dapp_tim/fhm_dapp/v18_4_realdata_smoke.py
  fhm_dapp_tim/fhm_dapp/v18_4_revision_audit.py
  tests/test_v18_4_core.py
  tests/test_v18_4_model.py
  tests/test_uji_v18_4_development.py
  tests/test_v18_4_external.py
```

The required regression suite ran once (exit 1):

```text
D:\miniminiconda\envs\liuyan\python.exe -m pytest -q
  tests/test_v18_4_core.py
  tests/test_v18_4_model.py
  tests/test_uji_v18_4_development.py
  tests/test_v18_4_external.py
  tests/test_v18_4_protocol.py
  tests/test_v18_4_datasets.py
  tests/test_v18_4_capacity_audit.py

5 failed, 64 passed in 77.23s (0:01:17)
```

The JSON record contains the exact command, UTC timestamps, process duration,
exit code and complete pytest output. `git diff --check` passed before tests.
Warnings about the global Git ignore file's read permission and LF/CRLF did
not cause any of these five failures; the global configuration was not edited.

## Failure diagnosis and unapplied repair proposals

The following diagnosis is based on the failure stack and source inspection.
These are three test-construction/isolation mistakes introduced during this
revision. They do not establish that the corresponding scientific behavior
is correct: the failed assertions must still run successfully after repair.
No tolerance was relaxed and none of the proposals below has been applied.

### 1. Anisotropic transform expectation

Failed node:
`tests/test_v18_4_core.py::test_anisotropic_cap_preserves_axes_and_ratio`.

The test computes sine/cosine from an implicitly float32 angle and then casts
the resulting matrix to float64. That matrix is not orthonormal to the
test's `1e-12` tolerance, so the constructed matrix's largest singular value
is not exactly the assumed 300. The implementation computes its actual
singular value, while the expected tensor assumes a divisor of exactly 10.
The observed maximum absolute discrepancy was `4.4784704300582234e-07`.

Proposed minimal repair: construct the angle as float64 *before* computing
sine and cosine. Retain the original strict matrix-comparison tolerance and
the singular-value/anisotropy assertions. The cap assertion after the failed
comparison was not reached in this test.

### 2. Zero-reliability mock signature

Failed node:
`tests/test_v18_4_model.py::test_v184_zero_reliability_returns_q_T_bitwise`.

`empirical_survival_reliability` receives both the mismatch tensor and a
reference tensor. The new test used `torch.zeros_like` directly as its mock
side effect; that function takes only one positional argument. It raised
`TypeError: zeros_like() takes 1 positional argument but 2 were given` before
the bitwise-identity and zero-KL assertions could execute.

Proposed minimal repair: use a side effect with both arguments, such as
`lambda mismatch, reference: torch.zeros_like(mismatch)`. Preserve all
bitwise-identity, zero-effective-budget, zero-rho and zero-realized-KL
assertions. This failed test did not observe a changed zero-reliability
posterior; it failed before a posterior was returned.

### 3. External training-role fixture isolation

Failed nodes:

- `tests/test_v18_4_external.py::test_prepare_external_uts_uses_training_only`
- `tests/test_v18_4_external.py::test_prepare_external_tampere_uses_training_only`
- `tests/test_v18_4_external.py::test_prepare_external_tuji_lodo_excludes_heldout_device`

The tests import `_current_head` directly. The new fixture patched only the
symbol inside `v18_4_external`, so `_fake_uji_freeze` still called the
previously imported original function. Because this revision necessarily
leaves tracked edits in the working tree, the production source-freeze gate
correctly raised `formal external phases require no tracked working-tree
changes` before any of these role assertions ran.

Proposed minimal repair: isolate the test module's imported `_current_head`
reference as well as the production-module symbol, scoped only to these three
synthetic freeze/manifest tests. Another narrow implementation is to have the
synthetic helper use the already patched module symbol. Keep the production
clean-tree and method-source compatibility gates unchanged; keep the
dedicated dirty-tree and changed-source rejection tests outside this fixture.
Retain every dataset-role and held-out-device exclusion assertion.

These three existing regression failures trigger the required stop boundary,
even though the observed error occurs in fixture construction rather than in
the data-role assertions themselves.

## Runs not executed and unresolved scientific concerns

- Real-data smoke: **NOT RUN**, blocked by the failed regression gate.
- New revision audit: **NOT RUN**, blocked by the failed regression gate.
- CUDA smoke: **NOT RUN**. CUDA availability was checked before editing:
  PyTorch `2.10.0+cu128`, CUDA available, NVIDIA GeForce RTX 2060.
- Formal UJI `prepare`, `develop` and `calibrate-freeze`: **NOT RUN**.
- UTS/Tampere/TUJI formal evaluation: **NOT RUN**.

The two requested scientific audit files do not exist:

```text
D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.json
D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\PRE_RUN_REVISION_AUDIT_V18_4.md
```

No fit/development counts, anchors, AP counts, real-data spatial scale,
real-data losses/gradients/predictions, real-data residual bounds or support
statistics are claimed. The required audit program has been created and
compiled but not executed. Its training-only checks, including CUDA behavior,
remain unverified. The reference-at-T=1 / deployment-at-q_T semantics are
documented but have not been validated on the required real-data sample.
Neither this failure record nor the passing subset of unit tests establishes
accuracy, convergence, full-dataset support behavior or formal-run readiness.

If a subsequent instruction authorizes proceeding after this stop report,
the minimal path is to apply the three test-only repairs without weakening
assertions, preserve this failed run, rerun the entire prescribed suite, then
run the CPU real-data smoke and revision audit (with the planned lightweight
CUDA checks). Formal execution remains outside this task's boundary.

## Exact changed/new files

Modified implementation files:

- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\v18_4_core.py`
- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\v18_4_model.py`
- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\uji_v18_4_development.py`
- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\v18_4_external.py`
- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\v18_4_realdata_smoke.py`

New implementation file:

- `D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\v18_4_revision_audit.py`

Modified tests:

- `D:\liuyanlunwen\tests\test_v18_4_core.py`
- `D:\liuyanlunwen\tests\test_v18_4_model.py`
- `D:\liuyanlunwen\tests\test_uji_v18_4_development.py`
- `D:\liuyanlunwen\tests\test_v18_4_external.py`

Modified documentation:

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\DESIGN_V18_4.md`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\RUNBOOK_UJI_V18_4.md`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\RUNBOOK_EXTERNAL_V18_4.md`

New execution records:

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\pre_run_revision_execution_v18_4.json`
- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\PRE_RUN_REVISION_EXECUTION_V18_4.md`

New ignored raw log (preserved locally; the JSON also embeds its entire output):

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\pre_run_revision_pytest.log`

No checkpoint was created or modified by these commands. No formal V18.4 UJI
registry/development/calibration/freeze artifact was created. No commit or
push was performed during this pre-run revision.
