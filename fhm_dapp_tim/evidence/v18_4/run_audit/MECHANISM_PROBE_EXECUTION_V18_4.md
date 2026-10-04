# V18.4 Mechanism probe execution record


Status: **PASS**。

Evidence HEAD: `2ba86d77ce710b714441de73e0c2e77e88813736`；frozen source: `5b3a1888707e193c649007ebeb208c2a9d02ff45`。

人工例外只允许生产 radio profiler 的两个临时张量 Adam；每批精确 6 步。禁止网络训练、网络优化和网络损失 backward。

## 门禁及执行


| Step | Exit | Seconds | Passed tests | Command |
| --- | --- | --- | --- | --- |
| compile | 0 | 0.111349 |  | D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py tests/test_v18_4_mechanism_probe.py |
| pytest | 0 | 76.713858 | 128 | D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py tests/test_v18_4_external.py tests/test_v18_4_protocol.py tests/test_v18_4_datasets.py tests/test_v18_4_capacity_audit.py tests/test_v18_4_mechanism_probe.py |
| probe | 0 | 1938.888370 |  | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.v18_4_mechanism_probe --device cuda |

## 输出 SHA256


| Path | File-byte SHA256 |
| --- | --- |
| fhm_dapp_tim/evidence/v18_4/MECHANISM_PROBE_V18_4.json | 790b667892d4b21ec8eb062d6fc27c9cf6d63196d680d300648bf94c1a983d67 |
| fhm_dapp_tim/evidence/v18_4/MECHANISM_PROBE_V18_4.md | 31affba0655b847e0c2fd742169a4355eb1657c765ea72db6e3577c0be90b627 |

## 问题与处理


- Sections 6 and 17 conflict: the required frozen radio profiling uses a temporary Adam despite No optimizer.
  Asked before dependent execution; human explicitly authorized this exception. New guard allows only two nuisance leaves and six steps/backwards, rejects network training/optimizers/backward and checks model state bitwise.

- Static implementation review identified JSON lists versus frozen Python tuple grids in config comparison.
  Compare JSON-canonical config values, preserving every numeric grid value. Added regression coverage before first compile/test gate; no production/config modification.

- An exploratory metadata read used the Windows default GBK encoding and failed decoding UTF-8 Chinese text.
  Read all JSON/Markdown using explicit UTF-8. This occurred in metadata inspection; no scientific run, source change or artifact mutation.

- An exploratory rg command passed literal wildcard file paths unsupported by PowerShell.
  Use rg directory searches with -g patterns. No scientific execution or data/artifact changes.

- Git emits Permission denied for the global ignore file.
  Read-only status commands still succeeded. No global Git configuration changes and no commit/push in this task.

## 完整性及边界


```json
{
  "boundary_confirmations": {
    "training_performed": false,
    "model_parameters_modified": false,
    "model_gradients_created": false,
    "official_validation_accessed": false,
    "external_data_accessed": false,
    "external_evaluation_run": false,
    "v18_5_changes": false,
    "correction_references_refitted": false,
    "method_freeze_modified": false,
    "network_optimizer_invoked": false,
    "network_loss_backward_invoked": false
  },
  "dataset_opens": [
    {
      "path": "d:\\liuyanlunwen\\data\\ujindoorloc\\trainingdata.csv",
      "mode": "r",
      "flags": 32896,
      "allowed": true
    },
    {
      "path": "d:\\liuyanlunwen\\data\\ujindoorloc\\trainingdata.csv",
      "mode": "r",
      "flags": 32896,
      "allowed": true
    },
    {
      "path": "d:\\liuyanlunwen\\data\\ujindoorloc\\trainingdata.csv",
      "mode": "r",
      "flags": 32896,
      "allowed": true
    }
  ],
  "model_integrity": {
    "development_native": {
      "before_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "after_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 46,
      "temporary_adam_steps": 276,
      "temporary_nuisance_backward_calls": 276,
      "radio_batches": 46
    },
    "calibration_frozen": {
      "before_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "after_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 45,
      "temporary_adam_steps": 270,
      "temporary_nuisance_backward_calls": 270,
      "radio_batches": 45
    },
    "calibration_T1_control": {
      "before_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "after_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 45,
      "temporary_adam_steps": 270,
      "temporary_nuisance_backward_calls": 270,
      "radio_batches": 45
    }
  },
  "source_artifact_postcheck": "PASS: all frozen sources and pre-existing formal/evidence/checkpoint bytes unchanged"
}
```

Execution canonical SHA256: `deac156cdb160e87f03bed7735d20a237a03d6be4dae4d1629bc2f59780d3c89`。硬停止；下一步为科学审阅。
