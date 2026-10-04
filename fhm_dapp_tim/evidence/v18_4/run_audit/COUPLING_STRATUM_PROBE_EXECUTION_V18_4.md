# V18.4 coupling × stratum execution record


Status: **PASS**。

Evidence HEAD: `bbd9c4c393413207ffc7f08d518162acf38e52e4`；frozen source: `5b3a1888707e193c649007ebeb208c2a9d02ff45`。

## 环境与门禁


```json
{
  "python_executable": "D:\\miniminiconda\\envs\\liuyan\\python.exe",
  "python_version": "3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.19045-SP0",
  "torch_version": "2.10.0+cu128",
  "numpy_version": "2.4.4",
  "cuda_runtime": "12.8",
  "cuda_available": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 2060",
  "deterministic_algorithms": true,
  "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
}
```

| Step | Exit | Runtime seconds | Tests | Command |
| --- | --- | --- | --- | --- |
| compile | 0 | 0.101409 |  | D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/v18_4_coupling_stratum_probe.py tests/test_v18_4_coupling_stratum_probe.py |
| pytest | 0 | 80.843529 | 227 | D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v18_4_core.py tests/test_v18_4_model.py tests/test_uji_v18_4_development.py tests/test_v18_4_external.py tests/test_v18_4_protocol.py tests/test_v18_4_datasets.py tests/test_v18_4_capacity_audit.py tests/test_v18_4_mechanism_probe.py tests/test_v18_4_coupling_stratum_probe.py |
| probe | 0 | 20849.517516 |  | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.v18_4_coupling_stratum_probe --device cuda |

## 完整性、模型、数据访问


```json
{
  "source_artifact_postcheck": "PASS: every frozen/prior/pre-existing Python/evidence/checkpoint/tested file unchanged",
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
  "denied_file_opens": [],
  "model_integrity": {
    "fit_development": {
      "before_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "after_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 368,
      "temporary_adam_steps": 2208,
      "temporary_nuisance_backward_calls": 2208,
      "radio_batches": 368
    },
    "fit_calibration": {
      "before_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "after_sha256": "d7d5804c1e9179fe6ac992714477a7a9c9395499eccc9ee0cbdec6578a6dffc7",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 360,
      "temporary_adam_steps": 2160,
      "temporary_nuisance_backward_calls": 2160,
      "radio_batches": 360
    },
    "refit_calibration": {
      "before_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "after_sha256": "df8d7157636bcece82805af297454958b6e4ef488064767840bcd04998f584a1",
      "all_parameters_and_buffers_bitwise_unchanged": true,
      "all_parameter_gradients_none": true,
      "training_mode_invoked": false,
      "temporary_adam_initializations": 360,
      "temporary_adam_steps": 2160,
      "temporary_nuisance_backward_calls": 2160,
      "radio_batches": 360
    }
  },
  "radio_temporary_adam_totals": {
    "temporary_adam_initializations": 1088,
    "temporary_adam_steps": 6528,
    "temporary_nuisance_backward_calls": 6528,
    "radio_batches": 1088
  },
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
    "network_loss_backward_invoked": false,
    "model_buffers_modified": false
  }
}
```

## 问题与处理


- Final audit authorizes a previously approved radio optimizer exception; blanket optimizer bans must still cover the network.
  Reuse the prior human authorization and ReadOnlyModelGuard. Permit only two fresh nuisance leaf tensors and precisely six registered Adam/backward calls per batch; hash model state each batch and check all gradients None.

- Production apply_correction accepts only the three positive registered epsilons; this instruction additionally requires epsilon=0 identity controls.
  Use apply_correction once at .025 for production normalization/reliability, then call the unchanged production core law for the epsilon grid, including zero. No production config or reference change.

- Naively scoring all identity controls and calling the tail utility would duplicate equal posteriors and geometric medians.
  Verify zero-budget posteriors are bitwise q_T, reuse only their score/point/tail arrays while retaining each mode reliability, and compute one frozen production GM per distinct posterior with the exact production tail-mass formula.

- Local Git emits a user global-ignore permission warning on read-only status.
  No repository failure; global Git settings are unchanged.

## 科学输出字节 SHA256


| Path | SHA256 |
| --- | --- |
| fhm_dapp_tim/evidence/v18_4/COUPLING_STRATUM_PROBE_V18_4.json | 091a6212b0963eb00f69435bb65a9bfe45d5b89fe9757b1a59b4e799988ad0da |
| fhm_dapp_tim/evidence/v18_4/COUPLING_STRATUM_PROBE_V18_4.md | ce52bbf758489e5abd86ecea7dafee05b04b716d69fa566361e8f92b6bb07973 |

Execution canonical SHA256: `e27051e39afb51eb3b6ac4aecc32616f156f8daf801727ff7149179530c13500`。输出路径从未覆盖；结束后硬停止。
