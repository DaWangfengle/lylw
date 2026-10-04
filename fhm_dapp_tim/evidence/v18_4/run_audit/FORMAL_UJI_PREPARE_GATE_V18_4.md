# V18.4 formal UJI prepare gate

Status: **REQUIREMENT CLARIFICATION PENDING**. Prepare completed; development and calibration/freeze have not started. This is a descriptive gate record, not a successful full formal execution record.

Recorded UTC: `2026-10-03T18:43:12.047629+00:00`.
Source commit: `5b3a1888707e193c649007ebeb208c2a9d02ff45`.
Branch: `v18-4-trust-region-posterior`.

## Verified pre-run gate

The existing scientific revision audit and acceptance JSON both passed the required status/flags/hash checks. The write-once revision audit was not rerun. Historical records and all 157 snapshotted tracked Python/instruction sources remain unchanged.

| Fresh check | Result |
|---|---|
| py_compile | PASS, 4 prescribed formal modules |
| Complete prescribed pytest suite | 69 passed, 0 failed; no skip/xfail; one invocation |
| CPU smoke | success |
| CUDA smoke | success |

Environment:
```json
{
  "source_commit_sha": "5b3a1888707e193c649007ebeb208c2a9d02ff45",
  "branch": "v18-4-trust-region-posterior",
  "python_executable": "D:\\miniminiconda\\envs\\liuyan\\python.exe",
  "python_version": "3.12.12",
  "torch_version": "2.10.0+cu128",
  "cuda_available": true,
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 2060",
  "gpu_memory_bytes": 6442123264
}
```

## Prepare result

Registry: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\uji_registry_v18_4.json`.
Canonical registry self-hash: `7630e6f8459b055268226ae7e83a0fe014337e0a3ef491d661d00a301e2001b0`.
Registry file-byte SHA256: `3470b9ff247117fbe9cefcd2f8135a55a8efe440b497a4ac96e8346339c26fb3`.
UJI trainingData SHA256: `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`.

Protocol version, frozen source commit, training-only path, 19,937 rows, 933 physical coordinate groups and 520 APs all match the instruction. The training file hash and registry self-hash are valid. Roles were independently reconstructed from UJI trainingData; the stored rows match exactly, row hashes are valid, role groups are pairwise disjoint, the union is exact, hull support is retained and no role loses a training stratum.

| Role | Rows | Coordinate groups |
|---|---:|---:|
| fit | 14126 | 655 |
| development | 2936 | 139 |
| selection_refit | 17062 | 794 |
| calibration | 2875 | 139 |

## Pending instruction/schema interpretation

Instruction section 5 requires:

> recorded row/group hashes present

The frozen prepare implementation writes `row_index_sha256` and `coordinate_count` for every role. It does not write a separate coordinate-group SHA256 field. The registry binds group identities indirectly through the frozen trainingData SHA256 and explicit role row indices; the full group membership can be independently reproduced.

The source and registry were not edited. No source-cleanliness guard was bypassed. Independent group hashes were computed descriptively and saved in the new run-state record; they were not inserted into the formal registry or represented as originally registered fields.

A user clarification was requested: whether independent group hashes must appear in the formal registry itself, or whether verification/provenance hashes in the new execution record satisfy this requirement. No answer/exception has been assumed and development remains gated.

Derived hash definition: SHA256 of canonical UTF-8 JSON `{"coordinate_keys": sorted(unique coordinate_keys(data)[role rows])}`. Keys are the existing registered `(building, floor, rounded x, rounded y)` coordinate keys. This adds no model-selection information or hyperparameter.

| Role | Registered row-index SHA256 | Derived coordinate-group SHA256 |
|---|---|---|
| fit | `4597c4dde162707fb0038c3b20bd6b097a65a56760477734ab7a676dfc054ea5` | `fa9abf2a01836ae49491bb2382b066ff7a28af3eb1bcd38e18b5ce75b4909c76` |
| development | `25548c3b1cb9ab1ce895ff56cf4a8ca7379ce9c8a37cded84e426c3b60b1891f` | `edc298ebdf112710c6061aa5708e2b4e26941bbdd2aad55681dd4db3be683b99` |
| selection_refit | `717ee05b6177db1dde486da49e3338e073ef55d1cb71262699f64771c5bdda5f` | `cd958befea221c6b1c7dfb26f39e53122c2c3478707b9801bbf46700b326073a` |
| calibration | `92d806ec05b068741ca2b145d7006293fb99ac41e3b44c376d7bd87780c4598c` | `539bcec2f1bdae46e1ee0c6e6e2aa162926f8cbc87abf2fd0f0fa4edcbe72a33` |

## Boundaries and retained artifacts

Formal prepare: completed once.
Formal development: **NOT RUN**.
Formal selection-refit/calibration/freeze: **NOT RUN**.
UJI validationData evaluation: **NOT RUN**.
External formal evaluation: **NOT RUN**.
No model, protocol, source revision or hyperparameter/grid law was changed.

Full commands, outputs, timing, verification and role-hash provenance: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\formal_uji_run_state_v18_4.json`.
Raw command logs are preserved under `run_audit/formal_uji_*.log`.
The three lambda checkpoints, refit checkpoint, development result, calibration result and method freeze remain absent. No previous formal attempt was overwritten or deleted.

If the derived-group-hash interpretation is explicitly accepted, the remaining stages can use this unchanged source and unchanged registry; prepare must not be rerun. If separate registry group hashes are mandatory, the current formal attempt must remain stopped and preserved.
