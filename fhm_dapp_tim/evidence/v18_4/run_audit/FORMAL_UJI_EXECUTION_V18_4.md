# V18.4 Formal UJI Execution Record

Status: PASS. Formal prepare, develop and calibrate-freeze completed and independently validated. The method is frozen; execution stops here for scientific review.

Created: 2026-10-04T13:59:50.940247+08:00 (Asia/Shanghai).

## Environment and verification

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

Four modules compiled successfully. The accepted seven-file regression suite ran once: 69 passed, 0 failed, no skip/xfail. Fresh CPU and CUDA smoke returned success. Existing pre-run audit and acceptance records were PASS; they were not rerun or overwritten.

```json
{
  "audit_status": "PASS",
  "audit_rerun": false,
  "acceptance_status": "PASS",
  "reviewed_implementation_unchanged": true,
  "historical_failed_run_preserved": true,
  "method_combinations": 9,
  "validation_accessed_by_pre_run_audit": false,
  "external_test_performance_accessed_by_pre_run_audit": false
}
```

| Stage | Exit | Wall seconds | Started UTC | Finished UTC |
| --- | --- | --- | --- | --- |
| compile | 0 | 0.3153601 | 2026-10-03T18:35:55.813546+00:00 | 2026-10-03T18:35:56.134964+00:00 |
| pytest | 0 | 70.5581525 | 2026-10-03T18:36:37.866341+00:00 | 2026-10-03T18:37:48.432593+00:00 |
| cpu_smoke | 0 | 8.8144084 | 2026-10-03T18:38:45.296501+00:00 | 2026-10-03T18:38:54.109736+00:00 |
| cuda_smoke | 0 | 9.3305628 | 2026-10-03T18:39:01.733870+00:00 | 2026-10-03T18:39:11.059041+00:00 |
| prepare | 0 | 6.5453482 | 2026-10-03T18:39:33.022273+00:00 | 2026-10-03T18:39:39.568342+00:00 |
| develop | 0 | 4397.23945 | 2026-10-04T04:35:02.006750+00:00 | 2026-10-04T05:48:19.243202+00:00 |
| calibrate_freeze | 0 | 533.073082 | 2026-10-04T05:49:10.697474+00:00 | 2026-10-04T05:58:03.784218+00:00 |

Exact commands, smoke diagnostics and log hashes are in the companion JSON. Raw command outputs remain in local logs and the intermediate run state.

## Prepare and role provenance

Training source: data/UJIndoorLoc/trainingData.csv. SHA256: `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`. Rows: 19937; physical coordinate groups: 933; APs: 520.

Registry canonical SHA256: `7630e6f8459b055268226ae7e83a0fe014337e0a3ef491d661d00a301e2001b0`.

Registry file SHA256: `3470b9ff247117fbe9cefcd2f8135a55a8efe440b497a4ac96e8346339c26fb3`.

| Role | Rows | Coordinate groups |
| --- | --- | --- |
| fit | 14126 | 655 |
| development | 2936 | 139 |
| selection_refit | 17062 | 794 |
| calibration | 2875 | 139 |

FIT/development/calibration have exact union and pairwise disjointness. selection_refit is FIT union development. All original strata and required hull support are retained. Row hashes and independently reconstructed coordinate-group hashes:

```json
{
  "fit": {
    "row_count": 14126,
    "coordinate_count": 655,
    "row_index_sha256": "4597c4dde162707fb0038c3b20bd6b097a65a56760477734ab7a676dfc054ea5",
    "derived_coordinate_group_sha256": "fa9abf2a01836ae49491bb2382b066ff7a28af3eb1bcd38e18b5ce75b4909c76",
    "registered_coordinate_group_hash_fields": {}
  },
  "development": {
    "row_count": 2936,
    "coordinate_count": 139,
    "row_index_sha256": "25548c3b1cb9ab1ce895ff56cf4a8ca7379ce9c8a37cded84e426c3b60b1891f",
    "derived_coordinate_group_sha256": "edc298ebdf112710c6061aa5708e2b4e26941bbdd2aad55681dd4db3be683b99",
    "registered_coordinate_group_hash_fields": {}
  },
  "selection_refit": {
    "row_count": 17062,
    "coordinate_count": 794,
    "row_index_sha256": "717ee05b6177db1dde486da49e3338e073ef55d1cb71262699f64771c5bdda5f",
    "derived_coordinate_group_sha256": "cd958befea221c6b1c7dfb26f39e53122c2c3478707b9801bbf46700b326073a",
    "registered_coordinate_group_hash_fields": {}
  },
  "calibration": {
    "row_count": 2875,
    "coordinate_count": 139,
    "row_index_sha256": "92d806ec05b068741ca2b145d7006293fb99ac41e3b44c376d7bd87780c4598c",
    "derived_coordinate_group_sha256": "539bcec2f1bdae46e1ee0c6e6e2aa162926f8cbc87abf2fd0f0fa4edcbe72a33",
    "registered_coordinate_group_hash_fields": {}
  }
}
```

Group-hash derivation: _canonical_sha({coordinate_keys: sorted(unique coordinate_keys(data)[role rows])}); SHA256 of canonical UTF-8 JSON; trainingData and row-index SHA256 frozen in registry.

## Problems and authorized resolution

The frozen registry lacks separate coordinate-group hash fields. Execution stopped after prepare and preserved the initial gate report. The user replied "允许，继续", authorizing unchanged registry bytes and verified group hashes in this new record. Execution resumed from develop without repeating prepare or editing model/protocol source.

```json
{
  "recorded_utc": "2026-10-04T04:35:00.727055+00:00",
  "human_reply": "允许，继续",
  "accepted_scope": "Keep the formal registry byte-for-byte unchanged; independently compute and verify role coordinate-group hashes in the new execution record; continue the existing formal run from develop without rerunning prepare.",
  "source_revision_unchanged": true,
  "registry_unchanged": true
}
```

The user-level Git ignore warning is retained in logs. Revision/cleanliness guards and all required commands succeeded; no global Git configuration changed.

Historical prepare-gate report: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\FORMAL_UJI_PREPARE_GATE_V18_4.md`. Its paused status remains an original historical snapshot; the final resolution is recorded here.

## Development: full nine-combination summary

Metrics are coordinate-macro where defined by the frozen protocol. ES/GM/mean error and their SE values are in meters. All reported metrics are finite. Three checkpoint hashes, result self-hash, registry/source links, and registered epoch/global selection rules were independently verified.

| lambda | Epoch | epsilon_KL | ES m | ES SE m | GM MAE m | GM SE m | Mean MAE m | NLL | NLL SE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 16 | 0.025 | 5.74126095 | 0.336991437 | 8.09649495 | 0.476529689 | 7.74931997 | 6.38506254 | 0.137224071 |
| 0.1 | 16 | 0.05 | 5.75796544 | 0.338597454 | 8.1216834 | 0.478493753 | 7.77529457 | 6.39393431 | 0.13961076 |
| 0.1 | 16 | 0.1 | 5.77815385 | 0.341305452 | 8.16445647 | 0.484150721 | 7.80314497 | 6.40679104 | 0.142719235 |
| 0.25 | 13 | 0.025 | 5.69493299 | 0.320426312 | 8.14936397 | 0.465193099 | 7.71908444 | 6.38272058 | 0.130523131 |
| 0.25 | 13 | 0.05 | 5.70871953 | 0.322872586 | 8.17917438 | 0.465876973 | 7.74883841 | 6.39139256 | 0.132623384 |
| 0.25 | 13 | 0.1 | 5.71709433 | 0.324936505 | 8.20139863 | 0.466028689 | 7.78147267 | 6.40419032 | 0.135186363 |
| 0.5 | 20 | 0.025 | 5.74296729 | 0.334874259 | 8.1133204 | 0.488979412 | 7.66547992 | 6.51498417 | 0.144624368 |
| 0.5 | 20 | 0.05 | 5.77123371 | 0.336478564 | 8.13322552 | 0.488937268 | 7.69911185 | 6.52907234 | 0.14712257 |
| 0.5 | 20 | 0.1 | 5.80117432 | 0.339928961 | 8.15424906 | 0.48832736 | 7.73958358 | 6.54665507 | 0.150141902 |

| lambda | Epoch | epsilon_KL | Entropy | Effective anchors | Top-1 | rho | Realized KL | Radio reliability | Geometry reliability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 16 | 0.025 | 2.01463955 | 7.85941608 | 0.277758858 | 0.349986653 | 0.0154335602 | 0.559113878 | 0.435569242 |
| 0.1 | 16 | 0.05 | 1.9737161 | 7.55996553 | 0.286921616 | 0.5059767 | 0.02965769 | 0.559113878 | 0.435569242 |
| 0.1 | 16 | 0.1 | 1.92474787 | 7.2150486 | 0.298409251 | 0.681628723 | 0.0515343436 | 0.559113878 | 0.435569242 |
| 0.25 | 13 | 0.025 | 2.01729306 | 7.90627748 | 0.279220371 | 0.354762916 | 0.0154595486 | 0.561082952 | 0.4407022 |
| 0.25 | 13 | 0.05 | 1.97439528 | 7.59075223 | 0.288671069 | 0.513435944 | 0.0296754186 | 0.561082952 | 0.4407022 |
| 0.25 | 13 | 0.1 | 1.92388671 | 7.2344698 | 0.300189699 | 0.690084594 | 0.0510817305 | 0.561082952 | 0.4407022 |
| 0.5 | 20 | 0.025 | 1.98301083 | 7.63967722 | 0.285947813 | 0.363468609 | 0.0153862135 | 0.559883198 | 0.438943122 |
| 0.5 | 20 | 0.05 | 1.9431526 | 7.35635438 | 0.295011118 | 0.523756915 | 0.0293641604 | 0.559883198 | 0.438943122 |
| 0.5 | 20 | 0.1 | 1.8967471 | 7.04243448 | 0.306382905 | 0.690824292 | 0.0499662652 | 0.559883198 | 0.438943122 |

Per-lambda base selection and FIT-only correction references:

| lambda | Epoch | Base ES m | Base GM MAE m | Base NLL | Spatial scale m | Parameters | Radio energy scale | Geometry energy scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 16 | 5.72131937 | 8.04243093 | 6.37587094 | 10.4923105 | 614553 | 0.500930548 | 1.93207383 |
| 0.25 | 13 | 5.66279256 | 8.08715498 | 6.37442409 | 10.4923105 | 614553 | 0.502914906 | 1.97736263 |
| 0.5 | 20 | 5.72153083 | 8.06111733 | 6.49244454 | 10.4923105 | 614553 | 0.50988698 | 1.92125428 |

Selected global method:

```json
{
  "checkpoint_path": "fhm_dapp_tim\\models\\v18_4\\uji_lambda_0p10.pt",
  "checkpoint_sha256": "460ce4c0a3fa22976bca164deec6ce0cbf60c4de796811acf7df22a880bf6419",
  "coordinate_groups": 139,
  "energy_score_m": 5.741260946927926,
  "energy_score_se_m": 0.33699143675179805,
  "epsilon_kl": 0.025,
  "gm_mae_m": 8.096494945204615,
  "gm_mae_se_m": 0.4765296893704759,
  "lambda_spatial": 0.1,
  "mean_effective_anchors": 7.859416084247324,
  "mean_entropy_nats": 2.0146395507235617,
  "mean_geometry_reliability": 0.43556924160234933,
  "mean_radio_reliability": 0.55911387842179,
  "mean_realized_kl": 0.01543356023096806,
  "mean_rho": 0.3499866533565051,
  "mean_top1_probability": 0.2777588582326143,
  "nll": 6.385062537315427,
  "nll_se": 0.13722407093798766,
  "posterior_mean_mae_m": 7.749319970895107,
  "rows": 2936,
  "selected_epoch": 16
}
```

Development artifact runtime_seconds: 4387.24548.

## Refit, calibration and freeze

Selection refit used only FIT union development at the already selected epoch/lambda/epsilon. Correction references were fitted on selection_refit at T=1. Calibration used only the registered calibration role. Every T recomputed q -> q_T -> reliability-bounded correction; scale changed only component scales. Exactly 48 registered combinations, finite NLL, and the deterministic tie law were verified.

Selected calibration:

```json
{
  "coordinate_macro_nll": 6.059140452836212,
  "scale_multiplier": 1.0,
  "temperature": 1.5
}
```

ALL selected_metrics, copied without omission:

```json
{
  "coordinate_macro_energy_score_m": 5.0794432681308,
  "coordinate_macro_gm_mae_m": 7.149656005812268,
  "coordinate_macro_nll": 6.059140452836212,
  "coordinate_macro_posterior_mean_mae_m": 6.942730971621736,
  "corr_mode_far_mass_20m_mean_gm_displacement": 0.664014091510244,
  "corr_truth_far_mass_20m_mean_error": 0.6368231587068544,
  "mean_effective_anchors": 11.667896405427353,
  "mean_effective_epsilon_kl": 0.013292343249010003,
  "mean_entropy_nats": 2.417949200091155,
  "mean_gm_displacement_from_mean_m": 2.5662788588689724,
  "mean_gm_spatial_leverage_m": 7.014042273314104,
  "mean_natural_pool_size_before_ceiling": 49.75895652173913,
  "mean_realized_kl_against_q_T": 0.013244586963766649,
  "mean_top1_probability": 0.19872138546990312,
  "natural_truncation_fraction": 0.0,
  "tail_mass": {
    "mode_far_mass_10m": 0.21581374954260157,
    "mode_far_mass_20m": 0.0777453778611286,
    "mode_far_mass_40m": 0.024190499833937874,
    "truth_far_mass_10m": 0.32397380451040103,
    "truth_far_mass_20m": 0.10273575902509076,
    "truth_far_mass_40m": 0.024504325710089644
  }
}
```

Best 10 calibration rows by NLL and the registered tie law:

| Temperature | Scale | Coordinate-macro NLL |
| --- | --- | --- |
| 1.5 | 1 | 6.05914045 |
| 1.25 | 1 | 6.06224929 |
| 1.25 | 1.25 | 6.07334929 |
| 1.5 | 1.25 | 6.08086078 |
| 1 | 1.25 | 6.08177224 |
| 2 | 1 | 6.08359328 |
| 1 | 1 | 6.08545569 |
| 1.5 | 0.75 | 6.10808405 |
| 2 | 0.75 | 6.10952627 |
| 0.75 | 1.25 | 6.11796942 |

Unit calibration (T=1, scale=1):

```json
{
  "coordinate_macro_nll": 6.085455692287693,
  "scale_multiplier": 1.0,
  "temperature": 1.0
}
```

Selected minus unit NLL: -0.0263152395.

Calibration artifact runtime_seconds: 526.396726. Includes refit, reference fitting and the complete calibration surface.

Frozen decoder: `posterior_weighted_component_center_geometric_median`.

## Generated formal artifacts and SHA256

File hashes cover exact bytes; canonical JSON self-hashes omit their own hash fields.

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\uji_registry_v18_4.json`
  File SHA256: `3470b9ff247117fbe9cefcd2f8135a55a8efe440b497a4ac96e8346339c26fb3`.
  registry_sha256: `7630e6f8459b055268226ae7e83a0fe014337e0a3ef491d661d00a301e2001b0`.

- `D:\liuyanlunwen\fhm_dapp_tim\models\v18_4\uji_lambda_0p10.pt`
  File SHA256: `460ce4c0a3fa22976bca164deec6ce0cbf60c4de796811acf7df22a880bf6419`.

- `D:\liuyanlunwen\fhm_dapp_tim\models\v18_4\uji_lambda_0p25.pt`
  File SHA256: `b2c33a03268b0f53d659189b1f98c4910635acf6b060d5dd8ebb4c10ff605fea`.

- `D:\liuyanlunwen\fhm_dapp_tim\models\v18_4\uji_lambda_0p50.pt`
  File SHA256: `716e6886e4aae5cce6dc5184e331e1941730ef7479c6d20d5e9c751c988815c9`.

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\uji_development_v18_4.json`
  File SHA256: `8b7a30e2de836c26631d8fcda965dc6d7bbeb522c6d01ed53f3a4c25b1dc2019`.
  development_result_sha256: `f60f98971bf9b57d61f9d52dc408b5600a039acdb841086552b3a0ae98710d78`.

- `D:\liuyanlunwen\fhm_dapp_tim\models\v18_4\uji_selection_refit.pt`
  File SHA256: `e460d90876deb893466c8e8af9ac167b0f2deb145ed47e0698362ee8fc08b72b`.

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\uji_calibration_v18_4.json`
  File SHA256: `77500418abd927629b5687e2639c6d77a56c7464f144fab81089899dc4076082`.
  calibration_result_sha256: `9c2c23169a5739c7c09a438b45dfd6e7144d2ae824fe312dfe9ba65e241498b8`.

- `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\uji_method_freeze_v18_4.json`
  File SHA256: `9901da1203e05af9a73ff8b7a3d6c6f75d77364b2f6d3a52d093e7d72b6c874f`.
  freeze_sha256: `bcf631c01e15784baa4ea822a18bba344daad7f921ecb083eef64618fefee432`.

## Execution integrity and boundaries

Companion JSON: `D:\liuyanlunwen\fhm_dapp_tim\evidence\v18_4\run_audit\FORMAL_UJI_EXECUTION_V18_4.json`.

Execution canonical self-hash: `d80db2d1c293dbabe320c20d862a42c6ddcd64068a432fa26fdcdb9e42b64a68`.

```json
{
  "formal_uji_prepare_completed": true,
  "formal_uji_develop_completed": true,
  "formal_uji_calibrate_freeze_completed": true,
  "uji_validationData_evaluated_by_this_formal_run": false,
  "uji_official_validation_used_for_v184_selection": false,
  "external_test_performance_used_for_v184_selection": false,
  "formal_uts_evaluation_run": false,
  "formal_tampere_evaluation_run": false,
  "formal_tuji_pooled_evaluation_run": false,
  "formal_tuji_leave_one_device_out_evaluation_run": false,
  "post_outcome_source_model_protocol_changes": false,
  "hyperparameter_grid_widened": false,
  "post_outcome_method_changes": false,
  "pre_run_revision_audit_rerun": false,
  "historical_pre_run_artifacts_preserved": true,
  "source_revision_and_tracked_files_unchanged": true,
  "registry_bytes_unchanged": true,
  "temperature_applied_before_correction": true,
  "post_correction_temperature_applied": false,
  "scale_changes_anchor_probabilities": false,
  "historical_preexposure_disclosures_preserved": true,
  "next_step": "STOP for scientific review of UJI formal artifacts"
}
```

Historical preexposure metadata is retained. These stages used trainingData internal roles only; UJI validationData performance and external formal evaluations were not run. No source, grid or model law changed after outcomes appeared. This record is descriptive and does not retune or reinterpret results.

All six preserved pre-run artifacts and all snapshotted tracked source files retain original bytes. Their hashes are in the companion JSON.

## Execution-record QA note

A post-write display check initially counted lambda/epoch prefixes across three tables and incorrectly expected six hits. Each prefix correctly appears seven times: three ES rows, three posterior rows, and one per-lambda base row. The corrected check counts each table by its own header and verifies 9 ES rows, 9 posterior rows, 3 base rows, and 10 calibration best rows. All selected_metrics match the calibration JSON exactly. This was a record-checker error; no registered scientific check failed, no formal stage was rerun, and no formal artifact, source, grid, or selected method changed.
