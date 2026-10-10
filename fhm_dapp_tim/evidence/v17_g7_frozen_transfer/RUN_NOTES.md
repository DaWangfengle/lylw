# V17-G7 execution notes

Started: 2026-10-10T10:30:59+08:00

## Environment and scope

{
  "python": "3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]",
  "executable": "D:\\miniminiconda\\envs\\liuyan\\python.exe",
  "platform": "Windows-10-10.0.19045-SP0",
  "numpy": "2.4.4",
  "pandas": "3.0.1",
  "scipy": "1.17.1",
  "torch": "2.10.0+cu128",
  "cuda": "12.8",
  "gpu": "NVIDIA GeForce RTX 2060",
  "gpu_memory_bytes": 6442123264,
  "cuda_free_total_bytes": [
    5361369088,
    6442123264
  ]
}

Only UTS/Tampere training inputs and original historical panel identities are used. No training; no Git operations; fixed G6 policy, no target selection. G4/G5 historical outcome vectors are not parsed. Historical evidence is hashed only for preservation checks.

G4/G5/G6 evidence before: {"files": 281, "sha256": "295f05057ef5b0d2698a086333be24265705e5823bb57590572c59eeb30f9898"}

Initial code/design hashes: {"g4_uts_historical_protocol": "e91cf79a3d691e5b932b6387f9ed7150278f53de003dc6a67a19d114854913f3", "g5_tampere_historical_protocol": "0953f4726d7b769b140f9d0e20e8ed7481a2d2b144ecc875368e5bc764945d8b", "g6_calibration_and_scorer": "e8f42637b24a69c664dd8726c9b15974fb51f219d2c373d1d0b3fce8d42cffe1", "g7_code": "0a4f3b3862b636f78f16719bb49aff79c451af0b3cdb8b51588bf7cdf365ce1f", "g7_design": "bdf3ab9ddb53f0e374a758abeaeaaac36fe43ed9d643f496221266fe324f5be4", "g7_tests": "71527124526bbefb3d5d0ae71c5c7017382ea386d53e92d6173ca7ca3b4e2584"}

## Commands

| Stage | Started | Seconds | Exit code | Command |
|---|---|---:|---:|---|
| compile-initial | 2026-10-10T10:31:05+08:00 | 0.098 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m compileall -q fhm_dapp_tim/fhm_dapp/uji_v17_g7_frozen_transfer.py tests/test_uji_v17_g7_frozen_transfer.py |
| tests-initial | 2026-10-10T10:31:09+08:00 | 9.442 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_uji_v17_g7_frozen_transfer.py tests/test_uji_v17_g6_calibration.py |
| preflight-initial | 2026-10-10T10:31:24+08:00 | 7.984 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.uji_v17_g7_frozen_transfer preflight |
| cuda-fit-smoke | 2026-10-10T10:33:01+08:00 | 12.088 | 0 | D:\miniminiconda\envs\liuyan\python.exe -c "<inline code SHA256=08bbe2ac7f7b542568f4828cd590e071178feb9f002c5c7ff267e7fc277ad99e>" |
| prepare | 2026-10-10T10:33:35+08:00 | 7.755 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.uji_v17_g7_frozen_transfer prepare |
| check-before-run | 2026-10-10T10:33:51+08:00 | 7.737 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.uji_v17_g7_frozen_transfer check |
| run-once | 2026-10-10T10:34:14+08:00 | 26.000 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.uji_v17_g7_frozen_transfer run --device cuda |
| check-after-run | 2026-10-10T10:34:56+08:00 | 7.819 | 0 | D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.uji_v17_g7_frozen_transfer check |

## Completed execution and preservation

All seven requested command stages plus the FIT-only CUDA smoke exited 0. CPU tests: 33 passed. Exactly one G7 transfer run; immutable lock retained. Final check: VERIFIED. No experiment failures, warning messages, source repairs or scientific changes. An initial read-only adapter-path lookup typo was corrected to v18_4_datasets.py; it did not invoke experiment stages or touch scientific files.

Both five-member historical panels match original frozen SHA256 identities. Every model shares the exact FIT-only memory and origin rebuilt from its own original FIT rows. FIT smoke: two rows per domain; state_dict unchanged, gradients None. G6 non-container code/design/test hashes match its original manifest; no TUJI MAT container was accessed.

G4/G5/G6 preserved evidence after: {"files": 281, "sha256": "295f05057ef5b0d2698a086333be24265705e5823bb57590572c59eeb30f9898"}; matches all 281 initial file hashes. Frozen G7 source, test and design hashes unchanged. Historical G1-G6 claims/gates are preserved.

UTS roles: FIT 945 sites/5791 rows, CAL 258/1625, CONFIRM 263/1692; all pairwise physical overlaps 0. Tampere: FIT 334/334, CONFIRM 112/112; overlap 0. CUDA FIT peak allocation: UTS 64692224 bytes; Tampere 51024896 bytes.

Only three training CSV files are used. A Python open audit during FIT smoke allowed only those three raw-data paths and observed no provided-test access. Canonical train-only loaders and the unchanged scoring call graph are used during the formal run. Previous G4/G5 outcome JSONs are not parsed; G6 CAL is read solely for frozen policy provenance.

## Fixed descriptive result and numerical checks

Decision: TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL. No p-values or new confidence/significance claims are computed. All three stored roles were verified against frozen sorted physical-coordinate key digests. Means recomputed from saved MAE/ES and radius/coverage vectors; device-weighted row metrics verified. Same-T MAE vectors are exactly equal across scale settings for every physical coordinate. No further inference or random draws were used for this review.

| Domain / historical role | Sites / rows | Baseline MAE / ES (m) | Fixed T0.6_S0.5 MAE / ES (m) | Delta MAE / ES (m) | MAE improved / worsened / tied |

|---|---:|---:|---:|---:|---:|
| uts/calibration | 258/1625 | 7.155117679 / 5.024128699 | 5.380405089 / 4.166231202 | -1.774712590 / -0.857897496 | 195/63/0 |
| uts/confirmation | 263/1692 | 6.819208114 / 4.786196406 | 5.107270462 / 3.983257221 | -1.711937652 / -0.802939185 | 192/71/0 |
| tampere/confirmation | 112/112 | 7.202666301 / 5.212626871 | 5.986883910 / 5.019884989 | -1.215782391 / -0.192741881 | 71/41/0 |

## Mechanism controls (descriptive only)

| Role | T-only ES delta vs baseline (m) | s-only ES delta vs baseline (m) | Added scale shrink ES delta at T0.6 (m) |

|---|---:|---:|---:|
| uts/calibration | -0.883286888 | +0.007067910 | +0.025389392 |
| uts/confirmation | -0.831413743 | +0.008183067 | +0.028474558 |
| tampere/confirmation | -0.244536690 | +0.023657801 | +0.051794809 |

MAE improvement arises from sharpening mixture weights: T-only and T+s MAE vectors are exactly equal. Shrinking scale alone leaves every MAE unchanged and slightly increases ES on both target domains. At T0.6, further scale shrinking increases ES relative to T-only, while the pre-frozen joint policy still improves ES against baseline. These paired Monte Carlo values do not justify changing the primary candidate.

## Building/floor heterogeneity

| Role | Building/floor | Sites | Baseline/candidate MAE (m) | Delta MAE (m) | Delta ES (m) |

|---|---|---:|---:|---:|---:|
| uts/calibration | 0/0 | 17 | 5.503215 / 4.585202 | -0.918013 | -0.435910 |
| uts/calibration | 0/1 | 6 | 3.668680 / 2.617114 | -1.051566 | -0.549737 |
| uts/calibration | 0/2 | 5 | 5.908123 / 4.437172 | -1.470950 | -0.532889 |
| uts/calibration | 0/3 | 21 | 7.642830 / 5.215109 | -2.427720 | -1.047926 |
| uts/calibration | 0/4 | 13 | 7.572406 / 5.845623 | -1.726782 | -1.305652 |
| uts/calibration | 0/5 | 15 | 5.348213 / 4.781697 | -0.566516 | -0.029520 |
| uts/calibration | 0/6 | 15 | 5.556108 / 4.487546 | -1.068562 | -0.621846 |
| uts/calibration | 0/7 | 17 | 6.167669 / 4.655129 | -1.512540 | -0.625130 |
| uts/calibration | 0/8 | 13 | 9.513295 / 7.156048 | -2.357246 | -1.632346 |
| uts/calibration | 0/9 | 12 | 5.728256 / 4.848279 | -0.879976 | -0.941941 |
| uts/calibration | 0/10 | 21 | 8.373808 / 6.352121 | -2.021687 | -0.951689 |
| uts/calibration | 0/11 | 17 | 6.831241 / 5.234846 | -1.596395 | -0.578330 |
| uts/calibration | 0/12 | 20 | 6.112994 / 3.841661 | -2.271334 | -1.049893 |
| uts/calibration | 0/13 | 23 | 8.722635 / 5.718398 | -3.004237 | -1.805932 |
| uts/calibration | 0/14 | 26 | 7.560844 / 6.447461 | -1.113383 | -0.207208 |
| uts/calibration | 0/15 | 17 | 9.982455 / 7.002118 | -2.980336 | -1.077886 |
| uts/confirmation | 0/0 | 17 | 5.625460 / 4.110154 | -1.515306 | -0.593776 |
| uts/confirmation | 0/1 | 6 | 7.341745 / 6.398619 | -0.943126 | +0.501374 |
| uts/confirmation | 0/2 | 6 | 6.137628 / 4.559878 | -1.577751 | -0.795077 |
| uts/confirmation | 0/3 | 21 | 7.677182 / 5.771375 | -1.905806 | -1.100527 |
| uts/confirmation | 0/4 | 14 | 7.025789 / 4.903253 | -2.122536 | -1.019162 |
| uts/confirmation | 0/5 | 15 | 5.365166 / 3.453636 | -1.911530 | -0.867189 |
| uts/confirmation | 0/6 | 16 | 5.265003 / 4.777131 | -0.487872 | -0.171858 |
| uts/confirmation | 0/7 | 17 | 6.382782 / 5.363161 | -1.019621 | -0.505956 |
| uts/confirmation | 0/8 | 13 | 7.868840 / 4.489277 | -3.379563 | -1.533073 |
| uts/confirmation | 0/9 | 13 | 8.827977 / 6.146872 | -2.681105 | -1.070205 |
| uts/confirmation | 0/10 | 21 | 5.311574 / 3.739886 | -1.571688 | -0.923157 |
| uts/confirmation | 0/11 | 17 | 7.016304 / 5.204239 | -1.812065 | -1.084886 |
| uts/confirmation | 0/12 | 21 | 7.641757 / 5.828720 | -1.813037 | -0.784151 |
| uts/confirmation | 0/13 | 23 | 7.765959 / 6.727024 | -1.038936 | -0.607535 |
| uts/confirmation | 0/14 | 26 | 6.830200 / 5.402352 | -1.427848 | -0.582585 |
| uts/confirmation | 0/15 | 17 | 7.033999 / 4.439192 | -2.594806 | -1.195223 |
| tampere/confirmation | 0/0 | 31 | 8.345234 / 5.425392 | -2.919843 | -1.073517 |
| tampere/confirmation | 0/1 | 55 | 7.536337 / 7.020032 | -0.516305 | +0.337022 |
| tampere/confirmation | 0/2 | 26 | 5.134531 / 4.470849 | -0.663682 | -0.263241 |

## Three largest coordinate-MAE harms per role

| Role | Physical coordinate key (building,floor,x,y) | Baseline MAE (m) | Candidate MAE (m) | Delta MAE (m) | Delta ES (m) |

|---|---|---:|---:|---:|---:|
| uts/calibration | [0,14,19.3671,17.8812] | 2.535445 | 7.549277 | +5.013832 | +1.523909 |
| uts/calibration | [0,14,84.6368,10.1681] | 9.228849 | 13.876946 | +4.648097 | +5.741589 |
| uts/calibration | [0,7,55.7736,17.017] | 5.400112 | 8.557059 | +3.156947 | +1.733330 |
| uts/confirmation | [0,6,29.5159,14.2223] | 3.970618 | 9.044249 | +5.073632 | +4.267972 |
| uts/confirmation | [0,13,20.4215,16.9571] | 12.401771 | 15.295283 | +2.893512 | +4.266496 |
| uts/confirmation | [0,1,75.1286,15.1718] | 10.467567 | 12.940788 | +2.473220 | +1.965123 |
| tampere/confirmation | [0,1,262.94,107.04] | 45.718090 | 60.967976 | +15.249885 | +24.282812 |
| tampere/confirmation | [0,1,262.94,112.04] | 42.114694 | 50.576828 | +8.462135 | +7.717421 |
| tampere/confirmation | [0,1,127.94,167.04] | 2.723615 | 8.408807 | +5.685192 | +0.697275 |

These harms are preserved despite favorable macro means. Full aligned coordinate vectors for all four fixed policies remain in transfer_v17_g7.json; paired differences are the exact candidate minus baseline subtraction on the same saved coordinate order.

## Radius and radial coverage diagnostics

| Role | Policy | Macro predicted radial distance (m) | Empirical coverage at nominal 50%/80%/90% |

|---|---|---:|---|
| uts/calibration | T1_S1 | 12.033409 | 0.774315/0.953140/0.991764 |
| uts/calibration | T0.6_S0.5 | 7.695598 | 0.598211/0.849293/0.926586 |
| uts/calibration | T0.6_S1 | 7.714848 | 0.602863/0.851809/0.927150 |
| uts/calibration | T1_S0.5 | 12.024400 | 0.769721/0.948941/0.991764 |
| uts/confirmation | T1_S1 | 11.775537 | 0.701497/0.935334/0.991099 |
| uts/confirmation | T0.6_S0.5 | 7.084047 | 0.541465/0.799645/0.905250 |
| uts/confirmation | T0.6_S1 | 7.105415 | 0.545008/0.798818/0.904574 |
| uts/confirmation | T1_S0.5 | 11.766811 | 0.701997/0.935334/0.991099 |
| tampere/confirmation | T1_S1 | 14.297442 | 0.741071/0.973214/1.000000 |
| tampere/confirmation | T0.6_S0.5 | 6.552897 | 0.428571/0.776786/0.857143 |
| tampere/confirmation | T0.6_S1 | 6.584924 | 0.437500/0.776786/0.857143 |
| tampere/confirmation | T1_S0.5 | 14.282164 | 0.741071/0.973214/1.000000 |

UTS device breakdown has only mapped device 0, and Tampere device -1 denotes unknown; no cross-device generalization can be inferred. All device row MAE/ES and per-coordinate credible radii/coverage are retained in the result JSON.

This is a retrospective analysis of previously seen G4/G5 internal partitions, not a blind external or prospective statistical test. UTS CAL is descriptive and excluded from the two-domain gate. E5 overlapping models share training data; domain replication does not imply universal transfer. Spatial dependence and 64+64 Monte Carlo samples limit uncertainty and coverage interpretation. No new bootstrap, parameter search, training or follow-up model is performed. G7 is not a JCR Q1 final model.

## Artifact identities

- manifest_v17_g7.json: 76f01a1ad317db4c01c43ffdf9b2ce50b26932f36e368235e54b5558be0a7763
- transfer_v17_g7.lock: c1484d4c02b84a7cd84ab566557823370b83ec276372b8f5bdb77c42fa2f74e6
- transfer_v17_g7.json: 01f731e39ec6cc1521b230cb7ef147dd862a2482de06a2c0bfc9e5ff52bd3a32
- RUN_REPORT_V17_G7.md: 002461532de1de2e0d2e5920fa1436acaa83b77d79787227d4356ecb3ad74ffd
- RUN_STDOUT.log: fdc27a28e3b6d2da9818b85de32818053692875ed67e15ee0bc6d20618a2768c
- RUN_STDERR.log: 47a0f444d3c64764e0d1bef0048da1f38ff3a263a698871b4f0ccbf2431f89e2
- data/UTSIndoorLoc/UTS_training.csv: a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b
- data/Tampere/Training_rss.csv: 7922efcd4ea14b1a5453a69ba35bc021a67b915b1b3076fa4ccf6da3e98e31b3
- data/Tampere/Training_coordinates.csv: 22f0894e80966ef12d5f4e73f6e0b466d5a5781dda1f76c35ae6d276a929afff

All ten checkpoint SHA256 identities and all scored-role coordinate-key digests are stored in manifest_v17_g7.json. Final vector/preservation review: exit 0; source/result/report/lock remain unchanged.

Finished: 2026-10-10T10:38:12+08:00; recorded execution+review elapsed 433.325s; summed stage process runtime 78.923s.
