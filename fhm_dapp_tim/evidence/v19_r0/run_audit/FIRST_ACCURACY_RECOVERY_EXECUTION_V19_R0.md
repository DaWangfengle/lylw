# V19-R0 First Accuracy-Recovery Execution Record

Status: **DEVELOPMENT_COMPLETE_WITH_USER_AUTHORIZED_BASELINE_GATE_WAIVER**.

The complete registered member panel, caches, and five E profiles were produced. The original frozen B32 historical-reference gate FAILED; it was not repaired or certified PASS. After the user explicitly instructed continuation, only the two named MAE/NLL stop raises were waived in memory. A separate read-only historical B64 diagnostic reproduced both numerical targets. This run requires scientific review with that exception visible.

| Binding | Value |
| --- | --- |
| Branch | v19-r0-accuracy-recovery |
| Exact source HEAD | 8e694620cd6fef442686b7f7d8ed2336c2cfc2d2 |
| Execution started UTC | 2026-10-06T12:15:24.608Z |
| JSON record snapshot UTC | 2026-10-06T14:20:17.408Z |
| Training command wall total (s) | 6022.658495 |
| Cache command wall total (s) | 130.829419 |
| Frozen initial evaluate runtime (s) | 28.921242 |
| Read-only baseline diagnostic runtime (s) | 26.078000 |
| User-waived evaluate wrapper runtime (s) | 25.031000 |
| Development-result internal runtime (s) | 21.431741 |

## Environment and precheck

| Field | Value |
| --- | --- |
| cuda_available | True |
| gpu | NVIDIA GeForce RTX 2060 |
| gpu_total_memory_bytes | 6442123264 |
| os | Windows-10-10.0.19045-SP0 |
| python | 3.12.12 \| packaged by Anaconda, Inc. \| (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)] |
| python_executable | D:\miniminiconda\envs\liuyan\python.exe |
| torch | 2.10.0+cu128 |
| torch_cuda | 12.8 |

Before starting, tracked source was clean, branch and exact HEAD matched, all prescribed write-once outputs were absent, and no partial V19 member/cache artifacts existed. CUDA was available.

| Compilation target | Exit code |
| --- | --- |
| fhm_dapp_tim/fhm_dapp/v19_r0_core.py | 0 |
| fhm_dapp_tim/fhm_dapp/v18_5_predictive_decision_probe.py | 0 |
| fhm_dapp_tim/fhm_dapp/uji_v19_r0_accuracy_recovery.py | 0 |

Exact pytest command:

```text
& 'D:\miniminiconda\envs\liuyan\python.exe' -m pytest -q tests/test_v18_5_core.py tests/test_v18_5_model.py tests/test_v18_5_radio.py tests/test_v19_r0_core.py tests/test_v18_5_predictive_decision_probe.py tests/test_uji_v19_r0_accuracy_recovery.py
```

```text
..........................................                               [100%]
42 passed in 6.79s
```

**42 passed; 0 failed; 0 skipped; 0 xfailed.** No source or tests were edited to achieve this.

## Historical artifact and role bindings

| Local file | Verified SHA256 |
| --- | --- |
| data/UJIndoorLoc/trainingData.csv | 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168 |
| fhm_dapp_tim/models/v18_5/uji_lambda_0p00.pt | ae1a2c6b46afca338a824bd45c81f6762cdb5744dddaf2ef8dc24518af2add3c |
| fhm_dapp_tim/models/v18_5/uji_selection_refit.pt | ec9cc9ed94c05f15a07515c3d2c7d55de5b497c788ee5fc15dacf689c1d4b3c2 |

| Historical JSON | File SHA256 | Canonical SHA256 |
| --- | --- | --- |
| uji_development_v18_5.json | 59dab320819914144dc226418cbc097f5fd602973cb79c5fd2de1e2d6871e4db | 86c5b7736a4bf2b0f318ae3bb7190b8f8c01b8e1a97d6cc0d5311ce04f898869 |
| uji_method_freeze_v18_5.json | 958bf49d9d88e0ce6f0873482e8501025f16e8cd9cbbb318f37d390c7307c130 | 20a5a0b81c44d97fbb937b21bf852285c676ccd8f1370ef031ca03bb65e8e427 |
| uji_registry_v18_5.json | 2d72c8a1358548f687ffb2e1001b3bf0be9147a6627465d655c32fa65feea4cd | 3ba80eb59ed6f48f9b71186cf98c155a88d192ba8a5cbd751dd051e0febf93fb |

| Role | Rows | Coordinate groups | Row-index SHA256 |
| --- | --- | --- | --- |
| calibration | 2905 | 139 | 39291a0b9814a626afdbbbf8fa83bd9a37f163b9f47a39807ec68d91a242c2c9 |
| development | 3005 | 139 | 2f0443217c175ab6cfc5dbe2be7e548bb68174422dce4f78120c57c27d7e4096 |
| fit | 14027 | 655 | 217df510b65837b649d16118b106904e9f542404ffa4454b0342215d777b1e1e |

Only trainingData.csv was used. V19 training used historical FIT only; evaluation used historical DEVELOPMENT only. Historical CALIBRATION was consumed solely by the frozen read-only V18.5 decision probe and did not train or select V19.

## Read-only V18.5 predictive-decision probe

Runtime: 99.626908 s. 2905 rows / 139 coordinate groups; 64 Student-t draws per stream; scoring seed 185512. No network training or checkpoint mutation.

| Cell | T | epsilon | Scale | Posterior-mean macro MAE +/- SE (m) | Predictive spatial-median macro MAE +/- SE (m) | Energy Score (m) | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1 | 0 | 1 | 7.284263 +/- 0.493030 | 7.366557 +/- 0.612004 | 5.346126 | 6.231538 |
| calibration_only | 1.25 | 0 | 1.25 | 7.384087 +/- 0.490456 | 7.331935 +/- 0.607355 | 5.419779 | 6.185489 |
| selected_final | 1.25 | 0.1 | 1.25 | 7.166467 +/- 0.447844 | 7.232601 +/- 0.509204 | 5.360942 | 6.186083 |

Paired candidate = spatial median, reference = posterior mean; negative delta is improvement.

| Cell | Delta (m) | Paired SE (m) | 95% CI (m) | Groups improved |
| --- | --- | --- | --- | --- |
| base | 0.082295 | 0.235508 | [-0.379301, 0.543890] | 44.604317% |
| calibration_only | -0.052152 | 0.257917 | [-0.557669, 0.453364] | 44.604317% |
| selected_final | 0.066134 | 0.194368 | [-0.314827, 0.447095] | 40.287770% |

| Cell | Row-mean posterior-mean MAE (m) | Row-mean spatial-median MAE (m) |
| --- | --- | --- |
| base | 6.839910 | 6.827461 |
| calibration_only | 6.942845 | 6.781607 |
| selected_final | 6.778909 | 6.786429 |

Selected-final posterior-mean reproduction: PASS; difference -1.353663039438402e-07 m. Every mean-vs-median paired 95% CI crosses zero. The probe did not trigger a V19 design change.

| Probe artifact | SHA256 |
| --- | --- |
| V18_5_PREDICTIVE_DECISION_PROBE.json (file) | 50058823f32711a00f1f18549ccbdf10af36ea75bd13df68cd072eb6cb8620f4 |
| V18_5_PREDICTIVE_DECISION_PROBE.md (file) | fabffad47a97139ece84141554dc119af9c751b46d35cafecd72505de5ad6dc6 |
| probe_sha256 (canonical) | 6194488728c2dd999a508c1f94cae2f9eaeff8108e19461903c3ae93320b6970 |

The selected-final frozen V18.5 diagnostic includes the already-selected radio correction and frozen T/scale. Any temporary radio nuisance-tensor optimization is historical inference, covered by the prior explicit user authorization; it did not optimize network parameters and was not introduced into V19-R0.

## Registered V19-R0 method and panel

| Registry hash | SHA256 |
| --- | --- |
| registry_v19_r0.json (file) | 28061b2004f3435cee3c53ac6ba47fcdbc8c19ceb396519a7b1e0cdb7f8be745 |
| registry_sha256 (canonical) | e9a2d7f9b694960d7210bea3dbaedca41ab1ccb06d7ec4f76f840ba6d1d0c957 |

Wide member law: **full_anchor_invariant_logits_no_candidate_reranker**. Full anchor dictionary count = 655. All 16 members have exactly equal anchor coordinates, building/floor identifiers, and coordinate origin. Anchor dictionary descriptor SHA256: e2e97262c988b130fbe09092cb04c2474b3f04965e9110a40ad8771e9da238fc.

| Parameter | Frozen value |
| --- | --- |
| Member seeds | 185511, 19301, 19302, 19303, 19304, 19305, 19306, 19307, 19308, 19309, 19310, 19311, 19312, 19313, 19314, 19315 |
| Member prefixes E | 1, 2, 4, 8, 16 |
| Top-L | 192 |
| Consensus K | 192 |
| alpha | 1 |
| lambda_spatial | 0 |
| Epochs | 27 |
| Student df | 4 |
| Predictive draws | 64 |
| Scoring seed | 19931 |
| Reliability law | V17 predictive radial second-moment; inverse radial variance member weights |
| Wide-path candidate reranker | false |
| Radio / temperature / scale calibration / geometry correction in V19 | none |
| Final E selected | false |

Member 1: seed 185511, exact inherited FIT lambda=0 epoch-27 checkpoint; **not retrained**. Members 2..16 were trained once, sequentially on CUDA, on FIT only, for exactly 27 epochs with lambda_spatial=0. No early stopping, epoch selection, seed replacement, or DEV-driven tuning.

## All 15 new member checkpoints

| Member | Seed | Checkpoint path | Checkpoint SHA256 | Training runtime (s) | Command wall (s) |
| --- | --- | --- | --- | --- | --- |
| 2 | 19301 | fhm_dapp_tim/models/v19_r0/member_02_seed19301.pt | 1d7eb305f8ca3c8635929e02732eb44784a10049682dd69ca5b81e3dc78b8042 | 373.482840 | 383.015751 |
| 3 | 19302 | fhm_dapp_tim/models/v19_r0/member_03_seed19302.pt | 3b3f15199fc93faf6ab1df20d5dd216b37ce74a8476dd10c3d5c31a5f127d3f5 | 405.461548 | 415.246261 |
| 4 | 19303 | fhm_dapp_tim/models/v19_r0/member_04_seed19303.pt | bccb7c3b7411b4c03922ccb3ce1e62c10d2512dccb42ffcec7d21c709ab20a46 | 394.151914 | 404.609848 |
| 5 | 19304 | fhm_dapp_tim/models/v19_r0/member_05_seed19304.pt | 64bbba38e51c64866d79691c80cf2023881e6db2342924290d239f697eac5af1 | 392.342181 | 402.146982 |
| 6 | 19305 | fhm_dapp_tim/models/v19_r0/member_06_seed19305.pt | 675ccc07eeb29ae686cd6b1872fa55b302f68203b84db927f21f89bf823d14ea | 392.036069 | 401.932283 |
| 7 | 19306 | fhm_dapp_tim/models/v19_r0/member_07_seed19306.pt | f3c5d2a62efcc156e73c6c6f061fc0010dfd899e3db8f8579fd0c273c44a5713 | 395.552320 | 405.554385 |
| 8 | 19307 | fhm_dapp_tim/models/v19_r0/member_08_seed19307.pt | a5e312cb4fb26fcbbe480e17cc0db8d8e8c80e5bdebf70571349cb4c63e0f82c | 390.603223 | 400.551370 |
| 9 | 19308 | fhm_dapp_tim/models/v19_r0/member_09_seed19308.pt | 58d4e1da6424449c0ee5b9dc0344c1a2f4502c43300f5e9bd2d631615c4b05ed | 391.189019 | 401.307962 |
| 10 | 19309 | fhm_dapp_tim/models/v19_r0/member_10_seed19309.pt | 8a3263beda095605aa60f0c964664efc304fa9ad163f1a2bb5d1c04457cd087f | 391.799054 | 401.773780 |
| 11 | 19310 | fhm_dapp_tim/models/v19_r0/member_11_seed19310.pt | 82a822c397f08b0bab5202bad3929c8fa73f10c78b886340d09f4f0140b8552c | 389.554308 | 399.525042 |
| 12 | 19311 | fhm_dapp_tim/models/v19_r0/member_12_seed19311.pt | eb676ef0772d269cf557f4bbf187becea84cec9a4038434cda797afbd67a3c8a | 392.952798 | 402.714057 |
| 13 | 19312 | fhm_dapp_tim/models/v19_r0/member_13_seed19312.pt | c66aa7e836562fd9b45f1926229d980b6533bb2ce6c0cd8fe717eb58fcfdb070 | 390.817786 | 400.642054 |
| 14 | 19313 | fhm_dapp_tim/models/v19_r0/member_14_seed19313.pt | 424464b437a98e3fe855205a91bb57a632437ee4c30a55e46e42c173cdcbaa7a | 393.177757 | 402.986904 |
| 15 | 19314 | fhm_dapp_tim/models/v19_r0/member_15_seed19314.pt | ded054a09c9133af59b879ae8c75a59c21e16268a53d051034e3485b0f476e89 | 391.439325 | 401.246621 |
| 16 | 19315 | fhm_dapp_tim/models/v19_r0/member_16_seed19315.pt | dbe73028a6a5d7f3042710e36f70cf81dff9833d2e6d861b945fd2c1c0e68a44 | 389.509811 | 399.405195 |

All 15 saved training trajectories and model parameters are finite. Final epoch-27 training losses/components:

| Member | loss | anchor_nll | coordinate_nll | domain_loss | hierarchy_loss | invariant_aux_loss | spatial_energy_loss | spatial_contribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 7.179783 | 1.959962 | 4.646582 | 0.869493 | 0.030647 | 2.057764 | 0.109762 | 0.000000 |
| 3 | 7.158626 | 1.955004 | 4.629129 | 0.882924 | 0.030154 | 2.061078 | 0.106009 | 0.000000 |
| 4 | 7.192261 | 1.962164 | 4.653618 | 0.868025 | 0.031507 | 2.069300 | 0.106822 | 0.000000 |
| 5 | 7.205262 | 1.964984 | 4.662586 | 0.860961 | 0.033492 | 2.071593 | 0.111518 | 0.000000 |
| 6 | 7.155781 | 1.959239 | 4.621131 | 0.896503 | 0.030455 | 2.061430 | 0.109028 | 0.000000 |
| 7 | 7.162470 | 1.951867 | 4.635917 | 0.884533 | 0.030848 | 2.060139 | 0.106636 | 0.000000 |
| 8 | 7.150851 | 1.956240 | 4.619962 | 0.886218 | 0.031071 | 2.059210 | 0.109327 | 0.000000 |
| 9 | 7.185904 | 1.965413 | 4.642833 | 0.923644 | 0.031930 | 2.062047 | 0.113617 | 0.000000 |
| 10 | 7.190884 | 1.961403 | 4.651520 | 0.896150 | 0.031778 | 2.069058 | 0.107773 | 0.000000 |
| 11 | 7.195396 | 1.963504 | 4.654874 | 0.880783 | 0.032605 | 2.066710 | 0.111486 | 0.000000 |
| 12 | 7.186473 | 1.969103 | 4.640174 | 0.882934 | 0.032380 | 2.067439 | 0.110631 | 0.000000 |
| 13 | 7.180958 | 1.955422 | 4.651918 | 0.876035 | 0.030036 | 2.059192 | 0.108211 | 0.000000 |
| 14 | 7.182134 | 1.959942 | 4.647387 | 0.878310 | 0.028787 | 2.065982 | 0.107952 | 0.000000 |
| 15 | 7.201212 | 1.965103 | 4.658148 | 0.898938 | 0.030797 | 2.070462 | 0.110257 | 0.000000 |
| 16 | 7.156822 | 1.959266 | 4.624741 | 0.867087 | 0.029012 | 2.059819 | 0.109742 | 0.000000 |

| Member | forced_support_size | natural_support_size | natural_recall_mass |
| --- | --- | --- | --- |
| 2 | 53.051330 | 53.045127 | 0.999276 |
| 3 | 52.955229 | 52.950096 | 0.999407 |
| 4 | 53.133243 | 53.128110 | 0.999351 |
| 5 | 53.109788 | 53.103942 | 0.999311 |
| 6 | 53.018251 | 53.012619 | 0.999231 |
| 7 | 53.012832 | 53.007200 | 0.999279 |
| 8 | 53.030299 | 53.023526 | 0.999009 |
| 9 | 52.892350 | 52.885435 | 0.998973 |
| 10 | 53.022100 | 53.016682 | 0.999275 |
| 11 | 52.935695 | 52.930206 | 0.999197 |
| 12 | 52.801953 | 52.795751 | 0.999340 |
| 13 | 52.949882 | 52.943751 | 0.999328 |
| 14 | 52.904898 | 52.898624 | 0.999217 |
| 15 | 53.109075 | 53.104228 | 0.999374 |
| 16 | 52.954516 | 52.948742 | 0.999300 |

## All 16 development caches

Each cache binds its exact checkpoint SHA, DEVELOPMENT row-index hash, 655-anchor dictionary, and the frozen wide law; candidate_reranker_applied=false. Dense probabilities [3005,655], member-specific locations/scales [3005,655,2], Top-192 indices/weights, retained mass, and positive finite radial scales were verified. All probability mixtures were finite and normalized.

| Member | Seed | Cache path | Cache SHA256 | Mean Top-192 mass | Minimum Top-192 mass | Cache runtime (s) | Command wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 185511 | fhm_dapp_tim/models/v19_r0/development_cache/member_01_seed185511_development.pt | 1540cde8fb8083542a6465c1066af200521903475e3211c412f20dd06058cb8c | 0.999594 | 0.980774 | 0.725806 | 8.122544 |
| 2 | 19301 | fhm_dapp_tim/models/v19_r0/development_cache/member_02_seed19301_development.pt | e4b4990cdf4fc7def96d5255295bb5f3b13d382e18102e9f9e0ed006b2d4c2b9 | 0.999583 | 0.985139 | 0.743962 | 8.123802 |
| 3 | 19302 | fhm_dapp_tim/models/v19_r0/development_cache/member_03_seed19302_development.pt | 447813d12a039bb8345aac19ed595374b0666c0ed82bdc5fd9f7785046a8e4d5 | 0.999619 | 0.981716 | 0.756204 | 8.165446 |
| 4 | 19303 | fhm_dapp_tim/models/v19_r0/development_cache/member_04_seed19303_development.pt | f6f86acde3bde55a32280f12127cbfaa55952882554326bcdf5fabb37cb4c0e1 | 0.999530 | 0.976585 | 0.740066 | 8.146161 |
| 5 | 19304 | fhm_dapp_tim/models/v19_r0/development_cache/member_05_seed19304_development.pt | e9fe9db88a5a5b3275a7166b10f717a066642680f30a9cbed090f868cc40a0f7 | 0.999553 | 0.986303 | 0.721183 | 8.071081 |
| 6 | 19305 | fhm_dapp_tim/models/v19_r0/development_cache/member_06_seed19305_development.pt | 4caebb957573d199faa215ffeed0e401642a0d1d3354b49b22a94e2691f20a97 | 0.999617 | 0.987889 | 0.859349 | 8.552227 |
| 7 | 19306 | fhm_dapp_tim/models/v19_r0/development_cache/member_07_seed19306_development.pt | 109c40c2e5b898f8c6cb8b73198c31f5db7d735e05f2136b1e638fa1d700f885 | 0.999581 | 0.983200 | 0.730054 | 8.114058 |
| 8 | 19307 | fhm_dapp_tim/models/v19_r0/development_cache/member_08_seed19307_development.pt | 9f7a26353bc36bd6b32cca937cf2894c7b117b2e87c7896fbc6fae2da7df4e0f | 0.999596 | 0.989969 | 0.766032 | 8.138959 |
| 9 | 19308 | fhm_dapp_tim/models/v19_r0/development_cache/member_09_seed19308_development.pt | 3f7780b96b43646551fd0c75375df227a7dfeded23a11e6c4f01db71931e5e2b | 0.999579 | 0.984961 | 0.722857 | 8.133813 |
| 10 | 19309 | fhm_dapp_tim/models/v19_r0/development_cache/member_10_seed19309_development.pt | 53c53dc5349b94fc640e189ddb0209691a724f53a92ad936df1c8d6655902ffc | 0.999555 | 0.981679 | 0.712296 | 8.177118 |
| 11 | 19310 | fhm_dapp_tim/models/v19_r0/development_cache/member_11_seed19310_development.pt | 3ff4623d9117c9dccef90b8d3f13559f81fef80baed7dfce462e004f6b0caaa5 | 0.999635 | 0.985717 | 0.734353 | 8.103166 |
| 12 | 19311 | fhm_dapp_tim/models/v19_r0/development_cache/member_12_seed19311_development.pt | e1a00a7644b46f21f07713c2029661d1e0e39df0be6f2ba982c0e6e388467cfe | 0.999652 | 0.986980 | 0.807337 | 8.392918 |
| 13 | 19312 | fhm_dapp_tim/models/v19_r0/development_cache/member_13_seed19312_development.pt | 66a84242e5be1845c1d5fe6681858b743f7b609a8f176026b3a02bbb6214c7ba | 0.999585 | 0.987551 | 0.743900 | 8.217922 |
| 14 | 19313 | fhm_dapp_tim/models/v19_r0/development_cache/member_14_seed19313_development.pt | bb3c19f2640f7fd379a1126d3b104d0537fe1236e83adabb0012b3f7e898e2bb | 0.999627 | 0.987914 | 0.725597 | 8.099856 |
| 15 | 19314 | fhm_dapp_tim/models/v19_r0/development_cache/member_15_seed19314_development.pt | 485ed875a2346cb4acc92661def7852b60dcedecada5d041b13762ae9d74c6b5 | 0.999565 | 0.990302 | 0.757581 | 8.121923 |
| 16 | 19315 | fhm_dapp_tim/models/v19_r0/development_cache/member_16_seed19315_development.pt | f8c247ffe31f9ad89e24bc1fe5e892c06ba71f7a278d23327f910b6eefa61950 | 0.999672 | 0.991220 | 0.754347 | 8.148425 |

## Original gate failure, batch diagnosis, and user-authorized continuation

The original registered evaluate command exited with RuntimeError at the historical MAE guard (uji_v19_r0_accuracy_recovery.py:893). It stopped before the NLL check and before a result file was written. All 15 new checkpoints, 16 caches, and the original failure record were retained.

```text
warning: unable to access 'C:\Users\LiuYan/.config/git/ignore': Permission denied
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\uji_v19_r0_accuracy_recovery.py", line 1096, in <module>
    main()
  File "D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\uji_v19_r0_accuracy_recovery.py", line 1091, in main
    payload = evaluate(device_name=args.device)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\liuyanlunwen\fhm_dapp_tim\fhm_dapp\uji_v19_r0_accuracy_recovery.py", line 893, in evaluate
    raise RuntimeError(
RuntimeError: V19-R0 failed to reproduce historical V18.5 development MAE
V19_STAGE_WALL_SECONDS=28.921242
```

Preserved failure log: D:\liuyanlunwen\fhm_dapp_tim\evidence\v19_r0\run_audit\FIRST_ACCURACY_RECOVERY_FAILURE_V19_R0.log. SHA256: c41868ad6d4ddefa4310c40174f8d057b9c4e7700c94e72614dcc29fece18858.

| Check | Historical target | Frozen B32 control | B32 minus target | Historical B64 diagnostic | B64 minus target |
| --- | --- | --- | --- | --- | --- |
| Mean macro MAE (m) | 7.400095281124617 | 7.392727511785563 | -0.007367769339054142 | 7.400095271100894 | -1.002372318481548e-08 |
| Mixture NLL | 6.29520105202517 | 6.295426654581435 | 0.0002256025562648034 | 6.29520105202517 | 0 |

Tolerance = 2e-5. **Registered B32 baseline: FAIL_USER_AUTHORIZED_WAIVER. Historical B64 read-only numerical reproduction: PASS.** The historical Energy Score was not a reproduction gate; the B64 diagnostic uses the registered R0 Student-t scoring randomness, so its Energy Score is not asserted to reproduce the prior historical scoring estimate.

The source explains batch dependence: v18_5_model.py:553 chooses candidate tensor width from the maximum pool size within the batch. model.py:729-732 standardizes candidate features without masking padded entries; v18_5_model.py:402-403 uses that standardization in reranking. Historical inference uses V185_CONFIG batch_size=64 (v18_5_model.py:71), while the V19 evaluate control uses batch_size=32 (uji_v19_r0_accuracy_recovery.py:700). The read-only B64 reproduction is consistent with this cause. The frozen source was not repaired.

User authorization: "忽视失败停止条件，继续" received 2026-10-06T14:12:45.357Z.

Replaced only the two named historical MAE/NLL RuntimeError raises with an audit-list append; inserted provenance before canonical result SHA/write. All scoring, batching, mixtures, seeds, E/L/K/alpha, other guards and on-disk source preserved.

The runtime AST wrapper replaced only these two raises with an audit-list append:

| Waived stop message |
| --- |
| V19-R0 failed to reproduce historical V18.5 development MAE |
| V19-R0 failed to reproduce historical V18.5 development NLL |

It inserted explicit failure/waiver provenance into the NEW result before the canonical SHA and first write. Frozen B32 scoring, all E/L/K/alpha/seeds/draws, mixtures, methods, source files, and every other guard remained unchanged. No checkpoint/cache retraining or scientific rerun was performed; only evaluation was repeated after the failed guard.

| Provenance binding | SHA256 |
| --- | --- |
| Executed evaluation AST | 9f872bfe7d3180571d10790ce16f9d341ef88716d0aa9889370a56778605737e |
| Unchanged registered evaluation source file | adfc3fae2532dfa9d3008f6e63dd6f8ba9b23b4699016da0b979aa203b7e6309 |

The exact runtime command is retained in the paired execution JSON under user_authorized_gate_waiver.runtime_execution_command. These results are not a clean PASS of the originally prescribed fail-closed acceptance.

## A/B/C controls and result decomposition

All subsequent result tables use 3005 DEVELOPMENT rows / 139 coordinate groups. Macro metrics use coordinate groups equally; +/- denotes SE. Original control A is the frozen as-implemented B32 candidate-reranked <=128 path with the explicitly waived historical gate. The separate B64 diagnostic is not silently substituted into paired comparisons.

| Control | Posterior-mean MAE +/- SE (m) | Predictive spatial-median MAE +/- SE (m) | Energy Score +/- SE (m) | NLL +/- SE |
| --- | --- | --- | --- | --- |
| A: Original V18.5 B32 (gate waived) | 7.392728 +/- 0.420075 | 7.228634 +/- 0.392028 | 5.261435 +/- 0.265493 | 6.295427 +/- 0.120199 |
| B: Same member, dense all-anchor invariant | 7.795709 +/- 0.426768 | 7.385929 +/- 0.448870 | 5.448163 +/- 0.284251 | 6.316940 +/- 0.116912 |
| C: E1 Top192 consensus | 7.771483 +/- 0.426333 | 7.400355 +/- 0.449871 | 5.449465 +/- 0.283945 | 6.316563 +/- 0.116917 |
| Read-only historical B64 diagnostic | 7.400095 +/- 0.419998 | 7.228944 +/- 0.391530 | 5.258546 +/- 0.264750 | 6.295201 +/- 0.119835 |

Within-control predictive-spatial-median minus posterior-mean paired effects:

| Comparison | Delta (m) | Paired SE (m) | 95% CI (m) | Coordinate groups improved |
| --- | --- | --- | --- | --- |
| A: spatial median - mean | -0.164094 | 0.116855 | [-0.393129, 0.064941] | 41.726619% |
| B: spatial median - mean | -0.409781 | 0.161831 | [-0.726970, -0.092591] | 48.920863% |
| C/E1: spatial median - mean | -0.371128 | 0.162402 | [-0.689437, -0.052820] | 48.920863% |

Same-member changes isolate support/inference law and Top192 consensus; they are not ensemble effects. Negative paired delta means improvement.

**Dense all-anchor minus original V18.5 B32**

| Comparison | Delta (m) | Paired SE (m) | 95% CI (m) | Coordinate groups improved |
| --- | --- | --- | --- | --- |
| Energy Score | 0.186728 | 0.044390 | [0.099724, 0.273732] | 38.129496% |
| Posterior-mean MAE | 0.402982 | 0.109987 | [0.187406, 0.618557] | 37.410072% |
| Predictive spatial-median MAE | 0.157295 | 0.131221 | [-0.099899, 0.414488] | 49.640288% |

**E1 Top192 minus dense all-anchor**

| Comparison | Delta (m) | Paired SE (m) | 95% CI (m) | Coordinate groups improved |
| --- | --- | --- | --- | --- |
| Energy Score | 0.001302 | 0.016605 | [-0.031244, 0.033847] | 47.482014% |
| Posterior-mean MAE | -0.024226 | 0.009157 | [-0.042173, -0.006279] | 61.151079% |
| Predictive spatial-median MAE | 0.014426 | 0.034853 | [-0.053886, 0.082739] | 46.043165% |

## D: Complete E={1,2,4,8,16} metric table

| Control | Posterior-mean MAE +/- SE (m) | Predictive spatial-median MAE +/- SE (m) | Energy Score +/- SE (m) | NLL +/- SE |
| --- | --- | --- | --- | --- |
| E=1 | 7.771483 +/- 0.426333 | 7.400355 +/- 0.449871 | 5.449465 +/- 0.283945 | 6.316563 +/- 0.116917 |
| E=2 | 7.535716 +/- 0.428035 | 7.296676 +/- 0.456089 | 5.387397 +/- 0.294576 | 6.243302 +/- 0.115366 |
| E=4 | 7.464233 +/- 0.429647 | 7.198662 +/- 0.453943 | 5.378646 +/- 0.305792 | 6.234543 +/- 0.115557 |
| E=8 | 7.431942 +/- 0.413425 | 7.231765 +/- 0.435235 | 5.341231 +/- 0.287737 | 6.233780 +/- 0.114862 |
| E=16 | 7.445961 +/- 0.407862 | 7.252396 +/- 0.435390 | 5.352445 +/- 0.283738 | 6.233554 +/- 0.114977 |

Within-E predictive-spatial-median minus posterior-mean paired effects:

| Comparison | Delta (m) | Paired SE (m) | 95% CI (m) | Coordinate groups improved |
| --- | --- | --- | --- | --- |
| E=1 | -0.371128 | 0.162402 | [-0.689437, -0.052820] | 48.920863% |
| E=2 | -0.239039 | 0.151657 | [-0.536287, 0.058208] | 44.604317% |
| E=4 | -0.265571 | 0.154334 | [-0.568065, 0.036924] | 41.726619% |
| E=8 | -0.200177 | 0.155987 | [-0.505912, 0.105558] | 41.726619% |
| E=16 | -0.193565 | 0.155861 | [-0.499052, 0.111922] | 43.884892% |

| E | Mean member-weight entropy | Mean maximum member weight | Mean consensus Top-1 probability | Mean member Top-192 retained mass |
| --- | --- | --- | --- | --- |
| 1 | 0.000000 | 1.000000 | 0.269800 | 0.999594 |
| 2 | 0.654733 | 0.604463 | 0.259520 | 0.999588 |
| 4 | 1.322794 | 0.359431 | 0.251719 | 0.999582 |
| 8 | 2.005394 | 0.203934 | 0.248094 | 0.999584 |
| 16 | 2.690351 | 0.115787 | 0.246706 | 0.999596 |

## E: Paired E2/E4/E8/E16 minus E1 effects

| Comparison | Delta (m) | Paired SE (m) | 95% CI (m) | Coordinate groups improved |
| --- | --- | --- | --- | --- |
| E=2 - E1: Posterior-mean MAE | -0.235767 | 0.110616 | [-0.452574, -0.018961] | 55.395683% |
| E=2 - E1: Predictive spatial-median MAE | -0.103679 | 0.101440 | [-0.302501, 0.095144] | 56.834532% |
| E=2 - E1: Energy Score | -0.062068 | 0.058401 | [-0.176534, 0.052397] | 55.395683% |
| E=4 - E1: Posterior-mean MAE | -0.307250 | 0.137667 | [-0.577077, -0.037424] | 58.992806% |
| E=4 - E1: Predictive spatial-median MAE | -0.201693 | 0.136278 | [-0.468798, 0.065413] | 54.676259% |
| E=4 - E1: Energy Score | -0.070819 | 0.093000 | [-0.253099, 0.111461] | 58.992806% |
| E=8 - E1: Posterior-mean MAE | -0.339541 | 0.155244 | [-0.643819, -0.035263] | 56.115108% |
| E=8 - E1: Predictive spatial-median MAE | -0.168590 | 0.145683 | [-0.454129, 0.116949] | 55.395683% |
| E=8 - E1: Energy Score | -0.108234 | 0.084766 | [-0.274375, 0.057908] | 54.676259% |
| E=16 - E1: Posterior-mean MAE | -0.325523 | 0.171077 | [-0.660834, 0.009789] | 53.956835% |
| E=16 - E1: Predictive spatial-median MAE | -0.147959 | 0.174409 | [-0.489800, 0.193881] | 50.359712% |
| E=16 - E1: Energy Score | -0.097020 | 0.092129 | [-0.277592, 0.083552] | 56.834532% |

## F: All 16 individual-member dense posterior-mean MAEs

| Member | Seed | Dense posterior-mean macro MAE +/- SE (m) |
| --- | --- | --- |
| 1 | 185511 | 7.795709 +/- 0.426768 |
| 2 | 19301 | 7.934525 +/- 0.442996 |
| 3 | 19302 | 7.998953 +/- 0.467008 |
| 4 | 19303 | 8.191247 +/- 0.460489 |
| 5 | 19304 | 8.090981 +/- 0.420105 |
| 6 | 19305 | 8.185872 +/- 0.432963 |
| 7 | 19306 | 7.554702 +/- 0.390424 |
| 8 | 19307 | 8.073010 +/- 0.408475 |
| 9 | 19308 | 7.977959 +/- 0.416824 |
| 10 | 19309 | 8.131074 +/- 0.400953 |
| 11 | 19310 | 7.877265 +/- 0.405259 |
| 12 | 19311 | 8.172809 +/- 0.450252 |
| 13 | 19312 | 8.391162 +/- 0.448128 |
| 14 | 19313 | 7.961781 +/- 0.410704 |
| 15 | 19314 | 7.825888 +/- 0.437488 |
| 16 | 19315 | 7.881798 +/- 0.389515 |

## G: Diversity diagnostics

| Diagnostic | Value |
| --- | --- |
| mean_member_point_rms_disagreement_m | 2.912348 |
| mean_pairwise_top1_anchor_agreement | 0.508400 |
| p90_member_point_rms_disagreement_m | 5.267547 |

## H: E-curve interpretation without selecting E

Posterior-mean MAE decreases E1 -> E2 -> E4 -> E8, then increases by 0.014018 m at E16. Predictive-spatial-median MAE decreases through E4 and increases at E8 and E16. Neither curve is monotone across all five prefixes.

The point-estimate curve largely saturates around E4-E8. E16 is not materially better than E8: mean MAE is 0.014018 m worse, spatial-median MAE 0.020631 m worse, Energy Score 0.011213 m worse, and NLL 0.000226 lower. A direct paired E16-vs-E8 interval was not registered or computed.

Relative to E1, posterior-mean paired 95% CIs exclude zero for E2, E4 and E8. The E16 interval [-0.660834, 0.009789] m includes zero. All spatial-median and Energy-Score paired E-vs-E1 intervals include zero.

Switching the inherited member from the as-implemented original B32 candidate-reranked control to dense all-anchor invariant inference increases mean MAE by 0.402982 m. Top192 consensus recovers only 0.024226 m; the E16 ensemble recovers 0.325523 m relative to E1. Member diversity primarily offsets the wide-law penalty rather than producing a new accuracy advantage over V18.5.

E16 mean MAE is 0.053233 m worse than the as-implemented original B32 control, and 0.045865 m worse than the historical B64 posterior-mean point estimate. E16 spatial-median MAE is 0.023762 m worse than the as-implemented original B32 spatial-median control. No paired B64-vs-V19 effects were newly computed.

The best individual dense mean point estimate is member 7 (seed 19306), 7.554702 +/- 0.390424 m. E2 and E4 improve before this member enters the E8 prefix; E16 mean MAE 7.445961 m is lower than every individual dense mean. This supports a multi-member contribution and does not establish that one strong seed explains all gains. Dense individual and joint-consensus laws differ, so this comparison is descriptive.

Diversity is present: row-mean RMS disagreement 2.912348 m, P90 5.267547 m, pairwise top1 agreement 0.508400. At E16 mean maximum member weight is 0.115787 and mean weight entropy 2.690351; the weights are distributed across the panel.

R0 did not demonstrate the hoped-for strong Euclidean accuracy recovery or 5-6 m accuracy. No final E is selected and no R1 work is performed.

## Issues and their handling

1. **initial_read_only_path_verification**: Inspector used unprefixed historical JSON filenames and raised FileNotFoundError before compilation or scientific execution.

Handling: Read registered DEFAULT_* constants, corrected inspector to uji_registry_v18_5.json / uji_development_v18_5.json / uji_method_freeze_v18_5.json. No repository source/test/design or historical artifact changed.

2. **continuation_source_inspection**: A read-only PowerShell inspection command included a stray '$ s=' token and emitted a shell error; subsequent reads and Git checks completed.

Handling: Removed the stray token from subsequent commands. No scientific command or repository file changed.

3. **failure_log_write**: Initial inline log command exceeded Windows command-line limit (CreateProcess error 206); logging process did not start.

Handling: Reduced operational log payload and successfully wrote the immutable failure .log; no scientific rerun or source modification for this logging error.

4. **registered_evaluation**: Original B32 V18.5 control failed MAE and NLL historical numerical reproduction tolerance.

Cause: Candidate support width depends on batch maximum; candidate standardization in model.py includes unmasked padded entries. Historical V18.5 uses B64; V19 control uses B32.

Handling: Preserved original failure log and all checkpoints/caches. User explicitly authorized continuing. Replaced only two named stop raises with audit recording in the evaluation process, leaving all frozen scoring/batching and on-disk source unchanged. Historical B64 diagnostic reproduces MAE/NLL; registered B32 control remains FAILED_AND_WAIVED, not PASS.

5. **git_read_commands**: Git emitted permission warnings for the global ignore file under C:/Users/LiuYan/.config/git/ignore.

Handling: Read-only branch/HEAD/clean-source checks completed. Global configuration was not modified.

6. During final read-only report verification, an inspection referenced nonexistent uji_v18_5_formal.py. rg reported a missing file; rg --files identified the actual uji_v18_5_development.py, and the correct source was read. This inspection error occurred after the JSON snapshot and changed no file or scientific result.

## Integrity, output hashes, and stop boundary

Immediately before creating these records, 68 protected/historical/generated scientific files were SHA-verified unchanged. Original evidence/checkpoints/data, V19 design/runbook/source, new checkpoints/caches, probe/registry/result, and failure log were retained. Exact branch/HEAD and clean tracked-source checks passed. The JSON records full protected-file SHA bindings and confirmation flags.

| Artifact | File SHA256 | Canonical SHA256 if applicable |
| --- | --- | --- |
| development_result_v19_r0.json | c61e35a9c30059d24c8cb0cfc263a3de7071d0f003b148d38d25d22f1d94b1fd | 732533b4b1b676bf584434fe6f4f396cfe9e67eba13b483c63d9dd7d3efa5ddf |
| FIRST_ACCURACY_RECOVERY_EXECUTION_V19_R0.json | a2adbd6db942fa930464682e4e36ecc553a8fb71e1a13749582581d0d7a4b897 | Not self-embedded |
| FIRST_ACCURACY_RECOVERY_FAILURE_V19_R0.log | c41868ad6d4ddefa4310c40174f8d057b9c4e7700c94e72614dcc29fece18858 | Not applicable |

This Markdown is created once; its SHA256 is computed externally after creation and reported in the final Codex response. All exact floating-point values and complete runtime provenance remain in the execution JSON; Markdown tables round display values to six decimals.

| Confirmation | Value |
| --- | --- |
| source_files_tests_design_runbook_unchanged | True |
| member_1_inherited_not_retrained | True |
| new_members_trained_once_sequentially | True |
| all_16_caches_sequential | True |
| wide_member_posterior | full_anchor_invariant_logits_no_candidate_reranker |
| candidate_reranker_applied_on_wide_path | False |
| no_v19_radio_projection | True |
| no_v19_temperature_calibration | True |
| no_v19_scale_calibration | True |
| no_geometry_posterior_correction | True |
| no_member_count_selection | True |
| no_uji_validationData_access | True |
| no_external_test_performance_access | True |
| no_v19_r1_implementation | True |
| no_post_outcome_scoring_method_grid_seed_change | True |
| user_authorized_stop_gate_exception_explicitly_recorded | True |
| no_git_checkout_commit_merge_push_upload | True |

No UJI validationData, no external test-performance access, no V19 radio/T/scale calibration, no geometry correction, no final E selection, no V19-R1, no source/test/design edits, and no post-outcome scoring/grid/seed changes. The sole post-outcome runtime exception was the explicitly authorized pair of stop-gate waivers. No Git checkout/commit/merge/push/upload was performed.

Development completion is qualified by the recorded gate waiver.

V19-R0 ACCURACY RECOVERY DEVELOPMENT COMPLETE.
STOPPED BEFORE RADIO/CALIBRATION, OFFICIAL VALIDATION, AND EXTERNAL TESTING.
READY FOR SCIENTIFIC REVIEW.
