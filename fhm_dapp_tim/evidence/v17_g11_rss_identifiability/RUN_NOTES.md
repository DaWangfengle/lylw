# G11 execution notes

Completed one frozen retrospective TRAIN-internal diagnostic. Final command check: VERIFIED.
Not blind, independent confirmation, statistical promotion, a Bayes lower bound, or a Q1-ready result.

## Environment and execution

- Python: D:\miniminiconda\envs\liuyan\python.exe; 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)].
- PyTorch 2.10.0+cu128, CUDA runtime 12.8, GPU NVIDIA GeForce RTX 2060 (6442123264 bytes); NumPy 2.4.4, SciPy 1.17.1.
- Windows-10-10.0.19045-SP0; audited start 2026-10-10T15:58:22.781229+08:00, notes finalized 2026-10-10T16:11:19.949370+08:00, elapsed 777.168s.
- Exactly one formal run (31.258s); no new network training, parameter fitting, Monte Carlo ES scoring, or outcome-driven method selection.
- Initial specified G11/G10/G9/G8 suite: 57 passed. Targeted pre-fix mapping regression: 1 failed. Final full suite: 58 passed.
- pytest used fresh local basetemp directories under models/v17_g11_rss_identifiability; the parent was created before testing. No historical fixture/cache permissions were changed.

| Command stage | Exit | Seconds |
|---|---:|---:|
| compile-initial | 0 | 0.096 |
| tests-initial | 0 | 7.391 |
| preflight-initial | 0 | 9.331 |
| identity-fit-map-ap-formula-cuda-smoke | 1 | 4.092 |
| identity-fit-map-ap-formula-cuda-smoke-corrected-mat-hook | 1 | 14.436 |
| map-distance-independent-fit-proof | 0 | 9.489 |
| regression-map-before-source-fix | 1 | 5.207 |
| compile-repaired | 0 | 0.095 |
| tests-repaired | 0 | 7.391 |
| preflight-repaired | 0 | 9.008 |
| identity-fit-map-ap-formula-cuda-smoke-repaired | 0 | 14.433 |
| prepare | 0 | 9.087 |
| check-before-run | 0 | 9.031 |
| run-once | 0 | 31.258 |
| check-after-run | 0 | 9.042 |

Exact arguments, timestamps, stdout/stderr byte hashes, raw output/tracebacks, original source text, and the unified repair diff are retained in RUN_STDOUT.log / RUN_STDERR.log. No failed record was deleted.

## Pre-manifest issue and minimal repair

Real new G11 bug: map_fit_rows_to_model_anchors used float32 torch.cdist's default GEMM norm expansion. It invented a 0.03125m self-distance for one Tampere FIT anchor, exceeding the unchanged 0.03m gate even though direct float64 distance was 0. CPU independent verification found 334/334 bijective matches, no ID disagreements and no actual out-of-tolerance Tampere coordinate.
Changed only new G11 mapping distance to direct float64 cdist, compute_mode=donot_use_mm_for_euclid_dist; input/checkpoint coordinates, building/floor filters and the 0.03m scientific gate stayed unchanged. The added synthetic 64-anchor permuted-memory regression failed before the repair, passes after it, and still rejects a genuine 0.031m displacement. No source/test/design changes after prepare.
- Original G11 source SHA256: 739206e7629dfcfc1f2b5d116f15ad0b21ee6e9ab75459168119306989c443c5
- Frozen repaired G11 source SHA256: 8faae18380463b916cc3d82076d51729d1635d94b636eed09a719bddf2bda59d
- Original G11 test SHA256: f614e28b754372aceb11514b43aac4a4803501076636e80c57d29bed726490f1
- Frozen repaired test SHA256: 31fedcdd18591e6e6f29e9a9e1eb021f41d152bcf060180a59cae5fcb0ada3e9
Auxiliary engineering issues were also retained: the first supplemental audit referenced g6.scipy_io.loadmat, while the module actually imports g6.loadmat directly; only the audit hook was corrected. A later inline baseline transport exceeded Windows CreateProcess length (error 206) before execution; recovery read the already-preserved baseline from the stdout log. Neither issue changed scientific data, produced formal results or required another formal run.

## Original data, roles and model identities

Only UTS_training.csv, Tampere Training_rss.csv / Training_coordinates.csv and the five original TUJI MAT containers were used. The original G6 loader requests only results_train; the restricted-load probe verified five such calls and no results_test object returned. MAT containers include test bytes and were physically opened/hashed; test variables were not decoded or used. TRAIN coordinates are parsed for original role/mapping integrity; query truth is used only for error and post-prediction diagnosis, never predictor inputs or anchor filtering.
The supplemental FIT-only audit observed exactly these eight permitted containers, blocked none because no forbidden path was requested, verified all pairwise physical role intersections empty, and independently rebuilt original R0 anchor memory from FIT scans only. Beta(.5,.5) AP frequencies and prototype RSS matched the historical formulas/memories exactly. No nonexistent anchor_detection_prob checkpoint cache was assumed.

| Domain | FIT sites/rows | CAL sites/rows | CONF sites/rows | Seen APs | Maximum FIT mapping error m |
|---|---:|---:|---:|---:|---:|
| tuji | 643/3199 | 137/695 | 139/701 | 115 | 1.821381129e-06 |
| uts | 945/5791 | 258/1625 | 263/1692 | 557 | 5.435222988e-05 |
| tampere | 334/334 | N/A | 112/112 | 483 | 7.143981872e-06 |

All three FIT-to-checkpoint mappings are complete bijections. The actual original order happened to agree, but it was independently matched, with the regression explicitly testing a permutation. FIT-only CUDA smoke passed in all domains; full network state stayed bitwise unchanged and all parameter grads were None. No runtime warning, OOM, nonfinite scoring, or CUDA determinism warning appeared in the formal command stderr.

| Domain/member | Seed | Epoch | Original checkpoint SHA256 |
|---|---:|---:|---|
| tuji/01 | 17301 | 65 | 20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a |
| tuji/02 | 17302 | 69 | febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8 |
| tuji/03 | 17303 | 56 | 73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a |
| tuji/04 | 17304 | 67 | 58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b |
| tuji/05 | 17305 | 62 | 124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c |
| uts/01 | 17301 | 65 | 299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5 |
| uts/02 | 17302 | 69 | 3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48 |
| uts/03 | 17303 | 56 | 4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548 |
| uts/04 | 17304 | 67 | 695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224 |
| uts/05 | 17305 | 62 | 0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c |
| tampere/01 | 17301 | 65 | 8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b |
| tampere/02 | 17302 | 69 | 37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294 |
| tampere/03 | 17303 | 56 | c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d |
| tampere/04 | 17304 | 67 | 0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1 |
| tampere/05 | 17305 | 62 | ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5 |

Post-run: all 15 checkpoint byte hashes still match original frozen G4/G5/G6 provenance.
All 341 snapshotted prior source/test/evidence/runbook files (excluding runtime caches and run_audit directories) remained byte-identical; inventory SHA256 038fe2410f213b767ec3f717a7477a05f40c3dbbf566fbaa69930c3e991d4ffd. Historical G1 STOP and earlier decisions were not changed.

## Independent result replication

Five fixed historical roles: 909 physical sites / 4825 RSS scans. Every vector length, finite/null status, site macro, P90/P95/max and quadrant count was independently recomputed from the frozen JSON.
Only after G11 result/report/lock were frozen, G8 was read for replication: every V17 Anchor T=.6 per-site MAE in all five roles agrees exactly (maximum absolute difference 0m; prescribed tolerance 1e-4m). No old result was used to generate predictions or choose methods.

| Domain/role | Sites/rows | Physical-key SHA256 | G8 max absolute MAE difference m |
|---|---:|---|---:|
| tuji/calibration | 137/695 | e1ad91fd38a086e445426f502bf24d5a3224823e48f46e6f99ea486c31c32e22 | 0.0 |
| tuji/confirmation | 139/701 | 1c46af2f1336fcc8e664f7c214ef876418a1c288df942db9d6f517e88d22542c | 0.0 |
| uts/calibration | 258/1625 | 60593e68e8b87ac284408acc5b4d0f5f6cb1c138677350ccf667aa29788da763 | 0.0 |
| uts/confirmation | 263/1692 | dee3633a2b4a4dfedb5054c420fef766e917c3a437303b08c7a7917373bdba93 | 0.0 |
| tampere/confirmation | 112/112 | 905866e37596155ee6131b271686a69ce3e0692f9d67d61b978d2c109e411b67 | 0.0 |

Values below are coordinate-macro MAE / P95 of physical-site mean error, in metres. P90, maximum and all complete per-site vectors are in the frozen JSON.

| Historical role | V17 Anchor T=.6 | Raw L2 1NN | Raw L2 5NN | AP visibility 1NN |
|---|---:|---:|---:|---:|
| tuji/calibration | 2.390619 / 3.675440 | 4.298767 / 6.882047 | 3.421057 / 5.688976 | 4.922795 / 7.695704 |
| tuji/confirmation | 2.657774 / 4.466831 | 4.485711 / 6.923201 | 3.751357 / 6.064429 | 4.872086 / 8.438101 |
| uts/calibration | 5.380266 / 12.260829 | 6.219443 / 15.992012 | 6.211548 / 13.427689 | 7.845195 / 19.127278 |
| uts/confirmation | 5.107529 / 12.221708 | 6.115539 / 15.640616 | 6.311442 / 15.858266 | 6.767924 / 15.897587 |
| tampere/confirmation | 5.987721 / 10.468628 | 6.903894 / 11.178262 | 5.764193 / 11.136824 | 8.098576 / 16.362488 |

| Historical role | Both NEAR | Raw NEAR/V17 non-NEAR | Raw non-NEAR/V17 NEAR | Both non-NEAR | AP rescue among both non-NEAR |
|---|---:|---:|---:|---:|---:|
| tuji/calibration | 137 | 0 | 0 | 0 | 0 |
| tuji/confirmation | 139 | 0 | 0 | 0 | 0 |
| uts/calibration | 195 | 16 | 15 | 32 | 5 |
| uts/confirmation | 203 | 16 | 10 | 34 | 8 |
| tampere/confirmation | 98 | 5 | 4 | 5 | 0 |

Quadrants use the predeclared site criterion: at least half of repeated scans have same-building/floor Top-1 within 10m. A quadrant label 'far' means majority non-NEAR; it does not guarantee every selected anchor is CLEARLY_WRONG (>=20m or wrong building/floor). The strict CLEARLY_WRONG rule applies to the separate near-vs-far margin. Top-1 neighborhood hits are not identical to the continuous V17 mean's location error.
On UTS CONF, 16 sites have raw majority-NEAR while V17 does not; 10 show the reverse; 8 of the 34 jointly non-NEAR sites are majority-NEAR under AP detection. TUJI has no jointly non-NEAR sites under this site criterion; Tampere has five, with zero AP rescues. Exact AP-rescue coordinate lists are frozen in the JSON.
Among UTS CONF jointly non-NEAR sites, 18/34 have positive site-mean raw near/far margin and 16/34 have nonpositive margin; mean 0.00686221648 in the original RSS distance units. Tampere is 1/5 positive and 4/5 nonpositive, mean -0.0122493535. Positive margin only ranks <=10m neighbors above CLEARLY_WRONG >=20m/wrong-floor impostors; an intermediate 10–20m anchor can still win, and site means can hide row variability.
V17 has lower mean and P95 error than all raw methods on TUJI and UTS. Tampere Raw 5NN has lower mean (5.764193 versus 5.987721m) and maximum (33.553520 versus 60.981600m), but higher P95 (11.136824 versus 10.468628m); this is mixed performance, not uniform superiority. AP detection is a different transformation of the same RSS scans, with clipping/missing ambiguity, not an independent sensor or causal explanation.
Evidence of recoverable missed neighborhoods is strongest in UTS and is domain-dependent. These retrospective observations neither prove information-theoretic ambiguity nor authorize a new method or promotion; any contrastive-training proposal needs fresh physically isolated development and independent evaluation, including proper continuous scoring and equal-budget full-panel baselines.

## Frozen evidence hashes

- Manifest canonical self SHA256: 568b7e28d43f50fc765ff28ac133ee659c9a72c3f0ce46eaae25d1d5e71cca21
- Result canonical self SHA256: f414517883ef959f3160deffd7f67eddb12a41765795328a6ff03170d50e578d
- fhm_dapp_tim/evidence/v17_g11_rss_identifiability/DESIGN_V17_G11.md: 11d3002b14ec07783d6a12ae29d715946d2e5935b32171c1df5ddc5cc71e4500
- fhm_dapp_tim/evidence/v17_g11_rss_identifiability/manifest_v17_g11.json: a65e5dfe608b46ad101893627f700378c5779a18677f1e273a00cfc2afe0176d
- fhm_dapp_tim/evidence/v17_g11_rss_identifiability/diagnostic_v17_g11.lock: e1729fb059f636a56dd52e96fe2d8bc436a48ea90a6ae3380e23a47538bfd4c5
- fhm_dapp_tim/evidence/v17_g11_rss_identifiability/rss_identifiability_v17_g11.json: 3fdd9908541b1c2c82c4d5ea548c3fc4ffd172ee96349b5e41ee6656684419fb
- fhm_dapp_tim/evidence/v17_g11_rss_identifiability/RUN_REPORT_V17_G11.md: 031fb416c7eee49225637e055ca8b61a4a74eac1d838552428c05d5138129e72
- fhm_dapp_tim/fhm_dapp/uji_v17_g11_rss_identifiability.py: 8faae18380463b916cc3d82076d51729d1635d94b636eed09a719bddf2bda59d
- tests/test_uji_v17_g11_rss_identifiability.py: 31fedcdd18591e6e6f29e9a9e1eb021f41d152bcf060180a59cae5fcb0ada3e9
- V17_G11_CODEX_RUN.txt: 48fa18148cfd5eff9813eff71ac6aa6db00842dea96d02e0b3aac425f9474ccb
- RUN_STDOUT.log: 9a2b0b837ba8adb9b320ff89cc7e83d4fe71ca9be79d48ab32425e2e7cadb14b
- RUN_STDERR.log: 9c1d5c570ccb62b49f246a84c9f23fe62782b637ed5bdf9c6140b32cc5476b9f

No Git commands were executed in this experiment. Original weights stay local. No supplied official test, external validation or historical UJI outer data was accessed.
