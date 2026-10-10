# G10 execution notes — 2026-10-10

Completed one frozen CUDA run; final CLI check: **VERIFIED**.
Scientific status: **ONE_POOLED_CAL_FIT_THREE_RETROSPECTIVE_CONFIRMS**.
This is a retrospective mechanism pilot, not independent confirmation or promotion.

Started 2026-10-10T14:49:56.727510+08:00; notes completed 2026-10-10T15:04:45.671538+08:00.
Wall time 888.944s; command-stage total 136.210s; formal run 47.239s.
Python: D:\miniminiconda\envs\liuyan\python.exe / 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)].
PyTorch 2.10.0+cu128; CUDA runtime 12.8; NVIDIA GeForce RTX 2060 (6442123264 bytes).
NumPy 2.4.4; SciPy 1.17.1; Windows-10-10.0.19045-SP0.

## Commands and failures

Exact argv, original stdout/stderr, start times, exits, durations and per-command log hashes are retained in RUN_STDOUT.log / RUN_STDERR.log.
The only formal run was `python -m fhm_dapp_tim.fhm_dapp.uji_v17_g10_evidence_corrected_law run --device cuda`.
CPU/GPU synthetic fits exercised implementation only; no real CAL/CONFIRM optimization or scoring preceded the frozen formal run.

| Stage | Exit | Seconds |
|---|---:|---:|
| compile-initial | 0 | 0.099 |
| tests-initial | 1 | 8.502 |
| preflight-initial | 0 | 9.325 |
| regression-before-fix | 1 | 6.722 |
| compile-repaired | 0 | 0.093 |
| tests-repaired | 0 | 7.176 |
| preflight-repaired | 0 | 9.196 |
| identity-and-synthetic-cuda | 1 | 6.054 |
| identity-and-synthetic-cuda-corrected-audit-key | 0 | 14.366 |
| prepare | 0 | 9.182 |
| check-before-run | 0 | 9.142 |
| run-once | 0 | 47.239 |
| check-after-run | 0 | 9.114 |

- Initial suite: 36 passed, 1 failed, 3 setup errors. Final G10 + G9 + G8 suite: **42 passed in 5.67s**.
- Setup errors were WinError 3: the new pytest basetemp parent directory was absent. Created only the G10 scratch parent; retained original failure logs. No historical fixture or ACL was changed.
- A float32 test offset of +100 quantized prototype inputs before centering (maximum feature difference 1.3768672943115234e-05). Changed only that synthetic offset to +10, keeping the original tolerance and scientific feature formula.
- A targeted synthetic R3 regression failed before repair: reconstructing joint probabilities from rounded marginals altered the baseline. G10 now retains original R3 joint probabilities and derives correction marginals from that exact law during scoring. CPU and all three CUDA panel smoke tests verify bitwise zero-theta probabilities, MAE and continuous ES against original G8. No tolerance was relaxed.
- Added exact-probability/full-score regression and a synthetic run-order test proving CAL-only fit, immutable coefficients before any CONFIRM target extraction, and no Tampere fit.
- Original G10 source SHA256: 9e0afbe514425323e6c6a9d3280dc932bd242485b5e2939af275fe3ed55b02c1; original tests: 12c2eb65a13038e442bd0f2e4f4b47c51980c0d704336aac48d5a35e128fe325. Full pre-manifest repair diffs are in RUN_STDOUT.log.
- Auxiliary smoke verifier initially expected checkpoint key seed; the historical key is member_seed. Corrected only the in-memory audit helper, then all 15 identities and synthetic CUDA checks passed. Original traceback is retained.
- Auxiliary tool-output truncation and note-script quoting errors were corrected without a scientific mutation or rerun; their records are retained. Baseline recovery reproduced the identical protected count/hash before edits.
- All G10 source/test repairs preceded prepare. No frozen source, design, model, manifest, lock, coefficients, result or generated report was edited after prepare. No formal runtime error, OOM, nonfinite loss or CUDA determinism warning was emitted.

## Data roles, weights and ordering

Only UTS_training.csv, Tampere Training_rss.csv/Training_coordinates.csv, and five TUJI train-variable MAT containers were used.
TUJI physical MAT containers include results_test bytes and were opened/hashed. The original restricted loader requests only variable_names=(results_train,) and returned no results_test object; test labels were not decoded. Do not claim MAT-container byte isolation.
Preflight and original loaders parse TRAIN records and role keys for provenance/disjointness. The enforced fit boundary concerns selected targets: only CAL rows enter the correction loss, and CONFIRM target extraction/scoring occurs after immutable coefficient saving. It does not claim that all TRAIN-container coordinate bytes remained unopened before fitting.
Tampere provides no target labels to coefficient fitting or parameter selection. G6 CAL JSON is read only for original frozen panel/policy identity; G4/G5 outcome JSON is not used for fitting or choosing parameters. Post-run G8 vectors were used solely to verify unchanged comparators.

| Domain | FIT sites/rows | CAL sites/rows | CONFIRM sites/rows |
|---|---:|---:|---:|
| tuji | 643/3199 | 137/695 | 139/701 |
| uts | 945/5791 | 258/1625 | 263/1692 |
| tampere | 334/334 | N/A | 112/112 |

Within each domain, all physical role sets are pairwise disjoint and match their original manifests exactly. CAL: 395 sites/2320 rows; CONFIRM: 514 sites/2505 rows.
Rows are weighted by 1/(rows at site × sites in domain × 2): physical CAL sites are equally weighted within domain; each of TUJI/UTS has total loss weight 1/2.
Seed 20261010; Adam lr0.06; batch24; exactly 24 fixed full-objective gradient-accumulation epochs. No best-epoch selection. Only four raw coefficients are optimized; theta=1.5*tanh(raw).
Recorded proxy loss: 0.412770444294 → 0.405932027090; Dirac-mixture ES + 0.30 point MAE, scaled by 10m, plus 0.10 KL. This is not continuous Student-t ES.
Frozen coefficients, in feature order: 0.262957721949, 0.151908710599, 0.061252161860, 0.169213622808.

Physical-key SHA256 uses compact JSON of sorted (building,floor,x,y) keys:
- tuji/calibration: `e1ad91fd38a086e445426f502bf24d5a3224823e48f46e6f99ea486c31c32e22`
- tuji/confirmation: `1c46af2f1336fcc8e664f7c214ef876418a1c288df942db9d6f517e88d22542c`
- tuji/fit: `0f166645db1c6bd58d101fc661a0ec62a589b6d89229989224c569092bb6ab95`
- uts/calibration: `60593e68e8b87ac284408acc5b4d0f5f6cb1c138677350ccf667aa29788da763`
- uts/confirmation: `dee3633a2b4a4dfedb5054c420fef766e917c3a437303b08c7a7917373bdba93`
- uts/fit: `470638ca0a79cf8154498dd900014de96af546d311fc32289ad3f777fde43bef`
- tampere/confirmation: `905866e37596155ee6131b271686a69ce3e0692f9d67d61b978d2c109e411b67`
- tampere/fit: `d917c5efe32c714267dd2b6785c79965d7948f3f484d8a02f8ab19d1c9798468`

## Original model identities

All original 15 checkpoint SHA256 values match frozen G4/G5/G6 identities. Member seeds 17301..17305; epochs 65/69/56/67/62; shared FIT anchor/prototype dictionary and origin within each panel.
Synthetic CUDA smoke verified all original network state tensors bitwise unchanged and parameter grads None. Formal extraction uses no_grad and releases each panel before coefficient optimization; no original network optimizer is constructed. Checkpoint bytes and historical files remained unchanged.

| Domain | Member | Seed | Epoch | SHA256 |
|---|---:|---:|---:|---|
| tuji | 1 | 17301 | 65 | `20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a` |
| tuji | 2 | 17302 | 69 | `febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8` |
| tuji | 3 | 17303 | 56 | `73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a` |
| tuji | 4 | 17304 | 67 | `58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b` |
| tuji | 5 | 17305 | 62 | `124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c` |
| uts | 1 | 17301 | 65 | `299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5` |
| uts | 2 | 17302 | 69 | `3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48` |
| uts | 3 | 17303 | 56 | `4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548` |
| uts | 4 | 17304 | 67 | `695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224` |
| uts | 5 | 17305 | 62 | `0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c` |
| tampere | 1 | 17301 | 65 | `8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b` |
| tampere | 2 | 17302 | 69 | `37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294` |
| tampere | 3 | 17303 | 56 | `c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d` |
| tampere | 4 | 17304 | 67 | `0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1` |
| tampere | 5 | 17305 | 62 | `ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5` |

## Outcomes and limits

All three method vectors preserve original sorted site alignment; macro means, P90/P95, paired gains/harm counts and extremes were independently recomputed without inference.
All Base and Anchor T=.6 per-site MAE/ES vectors are exactly equal to original G8 confirmation vectors. E5/L192/K192/alpha1/df4 are unchanged. Continuous ES uses original seed19602, paired two streams of 64 draws, fixed centers/scales.
P90/P95 are percentiles of physical-site mean Euclidean errors, not percentiles of RSS-row errors.

| CONFIRM | ΔMAE m | ΔES m | ΔP90 m | ΔP95 m | MAE improved/harmed |
|---|---:|---:|---:|---:|---:|
| tuji | -0.048581587 | -0.018242009 | -0.120364081 | -0.104520727 | 80/59 |
| uts | -0.144534022 | -0.065120477 | -0.296341730 | +0.023806723 | 155/108 |
| tampere | +0.003116447 | -0.015304187 | -0.190264614 | +0.082054808 | 46/66 |

Negative differences mean improvement over Anchor T=.6. **Three-domain dominance was not established:** TUJI/UTS macro MAE and ES improve, but Tampere macro MAE worsens, and UTS/Tampere P95 worsens. Small mean gains must not hide adverse sites or be called generalization success.
Worst MAE-harm locations (building,floor,x,y), with paired ES changes:
- tuji: [0, 0, 10.129, 15.501]; ΔMAE +0.649502227m; ΔES +0.396768848m.
- uts: [0, 5, 84.9937, 16.7134]; ΔMAE +2.502181033m; ΔES +0.810025616m.
- tampere: [0, 1, 262.94, 107.04]; ΔMAE +1.288968703m; ΔES +0.035865071m.
All site vectors, including those harmed, remain in evaluation_v17_g10.json. No confirmation-based revision, retry, temperature search, new model, p-value, confidence promotion or waiver was performed.
This single fit on already-viewed historical partitions is descriptive. Spatial dependence, finite 64+64 Monte Carlo ES, four correlated features and frozen E5 budgets limit interpretation. Coefficient signs do not establish unique causal contributions.
Prospective evidence requires previously unseen locations/external data and equal-budget E16/E25 comparisons; no Q1 or independent validation claim is supported here.

## Integrity and files

1360 pre-existing scientific source/test/evidence/runbook files are byte-identical before/after: inventory SHA256 `fbe40596012826ca4a92bf01228bc1ab42711e80f374fb38ee89f740730ee6e8`. Fifteen checkpoint files and allowed inputs match their historical hashes. No Git command was executed.
Core artifact file SHA256:
- manifest_v17_g10.json: `e118b2733a06718b5904e830eb078a81c2a361bd431c9ad5cd038eac126f2ff3`
- train_v17_g10.lock: `ce1df19d8d020d302132740a1b74e41fd38305db450db2b5669c3c5aadd4dce6`
- fitted_coefficients_v17_g10.json: `894dcbf5452d28bc788aaeac8c5a545c09a60c22ccb40bc1920673b689822e11`
- evaluation_v17_g10.json: `8898ebf3f4d4014e43af9e7547bf5ad9f5d86f051e30e4bbb4ac2c097a927604`
- RUN_REPORT_V17_G10.md: `96dd3200834cf3f1a2a67f0c8791a92b6be7e9c05ef59ea235d506fee407a2e7`
- DESIGN_V17_G10.md: `4a718dfb1d04fdbdc7bcd0ec556f6ff86cd14253d80e40f69b2ee3257b98c3eb`
- uji_v17_g10_evidence_corrected_law.py: `18aaf2f435e6a0dfee5ee290bb0d6ea3b9567876683bd27e4ec56b1597af5730`
- test_uji_v17_g10_evidence_corrected_law.py: `461e80b3d1235f201686a3f100bb7993b7ebf4a32ecf9918d423f1d075b2614a`
- V17_G10_CODEX_RUN.txt: `6f8942e4c636d33eb56e443cbc52f70386e6ac2fd738892bd6f2d892af1d96de`
- RUN_STDOUT.log: `e25348cb90e7c2aef884132cc0b26ba22c1f9c6758e907c74640bc32a47e37ab`
- RUN_STDERR.log: `55e792b05c7887439fc4dce8cfde0990605020121286dcb83e2170464083f7c3`
