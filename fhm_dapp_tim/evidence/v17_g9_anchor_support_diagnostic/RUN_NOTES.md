# V17-G9 execution notes

Started: 2026-10-10T13:15:56+08:00; interpreter: D:/miniminiconda/envs/liuyan/python.exe.

Retrospective TRAIN-internal diagnostic only; no training, tuning, new ES draws or statistical promotion. Original models and G1-G8 evidence are read-only.

## Command execution

| Stage | Start (+08:00) | Seconds | Exit | stdout SHA256 | stderr SHA256 |
|---|---|---:|---:|---|---|
| compile-initial | 2026-10-10T13:16:48+08:00 | 0.096 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| tests-initial | 2026-10-10T13:16:48+08:00 | 7.136 | 0 | 87f7d62a3baee641b434f914358083eec2d7af6fa2e4ec7b5313d9cdf1f99754 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| preflight-initial | 2026-10-10T13:17:31+08:00 | 9.411 | 0 | 846ba87f972d5831b7e1a8c98f1baaefb3d0490a466f65c1e99fb0e41a3dd63b | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| regression-before-fix | 2026-10-10T13:18:39+08:00 | 4.952 | 1 | e358d62e9448f328ccfe3016c8ef3258de5f115f5e6ae099710c880af99e8a9f | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| compile-repaired | 2026-10-10T13:19:04+08:00 | 0.092 | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| tests-repaired | 2026-10-10T13:19:04+08:00 | 5.158 | 0 | da1fd62cf451fdb1988d17008b94551e54dc388f91af9d5c427b5806355c00e0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| preflight-repaired | 2026-10-10T13:19:09+08:00 | 9.180 | 0 | 811a52faef979c6d4ff838cfa0fc3a4b3a00a880d5ac8d76cec327cfb9a7b93f | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| identity-and-synthetic-cuda | 2026-10-10T13:20:18+08:00 | 14.183 | 0 | a8d206c605025c4cdfb4f12dd49a25e2dc1f8de247751fb2e2b48fd14d855154 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| prepare | 2026-10-10T13:20:50+08:00 | 9.256 | 0 | f545f46905cda0cc109a7424e5f83209ef3ad165971b09a4203826c00378ab85 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| check-before-run | 2026-10-10T13:20:59+08:00 | 9.254 | 0 | 993389926eb6bf2c5fbfc9d837e94b64141c9b12194568538c46f3bca0ae1f3a | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| run-once | 2026-10-10T13:21:37+08:00 | 31.963 | 0 | f014a49e2ac3b5abd02324e159fb56f40ed44d63a6be89f40454b13088fd72ad | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| check-after-run | 2026-10-10T13:22:09+08:00 | 9.315 | 0 | 7520132f2f3447071bd5899211a5e7a3088ef94ae4bbbba8e0b4143c3c8dfffd | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

## Final execution review

Completed: 2026-10-10T13:27:25+08:00; wall time 689.834s; captured command stages 109.996s.
Initial compile/tests/preflight passed (26 tests); targeted synthetic regression then failed as expected, proving the defect. Repaired compile/tests/preflight passed (27 tests; 3.96s pytest time). Prepare/check/run/check all exit 0; final status VERIFIED. Exactly one formal CUDA run (31.963s).
Full stdout/stderr and arguments, including the original regression failure, are preserved in RUN_STDOUT.log / RUN_STDERR.log. Tests use separate workspace basetemp directories to preserve earlier fixtures; no cleanup or retraining was performed.

## Environment and pre-freeze repairs

Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]; executable D:\miniminiconda\envs\liuyan\python.exe; Torch 2.10.0+cu128; CUDA 12.8; NVIDIA GeForce RTX 2060 (6442123264 bytes); NumPy 2.4.4; SciPy 1.17.1; Windows-10-10.0.19045-SP0.
Only new G9 source/tests were edited BEFORE prepare. Original G9 cast centered evaluation targets to float32 and reconstructed the anchor action through a different reduction path. G8 scores against float64 centered targets and uses the joint-law marginals. The synthetic regression observed 60.6581031236965m instead of the reference 60.65810392196865m at one site. The fix preserves float64 targets and reuses the exact existing G8 Anchor-only action. No radius, model, retrieval/consensus operator, temperature, group, or mathematical tolerance was changed.
The new regression checks both actions against the original R3 joint and G8 law, includes repeated-site aggregation, and forbids an ES scoring call. After repair it passes exactly; the formal five roles also match all G8 coordinate MAE vectors exactly.

Source edit:
```diff
-                         device=device,dtype=torch.float32)
+                         device=device,dtype=torch.float64)
-        sharp=torch.softmax(c.clamp_min(1e-30).log()/T,dim=1)
-        means=(wm.transpose(0,1)[:,:,None,None]*
-               sharp[:,None,:,None]*
-               loc.reshape(b,E,K,2)).sum(dim=(1,2))
+        # Keep G8's marginalization and float32 reduction order exactly.
+        anchor_joint=g8.hierarchical_laws(p,E,K)["anchor"]
+        means=(anchor_joint[:,:,None]*loc).sum(1)
```
Original source SHA256: 5360a9ab49e55546447df7fba4f4eef459bfc04fef618c2a356e433a66bb00ad; original tests: 07afa373cec420c5756c8076986258eaa6ca7a19fcddfca8d592b46dcaea5375. Final source/tests hashes are listed below.

## Frozen identity and data boundary

All 15 original checkpoints passed SHA256, member index, seeds 17301..17305 and epochs 65/69/56/67/62. No checkpoint substitution/retraining. Shared anchor dictionaries, origins, prototype RSS and seen-AP masks match within each panel. Synthetic CUDA checks on two artificial queries per domain passed, with model state bitwise unchanged and every parameter grad None; peak allocated bytes: TUJI 68567040, UTS 89662976, Tampere 65300480. No real CAL/CONFIRM errors were computed for smoke checks.
The synthetic/identity helper enforced an allowed-data-file audit hook and observed only the three allowed CSVs and five allowed MAT containers. It also checked all five loadmat calls used variable_names=("results_train",) and returned no results_test. The same unchanged original loaders are used for formal scoring. MAT physical containers were opened and hashed; the results_test variable was not decoded or used. No official/provided test dataset was loaded.

| Domain | FIT sites/rows | Historical CAL sites/rows | Historical CONF sites/rows |
|---|---:|---:|---:|
| TUJI | 643/3199 | 137/695 | 139/701 |
| UTS | 945/5791 | 258/1625 | 263/1692 |
| Tampere | 334/334 | n/a | 112/112 |

All within-domain role intersections are empty. Frozen role membership and row counts match the original manifests.

| Domain/member | SHA256 |
|---|---|
| tuji/01 | 20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a |
| tuji/02 | febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8 |
| tuji/03 | 73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a |
| tuji/04 | 58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b |
| tuji/05 | 124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c |
| uts/01 | 299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5 |
| uts/02 | 3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48 |
| uts/03 | 4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548 |
| uts/04 | 695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224 |
| uts/05 | 0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c |
| tampere/01 | 8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b |
| tampere/02 | 37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294 |
| tampere/03 | c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d |
| tampere/04 | 0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1 |
| tampere/05 | ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5 |

## Both fixed radii and deterministic MAE

| Historical role / radius | FIT hit | Top-L union hit | Consensus hit | Base mass | Anchor-T mass | Residual centre hit |
|---|---:|---:|---:|---:|---:|---:|
| tuji/calibration/5m | 1.000000000 | 1.000000000 | 1.000000000 | 0.564102507 | 0.661796130 | 1.000000000 |
| tuji/calibration/10m | 1.000000000 | 1.000000000 | 1.000000000 | 0.921274431 | 0.955000828 | 1.000000000 |
| tuji/confirmation/5m | 1.000000000 | 1.000000000 | 0.998561151 | 0.545569128 | 0.632882324 | 0.998561151 |
| tuji/confirmation/10m | 1.000000000 | 1.000000000 | 1.000000000 | 0.928659935 | 0.954583754 | 1.000000000 |
| uts/calibration/5m | 0.937984496 | 0.932699084 | 0.927648579 | 0.189138944 | 0.348178111 | 0.927648579 |
| uts/calibration/10m | 0.996124031 | 0.996124031 | 0.996124031 | 0.373738620 | 0.581717739 | 0.996124031 |
| uts/confirmation/5m | 0.973384030 | 0.967046895 | 0.965779468 | 0.218417297 | 0.393139342 | 0.965779468 |
| uts/confirmation/10m | 1.000000000 | 1.000000000 | 1.000000000 | 0.398367302 | 0.610271553 | 1.000000000 |
| tampere/confirmation/5m | 0.982142857 | 0.982142857 | 0.982142857 | 0.304620168 | 0.485544183 | 0.982142857 |
| tampere/confirmation/10m | 1.000000000 | 1.000000000 | 1.000000000 | 0.486941878 | 0.707785988 | 1.000000000 |

| Historical role | Baseline macro MAE m | Anchor-only macro MAE m | Difference m |
|---|---:|---:|---:|
| tuji/calibration | 2.797311675 | 2.390619305 | -0.406692370 |
| tuji/confirmation | 2.969113295 | 2.657773678 | -0.311339617 |
| uts/calibration | 7.155117679 | 5.380265830 | -1.774851850 |
| uts/confirmation | 6.819208114 | 5.107529197 | -1.711678917 |
| tampere/confirmation | 7.202666301 | 5.987721130 | -1.214945171 |

Hit metrics are means over RSS rows within each physical location, then coordinate-macro means. Missing distances remain null when no finite row exists; other distance summaries average finite rows only. This diagnostic saved 27 aligned site vectors for each role, covering 909 sites and 4825 scans. Both radii are retained. Every original R3 consensus marginal and component gather assertion passed; no assertion or tolerance was removed. No selected consensus ID outside the Top-L union was observed in this panel; such IDs were permitted and counted.
Descriptively, near-neighbour candidate survival is high at 5m and almost complete at 10m, while correct-neighbourhood mass is much lower in UTS/Tampere than in TUJI and changes after sharpening. This supports examining probability allocation, without proving that the posterior is overdispersed or that weighting is the unique failure cause. There are still geometric/retrieval/consensus misses and heterogeneous locations. No observed FIT-unsupported 5m site gained residual-centre support, but this is not a general impossibility result for a continuous model.

## Integrity, anomalies and interpretation limits

All 1469 pre-existing scientific source/test/evidence files remained byte-identical. Before/after SHA256 of the sorted relative-path-to-SHA map: 88af8e328764f302c3c634ab7969f0af762e1f72630fe76ec2d53d351f026981. G1-G8 history and decisions remain unchanged.
A read-only post-run review reproduced all summary means/conditional recalls, sorted unique keys, vector lengths and radius hierarchy. G8 JSON was first parsed for this comparison AFTER G9 completion; baseline and Anchor-only per-site MAE arrays match exactly for all five roles. No new inference, ES Monte Carlo draw or bootstrap was needed.
Scientific command anomaly: only the intentionally failing synthetic regression before the repair. No formal runtime error, CUDA determinism warning, OOM or stderr warning was emitted.
Auxiliary audit issues had no scientific side effects: an oversized initial inventory output was truncated, so its same unchanged inventory was recovered as a compact digest before any source edit; an attempted in-memory store of a JavaScript function and Markdown-fence quoting were rejected before command execution. The post-run read-only comparison helper initially used the G9 by_domain layout for G8 (KeyError), then miscounted columns as 28 instead of the actual documented 27 (AssertionError); its assumptions were corrected, and the complete review passed without changing frozen files or rerunning inference. The two original review tracebacks are retained in RUN_STDERR.log.
All analyzed labels were previously exposed TRAIN-internal roles. These are retrospective descriptive diagnostics, with spatial dependence and repeated-device/site scans; no new p-values, independent confirmation, model promotion or causal claim. Same building/floor is required for support, although the historical location MAE is 2D. Residual-adjusted centres and anchor geometry remain separate, and correct-neighbourhood truth/mass/distances must never be deployed as features or gates. The five members share FIT data and evaluation sites, and are not five independent confirmations. G9 computed no Energy Score; G8 ES was not reused as a new G9 metric. Future mechanisms require new unviewed partitions and comparison to the fixed Anchor-only T=.6 baseline.
No Git command was executed.

## Artifact/source SHA256

- fhm_dapp_tim/evidence/v17_g9_anchor_support_diagnostic/manifest_v17_g9.json: 853b54874c9581288d2c8a97b94c3a66ee8ee3494f227a8df0a30901c96785aa
- fhm_dapp_tim/evidence/v17_g9_anchor_support_diagnostic/diagnosis_v17_g9.lock: 3ce05a5dd660ad58012bcec59abb3b93bbf50c0a2e1f8c6925af138be2c930d6
- fhm_dapp_tim/evidence/v17_g9_anchor_support_diagnostic/diagnosis_v17_g9.json: 4920e07aa18a4353809f88a90695f50c28723ca7fc01400ada0f5894d3ae09f5
- fhm_dapp_tim/evidence/v17_g9_anchor_support_diagnostic/RUN_REPORT_V17_G9.md: 9b65983cbb23ed5946265c39eb8cd54b0a92da41817e3c0f85edb5319b7ca7d4
- fhm_dapp_tim/evidence/v17_g9_anchor_support_diagnostic/DESIGN_V17_G9.md: 8e30bedc08499bafe9cc47542f8a7180999ee1338706e100b6871bde322d3366
- fhm_dapp_tim/fhm_dapp/uji_v17_g9_anchor_support_diagnostic.py: fd4617c9e3aa83a11af63cc5c29ca8ad5458f1d390881466899f8e1a4333dd65
- tests/test_uji_v17_g9_anchor_support_diagnostic.py: d3b65b2d1992f24e2a4c180f5436c7317ccc0e0303c721dc4dd6a6fae1e7093e
- V17_G9_CODEX_RUN.txt: 44d4a2ed89d67d6a30b346b1e6f4bcbc271193f35506c99384f41024300b75d5

Manifest canonical self hash: d6e291ad705d316b7b0df0035e7579e6beb2554d2a976b348d38acf116f7e264; result canonical self hash: 5d5dfd911e56ff13f3d6335ff05275c0b2caad182ddfc563b514ae210c4984bd.
