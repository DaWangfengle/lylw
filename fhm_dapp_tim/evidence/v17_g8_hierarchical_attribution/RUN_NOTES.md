# V17-G8 execution notes

Started: 2026-10-10T12:25:17+08:00.
Python: D:\miniminiconda\envs\liuyan\python.exe; 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)].
Torch 2.10.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 2060 (6442123264 bytes).
NumPy 2.4.4, SciPy 1.17.1; Windows-10-10.0.19045-SP0.

Before execution: 893 historical evidence/source/test files snapshotted by SHA256; digest b71ffdca1bd158643338f273386cb9ce0e826cb060148b899f217b27072b3458.
No G8 manifest, lock, result or report existed. Historical outcome files were hashed only; no scores were used to choose parameters.
TUJI MAT containers may be opened/hashed; only results_train may be decoded. UTS/Tampere provided tests and UJI official/outer are prohibited.
No Git commands, training, new T/s search or candidate promotion.

| Stage | Exit | Seconds | Started (local) |
|---|---:|---:|---|
| compile-initial | 0 | 0.101 | 2026-10-10T12:26:39+08:00 |
| tests-initial | 1 | 6.939 | 2026-10-10T12:26:40+08:00 |

## Pre-manifest implementation repairs

- Initial regression: 19 passed, 2 failed, 3 fixture errors (exit 1). Full failures remain in RUN_STDOUT.log.
- Normalization reference uses input probability dtype; the float32 scoring path is unchanged. Float32/float64 parameterization and the original float64 commutation test cover the repair.
- Report test no longer demands an absent literal NOT; it checks the actual no-training/no-blind-confirmation, already-viewed, and no-p-values/no-promotion statements.
- Created the missing workspace pytest temporary parent. No fixtures were deleted and no historical tests edited.
- Audit-helper issues, before scientific stages: JavaScript template quoting; oversized display/JSON parsing; then Windows command-line limit (CreateProcess error 206) when embedding whole sources for diff generation. Corrected helper formatting and shortened the audit append; no scientific command was rerun for these issues. The original 893-file historical digest was recovered and verified.
- No scientific changes: temperatures, scales, df, random seeds, sampling, roles, models, and design remain fixed.

```diff
--- before/fhm_dapp_tim/fhm_dapp/uji_v17_g8_hierarchical_attribution.py
+++ after/fhm_dapp_tim/fhm_dapp/uji_v17_g8_hierarchical_attribution.py
@@ -210 +210 @@
-    if not torch.allclose(p.sum(dim=(1,2)),torch.ones(n,device=p.device),
+    if not torch.allclose(p.sum(dim=(1,2)),torch.ones(n,device=p.device,dtype=p.dtype),
--- before/tests/test_uji_v17_g8_hierarchical_attribution.py
+++ after/tests/test_uji_v17_g8_hierarchical_attribution.py
@@ -28,2 +28,3 @@
-def test_joint_factorization_matches_original_g6_temperature():
-    w,c,p=_factorized()
+@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
+def test_joint_factorization_matches_original_g6_temperature(dtype):
+    w,c,p=_factorized(dtype=dtype)
@@ -35,2 +36,3 @@
         assert variants[label].shape==p.shape
+        assert variants[label].dtype==dtype
         assert torch.allclose(variants[label].sum(dim=-1),
@@ -37 +39 @@
-                              torch.ones(len(p)),atol=1e-6)
+                              torch.ones(len(p),dtype=dtype),atol=1e-6)
@@ -223 +225,4 @@
-    assert "NOT" in g8._overview(j)
+    report=g8._overview(j)
+    assert "No training or new blind confirmation." in report
+    assert "already-viewed" in report
+    assert "No p-values or model promotion." in report
```

Post-repair fhm_dapp_tim/fhm_dapp/uji_v17_g8_hierarchical_attribution.py SHA256: 1cde6ae3a67e7736840c82cbdf12bff52b06f17e573ed85cb6e4421aa6ff4166.
Post-repair tests/test_uji_v17_g8_hierarchical_attribution.py SHA256: bea6856e822e84f2678d2857d51ce9a9f25bdd213d106edbc4cb2fda56bf5593.

| Stage | Exit | Seconds | Started (local) |
|---|---:|---:|---|
| compile-repaired | 0 | 0.093 | 2026-10-10T12:28:28+08:00 |
| tests-repaired | 0 | 5.092 | 2026-10-10T12:28:30+08:00 |
| preflight | 0 | 9.312 | 2026-10-10T12:28:36+08:00 |
| identity-and-synthetic-cuda | 0 | 13.316 | 2026-10-10T12:30:16+08:00 |
| prepare | 0 | 9.129 | 2026-10-10T12:30:47+08:00 |
| check-before-run | 0 | 9.173 | 2026-10-10T12:30:58+08:00 |
| run-once | 0 | 32.550 | 2026-10-10T12:31:13+08:00 |
| check-after-run | 0 | 9.202 | 2026-10-10T12:31:54+08:00 |

## Final verification

Completed: 2026-10-10T12:35:16+08:00; wall time 599.600s from audit initialization, summed stage process time 94.907s.
Repaired regression: 25 passed in 3.88s. Preflight, prepare, checks and the single formal run exited 0. Final check: VERIFIED / DESCRIPTIVE_FACTORIAL_ATTRIBUTION_COMPLETE.
All 893 snapshotted G1-G7 evidence/source/test files remained byte-identical; before/after combined digest b71ffdca1bd158643338f273386cb9ce0e826cb060148b899f217b27072b3458.
No G8 code/test/design edits after manifest. All 15 checkpoint file hashes still match original G4/G5/G6 identities. No model training, provided-test evaluation, policy reselection or Git operation.

Original G8 code SHA256 before repair: 5c2630755f711bfbd02eceae76e6da4ac29967774be40b73247b5f07e75dc2b9.
Original G8 test SHA256 before repair: 1f22a64f2166f65a4dc6b15152d59f0e1c2e47cddffea16e194acb59b37b1180.
Frozen G8 code SHA256: 1cde6ae3a67e7736840c82cbdf12bff52b06f17e573ed85cb6e4421aa6ff4166.
Frozen G8 test SHA256: bea6856e822e84f2678d2857d51ce9a9f25bdd213d106edbc4cb2fda56bf5593.

| Domain | Member | Seed | Epoch | Original checkpoint SHA256 |
|---|---:|---:|---:|---|
| tuji | 1 | 17301 | 65 | 20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a |
| tuji | 2 | 17302 | 69 | febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8 |
| tuji | 3 | 17303 | 56 | 73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a |
| tuji | 4 | 17304 | 67 | 58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b |
| tuji | 5 | 17305 | 62 | 124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c |
| uts | 1 | 17301 | 65 | 299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5 |
| uts | 2 | 17302 | 69 | 3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48 |
| uts | 3 | 17303 | 56 | 4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548 |
| uts | 4 | 17304 | 67 | 695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224 |
| uts | 5 | 17305 | 62 | 0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c |
| tampere | 1 | 17301 | 65 | 8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b |
| tampere | 2 | 17302 | 69 | 37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294 |
| tampere | 3 | 17303 | 56 | c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d |
| tampere | 4 | 17304 | 67 | 0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1 |
| tampere | 5 | 17305 | 62 | ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5 |

| Domain / historical role | Coordinates / rows | base MAE / ES m | member MAE / ES m | anchor MAE / ES m | both MAE / ES m |
|---|---:|---:|---:|---:|---:|
| tuji / calibration | 137 / 695 | 2.797311675 / 1.969971332 | 2.797118118 / 1.985455670 | 2.390619305 / 1.713442926 | 2.390662751 / 1.714639863 |
| tuji / confirmation | 139 / 701 | 2.969113295 / 2.102096869 | 2.968962391 / 2.081671783 | 2.657773678 / 1.867255089 | 2.657762633 / 1.867301052 |
| uts / calibration | 258 / 1625 | 7.155117679 / 5.024128699 | 7.155526375 / 5.007852435 | 5.380265830 / 4.123892143 | 5.380405089 / 4.140841811 |
| uts / confirmation | 263 / 1692 | 6.819208114 / 4.786196406 | 6.819075237 / 4.772177737 | 5.107529197 / 3.937964600 | 5.107270462 / 3.954782663 |
| tampere / confirmation | 112 / 112 | 7.202666301 / 5.212626871 | 7.202235414 / 5.152497602 | 5.987721130 / 4.996576132 | 5.986883910 / 4.968090181 |

Saved vectors independently reviewed without new inference: role keys/rows, finite metrics, exact vector means, declared factorial contrasts, and all BASE/BOTH trace vectors agree. All trace units are m².
Both nesting identities pass original atol=1e-5, rtol=1e-6; largest saved-coordinate absolute roundoff residual 0.000245460726319 m². The relative term accommodates larger spatial traces.
Original q=w*c and BOTH vs G6 whole-joint T=.6 were checked by the original hard-stop guards in every scored batch (atol=rtol=3e-6). The analytic Student-t within trace uses 2*sigma² per axis.
G7 JSON was read only AFTER G8 finished for identity verification: UTS CAL/CONFIRM and Tampere CONFIRM base/both MAE+ES vectors equal G7 baseline/T-only exactly; BOTH MAE also equals the G7 fixed T+s candidate exactly. No old scores selected a parameter.
Member/anchor deltas in JSON are the declared one-layer-vs-base contrasts; interaction = both-base-member_delta-anchor_delta. These operator contrasts and the two variance nesting orders do not identify unique causal contributions.
All three historical CONFIRM cohorts show anchor-dominated MAE contrasts (member-only absolute MAE gain <0.001m); ES member contrasts and interactions are nonzero. TUJI CAL member-only ES worsens by 0.015484338m. No layer is selected or promoted.
Formal source functions, RNG seed19602, two64 streams, E5/L192/K192/alpha1, df4 and scale s=1 stayed fixed. G8 BOTH ES is compared to G7 T-only s=1, never asserted equal to G7 primary s=.5.

TUJI input proof: original G6 loader used variable_names=('results_train',) for all five MATs; no results_test object was returned. Containers were opened/hashed, so their bytes were not isolated. No provided UTS/Tampere test or UJI official/outer file was opened.
The synthetic CUDA supplement scored no CAL/CONFIRM role; peak allocated 221696 bytes. All 15 loaded CPU models shared their domain's original anchors/prototype RSS/seen-AP mask/origin, were in eval mode with grad=None, and matched frozen seeds/epochs. FIT/CAL/CONFIRM location intersections were empty.
Preflight field all_test_labels_previously_seen refers to the five historical TRAIN-internal scoring roles, not to provided TUJI/Tampere test labels.
No scientific/runtime warning or failure after the repairs; original failures and helper issues remain documented above. No CUDA nondeterminism warning was emitted. Monte Carlo ES remains a finite 128-draw paired approximation.
All roles are previously exposed, spatially correlated internal cohorts; this is retrospective mechanism attribution, not blind/independent/external confirmation, a new promotion or a Q1 final model. Negative transfer is retained in unfiltered per-coordinate vectors for subsequent independent analysis.

| Allowed raw input | SHA256 |
|---|---|
| UTS_training.csv | a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b |
| Training_rss.csv | 7922efcd4ea14b1a5453a69ba35bc021a67b915b1b3076fa4ccf6da3e98e31b3 |
| Training_coordinates.csv | 22f0894e80966ef12d5f4e73f6e0b466d5a5781dda1f76c35ae6d276a929afff |
| Formatted_data_A12.mat | 73c59c6a8a12c45c4b6bf9b93b45cb89d0cfb7bb147260dc763f0d5f5af314a5 |
| Formatted_data_POCO.mat | 9973384ac04365a55a2e6b33b16b61724cf41869e828b1544c7c4bc61c9d2495 |
| Formatted_data_S20.mat | e3e327243852ba38087c4e8f07c4d94c7b8acc0987e961d5dc6e8fcf23dd354e |
| Formatted_data_S7.mat | c33ac65f9b61f6b24272b256b091c54747d5bce3f186a8a96e83cbc3e9f9fdab |
| Formatted_data_tabS7.mat | f8a9bf0a94367f2255a3c94b2bd4b633d73d7000a8f1a0cf5c25e90d231b0e23 |

| Completed artifact | File-byte SHA256 |
|---|---|
| manifest_v17_g8.json | 0c89c4deb72fcada62f3ff0d16fa219a894b07626883309ede76b22f06504686 |
| attribution_v17_g8.lock | 31b6023502f1d170c62dc8f625d066a82667f0fbff92db824b9621a203e8c1d2 |
| attribution_v17_g8.json | 036bedc28798330d1cbd75b2a3de6db60e2d12d99a79d05d82c2e77355862537 |
| RUN_REPORT_V17_G8.md | b80a32c8e1d30af2ef1186ebcd37515bbbdd262a03eaf4c9807d5f436cabdcfa |
| RUN_STDOUT.log | 208d8a000e358bbcad37798ae5e5eebcb5a6e77af96e974f89a3f917455e932e |
| RUN_STDERR.log | 303f81b01bb23fb2e1e99aa2e6f79b8875d82cf73906b15972f0bf25029b516a |

Manifest canonical self hash: d8ccabe4ba53c7cf47379ed48f7fb299c037d4e681247a80995bbbaf0095f4bc.
Result canonical self hash: 3ba8ddf35d212ae8d2e61e119727893a82d5fd284f293e8c178676e057466988.
Raw stage stdout/stderr are retained in two aggregate logs; no extra audit tree was created.
