# V17-G3 UTS 执行摘要

**执行完成；科学判定：TRANSFER_NOT_ESTABLISHED_ON_UTS（STOP）。最终 check=VERIFIED 仅表示产物一致性通过，不是科学 GO。**

当前版本开始至最终 check（Asia/Shanghai）：2026-10-08T21:46:24.284664+08:00 → 2026-10-08T23:14:46.468552+08:00；历时 88.370 分钟。25 名串行训练命令合计 79.707 分钟；唯一一次测试命令 30.293 秒。

Python：D:\miniminiconda\envs\liuyan\python.exe；3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]。Torch 2.10.0+cu128 / CUDA 12.8；显卡 NVIDIA GeForce RTX 2060（6GB）。NumPy 2.4.4，pandas 3.0.1，SciPy 1.17.1，pytest 9.0.3。

## 冻结、测试访问与设计边界

- 用户确认结果盲评，曾知晓文件格式；不声称所有测试文件字节从未被任何人打开。
- 训练仅使用 UTS_training.csv：9,108 行、1,466 个物理锚点、589 AP、1 个楼宇、16 个训练编码楼层；原始位置到 float32/四位锚点键无合并。Pos_x/Pos_y 按冻结设计与适配器的米单位解释；只核对了数值范围，没有另作外部单位验证。
- 训练 RSS 为原 V17 float32 [-1,1] 标度。5,045,233 个缺失单元均为 -1；可见 RSS 范围 [-96,-37] dBm，裁剪到 -1 的可见值数量为 0。
- gamma=0.70、10m、E16/L192/K192/alpha1、原 V17 R0 模型/损失/优化器/增强、seeds 17301..17325、原固定 epochs 全部保持。没有 UTS epoch 搜索。
- 25/25 完整训练，无失败/重启/恢复；25 份哈希唯一，锚点字典、原点和原型记忆逐位一致，均仅来自 UTS_training.csv。
- 训练和冻结前后的测试前共 40 份 Python 数据 open 轨迹仅见训练文件，无测试读取或拒绝尝试。该轨迹不是 OS 全局文件访问监控。
- 首次测试 open（Asia/Shanghai）：2026-10-08T23:13:38.947507+08:00，当时锁与 freeze 已存在，哈希与最终文件一致。
- evaluate-test 仅执行 1 次，退出 0；同一次调用内部测试 open 2 次（加载 CSV 与最终 SHA256），不是两次评估。之后仅读取结果 JSON 做本摘要。
- 测试：388 行 / 386 个坐标组。按冻结 V17 键定义，train/test 坐标组交集为 0。未访问 Tampere/TUJI 测试、UJI outer/official 文件，未执行 Git 操作。
- 25 个 E16 子集共享成员与测试坐标，不是 25 次独立实验。历史 G1 STOP、全部 G1/G2 输入哈希及所绑定科学源码保持不变。
- checkpoint val_mae=0 是旧 schema 的兼容占位字段；没有验证评分，不能把该值当成测量结果。

## 唯一测试结果（m）

| 方法 | Coordinate macro MAE | Coordinate macro ES | Row MAE | Row ES |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7.684425323 | 5.468975076 | 7.687076142 | 5.470877735 |
| g1_gamma_0p7 | 7.755909200 | 5.627000145 | 7.761379942 | 5.630207485 |
| permuted_density | 7.702339346 | 5.496777411 | 7.701151054 | 5.497178135 |

差值方向统一为 G1−baseline；正值表示恶化。

MAE 差 +0.071483878m；ES 差 +0.158025069m。placebo−baseline MAE +0.017914023m。

原 G1 studentized-maxT：5000 次坐标级配对重采样，family=MAE+ES，confidence=0.95，seed=18113。

| 固定条件 | 结果 | 是否通过 |
| --- | ---: | --- |
| MAE 差 ≤ −0.15m | +0.071483878 | FAIL |
| MAE 同时上界 < 0 | +0.159362593 | FAIL |
| ES 同时上界 ≤ 0 | +0.205234985 | FAIL |
| placebo MAE > baseline | +0.017914023 | PASS |

| 坐标级变化 | 改善 | 恶化 | 不变 | 差值中位数 (m) |
| --- | ---: | ---: | ---: | ---: |
| exact_mixture_mean_mae_m | 179 | 207 | 0 | +0.048647163 |
| energy_score_2d_m | 130 | 256 | 0 | +0.155503401 |

逐坐标三种方法的 MAE/ES 与 25 子集分布在不可变 external_result JSON；显式两种配对差在 POST_RESULT_PAIRED_DIFFERENCES.json。未再次推断、重采样或评分。点估计及门槛支持‘UTS 固定迁移未建立’；不把 MAE 的正点差说成已证明显著恶化。历史 UJI 5.726m 来自不同数据与训练参考，不能直接排名；本轮也不证明普遍外部泛化、因果机制或 Q1 完整证据链。

## 错误、修复和警告

- 早期预检发现 G3 错误继承 G1 的 candidate<placebo 判据，而本轮预注册的是 placebo>baseline；合成复现失败日志 gate_reproduction_attempt01 保留。PREPARE_GATE_IMPLEMENTATION_FIX.json/.diff、原始源码快照及新版 V17_G3_CODEX_RUN.txt 的修正说明均保留。
- 当前用户提供版本已使用独立 g3_transfer_gate，并含反例、严格零边界、原阈值和非有限输入回归。当前 compile、60 项 tests、preflight 全部通过后才 prepare；本次执行没有再改源码，prepare 后至结束源码哈希未变。
- 25 个训练 stderr 均记录同一 Memory Efficient attention backward 非确定性警告。固定 seeds / CUBLAS_WORKSPACE_CONFIG=:4096:8 不保证该 CUDA backward 逐位可复现；保存实际训练权重哈希，不为消除警告更换算法或重训。测试 stderr 为空。
- 当前版本所有命令退出码为 0；不利结果是科学门槛失败，不能当作工程故障重跑。

## 单测与命令记录

每条指定命令的原始 stdout/stderr、退出码、UTC 起止时间、运行秒数和日志 SHA256 均保存在同名 run_audit/*.json/*.stdout.log/*.stderr.log；不重复复制全部日志内容。

| 阶段 | 退出码 | 秒数 |
| --- | ---: | ---: |
| compile_attempt01 | 0 | 0.335 |
| pytest_attempt01 | 0 | 12.976 |
| preflight_attempt01 | 0 | 11.944 |
| gate_reproduction_attempt01 | 1 | 9.821 |
| compile_attempt02 | 0 | 0.295 |
| pytest_attempt02 | 0 | 11.729 |
| compile_attempt03 | 0 | 0.289 |
| pytest_attempt03 | 0 | 11.310 |
| preflight_attempt02 | 0 | 12.046 |
| training_contract_review_attempt01 | 0 | 12.123 |
| prepare_attempt01 | 0 | 12.092 |
| train_member_01_attempt01 | 0 | 201.960 |
| train_member_02_attempt01 | 0 | 213.034 |
| train_member_03_attempt01 | 0 | 175.861 |
| train_member_04_attempt01 | 0 | 206.747 |
| train_member_05_attempt01 | 0 | 192.248 |
| train_member_06_attempt01 | 0 | 188.628 |
| train_member_07_attempt01 | 0 | 163.776 |
| train_member_08_attempt01 | 0 | 227.764 |
| train_member_09_attempt01 | 0 | 238.178 |
| train_member_10_attempt01 | 0 | 192.572 |
| train_member_11_attempt01 | 0 | 202.070 |
| train_member_12_attempt01 | 0 | 178.619 |
| train_member_13_attempt01 | 0 | 198.120 |
| train_member_14_attempt01 | 0 | 189.846 |
| train_member_15_attempt01 | 0 | 158.845 |
| train_member_16_attempt01 | 0 | 227.333 |
| train_member_17_attempt01 | 0 | 178.921 |
| train_member_18_attempt01 | 0 | 215.470 |
| train_member_19_attempt01 | 0 | 183.239 |
| train_member_20_attempt01 | 0 | 200.622 |
| train_member_21_attempt01 | 0 | 159.592 |
| train_member_22_attempt01 | 0 | 175.323 |
| train_member_23_attempt01 | 0 | 161.350 |
| train_member_24_attempt01 | 0 | 183.226 |
| train_member_25_attempt01 | 0 | 169.049 |
| check_members_attempt01 | 0 | 13.019 |
| pretest_panel_review_attempt01 | 0 | 10.666 |
| freeze_attempt01 | 0 | 14.634 |
| check_before_test_attempt01 | 0 | 14.643 |
| evaluate_test_attempt01 | 0 | 30.293 |
| check_after_test_attempt01 | 0 | 14.571 |
pytest_attempt01：42 passed in 10.70s。
pytest_attempt02：45 passed in 9.54s。
pytest_attempt03：60 passed in 9.19s。

## 全 25 权重（均为 UTS train-only；保留本地）

| 成员 | Seed | Epoch | Checkpoint SHA256 |
| --- | ---: | ---: | --- |
| 1 | 17301 | 65 | bbcf0a2b61fe871ace46ef84bb701e1badd7b505b572b41b19f1914f2a71fb65 |
| 2 | 17302 | 69 | bb21e7962f9820ef4a235bf27bc37fc853d68547bf6fdfa4d3fb387aa057dec2 |
| 3 | 17303 | 56 | d3c6cf6cc67bff90d18eaf7294a3cc25fdf3c18e70a1090f0bb7361841f0424a |
| 4 | 17304 | 67 | 362866b224519eb5b3d6ea7b845596131bb3f8529ade556dbf9e4abf1fab9dec |
| 5 | 17305 | 62 | 51ab7b76b4036cdfa66cef638ae933f4beabb865a3d57f66ecf8092aaec85b67 |
| 6 | 17306 | 60 | d6cabaee426cf109a92d8a2653e3a12a305829b0dc2620a8042d6090e2ef133d |
| 7 | 17307 | 52 | 4877a91cd7ee62fd6005379245f474bd70f1ebabc95da6b5649553dea0579068 |
| 8 | 17308 | 74 | 8a0b82640df8acba27c81e4522ec376893f102c48fc65557410ebe49fcd67a41 |
| 9 | 17309 | 79 | d63d16d8f7a04a2d423d3a8e4dbedabf0d600089204bda75280d66eb13958a02 |
| 10 | 17310 | 62 | c4da9eba3cb3cab8385511b1442a1094debc627659960ed7297a4fd46d6650d2 |
| 11 | 17311 | 65 | 615df46e255ed7f37618e7fea47cf10d8d9a4979b9f67fe80a84b58f528a3d90 |
| 12 | 17312 | 57 | 3d9785dc9256924d61d57b2813b95f7a67bbc79ab3bcf426605678c5aa6d957d |
| 13 | 17313 | 64 | ee0f308a8994ec51f4ccc34affafed9525182dae647416ff8b5c8a71118255ac |
| 14 | 17314 | 61 | 4f15f87dc12cbc80287b4a03bd9eac0c196d9201bde9dc62ac8fa403de34956e |
| 15 | 17315 | 50 | 7b3302ca8ce3a9cc65c2267036805478860aaa18bcbaa84406045fa98b0d87ac |
| 16 | 17316 | 74 | 8aacab850d32b3589131c7317b72b3bbd0f71c47bd9f50482a7c3639682a12a3 |
| 17 | 17317 | 59 | 49cc00c65b1fc357add3766de7625370b8a953a903ffd03f540ae4a16f031a75 |
| 18 | 17318 | 73 | 340853938d2e466ae29f370aa8671fa4b1f4c174f11450b648029d4023cba1a7 |
| 19 | 17319 | 60 | ce8e2f5d0d4dea2001bcbb5522ba38f2265d99cda791a4621ee873a75f1085e0 |
| 20 | 17320 | 68 | 76a5d8e3550978deafddd0f9df5056f494216741ed6e1e7454edf278b06e0f26 |
| 21 | 17321 | 52 | 50beff71f3ef7aa4665549a55ccfc07155c2e9d0fe77bb640c8ff23e6d0a2437 |
| 22 | 17322 | 58 | e62582179cde69bf5b6d2b68a05613a0ab6e7b5a0e30110439206ae54b5630b7 |
| 23 | 17323 | 53 | 51fb6508218328b6e1a0638b8aa4af945c065c5a159d7f56073b57af0f5b758f |
| 24 | 17324 | 61 | b1edb2236c4ab0662a00f3dd9e14e1ec9a24a4322cb5591d0cf2059bdc7e3d24 |
| 25 | 17325 | 56 | bd34525cfb299ec3531f68bfd92ba1744d1734b885724b76a8d61a37d90e9c85 |

## 关键哈希

| 文件/字段 | SHA256 |
| --- | --- |
| UTS_training.csv | a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b |
| UTS_test.csv（仅引用评估内生成的值） | 8c38273a5013fbe63984a235b92f72142aa52f19008fc213f4428a03661d8916 |
| G3 source | b7a7097fa3654dfc516d01c6aa2ab5acca80b7ca3d1ab87c463f085494576676 |
| G3 design | 2deda4ac5256341244c04c6989c61c091b99398c68e45aab483c48509b873793 |
| Manifest self hash | ba71edcc44ffe8caefa350e9ac733e5d62e9336d30bb4c540f400e3020d3d356 |
| Freeze self hash | e1f1ac8c8e1f3f8bbb10c4abba4181b1c549b2ba4911cc3a094ee0e53386e81d |
| Lock self hash | 58e3a1e3ef57d6d22c4f73e22341c77f13a6c61c0d265cfc1c066051235d3e90 |
| Result self hash | 8e9953ec4191676d1cc070bda757b83f1f5382703a39a2a4d0e4de97c1add0a0 |
| manifest_v17_g3_uts.json | 5387ae4e07dc01fc16dbc0c15c21e01a034c2d4efe6cd908f34a3b97dfcdf09b |
| freeze_v17_g3_uts.json | 4cfda131e0342a39ccf7ee043eed78f4ed59389bad831d5d6b6c77a391b1f1a4 |
| external_test_invocation.lock | e1ca9cd845e74fc75eb6e5391073dd0b4dab55efe859467f78943358cbc49ed5 |
| external_result_v17_g3_uts.json | fddaa94fe4a2ecc00b06d7c437df8c86265af57b7706d00089a81032f3efdff2 |
| RUN_REPORT_V17_G3_UTS.md | a199867c930b743f75b72dedd81470a9d189da31388a7a555196771848ab70fb |
| run_audit/POST_RESULT_PAIRED_DIFFERENCES.json | 8790587b5bb2ce8d8d5445b307b8595702b51727c70349ccaae4098a5f384cec |

Manifest/source_sha256 保存完整科学输入与代码哈希；CURRENT_EXECUTION_CONTEXT 保存当前测试和辅助代码版本；G1/G2 原始输入已逐一复核。未改写正式报告、结果、manifest、freeze 或 lock。

结论保持 TRANSFER_NOT_ESTABLISHED_ON_UTS；原 G1 STOP 保持。没有后续调参、新架构、其他外部测试或 Git 操作。
