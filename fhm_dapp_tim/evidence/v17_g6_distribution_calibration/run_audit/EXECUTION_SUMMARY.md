# V17-G6 执行摘要

最终检查 VERIFIED；科学决定 **PROMISING_JOINT_CALIBRATION**。CAL 冻结 T0.6_S0.5（T=0.6, s=0.5），CAL/CONFIRM 各一次。

北京时间 2026-10-09T21:23:45+08:00 至 2026-10-09T21:49:33+08:00，约 25.81 分钟；五成员训练命令合计 431.17 秒。

Python 3.12.12；Torch 2.10.0+cu128；CUDA 12.8；NVIDIA GeForce RTX 2060（6.00 GiB）。NumPy 2.4.4，SciPy 1.17.1，pandas 3.0.1。解释器 D:\miniminiconda\envs\liuyan\python.exe。

仅用 TUJI1 五个 MAT 中的 results_train；每次 scipy.io.loadmat 均限制 variable_names=('results_train',)，返回字典无 results_test。物理容器被打开并整体哈希，不能声称测试字节隔离；测试变量未请求/解码/读取坐标或评分。未调用原 load_tuji1，未读取其他提供测试。历史证据只在命令外按原始字节哈希核验，没有解析或用于选择。

此前 21 条命令的 Python open/loadmat 轨迹核验通过，限制变量读取 235 次、禁止路径访问 0；不是操作系统全局监控。历史科学代码/证据 63 文件 SHA256 不变。

4595 行 / 919 物理位置 / 259 AP，building/floor=0/0；角色位置交集0。原始 double→适配 float32→FIT 原点居中四位键全角色一一对应，原始与适配四位键数值变化0。

| 角色 | 位置 | 行 | 键 SHA256 |
|---|---:|---:|---|
| fit | 643 | 3199 | 0f166645db1c6bd58d101fc661a0ec62a589b6d89229989224c569092bb6ab95 |
| calibration | 137 | 695 | e1ad91fd38a086e445426f502bf24d5a3224823e48f46e6f99ea486c31c32e22 |
| confirmation | 139 | 701 | 1c46af2f1336fcc8e664f7c214ef876418a1c288df942db9d6f517e88d22542c |

| 设备 | FIT 行 | CAL 行 | CONFIRM 行 |
|---|---:|---:|---:|
| A12 | 641 | 137 | 139 |
| POCO | 619 | 135 | 137 |
| S20 | 685 | 152 | 149 |
| S7 | 638 | 137 | 139 |
| tabS7 | 616 | 134 | 137 |

共享 FIT 原点 [7.13189697265625, 15.915324211120605]。5/5 完整训练、hash 各异、未复用历史权重，共享 FIT 锚点/原型/楼宇楼层/原点/encoder AP 掩码；只有 FIT 行参与 loss，没有 CAL/CONFIRM 早停。

| 成员 | seed | epoch | checkpoint SHA256 |
|---:|---:|---:|---|
| 1 | 17301 | 65 | 20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a |
| 2 | 17302 | 69 | febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8 |
| 3 | 17303 | 56 | 73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a |
| 4 | 17304 | 67 | 58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b |
| 5 | 17305 | 62 | 124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c |

权重保留在本地 fhm_dapp_tim/models/v17_g6_distribution_calibration/。原 R0 网络/loss/AdamW/增强不变。E5/L192/K192/alpha1、df4、两流各64 draws、seed19602。FIT RSS/合成零目标 CUDA 检查证明基线 law/均值/MAE/ES/draws 与原 R3 逐位一致，同 T 所有 s 的 MAE 逐位一致，grad 为 None。

| CAL 候选 | macro MAE m | macro ES m | MAE 差 m | ES 差 m | 合格 |
|---|---:|---:|---:|---:|---|
| T1_S1 | 2.797311675 | 1.969971332 | +0.000000000 | +0.000000000 | False |
| T1_S0.75 | 2.797311675 | 1.968483436 | +0.000000000 | -0.001487896 | False |
| T1_S0.5 | 2.797311675 | 1.967545727 | +0.000000000 | -0.002425605 | False |
| T0.8_S1 | 2.590559130 | 1.844575380 | -0.206752546 | -0.125395952 | True |
| T0.8_S0.75 | 2.590559130 | 1.843286796 | -0.206752546 | -0.126684536 | True |
| T0.8_S0.5 | 2.590559130 | 1.842622060 | -0.206752546 | -0.127349271 | True |
| T0.6_S1 | 2.390662751 | 1.714639863 | -0.406648924 | -0.255331469 | True |
| T0.6_S0.75 | 2.390662751 | 1.713562481 | -0.406648924 | -0.256408851 | True |
| T0.6_S0.5 | 2.390662751 | 1.713302857 | -0.406648924 | -0.256668475 | True |

仅用 CAL 的 MAE 改善≥0.10m 且 ES 改善≥0.02m 筛选，最小 MAE、ES/名称破平局；用保存指标复核选择一致，没有人工更选。

| CONFIRM 方法 | macro MAE m | macro ES m | row MAE m | row ES m |
|---|---:|---:|---:|---:|
| T0.6_S0.5 | 2.657762633 | 1.866291035 | 2.650049767 | 1.861136590 |
| T0.6_S1 | 2.657762633 | 1.867301052 | 2.650049767 | 1.862074350 |
| T1_S0.5 | 2.969113295 | 2.100576019 | 2.962000757 | 2.097249260 |
| T1_S1 | 2.969113295 | 2.102096869 | 2.962000757 | 2.098678693 |

正式 candidate−baseline：MAE **-0.311350662m**，ES **-0.235805834m**；95% familywise 同时上界 MAE **-0.239044678m**、ES **-0.198812604m**。

Gate：非 baseline=True；MAE 改善≥0.15m=True；MAE 上界<0=True；ES 上界≤0=True；总 gate=True。

描述性：改善 113 位置，恶化 26，相等 0；配对 MAE 差中位数 -0.224534137m。单因素对照只能解释机制，不能替代冻结候选。
5000 次坐标配对 maxT，seed19603，MAE/ES 两指标 family，临界值 1.815861914。用保存向量和原临界值复核均值/SE/上界，没有重新 bootstrap。

| CONFIRM 方法 | predicted radial spread m | 50%覆盖 | 80%覆盖 | 90%覆盖 |
|---|---:|---:|---:|---:|
| T0.6_S0.5 | 3.776843171 | 0.754676 | 0.949161 | 0.994245 |
| T0.6_S1 | 3.798080479 | 0.758993 | 0.953477 | 0.994245 |
| T1_S0.5 | 4.386635022 | 0.772422 | 0.976978 | 0.992806 |
| T1_S1 | 4.404312336 | 0.775300 | 0.975540 | 0.994245 |

所有被评分候选的逐坐标半径、50/80/90% credible-ball 半径/经验覆盖在 CAL/CONFIRM JSON；有限128样本估计，不是 conformal 保证，不参与选择。

| CONFIRM 坐标尾部 | p90 MAE m | 最大 MAE m | >20m 位置 | >50m 位置 |
|---|---:|---:|---:|---:|
| T1_S1 | 4.392200290 | 6.159217323 | 0 | 0 |
| T0.6_S0.5 | 3.981617098 | 5.934683186 | 0 | 0 |

这些尾部数值是每物理坐标扫描平均后的误差，不是单扫描极值。设备 row MAE/ES、原始坐标键和对齐向量在结果 JSON。

错误与修复：
- 原始测试43 passed/1 failed：报告缺测试要求的 TRAIN 标识。三项新增回归在原 G6 源码上全部 FAIL，复现原始 double→float32 合并检查缺失、排序 JSON 后表格不一致、九候选可靠性输出缺失。
- manifest 前仅修改新增 G6 源码/测试：补原始坐标映射断言和证明；固定 TRAIN 标识/表格排序；补原设计要求的半径/覆盖诊断。原始文件、失败日志和精确 diff 保留；分区/指标公式/随机流/训练/门槛不变。最终指定测试47 passed，编译/预检成功。
- manifest 后，我编写的只读辅助核验错误引用 model.seen_ap_mask（正确位置 model.encoder.seen_ap_mask）。仅更正非科学性辅助检查，保留原稿/差异/失败 stderr，重跑 FIT/合成核验通过。冻结的科学源码/测试/manifest/模型未变，CAL/CONFIRM 未重跑。
- 五训练 stderr 均保留 Memory Efficient attention 非确定性 backward warning；固定种子/CUBLAS 不保证再次训练逐位一致，未改变训练算法。无 OOM、训练失败、成员重启，CAL/CONFIRM 各一次成功。
- 初始 apply_patch 创建日志辅助文件失败，改用 PowerShell 写入；一次摘要工具 JS 字符串解析失败，未启动命令，改为安全字面量。无科学产物受影响；未执行 Git。

| 命令阶段 | exit code | 秒 |
|---|---:|---:|
| compile_initial | 0 | 3.389 |
| tests_initial | 1 | 19.733 |
| preflight_initial | 0 | 14.572 |
| regression_before_fix | 1 | 22.798 |
| compile_fixed | 0 | 2.683 |
| tests_fixed | 0 | 21.827 |
| preflight_fixed | 0 | 14.669 |
| pre_manifest_review | 0 | 23.245 |
| prepare | 0 | 15.912 |
| train_member_01 | 0 | 86.462 |
| train_member_02 | 0 | 91.323 |
| train_member_03 | 0 | 79.037 |
| train_member_04 | 0 | 89.713 |
| train_member_05 | 0 | 84.639 |
| check_members | 0 | 16.208 |
| trained_panel_review | 1 | 24.779 |
| trained_panel_review_fixed | 0 | 24.439 |
| calibrate | 0 | 30.756 |
| check_after_calibration | 0 | 18.536 |
| confirm | 0 | 29.524 |
| final_check | 0 | 18.405 |

本摘要生成命令记录另见 COMMANDS.jsonl。全部 stdout/stderr/退出码/时间/hash/访问轨迹保留，上表 FAIL 没有删除或豁免。

关键 SHA256：
- manifest file: 45896bed26ca192ad35422efd657e373b8878d6e7870eac8f17447d5b26d08c7
- calibration file: 77ae0e137335dc6e99f1bf9d26b135fc9979bfc89ae26c881c207dae4fc2b17a
- confirmation lock: c7441f8569ee34eae74aca7a2d2f1191b4b7062964253bbda634823ce2d180b0
- confirmation file: f8ac85cf0fc1010261e7401b15305475dbe13d74d3289291b6c653c1fdbc7c5a
- report: a1519511a17ecd226ff715c84ed0741fc67ce0cbe424cd1ceb2bbde1cba36495
- manifest self: ef4f370ea86f90d2f435d99993911d25b7c9d529e65a63a7671b1cf720eac19e；confirmation self: e068b40149201b7c9d32577e0771a1c1cf27c271a2170da555cee0dd78c292b4
- G6 science source: e8f42637b24a69c664dd8726c9b15974fb51f219d2c373d1d0b3fce8d42cffe1；tests: 12fafc6fee8a630ac482adfca52fe8be440bc90b618b19dd40dea5ab060ad9f4
- train_container_A12: 73c59c6a8a12c45c4b6bf9b93b45cb89d0cfb7bb147260dc763f0d5f5af314a5
- train_container_POCO: 9973384ac04365a55a2e6b33b16b61724cf41869e828b1544c7c4bc61c9d2495
- train_container_S20: e3e327243852ba38087c4e8f07c4d94c7b8acc0987e961d5dc6e8fcf23dd354e
- train_container_S7: c33ac65f9b61f6b24272b256b091c54747d5bce3f186a8a96e83cbc3e9f9fdab
- train_container_tabS7: f8a9bf0a94367f2255a3c94b2bd4b633d73d7000a8f1a0cf5c25e90d231b0e23

仅为 TUJI1 训练变量内的探索性试验；五成员共享 FIT 数据，不是五次独立实验。空间相关可能使普通坐标 bootstrap 偏乐观；ES/覆盖有 Monte Carlo 数值误差，全局 T/s 可能依赖数据集。没有新外部泛化验证或 Q1 完整证据链，G1–G5 历史决定不变。
