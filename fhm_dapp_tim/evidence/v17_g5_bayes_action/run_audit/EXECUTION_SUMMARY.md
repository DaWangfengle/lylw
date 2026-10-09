# V17-G5-A 执行摘要

完整执行完成；最终 check=VERIFIED；科学决定 **NO_INTERNAL_PROMOTION**。仅 Tampere 训练文件内新分区的五成员探索，未进行提供测试集或外部泛化验证。

运行：2026-10-09T08:46:20.239945+00:00 至 2026-10-09T08:59:43.258402+00:00（UTC），至最终检查 13.384 分钟；五个训练命令合计 140.960 秒，包含进程启动/加载时间。
环境：Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；解释器 D:\miniminiconda\envs\liuyan\python.exe；Torch 2.10.0+cu128 / CUDA 12.8；NVIDIA GeForce RTX 2060 6GB；NumPy 2.4.4、pandas 3.0.1、SciPy 1.17.1。

数据只读 Training_rss.csv 与 Training_coordinates.csv：446 行、489 AP、446 物理坐标；FIT 334 行/334 坐标，CONFIRM 112 行/112 坐标。物理键及行交集均为 0。原始浮点坐标、适配器 float32、R0 居中后四位键在全部 446 个位置上一一对应，无压合。
楼层 z 映射：{'0.0': 0, '3.5': 1, '7.0': 2}；只计算二维 x/y 欧氏误差。设备未知类别 -1 按原始 R0 映射为一个类别。FIT 坐标原点 [195.4998779296875, 126.66690063476562]。
FIT/CONFIRM 分楼层坐标数：{"0": {"confirmation": 31, "fit": 92}, "1": {"confirmation": 55, "fit": 164}, "2": {"confirmation": 26, "fit": 78}}。
键 SHA256（按排序列表、紧凑 JSON、UTF-8）：
- fit: d917c5efe32c714267dd2b6785c79965d7948f3f484d8a02f8ab19d1c9798468
- confirmation: 905866e37596155ee6131b271686a69ce3e0692f9d67d61b978d2c109e411b67

训练固定原始 V17 R0 网络、损失、AdamW 和增广；锚点字典、原点、可见 AP、设备映射与训练 loss 均由 FIT 行建立。CONFIRM 无训练损失或提前误差评估；未用 G3 完整权重。五成员是同一 FIT 上不同种子的共享锚点训练，不构成五次独立实验。

| 成员 | seed | 完整 epoch | checkpoint SHA256 |
|---|---:|---:|---|
| 1 | 17301 | 65 | 8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b |
| 2 | 17302 | 69 | 37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294 |
| 3 | 17303 | 56 | c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d |
| 4 | 17304 | 67 | 0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1 |
| 5 | 17305 | 62 | ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5 |

共享原型/RSS、锚点坐标/楼宇/楼层、坐标原点、可见 AP 掩码核验通过；五份哈希各异。freeze 后仅用前 16 个 FIT RSS 扫描进行 CUDA smoke；真实查询标签评分为 0，scorer 的目标为合成零向量。联合 law 16×960，动作 2×128 样本；峰值 allocated 65982464 bytes、reserved 75497472 bytes；原 R3 点均值逐位一致、grad 均为 None、checkpoint 文件未变。

确认只执行一次；锁在评分前创建并保留。三动作共用原 E5/L192/K192/alpha1、df4 Student-t law。动作 seed19503、风险 seed19504，均 2×128；共用 ES seed19505、2×64；Weiszfeld 64 步/1e-5m。

| 动作 | 坐标宏 MAE (m) | row MAE (m) | 独立抽样预测欧氏风险 (m) |
|---|---:|---:|---:|
| mean | 7.202666301054 | 7.202666301054 | 14.228222511953 |
| component_median | 6.680664314603 | 6.680664314603 | 13.292911399144 |
| continuous_median | 6.820972859802 | 6.820972859802 | 13.323578389011 |

唯一共用 ES：坐标宏 5.271725195366 m，row 5.271725195366 m；概率法则未改变，动作间 ES 差严格为零。不宣称概率预测改善。
连续中位数相对均值的宏平均点位移 4.524534842532 m；连续/分量中位数之间 0.835746898515 m，112 个配对位置位移向量已保存。

| 预声明对照（候选减均值） | 宏平均差 (m) | SE (m) | 95% familywise 同时上界 (m) |
|---|---:|---:|---:|
| continuous_median | -0.381693441252 | 0.395454293104 | +0.350248391929 |
| component_median | -0.522001986451 | 0.379002188615 | +0.179488834491 |

原始 studentized maxT：112 个物理坐标、5,000 次共同配对重采样、seed19502、两个 MAE 对照一个 family、单侧 confidence95%；critical=1.8508885753542097。最终核验只从保存向量和保存 critical 复原均值/SE/上界，没有重跑 bootstrap。

主要改善 >=0.15m：PASS；同时上界 <0：FAIL。固定最终决定 NO_INTERNAL_PROMOTION；没有豁免、不选次要动作晋级、不调整种子/阈值。

描述性：连续中位数改善 52 坐标、恶化 60 坐标、相同 0；配对差中位数 +0.172663103006 m。
分楼层主要对照（描述性，不能作为新的 gate）：
| 编码楼层 | 坐标数 | 平均差 (m) | 中位差 (m) |
|---|---:|---:|---:|
| 0 | 31 | -2.057398299827 | -1.492831541516 |
| 1 | 55 | +0.380377431556 | +0.270155963951 |
| 2 | 26 | +0.004189351492 | +0.378161468893 |

row P90/P95 误差：{"row_error_p90_m": {"component_median": 10.027554775432435, "continuous_median": 9.4292005220036, "mean": 12.911591752478895}, "row_error_p95_m": {"component_median": 11.085929496695538, "continuous_median": 11.598667314049626, "mean": 18.7339344719072}}。

测试/修复：首次指定测试 39 passed；额外针对性回归在原实现复现两项 FAIL（regression_before_fix，退出1）：重复归一化破坏原 R3 float32 均值逐位一致，以及风险字典顺序导致报告 JSON 往返不一致。仅在 manifest 前修复 G5 新源码/测试，并补存指令要求的两种中位数位移诊断；最终完整测试 40 passed。原始失败、源码/测试快照、修复 diff/哈希完整保留在 PRE_MANIFEST_REPAIRS.* 与 G5_*_BEFORE_FIX.txt。未改科学参数或研究设计，冻结后未改源码。
工具环境：首次默认 shell 进程因 helper setup refresh 错误未能启动；文件补丁工具两次未能写入新的审计 helper，随后用获准的 PowerShell 原生文件写入完成日志目录。均发生在 manifest 前，未执行科学命令、读取数据或覆盖结果；命令级实际输出见下表及原始日志。
全部五名成员保留相同 CUDA Memory Efficient attention 后向非确定性 warning；沿用原训练实现，不为消除警告改变算法。固定 seed/CUBLAS 配置不保证重新训练逐位一致。无 OOM、训练中断、checkpoint 复用或确认评分失败；确认及最终 check stderr 为空。

| 阶段 | exit | 秒 |
|---|---:|---:|
| compile_attempt01 | 0 | 0.395 |
| pytest_attempt01 | 0 | 18.005 |
| preflight_attempt01 | 0 | 14.291 |
| regression_before_fix | 1 | 15.976 |
| compile_attempt02 | 0 | 0.324 |
| pytest_attempt02 | 0 | 16.138 |
| preflight_attempt02 | 0 | 14.255 |
| preparation_review | 0 | 14.538 |
| prepare | 0 | 14.344 |
| train_member_01 | 0 | 28.645 |
| train_member_02 | 0 | 28.522 |
| train_member_03 | 0 | 27.468 |
| train_member_04 | 0 | 28.384 |
| train_member_05 | 0 | 27.942 |
| check_members | 0 | 14.532 |
| freeze | 0 | 14.570 |
| check_before_confirm | 0 | 14.905 |
| frozen_fit_gpu_smoke | 0 | 15.324 |
| confirm_once | 0 | 16.115 |
| check_after_confirm | 0 | 14.491 |

20 个记录阶段均保存独立原始 stdout/stderr、UTC 时间、退出码和日志 SHA256（COMMANDS.jsonl）。每阶段 Python open 记录只包含两份允许的训练文件，拒绝访问尝试 0；此证据是 Python 进程内 open hook，不是操作系统全局监控。历史证据仅作字节哈希核验，未解析标签结果或参与 G5 调参。
654 个历史源代码/证据文件与初始哈希逐字匹配；G1/G3/G4 既有决定未修改。本轮未执行任何 Git 操作。五份模型保留本地 fhm_dapp_tim/models/v17_g5_bayes_action/。

输入/算法/设计 SHA256：
- g5_code: 0953f4726d7b769b140f9d0e20e8ed7481a2d2b144ecc875368e5bc764945d8b
- g5_design: d0756575000de5f69a772a94cfb27703b753d7b78ef478912f91a59ad972041b
- tampere_adapter: 3464d3d131cfaa74a10cc67ee8cdf60b5a5497addaf910d828ab4af65d9b0f63
- training_rss: 7922efcd4ea14b1a5453a69ba35bc021a67b915b1b3076fa4ccf6da3e98e31b3
- training_xyz: 22f0894e80966ef12d5f4e73f6e0b466d5a5781dda1f76c35ae6d276a929afff
- v17_anchor_loader: a580ddd80bd9adef18953c66e3ce295c6c200e6d6d387184fcd3ec89f8fd10dd
- v17_score: 4e17632041fe1364e253a6e8275772f45479b6cca454ce8351b6211e87139082
- v17_score_protocol: 7d2ae1aa0a8c1e72d8ffc4adff8c784ad68a0726b6f0d1d259cfbd28cdeae03f
- v17_training: 9b8afdbdd5475d0a94cb8e3cacfc66a62796b72964c3f5414eb7bf524c050b4d
- v9_continuous_sampler_and_median: 937d7c6d6b2e094322d16ba6d5fb79108a7e1d49aaa6ebf0fb80bfca954f853d
- g5_tests: 92eccc06ee907a5e216f7af39baf314e8e98a4817b938430a9a56bbc69d511a3

冻结正式产物 SHA256：
- manifest_v17_g5.json: f3c92a952566787039cab02ec42fbcf3a493aa3b34b7194ba4b26989d450c812
- freeze_v17_g5.json: d1e0d97f102fd85931332a4dbf5139dcd368667366068d9d8903910c96b4523b
- internal_confirmation.lock: bfbb660b2cc8b28b9329f16f14e187db3f4ec7c3918c5e6c812e1aff782fafe0
- confirmation_v17_g5.json: d60bb5c25a577ed637be49613f0218b6d6c328acd18cd4d725873406342a8b09
- RUN_REPORT_V17_G5.md: ebfb41d8b71c60557223fca93f86bf0a582ee84b798ed37cc64b31a16157c9ce
- manifest self: 0ce50ed637ef6987b915a4a94169400782eb1a4b638c32869c450eb40319bfbc
- freeze self: 9414bbe69deb94c52c3d8641f1469e9c46e0d980d874776eb699235045b0496f
- lock self: 1427b3baa91db5c5206f313d4fe3bd17073b851e6baec46d8d869d517796ef10
- result self: 16d622c8604ad9dd1a79663b129767370e5357ce91db2d2a3477c2f394271c71

局限：这是训练文件内一次固定五成员内部探索，未完成独立提供测试集、全成员跨数据集或 Q1 证据链；绝不能与历史 E16/25 或 G3/G4 不同分区的绝对 MAE 直接排名。坐标空间相关使普通坐标 bootstrap 可能高估有效样本量。连续中位数是固定 256 点、最多64步的 Monte Carlo/Weiszfeld 近似；独立风险也只有256点，不能据小数差值认定连续中位数已达到精确 Bayes 最优。风险估计中分量中位数略低于连续中位数，按原样报告，不重抽种子或增大样本寻优。Student-t 抽样/Weiszfeld 与标准决策理论已在 V9 存在，不属于原创理论。
