# V17-G4 执行摘要

完成：5/5 FIT-only 成员、CAL 自动冻结、一次内部 CONFIRM；最终 check=VERIFIED。科学结论：**NO_INTERNAL_PROMOTION**。

Asia/Shanghai：2026-10-09T13:48:06.744232+08:00 → 2026-10-09T14:16:47.710109+08:00；历时 28.683 分钟。
Python 3.12.12；D:\miniminiconda\envs\liuyan\python.exe；Torch 2.10.0+cu128 / CUDA 12.8；NVIDIA GeForce RTX 2060（6GB）。
NumPy 2.4.4 / pandas 3.0.1 / SciPy 1.17.1 / pytest 9.0.3。

## 冻结输入与分区

只读取 UTS_training.csv。角色均按同一物理坐标的全部扫描分组；三个两两交集为 0，各角色覆盖同一 16 个编码楼层。FIT 的原始物理键与 R0 中心化 float32/四位键一一对应。
门控只使用扫描 RSS、5 成员基线后验、FIT 锚点密度；真实查询位置仅用于离线 MAE/ES。CAL 阈值按坐标组平均支持度的 q25/q50/q75 固定计算，CONFIRM 不重新计算阈值。

| 角色 | 坐标 | 行 | 排序键 SHA256（紧凑 JSON） |
|---|---:|---:|---|
| fit | 945 | 5791 | 470638ca0a79cf8154498dd900014de96af546d311fc32289ad3f777fde43bef |
| calibration | 258 | 1625 | 60593e68e8b87ac284408acc5b4d0f5f6cb1c138677350ccf667aa29788da763 |
| confirmation | 263 | 1692 | dee3633a2b4a4dfedb5054c420fef766e917c3a437303b08c7a7917373bdba93 |

训练 CSV SHA256：a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b
G4 源码 SHA256：e91cf79a3d691e5b932b6387f9ed7150278f53de003dc6a67a19d114854913f3
最终 G4 测试 SHA256：87b0321871fb9e977a04d74bcf568876e8f0e2567f5455f3f2b91be66beafdce
manifest 自哈希：9c430315b50c7457e9af2da2e0cc0a8569cba9e8f03247f9ef4a5e15bf966b31；文件 SHA256：24b5de94db3c2539daef4fb28b808eccf762b795709db8430bd5a967f86699f5
完整算法、训练、评分及设计哈希绑定在 manifest。E5/L192/K192/alpha1，gamma=0.70、10m；原始 R0 模型/损失/优化器/增强及五个固定 epochs 未改。

## 五成员

| 成员 | seed | epoch | checkpoint SHA256 |
|---|---:|---:|---|
| 1 | 17301 | 65 | 299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5 |
| 2 | 17302 | 69 | 3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48 |
| 3 | 17303 | 56 | 4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548 |
| 4 | 17304 | 67 | 695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224 |
| 5 | 17305 | 62 | 0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c |

五名成员均从 FIT 5,791 行/945 锚点完整训练，未重启或复用旧 G3 权重。训练阶段只给原始 R0 loader/loss 传入 rows[fit]；CAL/CONFIRM 标签未进入训练。锚点/原型/原点的一致性由 calibration 的 _load_models 严格逐位核验。五份 .pt 保留本地。

## CAL 与冻结选择

| CAL 方法 | Macro MAE m | Macro ES m | Row MAE m | Row ES m |
|---|---:|---:|---:|---:|
| baseline | 7.155117801 | 5.031860275 | 7.037644008 | 4.978962319 |
| g1 | 7.200592487 | 5.064321730 | 7.035625633 | 5.003299129 |
| gate_q25 | 7.062688248 | 4.997567755 | 6.958962664 | 4.943445170 |
| gate_q50 | 7.115932718 | 4.997876637 | 6.990468456 | 4.952891438 |
| gate_q75 | 7.165083395 | 5.008974288 | 7.015244384 | 4.954409448 |
| placebo | 7.309820310 | 5.079788010 | 7.187614107 | 5.034456561 |

自动冻结：**gate_q25**；阈值 2.180776695410411；满足 CAL 改善≥0.05m、ES 增量≤0.02m 的候选：['gate_q25']。
预声明门控阈值：{"gate_q25": 2.180776695410411, "gate_q50": 2.289001067479451, "gate_q75": 2.3526904756824174}。placebo 不参与选择。
calibration 自哈希：d4a32891a1f059d92ad42998bb4311c651bba6dd3116b42d0b88aabe6676aee4；文件 SHA256：341155ed96ca7a75a12293a7741d0fc1d8d6e7f79a3942b33124e144fab0071a。

## 唯一内部 CONFIRM

| CONFIRM 方法 | Macro MAE m | Macro ES m | Row MAE m | Row ES m |
|---|---:|---:|---:|---:|
| baseline | 6.819208120 | 4.788740078 | 6.606762813 | 4.666061759 |
| g1 | 6.818034172 | 4.848046625 | 6.619545084 | 4.733248647 |
| gate_q25 | 6.688933862 | 4.741959984 | 6.526629175 | 4.632294309 |
| placebo | 6.905431226 | 4.852990106 | 6.667639446 | 4.727080537 |

差值统一为冻结候选−基线：MAE -0.130274259m；ES -0.046780095m。

| CONFIRM 固定条件 | 数值 | 是否通过 |
|---|---:|---|
| 候选非 baseline | gate_q25 | PASS |
| MAE 差≤−0.15m | -0.130274259 | FAIL |
| MAE 同时上界<0 | -0.043945384 | PASS |
| ES 同时上界≤0 | 0.001976138 | FAIL |

原 R3 studentized maxT：5000 次物理坐标配对 resampling，family=MAE+ES，95% 单侧同时上界，seed=19403。不是逐扫描独立重采样。

confirmation 自哈希：f0bd0416fa6d2c7c8665b760a4e8eb687085bdb9c478af4fcb22338ae433a765；文件 SHA256：cb37c9e40327a6e89b89aaec3bf0a49bd31cbc04de826b80676f14af30c8cd2d。
锁 SHA256：b7c56fac705fd9cac978f8e6c1f0300066e8291e09806be90d056bd7cd9b7533；只执行 1 次 confirm，保留锁，未清锁或重评。

## 描述性机制与竞争解释

gain=baseline MAE−global G1 MAE，正值表示 G1 收益。以下 Spearman 仅描述方向，不视为独立显著性或因果。

| 特征 vs G1 gain | CAL Spearman | CONFIRM Spearman |
|---|---:|---:|
| 后验期望 log 支持度 | -0.162907070 | -0.222399817 |
| entropy | -0.046833407 | -0.145172739 |
| visible_aps | 0.043750897 | -0.052492309 |

| 组/对照（MAE） | 改善坐标 | 恶化坐标 | 不变 | 差值中位数 m |
|---|---:|---:|---:|---:|
| CAL/g1−baseline | 116 | 142 | 0 | 0.107035707 |
| CAL/gate_q25−baseline | 39 | 26 | 193 | 0.000000000 |
| CONFIRM/g1−baseline | 129 | 134 | 0 | 0.032995110 |
| CONFIRM/gate_q25−baseline | 53 | 22 | 188 | 0.000000000 |

| CONFIRM 编码楼宇/楼层 | 坐标 | Baseline MAE m | 冻结候选 MAE m | 差 m | Global G1 差 m |
|---|---:|---:|---:|---:|---:|
| B0/F0 | 17 | 5.625459881 | 5.410152471 | -0.215307410 | -0.174745837 |
| B0/F1 | 6 | 7.341746228 | 7.798535637 | 0.456789410 | 0.920102814 |
| B0/F2 | 6 | 6.137628799 | 5.224714093 | -0.912914706 | -0.865160171 |
| B0/F3 | 21 | 7.677181070 | 7.584804107 | -0.092376963 | 0.248719500 |
| B0/F4 | 14 | 7.025789091 | 6.617603221 | -0.408185870 | -0.845280562 |
| B0/F5 | 15 | 5.365166225 | 5.474396837 | 0.109230612 | 0.179675366 |
| B0/F6 | 16 | 5.265003082 | 5.231106925 | -0.033896157 | -0.016950731 |
| B0/F7 | 17 | 6.382781703 | 6.130897971 | -0.251883732 | -0.251455177 |
| B0/F8 | 13 | 7.868839511 | 7.761942576 | -0.106896934 | -0.034943489 |
| B0/F9 | 13 | 8.827977458 | 8.953096714 | 0.125119256 | 0.115001424 |
| B0/F10 | 21 | 5.311574765 | 5.341480789 | 0.029906023 | 0.102140808 |
| B0/F11 | 17 | 7.016303601 | 7.004180569 | -0.012123033 | 0.470857339 |
| B0/F12 | 21 | 7.641757187 | 7.150158460 | -0.491598727 | -0.147816963 |
| B0/F13 | 23 | 7.765959082 | 7.548327153 | -0.217631929 | -0.280625468 |
| B0/F14 | 26 | 6.830200010 | 6.785066812 | -0.045133198 | 0.452332042 |
| B0/F15 | 17 | 7.033998479 | 6.964528747 | -0.069469732 | -0.152850793 |

楼层为适配器词典序编码，不能把编码直接当物理楼层；分层数值仅读取冻结 JSON 的同序坐标向量描述计算，不重新评分、选门槛或追加推断。

## 错误、修复与限制

- pytest_attempt01：43 passed/5 setup errors。新增日志包装器未建 basetemp 父目录；只补 mkdir，保存 PYTEST_TMP_PARENT_FIX.json/.diff。
- pytest_attempt02：47 passed/1 failed。新 G4 测试的错误文本 regex 不符；只改匹配并加强锁后不得调用 prepare 的回归，保存 G4_TEST_LOCK_MESSAGE_FIX.json/.diff。
- 最终 compile_attempt02、pytest_attempt03（48 passed）、preflight 全通过后才 prepare；G4 实验源码未改，manifest 后未改源码/测试/设计。全部原失败 stdout/stderr/hash 留存。
- check_after_confirm_attempt01 因 Markdown 表格行序与 JSON 回读顺序不一致失败；只重排本轮派生报告表格，原报告快照和 REPORT_TABLE_ORDER_FIX.json/.diff 保留。数值、正文、源码/测试/manifest/权重/CAL/CONFIRM/锁均未变；只重跑只读 check。渲染器源码在冻结后未改。
- CUDA：5 个训练日志记录 Memory Efficient attention backward 非确定性警告；固定 seed/CUBLAS 配置不保证逐位重训相同，保存实际 checkpoint 哈希，未更换训练算法。
- 其他非空 stderr：['check_after_confirm_attempt01']；完整内容在对应原始 stderr 日志。
- G1/G2/G3 历史科学代码及原始证据 SHA256 与开始快照一致，G1/G3 STOP 保持；既有权重未加载。命令中的 Python open 轨迹仅见 UTS_training.csv，没有其他提供测试或旧结果输入；这不是 OS 全局访问监控。
- 本轮仅为已公开 G3 历史背景下的 UTS 训练文件内部探索；5 成员共享 FIT 和评估坐标，不是5次独立实验，不能与G3 25成员绝对MAE直接排名。同楼层空间依赖降低有效样本量。
- 后验支持度可能代理熵/原始误差，相关性不能证明稀疏因果；任何内部正结果也不等于独立外部泛化、发表许可或 Q1 完整证据链。不训练下一模型，不读取其他测试，不执行 Git 操作。

## 指定命令退出码与时长

每条同名 .json 保存 UTC 起止、退出码及 stdout/stderr/access SHA256。失败未删除；训练、CAL 和 CONFIRM 未重复。

| 阶段 | 退出码 | 秒数 |
|---|---:|---:|
| compile_attempt01 | 0 | 0.364 |
| pytest_attempt01 | 1 | 16.002 |
| pytest_attempt02 | 1 | 11.465 |
| compile_attempt02 | 0 | 0.272 |
| pytest_attempt03 | 0 | 11.277 |
| preflight_attempt01 | 0 | 10.923 |
| preparation_review_attempt01 | 0 | 11.794 |
| prepare_attempt01 | 0 | 12.005 |
| train_member_01_attempt01 | 0 | 134.552 |
| train_member_02_attempt01 | 0 | 140.509 |
| train_member_03_attempt01 | 0 | 118.085 |
| train_member_04_attempt01 | 0 | 136.169 |
| train_member_05_attempt01 | 0 | 128.089 |
| check_members_attempt01 | 0 | 12.346 |
| calibrate_attempt01 | 0 | 25.589 |
| check_after_calibrate_attempt01 | 0 | 14.205 |
| confirm_attempt01 | 0 | 22.976 |
| check_after_confirm_attempt01 | 1 | 14.169 |
| report_table_order_repair_attempt01 | 0 | 9.903 |
| check_after_confirm_attempt02 | 0 | 14.167 |

训练五命令合计 10.957 分钟；记录 20 条命令、20 份访问轨迹。
