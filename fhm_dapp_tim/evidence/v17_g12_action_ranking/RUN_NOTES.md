# V17-G12 本地执行说明

执行完成：FROZEN_T06_ACTION_AND_RANK_DIAGNOSIS；最终 check=VERIFIED；独立逐站点复核通过。75 项指定 CPU 回归测试全部通过（7.28 s）；一次正式 CUDA run，未训练、未选择新方法、未执行任何 Git 命令。

运行区间 2026-10-10T16:58:40.003146+08:00 至 2026-10-10T17:12:59.281367+08:00；审计总历时 859.278 s，正式 run 45.756 s。Python D:\miniminiconda\envs\liuyan\python.exe，Python 3.12.12 / Torch 2.10.0+cu128 / CUDA build 12.8 / NVIDIA GeForce RTX 2060，显存 6442123264 bytes。

## 顺序与异常

| 命令阶段 | 退出码 | 秒 |
|---|---:|---:|---:|
| compile-initial | 0 | 0.109 |
| tests-initial | 0 | 8.869 |
| preflight-initial | 0 | 9.387 |
| fit-only-cuda-and-identity-smoke | 1 | 6.258 |
| fit-only-cuda-and-identity-smoke-corrected-helper | 0 | 13.948 |
| prepare | 0 | 9.113 |
| check-before-run | 0 | 9.271 |
| run-once | 0 | 45.756 |
| check-after-run | 0 | 9.118 |
| post-run-independent-review | 0 | 8.000 |

七个指定科学命令顺序完整执行。pytest 仅附加本轮独立 basetemp，避免触碰历史测试缓存。正式 run 首先写不可逆锁，之后覆盖全部五个原角色；锁未删除，正式 run 未重试。
补充审计助手误用不存在的 model.anchor_rss 字段，正确字段为 model.prototype_rss；同时在重试前按实际签名纠正 FIT 映射助手为 (fit_rows,data,ref,origin) 并解包 (row_anchor_ids,sorted_keys)。仅修改临时内存审计脚本。没有修改 G12 科学源码、测试或设计；没有发生科学性修复。原始 traceback、失败退出码、前后助手脚本和全部 stdout/stderr 保存在两份日志。另一次说明脚本的 JavaScript 引号构造错误在 Python/文件操作执行前被拦截，已纠正并记入 stderr。
修正后的补充检查仅使用各域前 16 条原 FIT 扫描；全部 960 分量动作/风险及六个排序地图均通过 RTX2060 烟雾测试。模型 state_dict 逐位不变、parameter.grad 全 None，禁止创建 optimizer。峰值 CUDA 分配：TUJI 83,373,056 / UTS 106,967,552 / Tampere 65,984,000 bytes（仅烟雾检查，非正式 run 的峰值）。所有正式命令无 stderr 警告；无 OOM、CUDA 确定性警告或失败。

## 数据与身份

仅解析 TUJI TRAIN 变量、UTS/Tampere TRAIN 文件；文件打开守卫在补充 FIT 检查中只观察到八个许可训练容器。五次 MAT 加载均限定 variable_names=('results_train',)，返回值无 results_test。物理 MAT 容器包含测试变量，容器被打开和 SHA256 哈希；没有解码或建模 results_test，不能称整个 MAT 测试字节隔离。未访问任何提供的 test、UJI official/outer。旧评分 JSON 在冻结 G12 结果后才解析作身份复核，不参与动作或参数选择；此前历史文件只做字节哈希保护。
原角色物理位置两两无重叠，FIT 键到 checkpoint 全局 anchor ID 一一映射并遵守原 0.03m 限制；三域 FIT 分别 643/3199、945/5791、334/334（位置/行）。各 E5 面板共享原 FIT 原型、AP mask、建筑/楼层/坐标原点。

| 原 TRAIN-internal 角色 | 物理位置 / RSS 行 | 物理键 SHA256 |
|---|---:|---|
| tuji/calibration | 137 / 695 | e1ad91fd38a086e445426f502bf24d5a3224823e48f46e6f99ea486c31c32e22 |
| tuji/confirmation | 139 / 701 | 1c46af2f1336fcc8e664f7c214ef876418a1c288df942db9d6f517e88d22542c |
| uts/calibration | 258 / 1625 | 60593e68e8b87ac284408acc5b4d0f5f6cb1c138677350ccf667aa29788da763 |
| uts/confirmation | 263 / 1692 | dee3633a2b4a4dfedb5054c420fef766e917c3a437303b08c7a7917373bdba93 |
| tampere/confirmation | 112 / 112 | 905866e37596155ee6131b271686a69ce3e0692f9d67d61b978d2c109e411b67 |

合计 909 位置 / 4825 行。每位置先平均其 RSS 行误差，再对位置等权；分位数及 >=10m/20m 计数均针对位置平均误差。
399 个受保护的既有源码、测试、历史证据及 runbook 文件逐字节未变（范围不含 runtime cache/run_audit）；15 个原 checkpoint SHA256 未变。保护清单规范 SHA256 ce34c7cb459d63630c35d2990fef631b0b8e72a5c883060c98dc9931cf519ce2。完整清单在 stdout 开头；manifest 保存原始面板和角色身份。
成员身份（各域顺序均 seed 17301..17305，epoch 65/69/56/67/62）：

- fhm_dapp_tim/models/v17_g4_observable_support/member_01_seed17301.pt — 299b5955967acc333fccef45d6a83ca45af4a6e073bf64a909ba9889dabb4fb5
- fhm_dapp_tim/models/v17_g4_observable_support/member_02_seed17302.pt — 3bf3db6c88d3189529623106af2e1463e44b17408965eeecd98a739efa8d9b48
- fhm_dapp_tim/models/v17_g4_observable_support/member_03_seed17303.pt — 4e56c499c837a40d45c9a40b984ec12eb7bc9f10dd8db5ae2b549103d05e7548
- fhm_dapp_tim/models/v17_g4_observable_support/member_04_seed17304.pt — 695b0afbc0c282f20f6f7f41d385b842b169ef366c75f0c010e5f27f24bd1224
- fhm_dapp_tim/models/v17_g4_observable_support/member_05_seed17305.pt — 0067c6a6f4d88c5dc49a9159d26304802be14e31f6cbcf23b91c29555c5ef25c
- fhm_dapp_tim/models/v17_g5_bayes_action/member_01_seed17301.pt — 8af93392f81c8cc71c7e44664da8aa3c62356f1eb8e067feff072dba3e67e60b
- fhm_dapp_tim/models/v17_g5_bayes_action/member_02_seed17302.pt — 37c8af1d611586456e6200d3fe196f111f4e93249613d4997ccdef6f9541e294
- fhm_dapp_tim/models/v17_g5_bayes_action/member_03_seed17303.pt — c1e173049c37ef26858ada993f8d3fa62ea150ff55b7c6ccd34bd31eb0b04f2d
- fhm_dapp_tim/models/v17_g5_bayes_action/member_04_seed17304.pt — 0f53c19750169c95327f501172e2da60a690b08dcf19db677a78365b848280f1
- fhm_dapp_tim/models/v17_g5_bayes_action/member_05_seed17305.pt — ccd93aee342b27b67b3f971f157286886d7090556bd28cdbdb4af82f350ef3b5
- fhm_dapp_tim/models/v17_g6_distribution_calibration/member_01_seed17301.pt — 20415c513976bbdd2e6aaf911c284d5a3aca271b037c41b859dce7cf89195f2a
- fhm_dapp_tim/models/v17_g6_distribution_calibration/member_02_seed17302.pt — febeb7b185cee8b05e748616d99342f352b77d6bd0a8a0f758ad3c5a484b44e8
- fhm_dapp_tim/models/v17_g6_distribution_calibration/member_03_seed17303.pt — 73ac17a495574be6ea80989982463083ab4db58b5f87f521fed82846a53ecf0a
- fhm_dapp_tim/models/v17_g6_distribution_calibration/member_04_seed17304.pt — 58bd77a131b59dae28d9e0d513055746383f6f7edf973650fa155cc94ff7611b
- fhm_dapp_tim/models/v17_g6_distribution_calibration/member_05_seed17305.pt — 124626cbf93dd9b834a5325b61c6ff7735ed89254c9a16fae174d0a9bd79d26c

训练输入 SHA256：

- data/UTSIndoorLoc/UTS_training.csv — a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b
- data/Tampere/Training_rss.csv — 7922efcd4ea14b1a5453a69ba35bc021a67b915b1b3076fa4ccf6da3e98e31b3
- data/Tampere/Training_coordinates.csv — 22f0894e80966ef12d5f4e73f6e0b466d5a5781dda1f76c35ae6d276a929afff
- data/TUJI1/Formatted_data_A12.mat — 73c59c6a8a12c45c4b6bf9b93b45cb89d0cfb7bb147260dc763f0d5f5af314a5
- data/TUJI1/Formatted_data_POCO.mat — 9973384ac04365a55a2e6b33b16b61724cf41869e828b1544c7c4bc61c9d2495
- data/TUJI1/Formatted_data_S20.mat — e3e327243852ba38087c4e8f07c4d94c7b8acc0987e961d5dc6e8fcf23dd354e
- data/TUJI1/Formatted_data_S7.mat — c33ac65f9b61f6b24272b256b091c54747d5bce3f186a8a96e83cbc3e9f9fdab
- data/TUJI1/Formatted_data_tabS7.mat — f8a9bf0a94367f2255a3c94b2bd4b633d73d7000a8f1a0cf5c25e90d231b0e23

## 动作与尾部结果

同一 E5/L192/K192/alpha1/df4/Anchor T=.6 后验；两种中位数复用原 G5/V9 算法，64 迭代、tol=1e-5m；动作抽样 seed19801、独立风险 seed19802 均 128+128；ES seed19602 为 64+64，仅共算一次。未查看真值选动作、阈值、抽样数量或停止条件。

| 角色 | 均值 MAE | 分量中位数 MAE | 连续中位数 MAE | 共用 ES | 连续风险 Δ / 实际 MAE Δ |
|---|---:|---:|---:|---:|---:|
| tuji/calibration | 2.390619 | 2.229682 | 2.245287 | 1.713443 | -0.032696 / -0.145332 |
| tuji/confirmation | 2.657774 | 2.502682 | 2.521668 | 1.867255 | -0.031145 / -0.136105 |
| uts/calibration | 5.380266 | 5.457483 | 5.446014 | 4.123892 | -0.370583 / +0.065748 |
| uts/confirmation | 5.107529 | 5.206837 | 5.166874 | 3.937965 | -0.365299 / +0.059345 |
| tampere/confirmation | 5.987721 | 7.015508 | 6.945454 | 4.996576 | -0.490126 / +0.957733 |

量均为 m；预测模型自身的欧氏风险不是实际定位误差。五角色原均值 MAE、共用 ES 的全部物理键及逐位置向量与 G8 Anchor-only 完全相同（绝对差为 0）；原型/R3 Top1 命中向量与 G11 完全相同。共同 ES 不构成中位数的概率预测改进。

| CONFIRM 域 | P95（均值 / 分量 / 连续） | 最差5%平均（均值 / 分量 / 连续） | 最大误差（均值 / 分量 / 连续） | >=10m / >=20m（三动作顺序） |
|---|---:|---:|---:|---|
| tuji | 4.466831 / 4.074676 / 4.153276 | 4.942760 / 4.816928 / 4.920907 | 5.934245 / 5.945197 / 5.795598 | 0 / 0; 0 / 0; 0 / 0 |
| uts | 12.221708 / 12.545960 / 12.474760 | 14.689228 / 15.363451 / 15.392964 | 20.059251 / 19.524779 / 20.046837 | 25 / 1; 29 / 0; 29 / 1 |
| tampere | 10.468628 / 10.877775 / 10.825699 | 27.308503 / 31.426522 / 31.315579 | 60.981600 / 70.856273 / 70.802494 | 8 / 2; 8 / 2; 8 / 2 |

TUJI 的中位数宏 MAE/P95 改善，但连续中位数 P99 由 5.050792 增至 5.255721m；不能声称全部尾部都改善。UTS 两角色的中位数自身风险下降而真实宏 MAE 上升；CONFIRM 连续中位数 P95、最差5%均变差。Tampere 两种中位数使宏 MAE、P95/P99、最差5%与最大误差恶化。
连续中位数出现“自身风险下降而实际误差变差”的站点：TUJI CAL/CONFIRM 36/33，UTS CAL/CONFIRM 129/119，Tampere CONFIRM 77。两次有限抽样给出风险近似，这种方向失配提示后验失配，不能视为唯一因果证明。

## 排序与预指定困难位置

下表为每位置内 RSS Top1 命中比例，再位置等权宏平均；百分数对应10m内同建筑/楼层，不能作为部署门控。

| 角色 | 原型 | 神经 | 融合等权 | 融合径向 | Top-L径向 | R3共识 |
|---|---:|---:|---:|---:|---:|---:|
| tuji/calibration | 91.2409% | 99.0268% | 99.4161% | 99.4161% | 99.4161% | 99.4161% |
| tuji/confirmation | 92.2782% | 98.6331% | 98.4652% | 98.4652% | 98.4652% | 98.4652% |
| uts/calibration | 81.7971% | 76.9008% | 79.4043% | 80.9547% | 80.9547% | 80.9547% |
| uts/confirmation | 82.8514% | 75.3183% | 81.3009% | 80.5405% | 80.4137% | 80.4137% |
| tampere/confirmation | 91.9643% | 83.9286% | 91.9643% | 91.0714% | 91.0714% | 91.0714% |

原型与神经是平行评分对照，彼此转移不是因果串行步骤。站点转换采用固定“行命中比例 >=0.5”标准，wrong 表示不占多数 NEAR；不等价于每一行都在 >=20m 的明确错误区。排名针对全部 FIT ID，零概率并列按全局 ID 稳定排序；物理近邻不存在时为 null，站点 rank 仅平均有效行。
UTS CAL 的 16 个“原型多数近、R3 多数不近”位置，神经/融合等权/融合径向/Top-L/共识多数近计数均 0；UTS CONFIRM 对应16个的计数是 1/1/0/0/0。该互补性事后复现，没有用真值选择预测算法。UTS CONFIRM 神经到等权融合挽回18站、失去2站；等权到径向失去2站、挽回0。Top-L 后未产生多数站点转换，但扫描命中宏比例小幅变化，不能称截断完全无影响。所有五角色 Top-L→共识多数站点挽回/损失均0；近邻质量和排名仍必须结合完整向量分析。
Tampere 神经到等权融合挽回10站、失去1站；随后径向加权失去1站。预指定两处巨大错误在六张地图中 Top1 全部不近、最佳正确邻域锚点名次均2：

| building/floor/x/y | 均值 MAE | 分量中位数 MAE | 连续中位数 MAE | 连续行动偏移 | 连续模型风险（原均值 → 中位数） |
|---|---:|---:|---:|---:|---:|
| [0, 1, 262.94, 107.04] | 60.981600 | 67.466167 | 67.277966 | 6.417174 | 11.583338 → 6.939621 |
| [0, 1, 262.94, 112.04] | 50.585653 | 70.856273 | 70.802494 | 20.241068 | 27.642779 → 20.054152 |

两处都有正确 FIT 近邻且名次2，模型自身风险下降却进一步远离真值；单换欧氏点动作不能修复这些错误空间模式。三域不存在稳定的真实 MAE/尾部联合改善；没有选择新动作，也没有正式晋级。历史标签早已观察：本轮仅为回顾性机制诊断，不能证明独立外部泛化、统计显著性或 JCR Q1 可发表创新。几何中位数与采样均为已有算法；空间相关、设备异质性、有限 Monte Carlo/Weiszfeld 近似限制保留。

## 冻结与日志 SHA256

- V17_G12_CODEX_RUN.txt — 53dba736d0f10b30f10df7057889b389aff9883c6e5e8f237e79f25836e9b91d（科学源码/测试/设计/runbook 无修改）
- fhm_dapp_tim/evidence/v17_g12_action_ranking/DESIGN_V17_G12.md — 1f7e93fdf94ac383317c62573886f3664c2b27daaef4eb492d89b987a0e7ded5（科学源码/测试/设计/runbook 无修改）
- fhm_dapp_tim/fhm_dapp/uji_v17_g12_action_ranking.py — d3ad96e17e9d47841f403ae8bcca6595851b9509e41a9444cace69e67c88912c（科学源码/测试/设计/runbook 无修改）
- tests/test_uji_v17_g12_action_ranking.py — 3becf787c03bd732c38058c25b685a012a3dcee1c9fee9a732819929fd871e57（科学源码/测试/设计/runbook 无修改）
- manifest_v17_g12.json — 552ff668c1483104de8b77acd08531f03980e2476189513337fc731f25e032ae
- run_v17_g12.lock — 8a235e128c8433a4839b737430f2b3d9c8594c03b556549199e65c97710c96a9
- diagnosis_v17_g12.json — 6816c2c60227a6a5deb4323c5504a50497f593238ef6faefd8950511bd6fb72a
- RUN_REPORT_V17_G12.md — 78ac1833bac3a5c0c3e3b39aebbd8c2962a1a08114a914fb8084f3ec33126279
- RUN_STDOUT.log — 1509fc376c27b35b1a00cac8ec65b11ada1bda15ed932b09c7ff0d796c44a87c
- RUN_STDERR.log — bd8fce5b470de2844dd2f46ab6c2e66f2cb26335d28ffc5d0c26d1129265b8cc
- manifest 规范自身哈希 d54c37dd1f82f841c42ddcee36fecd54844a80fec5315207d1ff61c039d0c6bb；result 规范自身哈希 30e661aa6784889d9315b318fa5f40020553d8b75891123f5b2f6f71b5a85ffc。
- 冻结后对照的 G8/G11 结果文件哈希 036bedc28798330d1cbd75b2a3de6db60e2d12d99a79d05d82c2e77355862537 / 3fdd9908541b1c2c82c4d5ea548c3fc4ffd172ee96349b5e41ee6656684419fb；完整来源 hashes 与逐位置复核输出保存在 manifest/stdout。

全部正式向量、配对差、风险方向、行动偏移、median/P90/P95/P99、严重尾部、六阶段质量/排名与转换计数已保存；JSON 无非标准 NaN/Inf。原报告与 JSON 冻结后未覆写，锁保留，15 份 .pt 留在原位置。本说明为另外新写的执行与解释文件。
