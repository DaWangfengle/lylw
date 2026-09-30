# V18.1 正式执行报告

全部六条注册命令已依次完成。官方 external 命令只执行一次，consumption lock 在首次验证 CSV 读取之前创建。

**EXTERNAL MAE = 7.367951 m**

**EXTERNAL ENERGY SCORE = 5.311939 m**

**EXTERNAL NLL = 5.737838**

MAE、Energy Score、NLL、building/floor error 按完整物理坐标组取 macro；其余不确定性、候选召回、coverage 和半径按注册实现对 scan 求均值。完整数值保留在 [JSON 报告](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/RUN_REPORT_V18_1.json>)。

## GIT

- branch: `v18-1-single-member`
- formal experiment SHA: `e87871d6f5d75c1c3e21e3911b2a3fb0b4a5e781`
- prepare 前代码修复：no；修复 commit：无；正式整轮重启：0。
- 五个预读文件的字节哈希从 prepare 前至报告生成保持一致；源码未修改。

## TESTS

- 三项 py_compile 全部通过：v18_1_single.py、uji_v18_1_experiment.py、test_uji_v18_1_experiment.py。
- 指定 pytest：22 passed，0 failed，11.42 s；无删除、skip/xfail 或放宽测试。
- 合成数据 CUDA 前向/反向/显式 radio 校正 finite，checkpoint 重建预测逐元素完全一致；未读取任何真实数据集。
- 严格 deterministic CUDA：selection、stability、final 均成功；warn_only=False；flash 和 memory-efficient SDPA 关闭，math SDPA 启用；cuDNN benchmark=False、deterministic=True；CUBLAS_WORKSPACE_CONFIG=:4096:8。
- calibration/external 独立进程执行原注册 CLI，使用其 evaluation 默认开关；逐阶段真实 runtime 开关见 COMPUTE 表及 JSON，无额外改动源码。

日志：[py_compile](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/py_compile.log>)；[pytest](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/pytest.log>)；[CUDA preflight](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/cuda_preflight.log>)。

## PROTOCOL

training rows=19937；physical coordinates=933；APs=520；building-floor strata=13。

| role | coordinates | rows |
| --- | --- | --- |
| member_fit | 635 | 13523 |
| early_stop | 111 | 2281 |
| selection_refit | 746 | 15804 |
| calibration | 187 | 4133 |
| full_train | 933 | 19937 |

member_fit ∩ early_stop=∅；两者并集=selection_refit。selection_refit ∩ calibration=∅；两者并集=全 trainingData 坐标。各 role 保留全部 13 个 strata，两个 hull missing 均为 0。registry self-hash、training CSV SHA 和四个 shard SHA 已独立核验。

只有四个物理 shard；full_train.pt 从未生成。final 使用两个已有 shard 在内存中组合完整 training development pool，未重新打开 trainingData.csv。

trainingData SHA256: `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`

| shard | SHA256 |
| --- | --- |
| calibration | `0e4eea5e364c2942f55394e40b43ae8d6119110a631f864252ad1c2166286a4d` |
| early_stop | `96f9bc5ec8ca549c1fa12806e67d2c8c161b37dc0f28ac9a0e2b89d205e0e5bb` |
| member_fit | `8649abeca73393512adac1e6165e015238da1e5f71887f662d759f5e564960fd` |
| selection_refit | `ee6cf897b3441478bb75fc6240690bf24ac0f60f4e61afd77f381160c9105830` |

registry self-hash: `04ce53f37f34d6129c6ef62901619e226591afa659090028a5c4ec2d68824006`

## MODEL

| 属性 | selection | final |
| --- | --- | --- |
| parameters | 626201 | 650137 |
| anchors | 746 | 933 |
| epoch | 18 | 18 |
| tensor finite | yes | yes |

| 冻结合同 | 值 |
| --- | --- |
| residual_experts | 1 |
| scan_feature_mode | bounded_maximal_invariant |
| prototype_metric | none |
| candidate_reranker | True |
| train_candidates | 64 |
| inference_top_k | 64 |
| radio_lambda | 1.0 |
| minimum_epochs | 12 |

单 checkpoint、单 residual expert；leave-self-out V9-style joint geometry loss 为主目标，真实 self anchor 的正 geometry target weight 被构造检查拒绝。训练候选及推理 K 均为 64。

推理顺序：neural retrieval → learned reranker → robust marked-radio energy → affine nuisance profiling → z'=z−E_radio/lambda → Student-t continuous mixture；仅使用 freeze 固定的 T 和 scale。

selection checkpoint SHA256: `9819e688d13b257b95068e2caf128b066bde3528b5db40b829e20d723f70a762`

final checkpoint SHA256: `543d8d532f8f63cf1b446a62cea4251ce8a70505960acaa36e825e748ff31cd5`；链接 freeze `0a3e7fa2bca99a3684dccf8843d6ae5dc1c26285bb115ec2b2005da3514ddbf0`。

## TRAINING

搜索共 26 epochs，只有 epoch≥12 eligible。best eligible Energy Score epoch=18（5.423627 m），注册 0.05 m shortlist 为：

| epoch | MAE (m) | Energy Score (m) |
| --- | --- | --- |
| 12 | 7.328479 | 5.448135 |
| 13 | 7.399448 | 5.442558 |
| 18 | 7.306506 | 5.423627 |

选中 epoch=18：同时是 eligible Energy 最小值和 shortlist MAE 最小值；NLL 未用于主选择标准。selection_refit 与 final 均固定重训 18 epochs。

| loss / diagnostic | search epoch 1→26 | selection refit 1→18 | final 1→18 |
| --- | --- | --- | --- |
| loss | 13.357491 → 8.333890 | 13.090002 → 8.600087 | 12.725311 → 8.307589 |
| joint_geometry_loss | 10.741226 → 7.096270 | 10.521270 → 7.291171 | 10.207533 → 6.979845 |
| retrieval_nll | 3.673668 → 2.113983 | 3.742911 → 2.228596 | 3.847774 → 2.266010 |
| coordinate_nll | 7.067558 → 4.982287 | 6.778359 → 5.062575 | 6.359759 → 4.713835 |
| hierarchy_loss | 0.739230 → 0.037193 | 0.723143 → 0.050319 | 0.691632 → 0.039162 |
| coordinate_loss | 12.008764 → 4.390364 | 11.171295 → 4.617908 | 10.252007 → 4.464506 |
| reranker_loss | 3.032189 → 1.907964 | 3.037123 → 2.007454 | 3.066063 → 2.071481 |
| domain_loss | 2.602339 → 0.910464 | 2.600684 → 0.982698 | 2.526595 → 0.983952 |
| natural_geometry_recall_mass | 0.962547 → 0.999409 | 0.962714 → 0.998933 | 0.961600 → 0.998810 |

selected early-stop 的全部诊断（未做 calibration，T=1、scale=1）：

| 指标 | 值 |
| --- | --- |
| MAE (m) | 7.306506 |
| Energy Score (m) | 5.423627 |
| mixture NLL | 6.290462 |
| building error | 0.005486 |
| floor error | 0.025425 |
| anchor entropy (nats) | 1.692587 |
| effective anchor count | 5.815798 |
| top1 probability | 0.382862 |
| radial scale (m) | 8.883701 |
| within-component radial SD (m) | 6.631009 |
| between-anchor radial SD (m) | 5.643103 |
| posterior radio energy | 0.393208 |
| mean absolute profiled style offset | 0.084222 |
| mean profiled style scale | 0.950782 |
| error/radial-scale correlation | 0.676480 |

| 最近同 building-floor anchor | recall |
| --- | --- |
| recall@8 | 93.512% |
| recall@16 | 98.422% |
| recall@32 | 99.737% |
| recall@64 | 100.000% |

| nominal coverage | actual coverage | 平均预测半径 (m) |
| --- | --- | --- |
| 50% | 40.903% | 5.442667 |
| 80% | 70.145% | 9.343496 |
| 90% | 84.042% | 12.542988 |
| 95% | 91.714% | 15.997505 |

全逐轮训练和 early-stop 指标：[selection history](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/selection_verification.json>)；[final history](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/final_verification.json>)。

## CALIBRATION

posterior temperature=1.25；scale multiplier=1.0；temperature 命中边界=no；scale 命中边界=no。

固定 8×6 网格全部 48 点均已评价，只以 calibration coordinate-macro mixture NLL 选择两个标量；selection backbone/checkpoint SHA 未变。freeze 固定完整配置、epoch、registry/selection/calibration SHA、T 和 scale。

| 指标 | 值 |
| --- | --- |
| MAE (m) | 6.368777 |
| Energy Score (m) | 4.492849 |
| mixture NLL | 5.334452 |
| building error | 0.006557 |
| floor error | 0.016713 |
| anchor entropy (nats) | 1.958782 |
| effective anchor count | 7.625079 |
| top1 probability | 0.301574 |
| radial scale (m) | 9.529463 |
| within-component radial SD (m) | 6.170543 |
| between-anchor radial SD (m) | 7.002299 |
| posterior radio energy | 0.353366 |
| mean absolute profiled style offset | 0.073096 |
| mean profiled style scale | 0.954256 |
| error/radial-scale correlation | 0.814291 |

| 最近同 building-floor anchor | recall |
| --- | --- |
| recall@8 | 94.750% |
| recall@16 | 98.476% |
| recall@32 | 99.347% |
| recall@64 | 99.758% |

| nominal coverage | actual coverage | 平均预测半径 (m) |
| --- | --- | --- |
| 50% | 54.512% | 5.360707 |
| 80% | 83.112% | 9.189260 |
| 90% | 92.814% | 12.575170 |
| 95% | 96.976% | 16.361858 |

完整 calibration surface（行 temperature、列 scale，值为 coordinate-macro NLL）：

| T / scale | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 6.057149 | 5.618113 | 5.479430 | 5.466592 | 5.513780 | 5.683747 |
| 0.75 | 5.817942 | 5.473143 | 5.388467 | 5.406863 | 5.473128 | 5.663280 |
| 1.0 | 5.697137 | 5.401546 | 5.346993 | 5.384166 | 5.462319 | 5.665370 |
| 1.25 | 5.633644 | 5.370517 | 5.334452 | 5.383447 | 5.469335 | 5.680925 |
| 1.5 | 5.601621 | 5.363861 | 5.340965 | 5.398033 | 5.489250 | 5.706770 |
| 2.0 | 5.592091 | 5.394231 | 5.390461 | 5.458504 | 5.556651 | 5.781746 |
| 2.5 | 5.624683 | 5.457065 | 5.467335 | 5.543035 | 5.645734 | 5.875439 |
| 3.0 | 5.677334 | 5.533086 | 5.554148 | 5.635627 | 5.741649 | 5.974474 |

网格缓存 NLL 和最终 evaluation NLL 的微小浮点差异分别原样保留在 JSON。

freeze self-hash: `0a3e7fa2bca99a3684dccf8843d6ae5dc1c26285bb115ec2b2005da3514ddbf0`；calibration result self-hash: `efaf760d77a3809a8d6a4d39a14c575f946808cfeb53002e28d25694c693c39d`。

## STABILITY

五个 seed 均在 member_fit 独立训练 18 epochs，仅在 early_stop 使用冻结 T=1.25、scale=1.00 评价。全部五个结果均保留；no best seed selection / no ensemble / no prediction averaging。

| seed | MAE (m) | Energy (m) | NLL | recall@64 | entropy (nats) | radial scale (m) |
| --- | --- | --- | --- | --- | --- | --- |
| 18311 | 7.235813 | 5.385769 | 6.251019 | 1.000000 | 1.915455 | 10.435546 |
| 18312 | 7.401915 | 5.481671 | 6.304380 | 0.999123 | 1.938994 | 10.602374 |
| 18313 | 7.719624 | 5.715607 | 6.385917 | 0.999562 | 1.937132 | 10.796468 |
| 18314 | 7.656363 | 5.668919 | 6.363215 | 1.000000 | 1.913276 | 10.859762 |
| 18315 | 7.338737 | 5.460051 | 6.315335 | 0.998685 | 1.912564 | 10.839638 |

| 指标 | mean ± sample SD (ddof=1) |
| --- | --- |
| coordinate_macro_mae_m | 7.470491 ± 0.208417 |
| coordinate_macro_energy_score_m | 5.542403 ± 0.142311 |
| coordinate_macro_mixture_nll | 6.323973 ± 0.052815 |
| nearest_same_stratum_anchor_recall_at_64 | 0.999474 ± 0.000572 |
| mean_anchor_entropy_nats | 1.923484 ± 0.013367 |
| mean_radial_scale_m | 10.706758 ± 0.182734 |

每个 seed 的 building/floor error、coverage、候选召回、不确定性与末轮 loss 也完整保留在 [stability result](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/stability_result_v18_1.json>)；未用于改 freeze 或正式 seed。

## EXTERNAL

**EXTERNAL MAE = 7.367951 m**

**EXTERNAL ENERGY SCORE = 5.311939 m**

**EXTERNAL NLL = 5.737838**

external rows=1111；physical coordinates=1052。

| 指标 | 值 |
| --- | --- |
| MAE (m) | 7.367951 |
| Energy Score (m) | 5.311939 |
| mixture NLL | 5.737838 |
| building error | 0.000951 |
| floor error | 0.069973 |
| anchor entropy (nats) | 2.162588 |
| effective anchor count | 9.410673 |
| top1 probability | 0.270455 |
| radial scale (m) | 11.557017 |
| within-component radial SD (m) | 6.581287 |
| between-anchor radial SD (m) | 9.119794 |
| posterior radio energy | 0.716661 |
| mean absolute profiled style offset | 0.104253 |
| mean profiled style scale | 0.941321 |
| error/radial-scale correlation | 0.686063 |

| 最近同 building-floor anchor | recall |
| --- | --- |
| recall@8 | 80.108% |
| recall@16 | 92.889% |
| recall@32 | 97.210% |
| recall@64 | 98.470% |

| nominal coverage | actual coverage | 平均预测半径 (m) |
| --- | --- | --- |
| 50% | 43.924% | 5.611224 |
| 80% | 76.688% | 9.980261 |
| 90% | 88.299% | 14.036177 |
| 95% | 94.779% | 19.104824 |

profiled style offset 按实现记录的是绝对值均值，不能解释为有符号偏移；scale 为无单位 affine scale。

| hash | SHA256 |
| --- | --- |
| external_data_sha256 | `5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0` |
| lock_canonical_self_sha256 | `1b9d008c4dbbf1111a25db8b1dfdc8381b082c63d89399b67649d2e32b73635a` |
| lock_file_sha256 | `4f9952a7555eebeebf16f0f9e995024c8d8930bb591421e075fd7fd2658a68f5` |
| external_result_self_sha256 | `4cc64a970681174b9c292d4625b974392f53585fe4252909b91feaf2e13b5e2d` |
| external_result_file_sha256 | `774c073842c904326992302328fdc501ef7f9b1e5e19e1c6d908dde4a3dd648e` |

lock SHA256 同时列出 canonical JSON self-hash 与实际文件字节 SHA；external result 同时列出 self-hash 与文件 SHA，二者用途不同。

## COMPUTE

GPU: NVIDIA GeForce RTX 2060（6,144 MiB）；Python: 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；解释器: `D:\miniminiconda\envs\liuyan\python.exe`；PyTorch: 2.10.0+cu128；CUDA runtime: 12.8。

selection training（搜索+selection_refit）=342.635 s；final training=169.106 s。

| phase | wall seconds | CUDA peak allocated (MiB) | CUDA peak reserved (MiB) |
| --- | --- | --- | --- |
| prepare | 7.342 | 0.000 | 0.000 |
| train-selection | 347.573 | 449.016 | 744.000 |
| calibrate-freeze | 13.397 | 410.918 | 632.000 |
| stability | 585.765 | 444.236 | 744.000 |
| train-final | 174.099 | 325.550 | 358.000 |
| external | 7.983 | 412.640 | 632.000 |

训练/评价子进程最大 allocator peak：allocated=449.016 MiB；reserved=744.000 MiB。此值来自 PyTorch allocator，未包括驱动/context 和其他进程显存。prepare 无 CUDA allocation。

| phase | strict deterministic | warn_only | flash | memory-efficient | math | cuDNN deterministic |
| --- | --- | --- | --- | --- | --- | --- |
| train-selection | yes | no | no | no | yes | yes |
| calibrate-freeze | no | no | yes | yes | yes | no |
| stability | yes | no | no | no | yes | yes |
| train-final | yes | no | no | no | yes | yes |
| external | no | no | yes | yes | yes | no |

训练三个阶段全部使用严格 deterministic；evaluation 两个阶段如表按原 CLI 默认运行。所有子进程 CUBLAS=:4096:8、cuDNN benchmark=False。

逐阶段命令、起止 UTC、退出码和 source hash check：[run_audit](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/source_snapshot.json>) 以及同目录 *_execution.json / *_runtime.json。

## AUDIT

| 核验项 | yes/no 或计数 |
| --- | --- |
| one inference checkpoint | yes |
| residual_experts == 1 | yes |
| positive self-anchor geometry target used | no |
| ensemble used | no |
| consensus used | no |
| member weighting used | no |
| calibration labels used for gradient training | yes，仅 freeze 后按协议用于 final 完整 development pool |
| calibration labels used before freeze for gradients / epoch selection | no |
| calibration backbone updated | no |
| validation accessed before freeze | no |
| validation accessed before final refit | no |
| external lock written before validation read | yes |
| external command invocation count | 1 |
| external evaluation count | 1 |
| external result used for tuning | no |
| V18.0/V9/V17 outcome 在本轮 external 前被查看 | no |
| source / architecture / registered config changed after prepare | no |
| formal run restart | 0 |
| stability best seed selection / ensemble | no |
| train-final reopened trainingData.csv | no |
| full_train.pt generated | no |

calibration 标签的梯度使用按阶段明确区分：selection/epoch selection/calibration-freeze 阶段没有用其更新网络；冻结后 train-final 的注册协议明确要求将 calibration 与 selection_refit 合并进行梯度训练，因此全流程此项如实为 yes。

唯一 external 命令内的 CSV 打开次数=2（加载与 SHA 计算），两次均发生在 lock 已存在后；这不是多次 external 评价。首次读取 audit time_ns=1790752908293648600，lock 写入 open time_ns=1790752908292070100。

本轮未重新打开历史 V18.0/V9/V17 outcome；没有用此前对话里已知的信息选择或修改本轮配置。external 前 13 项 gate 全部通过：[pre_external_gate](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_1/run_audit/pre_external_gate.json>)。

| phase | training CSV read opens | validation CSV read opens | denied |
| --- | --- | --- | --- |
| prepare | 2 | 0 | 0 |
| train-selection | 0 | 0 | 0 |
| calibrate-freeze | 0 | 0 | 0 |
| stability | 0 | 0 | 0 |
| train-final | 0 | 0 | 0 |
| external | 0 | 2 | 0 |

prepare 后额外一次 training SHA 独立核验发生在 verify-prepare；之后任何训练阶段均未重新打开训练 CSV。独立文件 I/O 记录：同目录 *_io.jsonl。

## INTERPRETATION

1. leave-self-out geometry 优化正常：epoch 搜索及两个固定重训的 joint geometry、retrieval、coordinate 和 reranker loss 均下降；内部留出指标有波动，已按注册 early stopping 和 shortlist 规则停止。训练损失下降不等同于证明外部泛化。
2. 自然 geometry target recall mass：搜索末轮 0.999409，selection refit 末轮 0.998933，final 末轮 0.998810。外部最近同层 anchor recall@64=0.984698；自然 geometry 邻居质量与推理最近 anchor recall 是两种不同诊断。
3. explicit radio correction 全流程 finite；合成 CUDA 检查验证 z'=z-E/lambda，外部平均 posterior radio energy=0.716661。这证明数值有效，不等同于 radio 模块的因果增益证明，当前协议没有额外 ablation。
4. calibration 在 48 个注册网格点中选出 T=1.25, s=1.0；temperature boundary=False, scale boundary=False。
5. 外部 50/80/90/95% 区间实际覆盖率分别为 43.924/76.688/88.299/94.779%；50–90% 区间显示 over-confidence，95% 接近 nominal。calibration 的相应覆盖率为 54.512/83.112/92.814/96.976%，偏保守。这是不同数据分布及不同 anchor pool 的描述性比较，没有追加置信度检验或根据外部结果重新校准。
6. 外部平均 within-component radial SD=6.581287 m，between-anchor radial SD=9.119794 m；between-anchor 项较大。这里比较的是各 scan SD 的均值，不把两个均值直接平方相加当作精确方差分解。
7. 值得设计 V18.2：五种子内部 MAE sample SD=0.208 m，当前主要可研究的问题是外部低/中 coverage 区间偏窄、recall@8=80.108% 与 floor error=6.997%。这些诊断不能单独确定原因；新的实验需另行预注册并使用新的未消费确认方案。当前 V18.1 的参数、seed、freeze 和结果均不修改、不重跑。

| nominal | observed | 差值 (percentage points) | 方向 |
| --- | --- | --- | --- |
| 50% | 43.924% | -6.076 | over-confidence |
| 80% | 76.688% | -3.312 | over-confidence |
| 90% | 88.299% | -1.701 | over-confidence |
| 95% | 94.779% | -0.221 | over-confidence |

以上是当前冻结实验的描述性诊断；没有追加训练、额外 external 评价或参数调整。

报告生成 UTC: 2026-09-30T07:22:58.172517+00:00
