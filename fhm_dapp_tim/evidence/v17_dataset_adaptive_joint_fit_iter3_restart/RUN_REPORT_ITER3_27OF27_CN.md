# ITER3 三域 27/27 实测与续跑报告

本报告从原程序完成的 CAL-fit 选择、CAL-eval 站点向量和独立只读核算生成。TUJI 9/9、UTS 9/9、UJI 9/9，原生汇总 PASS；不是只包含最后一个数据域的报告。

本轮为已观察训练语料内的校准/机制比较。CAL-fit 选参，CAL-eval 只报告；internal TEST 与 official TEST 评分数均为 0，没有训练或更新 checkpoint。CAL-fit 不退化约束不能保证 CAL-eval 不退化；两者必须区分。

## 执行、环境与版本

Python `D:\miniminiconda\envs\liuyan\python.exe`；3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；Torch 2.10.0+cu128；CUDA 12.8；NVIDIA GeForce RTX 2060，6GB。

UTS 完成并通过只读核算后才启动 UJI，两域没有并行。下表为真实日志元数据；首次 UTS 失败不计为成功实验，也没有删除。

| 步骤 | 退出码 | 耗时秒 | 开始 | 结束 |
|---|---|---|---|---|
| UTS failed before repair | 1 | 170.541 | 2026-10-23T12:51:27.527232+08:00 | 2026-10-23T12:54:18.068291+08:00 |
| UTS authorized restart | 0 | 1748.812 | 2026-10-23T13:56:08.175347+08:00 | 2026-10-23T14:25:16.987780+08:00 |
| UTS 18-run audit | 0 | 34.567 | 2026-10-23T14:25:24.867505+08:00 | 2026-10-23T14:25:59.434425+08:00 |
| UJI | 0 | 2525.576 | 2026-10-23T14:26:13.687616+08:00 | 2026-10-23T15:08:19.263157+08:00 |
| Final 27-run audit | 0 | 52.695 | 2026-10-23T15:08:27.129746+08:00 | 2026-10-23T15:09:19.824570+08:00 |

**UTS failed before repair**：`D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uts --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart --device cuda`

stdout：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_UTS.stdout.log`；stderr：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_UTS.stderr.log`。

**UTS authorized restart**：`D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uts --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart --device cuda`

stdout：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_15_run_UTS.stdout.log`；stderr：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_15_run_UTS.stderr.log`。

**UTS 18-run audit**：`D:\miniminiconda\envs\liuyan\python.exe tmp/iter3_continuation_readonly_check.py tuji uts`

stdout：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_16_audit_UTS.stdout.log`；stderr：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_16_audit_UTS.stderr.log`。

**UJI**：`D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uji --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart --device cuda`

stdout：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_17_run_UJI.stdout.log`；stderr：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_17_run_UJI.stderr.log`。

**Final 27-run audit**：`D:\miniminiconda\envs\liuyan\python.exe tmp/iter3_continuation_readonly_check.py tuji uts uji`

stdout：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_18_audit_ALL.stdout.log`；stderr：`fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\numerical_revision_20261023_133252_18_audit_ALL.stderr.log`。

原 TUJI 核心源码 SHA256：`2895555096b253f66a7eccf445510dccc5de027e0429960bdae337bf826fe498`。

授权修订后 UTS/UJI 核心源码 SHA256：`89a59244a52c47c6cb53938301f59962c4c4e0015f43c7d6fb5c64430884f0ca`。

重要版本限制：TUJI 保留原完成的 9 组结果，UTS/UJI 使用授权数值修订。27/27 是产物完整性，不能称为 27 次逐字节相同源码执行。所有科学常量、参数池、模型、优化目标、选择规则、数据划分和门禁保持原定义。

## 数值故障及修复

首次 UTS 17301/C 的 clean MAE 比参考多 1.430511474609375e-6m，触发原 1e-6m 门禁。独立回放证明根因是候选批量求和、附加零概率列以及宏平均的浮点归约顺序；与此前 TUJI 距离截断的 float32/float64 问题不同。

授权修复统一 beta=0 的物理 softmax、点乘求和及按站点宏平均路径，并将参考 BF accuracy 恢复为导出站点计数比例的双精度汇总。没有抄写基准 MAE、删断言或放宽门禁。修复过程中的夹具错误、残余数值差、被原测试拦截的准确率汇总尝试和基准舍入问题均保留。

最终编译通过、38 项测试通过；UTS/UJI 18 个 checkpoint × 12 几何配置，216 个 clean 对照 MAE 差全为 0，BF accuracy/BF-CE 门禁均通过。完整失败与修复版本见 `run_logs/numerical_revision_20261023_133252/FIX_RECORD_CN.md`、`APPLIED_NUMERICAL_FIX_FINAL.patch`、`REVISION_FREEZE.json`；独立根因证据见 `run_logs/diagnosis_clean_anchor_20261023_130525/CAUSE_REPORT_CN.md`。

## 分域完整指标

每行是同域 9 个冻结 checkpoint（3 seeds × C/D/E）的指标等权描述均值。同一域反复使用同一 CAL-eval 站点，这些均值及 checkpoint 计数不构成独立样本显著性检验。Transport 列为原程序导出的 transport risk（W1 项），保留其原定义。P95/P99/worst 是每 checkpoint 的站点尾部指标再求均值，不能解释为合并站点分布的分位数或全局最大。

| 域 | 方法 | MAE m | ES m | Transport m | Joint NLL | BF-CE | P95 m | P99 m | Worst m | BF acc |
|---|---|---|---|---|---|---|---|---|---|---|
| tuji | clean | 1.824964 | 1.292146 | 2.521923 | 3.848854 | 0.000001 | 2.741658 | 3.232489 | 3.415006 | 1.000000 |
| tuji | joint λ=.02 | 1.831356 | 1.312338 | 2.404696 | 3.861187 | 0.000000 | 2.804618 | 3.317988 | 3.541508 | 1.000000 |
| tuji | joint λ=.05 | 1.831665 | 1.307388 | 2.404034 | 3.843005 | 0.000001 | 2.805586 | 3.318914 | 3.541764 | 1.000000 |
| tuji | 旧 1D+2A | 1.859586 | 1.297667 | 2.518084 | 3.422627 | 0.000004 | 2.838151 | 3.278285 | 3.475808 | 1.000000 |
| uts | clean | 3.988478 | 3.333806 | 5.109560 | 10.855994 | 0.024996 | 9.155295 | 15.185089 | 17.890753 | 0.996779 |
| uts | joint λ=.02 | 4.056471 | 3.169922 | 4.947731 | 6.738104 | 0.017433 | 9.093190 | 15.900778 | 18.874794 | 0.998390 |
| uts | joint λ=.05 | 4.058809 | 3.160391 | 4.950500 | 6.581953 | 0.017495 | 9.087113 | 15.911106 | 18.861585 | 0.998390 |
| uts | 旧 1D+2A | 4.166438 | 3.012767 | 5.553142 | 5.755204 | 0.028289 | 8.509475 | 15.118652 | 17.061840 | 0.996779 |
| uji | clean | 7.309175 | 6.209477 | 9.447708 | 14.488800 | 0.122815 | 16.610555 | 22.082089 | 24.890337 | 0.963182 |
| uji | joint λ=.02 | 7.245675 | 5.593350 | 8.955948 | 10.651403 | 0.124509 | 16.078447 | 21.334774 | 23.771562 | 0.966240 |
| uji | joint λ=.05 | 7.245675 | 5.593350 | 8.955948 | 10.651403 | 0.124509 | 16.078447 | 21.334774 | 23.771562 | 0.966240 |
| uji | 旧 1D+2A | 7.413540 | 5.497706 | 9.875378 | 10.426175 | 0.101919 | 15.824936 | 22.043207 | 25.038456 | 0.968569 |

## 主要对照相对 clean 的变化

负 Δ MAE/ES/Transport/NLL/尾部表示下降。BF accuracy 越高越好；BF-CE 越低越好。

| 域 | λ | ΔMAE | ΔES | ΔTransport | ΔNLL | ΔBF-CE | ΔP95 | ΔP99 | ΔWorst | ΔBF acc | MAE 改/坏/同 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tuji | 0.02 | +0.006392 | +0.020192 | -0.117227 | +0.012333 | -0.000000 | +0.062960 | +0.085499 | +0.126502 | +0.000000 | 3/6/0 |
| tuji | 0.05 | +0.006701 | +0.015242 | -0.117889 | -0.005849 | +0.000000 | +0.063928 | +0.086425 | +0.126758 | +0.000000 | 2/7/0 |
| uts | 0.02 | +0.067993 | -0.163884 | -0.161829 | -4.117889 | -0.007563 | -0.062105 | +0.715690 | +0.984040 | +0.001610 | 0/9/0 |
| uts | 0.05 | +0.070331 | -0.173415 | -0.159059 | -4.274041 | -0.007501 | -0.068183 | +0.726017 | +0.970832 | +0.001610 | 0/9/0 |
| uji | 0.02 | -0.063500 | -0.616126 | -0.491759 | -3.837397 | +0.001693 | -0.532108 | -0.747315 | -1.118775 | +0.003058 | 8/1/0 |
| uji | 0.05 | -0.063500 | -0.616126 | -0.491759 | -3.837397 | +0.001693 | -0.532108 | -0.747315 | -1.118775 | +0.003058 | 8/1/0 |

旧组合相对 clean 的 checkpoint MAE 计数（严格差值符号，不加事后阈值）：


| 域 | 旧组合可用数 /9 | 旧组合 ΔMAE 均值 | 改善 | 恶化 | 不变 |
|---|---|---|---|---|---|
| tuji | 9 | +0.034622 | 0 | 9 | 0 |
| uts | 9 | +0.177960 | 0 | 9 | 0 |
| uji | 9 | +0.104365 | 3 | 6 | 0 |

## 分骨干（3 seeds 均值）


| 域 | 骨干 | λ | clean MAE | ΔMAE | ΔES | ΔTransport |
|---|---|---|---|---|---|---|
| tuji | C | 0.02 | 1.865847 | +0.010930 | +0.029276 | -0.142872 |
| tuji | C | 0.05 | 1.865847 | +0.010930 | +0.029276 | -0.142872 |
| tuji | D | 0.02 | 1.814602 | +0.002365 | +0.003870 | -0.084229 |
| tuji | D | 0.05 | 1.814602 | +0.002853 | -0.004263 | -0.085171 |
| tuji | E | 0.02 | 1.794441 | +0.005882 | +0.027429 | -0.124580 |
| tuji | E | 0.05 | 1.794441 | +0.006320 | +0.020712 | -0.125625 |
| uts | C | 0.02 | 3.984811 | +0.035211 | -0.222967 | -0.109312 |
| uts | C | 0.05 | 3.984811 | +0.037536 | -0.239871 | -0.105085 |
| uts | D | 0.02 | 4.012157 | +0.084747 | -0.207374 | -0.190861 |
| uts | D | 0.05 | 4.012157 | +0.084747 | -0.207374 | -0.190861 |
| uts | E | 0.02 | 3.968466 | +0.084021 | -0.061311 | -0.185312 |
| uts | E | 0.05 | 3.968466 | +0.088709 | -0.073000 | -0.181231 |
| uji | C | 0.02 | 7.407989 | +0.019725 | -0.554703 | -0.484087 |
| uji | C | 0.05 | 7.407989 | +0.019725 | -0.554703 | -0.484087 |
| uji | D | 0.02 | 7.106555 | -0.092743 | -0.592760 | -0.616937 |
| uji | D | 0.05 | 7.106555 | -0.092743 | -0.592760 | -0.616937 |
| uji | E | 0.02 | 7.412981 | -0.117483 | -0.700915 | -0.374254 |
| uji | E | 0.05 | 7.412981 | -0.117483 | -0.700915 | -0.374254 |

## 实测判读

TUJI λ=0.02：平均 ΔMAE=+0.006392m，ΔES=+0.020192m，ΔP99=+0.085499m，Δworst=+0.126502m；MAE 严格改善/恶化/不变 checkpoint 为 3/6/0。这组结果不支持同时改善 MAE 与 ES 均值。尾部存在代价，不能称为可靠性已解决。

TUJI λ=0.05：平均 ΔMAE=+0.006701m，ΔES=+0.015242m，ΔP99=+0.086425m，Δworst=+0.126758m；MAE 严格改善/恶化/不变 checkpoint 为 2/7/0。这组结果不支持同时改善 MAE 与 ES 均值。尾部存在代价，不能称为可靠性已解决。

UTS λ=0.02：平均 ΔMAE=+0.067993m，ΔES=-0.163884m，ΔP99=+0.715690m，Δworst=+0.984040m；MAE 严格改善/恶化/不变 checkpoint 为 0/9/0。这组结果不支持同时改善 MAE 与 ES 均值。尾部存在代价，不能称为可靠性已解决。

UTS λ=0.05：平均 ΔMAE=+0.070331m，ΔES=-0.173415m，ΔP99=+0.726017m，Δworst=+0.970832m；MAE 严格改善/恶化/不变 checkpoint 为 0/9/0。这组结果不支持同时改善 MAE 与 ES 均值。尾部存在代价，不能称为可靠性已解决。

UJI λ=0.02：平均 ΔMAE=-0.063500m，ΔES=-0.616126m，ΔP99=-0.747315m，Δworst=-1.118775m；MAE 严格改善/恶化/不变 checkpoint 为 8/1/0。MAE 与 ES 均值均下降。该尾部均值未上升；仍须查阅逐 checkpoint/逐位置伤害。

UJI λ=0.05：平均 ΔMAE=-0.063500m，ΔES=-0.616126m，ΔP99=-0.747315m，Δworst=-1.118775m；MAE 严格改善/恶化/不变 checkpoint 为 8/1/0。MAE 与 ES 均值均下降。该尾部均值未上升；仍须查阅逐 checkpoint/逐位置伤害。

## 尾部伤害与退化选择

以下位置计数在每 checkpoint 内配对，跨 checkpoint 汇总时重复位置会重复计数；不声称它们相互独立。

| 域 | λ | alpha_obs=0 /9 | beta=0 /9 | 三空间项均0 /9 | 位置改善计数和 | 位置恶化计数和 | 最大单位置恶化 m | Worst 退化 checkpoint /9 |
|---|---|---|---|---|---|---|---|---|
| tuji | 0.02 | 9 | 9 | 2 | 199 | 215 | 0.768161 | 9 |
| tuji | 0.05 | 9 | 7 | 0 | 198 | 216 | 0.768161 | 9 |
| uts | 0.02 | 9 | 0 | 0 | 290 | 331 | 2.341880 | 7 |
| uts | 0.05 | 9 | 0 | 0 | 289 | 332 | 2.341880 | 7 |
| uji | 0.02 | 8 | 0 | 0 | 187 | 209 | 4.133786 | 3 |
| uji | 0.05 | 8 | 0 | 0 | 187 | 209 | 4.133786 | 3 |

## 几何、门禁及身份核查


| 域 | 完成 | S1 边数范围 | S2 边数范围 | 空 S1 | 空 S2 | 可行 clean 行 | CAL-eval 位置 | 扫描 |
|---|---|---|---|---|---|---|---|---|
| tuji | 9 | 2209–2946 | 1473–2946 | 0 | 0 | 108 | 46 | 234 |
| uts | 9 | 3750–5000 | 2500–5000 | 0 | 0 | 108 | 69 | 414 |
| uji | 9 | 2424–3232 | 1617–3232 | 0 | 0 | 108 | 44 | 919 |

原生候选汇总 1,417,203 行；54 行近优敏感性摘要；27 份报告、27 份选择与27份候选表。

只读核算逐候选重算 MAE ≤ clean+1e-6m、BF accuracy ≥ clean−1e-12、BF-CE ≤ clean×1.02+0.005、仅惩罚 BF-CE 上升的归一化项、λ=.02/.05目标及原破平局顺序，并核对全部几何与各自域的预检、45% 支持门禁。没有把 TUJI 边数套到 UTS/UJI。

站点向量到宏指标最大核算误差 1.11851194795e-06；最终 324 个 clean 行最大 MAE 漂移 3.57627868652e-07m；历史 clean/old-combo 参考最大指标差 5.79077586993e-08。

138 个源码/测试/协议文件与授权修订清单匹配，其中136个与原冻结逐字节相同，另2个是记录在案的 core/test 数值修订。27 个 checkpoint、7 个训练输入、3 个预检文件、36 个 TUJI 原始结果文件及5个先前审计文件 SHA 不变。完整清单见独立复核 JSON 和修订 freeze。

所有选择和报告中的 internal_test_rows_scored=0、official_test_rows_scored=0。该结论结合源码角色守卫与导出计数核验；本轮未安装操作系统级文件访问跟踪器。CAL-eval BF-CE 的逐扫描真类概率未导出，因此该值只能核对保存的汇总，不能从站点 JSON 完整独立重建。

## 已知限制与材料位置

CAL-fit 多候选筛选后的 CAL-eval 改善或退化只作描述；没有新 TEST/独立外部验证，不能宣称模型晋级。λ=.02 和 .05 都完整保留，不按 CAL-eval 换赢家。均值改善不能掩盖 ES、楼层或尾部损害。

标准原生汇总：`CAL_EVAL_REPORT.md`、`JOINT_FIT_SELECTION.json`、`INTEGRITY_REVIEW.json`；每 checkpoint 科学原始结果在 `runs/<dataset>_seed<seed>/<C|D|E>/`；独立核算为 `CONTINUATION_READ_ONLY_VERIFICATION.json`。

旧 TUJI-only 报告、publication archive 与审计保持其原阶段快照，不是本次27/27的完整发布包。原生根汇总由原程序自动更新，旧7份汇总已有备份。大候选 CSV 按既有忽略规则留本地；本轮无 Git 操作。

两次成功正式命令的 stderr 原文随日志保留。下列长度为字节；非空内容必须结合原日志解释。


| 步骤 | stderr 字节 | 日志 SHA256 |
|---|---|---|
| UTS authorized restart | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| UJI | 0 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

## 训练输入 SHA256


| 文件 | SHA256 |
|---|---|
| data/TUJI1/Formatted_data_A12.mat | 73c59c6a8a12c45c4b6bf9b93b45cb89d0cfb7bb147260dc763f0d5f5af314a5 |
| data/TUJI1/Formatted_data_POCO.mat | 9973384ac04365a55a2e6b33b16b61724cf41869e828b1544c7c4bc61c9d2495 |
| data/TUJI1/Formatted_data_S20.mat | e3e327243852ba38087c4e8f07c4d94c7b8acc0987e961d5dc6e8fcf23dd354e |
| data/TUJI1/Formatted_data_S7.mat | c33ac65f9b61f6b24272b256b091c54747d5bce3f186a8a96e83cbc3e9f9fdab |
| data/TUJI1/Formatted_data_tabS7.mat | f8a9bf0a94367f2255a3c94b2bd4b633d73d7000a8f1a0cf5c25e90d231b0e23 |
| data/UTSIndoorLoc/UTS_training.csv | a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b |
| data/UJIndoorLoc/trainingData.csv | 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168 |

## 27 个冻结 checkpoint SHA256


| 文件 | SHA256 |
|---|---|
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17301/C/best_dev_epoch_21_80.pt | 2e8122dfee0c9168e9e5c60e6c3763d7ab3e3c55d69a46aae373f30f06a21fb0 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17301/D/best_dev_epoch_21_80.pt | 08f26213c2d6e84ebe0b4462660f445b3ffd3c458aea39426771786e9dda8821 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17301/E/best_dev_epoch_21_80.pt | 88811fb320cb04d1f9d90b7b5069f284574d662dc5f5b0acb79de95e1051e14b |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17302/C/best_dev_epoch_21_80.pt | 1e0c9486d24fd9b34c860a86d404a03d9c582cdf403d60160e5446be6e512f28 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17302/D/best_dev_epoch_21_80.pt | e64a5fbac3a3a3f914a7f84e892fe312b8e376adb140827a1f0682d6d67868a0 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17302/E/best_dev_epoch_21_80.pt | 8e157a074a9512d2280614c382128ce44ee5bf468a5738545c32a7a67ccd6e11 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17303/C/best_dev_epoch_21_80.pt | 0da46972e79de26c499308eeb5a8108de11b1091d4e4c3ccf0d9325af2d86e03 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17303/D/best_dev_epoch_21_80.pt | f07b2482093858a321122849b743330c15c1404729cb2b53d39ddd8c0dccea98 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/tuji_seed17303/E/best_dev_epoch_21_80.pt | f69a5f2fca4d23ffa261faa901171a47adf7738c8b1dae0d1431864a674bc40a |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17301/C/best_dev_epoch_21_80.pt | 7a7c1e28f18589cfd887ed15be1274c85910d6dca43ad90bc41e031d5fbc4e3f |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17301/D/best_dev_epoch_21_80.pt | 2605b9d88437f5b7cf23b6eedfb4f64583f710f5f865a746012f8c63677d5119 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17301/E/best_dev_epoch_21_80.pt | 531cd1525ecfb6369b12364814a66e34b4c89dc499695b246059dd0682a4561a |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17302/C/best_dev_epoch_21_80.pt | ecbf6d20c0159421f3ce59d681908f959d142d783ebae54a48516aefff13444d |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17302/D/best_dev_epoch_21_80.pt | 0ac425fb102d759097191d529b45d826feb2c54541d8fc28392aea895c6244e9 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17302/E/best_dev_epoch_21_80.pt | 9ea6c9b7c8ef98192a400c9a8999bdb52198c5ac6a1c451ed0c5fa975f5826e1 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17303/C/best_dev_epoch_21_80.pt | 0216a662b739b9779f5ec892d182fa274e8ad202003579d8c34eee10868ffc27 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17303/D/best_dev_epoch_21_80.pt | 7d7f940cc02a5219187e12ab36aa7b83085dffaafb58f918897015913f587b01 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uji_seed17303/E/best_dev_epoch_21_80.pt | 62e4ae6fedf3c028ca5dbd18e9bae7b1907a43a56c46decf8f7ba1c43709de4c |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17301/C/best_dev_epoch_21_80.pt | 6e7191a166218a6b48fd46d25a478c64e597c8697a615e564ad36825c5fa0d63 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17301/D/best_dev_epoch_21_80.pt | e1432d059779cedec4fb28b0d5b158499396c9a6f8ede4f2416fc6c90c5f5fa5 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17301/E/best_dev_epoch_21_80.pt | aa53a5ab96dc069b5905650265fe1fbaaa39171c0e6351c3f88cffe233219fdf |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17302/C/best_dev_epoch_21_80.pt | 7eebd7c30ac125a439530f544f8b8d6d2fce4b65d993a1d103e22e9c63cca05f |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17302/D/best_dev_epoch_21_80.pt | 2e0b3640bd850c36a4e3facbbc0f008f2a30170898dc4bbb0447c028b3ddeeb2 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17302/E/best_dev_epoch_21_80.pt | 0f579d789d69ef693e4b5de4281cf48130187ca3bf3359e9929381060dd28a56 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17303/C/best_dev_epoch_21_80.pt | cab935985cf8c0ca44203d1996a2de28c82aca0e4c27945f1578b12f36c2fb38 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17303/D/best_dev_epoch_21_80.pt | af110cd7b2b1eeee00c83dbbd8ecea16cf91089c69694aefc737bcf20aa89f77 |
| fhm_dapp_tim/evidence/v17_spatial_structure_v1/trained/uts_seed17303/E/best_dev_epoch_21_80.pt | 3afcb2f5b258f664cf1a537140298f5f1d8cc9b6f5755191e366eb4318c873e8 |

