# V18.3 执行记录：未完成

**BLOCKED: CAPACITY IMPLEMENTATION BUG; AUTHORIZATION PENDING**

Stage 1–4 已完成；Stage 5 构建时中止；Stage 6–8 未运行。未生成或删除 confirmation lock，未消费确认集，未运行当前 benchmark。


## GIT

分支 `v18-3-structured-consensus`；正式 SHA `342b73cf77222bd93d13eeb094858d569354f9ad`。
首次 API 失败按用户授权修复并提交为 `342b73c`，旧登记和日志完整归档；只在评价统计边界 detach，未改模型／损失／seed／epoch／校准网格。第二次容量失败后未改科学源码。

## TESTS

3 个 py_compile PASS；指定四个测试文件 **43 passed in 5.92s**。无删除、skip、xfail 或弱化测试。小模型与 520 AP CUDA 合成预检均完成有限前向／反向／优化器、精确 checkpoint 重建和完整评价。
另外在合成 CPU/CUDA 中验证尚未应用的计数补丁：模型状态、正目标与预测值逐项相同；16 个有效邻居通过，17 个仍拒绝。

## PROTOCOL


| role | coordinates | rows | strata |
| --- | --- | --- | --- |
| member_fit | 506 | 10601 | 13 |
| early_stop | 89 | 2064 | 13 |
| selection_refit | 595 | 12665 | 13 |
| calibration | 151 | 3139 | 13 |
| outer_refit | 746 | 15804 | 13 |
| confirmation | 187 | 4133 | 13 |

933 坐标 / 19,937 rows / 520 APs；三项交集均为 0，三个 union 恒等式成立，所有角色保留 13 strata，三项 hull_missing 均为 0。

| shard | registered SHA256 |
| --- | --- |
| calibration | cb1ef90851769f157ca186cf9c3667be40b3c21a75a5e0d5386002e9e0c2d517 |
| confirmation | 441c88c0962c0ed7b348104739db46891d981c531556a5790b3e4a4e937e28d9 |
| early_stop | c52b7f58c8bbe942ed217dd9d74de027c9fd08362b6875d29d8fd1059087ef69 |
| member_fit | 63d309dc6fdc6a9cad8f66561d607fd3813ede27bc5367bb6012c719a0b01891 |
| outer_refit | 6e4b19195437d5328131d8a11cb02cac970a3e27e1489a426e45abdaa2c4b658 |
| selection_refit | 4b5e02af1c777f6df73fe8a874265ac263faabef7737f2539d8becf271f82c0b |

五个开发 shard 已独立加载并核验 SHA；confirmation 仅在 prepare 写入并固定注册 hash，之后从未手工加载，未作确认评价。

## MODEL


| checkpoint | parameters | anchors | epoch |
| --- | --- | --- | --- |
| selection | 606873 | 595 | 17 |
| confirmation | 未生成 | 未构建 | 17 |
| final | 未生成 | 未构建 | 17 |

单 checkpoint 合同，E=1，L=96/K=64；无 trainable gate、teacher、distillation、ensemble、member axis 或多 seed prediction averaging；正式运行未 import v18_2_single.py。family floor=0.15，族内 source ∈[0.30,0.70]，训练 dropout p=0.30，T_dis∈[1,2]。

## TRAINING

搜索 23 轮，eligible best ES epoch=15，shortlist 为 15/17/20，按 MAE 选 epoch 17；selection refit 完成 17 轮，五个 stability seed 均完成 17 轮。confirmation 尚未开始梯度训练。

| metric | selection search first→last | selection refit first→last |
| --- | --- | --- |
| joint_geometry_loss | 10.793820 → 8.134389 | 10.524379 → 8.191188 |
| fused_anchor_nll | 3.461633 → 2.914313 | 3.484680 → 2.973105 |
| coordinate_nll | 7.332187 → 5.220076 | 7.039699 → 5.218084 |
| ranking_loss | 0.913588 → 0.346218 | 0.883573 → 0.378847 |
| invariant_recall_mass | 0.927137 → 0.996602 | 0.926445 → 0.995233 |
| prototype_recall_mass | 0.895732 → 0.894467 | 0.900346 → 0.900028 |
| floor_recall_mass | 0.869461 → 0.990396 | 0.859600 → 0.987146 |
| union_recall_mass | 0.991849 → 0.999678 | 0.991244 → 0.999485 |
| family_weight_representation | 0.435442 → 0.461995 | 0.433920 → 0.458743 |
| family_weight_radio | 0.227204 → 0.222690 | 0.228437 → 0.224490 |
| family_weight_structure | 0.337354 → 0.315315 | 0.337643 → 0.316766 |
| mean_global_disagreement | 0.160040 → 0.311978 | 0.159188 → 0.305450 |
| mean_disagreement_temperature | 1.160040 → 1.311978 | 1.159188 → 1.305450 |
| mean_radio_mismatch | 0.604339 → 0.569306 | 0.593540 → 0.558090 |
| family_dropout_fraction | 0.297519 → 0.307329 | 0.296723 → 0.305409 |

完整搜索／refit／五 seed 每轮 trajectory 均保存在 RUN_STATUS_V18_3.json 与 run_audit 对应记录；没有额外重跑训练。

## CALIBRATION

36 点全部评价；T=1.0≥1，scale=0.75；temperature boundary=True，scale boundary=False。epoch/config/T/scale 和关联 SHA 已冻结；未更新 backbone。

| metric | value |
| --- | --- |
| coordinate_groups | 151 |
| coordinate_macro_building_error | 0.001076 |
| coordinate_macro_energy_score_m | 7.308598 |
| coordinate_macro_energy_score_se_m | 0.270288 |
| coordinate_macro_floor_error | 0.010903 |
| coordinate_macro_mae_m | 12.698292 |
| coordinate_macro_mae_se_m | 0.575772 |
| coordinate_macro_mixture_nll | 6.285369 |
| coordinate_macro_mixture_nll_se | 0.098954 |
| error_evidence_disagreement_correlation | -0.028393 |
| error_radial_scale_correlation | 0.484439 |
| final_nearest_anchor_recall_at_16 | 0.993310 |
| final_nearest_anchor_recall_at_32 | 0.998726 |
| final_nearest_anchor_recall_at_64 | 0.999044 |
| final_nearest_anchor_recall_at_8 | 0.956037 |
| floor_nearest_anchor_recall | 0.990124 |
| invariant_nearest_anchor_recall | 0.997451 |
| mean_abs_profiled_style_offset | 0.085399 |
| mean_anchor_entropy_nats | 3.748924 |
| mean_between_anchor_radial_sd_m | 42.268738 |
| mean_disagreement_temperature | 1.301937 |
| mean_effective_anchor_count | 42.575874 |
| mean_family_disagreement_radio | 0.216951 |
| mean_family_disagreement_representation | 0.410931 |
| mean_family_disagreement_structure | 0.277929 |
| mean_family_entropy_radio | 0.896629 |
| mean_family_entropy_representation | 0.528856 |
| mean_family_entropy_structure | 0.967599 |
| mean_family_weight_radio | 0.227687 |
| mean_family_weight_representation | 0.455943 |
| mean_family_weight_structure | 0.316370 |
| mean_global_evidence_disagreement | 0.301937 |
| mean_posterior_radio_energy | 1.521209 |
| mean_predictive_radius_50_m | 16.377708 |
| mean_predictive_radius_80_m | 36.517807 |
| mean_predictive_radius_90_m | 62.608429 |
| mean_predictive_radius_95_m | 93.265259 |
| mean_profiled_style_scale | 0.928379 |
| mean_radial_scale_m | 44.159969 |
| mean_radio_mismatch | 0.529180 |
| mean_source_entropy_floor | 0.950418 |
| mean_source_entropy_geometry | 0.881406 |
| mean_source_entropy_invariant | 0.583538 |
| mean_source_entropy_learned | 0.458757 |
| mean_source_entropy_marked_radio | 0.904323 |
| mean_source_entropy_prototype | 0.867738 |
| mean_top1_probability | 0.092522 |
| mean_within_component_radial_sd_m | 12.066250 |
| mean_within_family_weight_radio_marked | 0.492741 |
| mean_within_family_weight_radio_prototype | 0.507259 |
| mean_within_family_weight_representation_invariant | 0.475257 |
| mean_within_family_weight_representation_learned | 0.524743 |
| mean_within_family_weight_structure_floor | 0.486238 |
| mean_within_family_weight_structure_geometry | 0.513762 |
| predictive_radial_coverage_50 | 0.832749 |
| predictive_radial_coverage_80 | 0.992036 |
| predictive_radial_coverage_90 | 1.000000 |
| predictive_radial_coverage_95 | 1.000000 |
| prototype_nearest_anchor_recall | 0.989806 |
| radio_mismatch_radio_weight_correlation | -0.946045 |
| rows | 3139 |
| union_nearest_anchor_recall | 0.999363 |


## STABILITY


| seed | MAE | ES | NLL | recall@8 | recall@64 |
| --- | --- | --- | --- | --- | --- |
| 18431 | 12.964417 | 8.465900 | 6.970012 | 0.939438 | 1.000000 |
| 18432 | 12.908610 | 8.688315 | 6.951642 | 0.938469 | 1.000000 |
| 18433 | 12.789115 | 8.559416 | 7.011047 | 0.935078 | 1.000000 |
| 18434 | 13.208419 | 8.716110 | 6.955004 | 0.937984 | 1.000000 |
| 18435 | 12.838564 | 8.714915 | 7.026291 | 0.937016 | 1.000000 |


| metric | mean±sample SD | min | max |
| --- | --- | --- | --- |
| coordinate_macro_energy_score_m | 8.628931 ± 0.111726 | 8.465900 | 8.716110 |
| coordinate_macro_mae_m | 12.941825 ± 0.163301 | 12.789115 | 13.208419 |
| coordinate_macro_mixture_nll | 6.982799 ± 0.033898 | 6.951642 | 7.026291 |
| final_nearest_anchor_recall_at_64 | 1.000000 ± 0.000000 | 1.000000 | 1.000000 |
| final_nearest_anchor_recall_at_8 | 0.937597 ± 0.001657 | 0.935078 | 0.939438 |


| seed | Representation | Radio | Structure |
| --- | --- | --- | --- |
| 18431 | 0.457416 | 0.220659 | 0.321925 |
| 18432 | 0.457421 | 0.220531 | 0.322049 |
| 18433 | 0.457232 | 0.220415 | 0.322352 |
| 18434 | 0.457301 | 0.220524 | 0.322175 |
| 18435 | 0.457361 | 0.220383 | 0.322255 |

无 best seed selection，无 ensemble，无 prediction averaging；全部诊断与训练 history 保留。

## CONFIRMATION

**CONFIRMATION MAE / ENERGY SCORE / NLL = 未运行**。confirm count=0，lock 不存在，不能声称取得确认结果。计划证据等级为 one-shot sequential internal confirmation；independent external confirmation=NO。

## PRE-EXPOSED BENCHMARK

**BENCHMARK MAE / ENERGY SCORE / NLL = 未运行**。benchmark count=0，没有结果文件。计划等级仅为 PRE-EXPOSED DESCRIPTIVE BENCHMARK；independent external confirmation=NO；benchmark used for tuning=NO。

## COMPUTE

NVIDIA GeForce RTX 2060 / Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)] / PyTorch 2.10.0+cu128 / CUDA 12.8；指定解释器 `D:\miniminiconda\envs\liuyan\python.exe`。

| phase | wall seconds | exit code | peak allocated MiB | peak reserved MiB |
| --- | --- | --- | --- | --- |
| prepare | 7.802427 | 0 | 0.000 | 0.000 |
| train-selection | 1147.192669 | 0 | 464.014 | 864.000 |
| calibrate-freeze | 15.170609 | 0 | 446.346 | 640.000 |
| stability | 2325.558026 | 0 | 460.365 | 868.000 |
| train-confirmation | 7.355575 | 1 | 0.000 | 0.000 |

显存为各子进程 PyTorch allocator 峰值，不含驱动或其它进程。三个已完成 CUDA 阶段及失败阶段的实际开关均保留：strict deterministic=True，warn_only=False，flash/memory-efficient SDPA=False，math SDPA=True，cuDNN benchmark=False，deterministic=True。

## AUDIT


| metric | value |
| --- | --- |
| confirm_command_count | 0 |
| benchmark_command_count | 0 |
| confirmation_consumed | NO |
| confirmation_shard_loaded_after_prepare | NO |
| benchmark_loaded | NO |
| model_config_changed | NO |
| positive_self_target_used | NO |
| positive_targets_pruned | NO |
| primary_quotas_reduced | NO |
| capacity_guard_bypassed | NO |
| confirmation_used_for_tuning | NO |
| benchmark_used_for_tuning | NO |
| calibration_labels_used_for_backbone_gradients_before_freeze | NO |
| formal_stages_use_historical_checkpoint | NO |
| exact_coordinate_capacity_preflight_authorized | NO |
| capacity_source_patch_applied | NO |
| only_explicitly_authorized_detach_fix_applied | YES |


| role | padded width | max positive nonself targets | primary quotas | padded check capacity | actual positive capacity | L |
| --- | --- | --- | --- | --- | --- | --- |
| member_fit | 14 | 13 | 80 | 94 | 93 | 96 |
| selection_refit | 14 | 13 | 80 | 94 | 93 | 96 |
| outer_refit | 17 | 16 | 80 | 97 | 96 | 96 |

outer_refit 的 guard 把零权重 self/padding 位计入 17，误算 80+17=97；真实正目标数为 16，恰好 80+16=96。只更正计数的补丁已准备和验证但未应用。
登记坐标元数据的初步 full-support probe 得到 18 个有效邻居、所需 98；这不是实际按行数均值生成的 full checkpoint geometry，尚待用户授权做坐标列静态预检，不能当成已运行 train-final 的结果。

## INTERPRETATION

1. 开发阶段没有可训练 gate；逐 scan family/source 权重边界检查通过。可确认结构防塌缩约束成立，不能据此声称新的 confirmation 或 benchmark 已验证改善。

2. 校准集三个 family 平均质量为 45.594%/22.769%/31.637%，五 seed 的族权重范围完整保留。非零权重不等于已识别的独立因果贡献。

3. 校准集 radio-mismatch/Radio-weight Pearson r=-0.946045；注册惩罚在固定其它可靠性项时单调抑制 Radio 权重。没有确认集或 benchmark 结果支持进一步判断。

4. 校准集 error/disagreement Pearson r=-0.028393，没有支持困难 scan 明显对应更高 disagreement 的证据。T_dis 按固定公式增加，但其对真实误差的诊断作用需区分数学定义与经验相关。

5. stability 的开发集 Top-8 mean=93.760%，Top-64 mean=100%。没有当前官方 benchmark，不能与 V18.2 判定同一 CSV 上的改善。

6. 校准集 50/80/90/95% nominal coverage 的实际值为 83.275%/99.204%/100.000%/100.000%，区间偏宽。confirmation 与 benchmark 未运行，不能判断跨数据集 coverage 改善。

7. 本版 benchmark 未运行，不能判断相较 V18.2 是否恢复；没有加载旧 benchmark 结果进行当前比较，也未追加任何模型评价。

8. 进一步工作应先在正式 prepare 前核验全部训练支持的真实有效目标宽度，修正零权重 padding 计数错误。若 full support 实际需要超过 L=96，则变更 L 或 source quotas 属于新的方法合同，需另行注册，不能裁剪正目标或绕过容量检查。

本报告明确区分已完成、失败和未运行阶段。没有填补未取得的 confirmation／benchmark 指标，没有绕过容量 guard，也没有以新参数悄悄重跑。
