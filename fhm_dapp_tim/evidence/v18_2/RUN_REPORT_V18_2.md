# V18.2 正式执行报告

八条注册命令全部按顺序完成，五个 stability seeds 全部保留。V18.2 的主确认结果来自 newly registered trainingData physical-coordinate holdout；官方 validationData 的结果严格属于 PRE-EXPOSED DESCRIPTIVE BENCHMARK。

**CONFIRMATION MAE = 6.409965 m**

**CONFIRMATION ENERGY SCORE = 4.579236 m**

**CONFIRMATION NLL = 5.394942**

MAE、ES、NLL 和 building/floor error 按完整物理坐标组取 macro；候选 recall、gate、entropy、uncertainty、coverage 和 radius 按注册代码对 scan 求均值。style offset 是绝对值均值，style gain 是 affine scale。source entropy 是归一化熵，posterior entropy 以 nats 计。

## GIT

- branch: `v18-2-adaptive-evidence`
- formal experiment SHA: `8d986195e4ea10c05943b63a17ad2ca0c54ba205`
- prepare 前 model/protocol bug fix: NO；fix commit: 无；科学源码和注册配置均未修改。

## TESTS

- 三个指定文件的 py_compile 全部通过。指定 pytest：28 passed，0 failed，5.80 s，无 skip/xfail、删测试或放宽合同。
- 合成 CUDA preflight 的两个用例（8 anchors/16 APs，以及 300 anchors/520 APs/batch 64）均通过 strict forward/backward/optimizer、gate gradient、有限值和 checkpoint 精确重建检查。
- 辅助 preflight 脚本首次因仓库 import path 缺失失败，修正辅助路径后通过；发生在正式 prepare 前，没有真实数据访问，不涉及模型源码修复。
- selection/stability/confirmation-refit/final-refit 的 strict CUDA 开关全部成功；evaluation CLI 的真实开关见 COMPUTE。
- 原注册 evaluation 的 scalar conversion warning 所在阶段：train-selection, calibrate-freeze, stability, confirm, benchmark；没有转换为 warn_only，没有据此更改模型，最终 finite/hash 检查均通过。

## PROTOCOL

trainingData：19,937 rows；933 physical coordinates；520 APs；13 building-floor strata。

| role | coordinates | rows |
| --- | --- | --- |
| member_fit | 506 | 10708 |
| early_stop | 89 | 1957 |
| selection_refit | 595 | 12665 |
| calibration | 151 | 3139 |
| outer_refit | 746 | 15804 |
| confirmation | 187 | 4133 |

member_fit 与 early_stop 不交且并集为 selection_refit；selection_refit 与 calibration 不交且并集为 outer_refit；outer_refit 与 confirmation 不交且并集为 trainingData。每个 role 保留全部 13 strata，三个 hull missing 都为 0。

prepare 按注册代码写入并 hash 六个 shard。之后独立验证五个 development shard，confirmation 在 confirm 写入 lock 后由正式 loader 验证哈希；没有提前手工读取或评价 confirmation。最终全部六个 shard 哈希有效且保持不变。

trainingData SHA256: `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`

| role shard | SHA256 |
| --- | --- |
| calibration | `e4357029b932bd990385313c319ceb0b7738dc59ffcbeee5798836c3a3c5bcf3` |
| confirmation | `faed56655888d1e9756c2daea45a6ca8c50e5acb6290bef22741645878283e74` |
| early_stop | `b828e6a52aa20a8aeadf77748e1f31c311d720388b267e919a9eda0bc00dcfb8` |
| member_fit | `e9267c1123616eff1a623f8beb7cb6e26b861a3da57fe4fa86493ae6fd3c340c` |
| outer_refit | `50db20cc51adfbea7a67724dd2df308e12d001358e48caaa40e222b8f3c00b3b` |
| selection_refit | `3bd7c4aa402e8d57eedaeadc3ba8ff79aa8982280be35a6937580865c1b0ac56` |

registry self-hash: `93a442bfe21e378ef87fee3c08442d49d54a41aa3e145cc0bab1dfe55f8a276a`

## MODEL

| 模型 | parameters | anchors | epoch | tensor finite |
| --- | --- | --- | --- | --- |
| selection | 607423 | 595 | 17 | YES |
| confirmation | 626751 | 746 | 17 | YES |
| final | 650687 | 933 | 17 | YES |

| 合同 | 值 |
| --- | --- |
| residual_experts | 1 |
| scan_feature_mode | bounded_maximal_invariant |
| prototype_metric | none |
| candidate_reranker | True |
| evidence_alpha | 1.0 |
| invariant_retrieval_k | 32 |
| prototype_retrieval_k | 32 |
| floor_retrieval_k | 16 |
| geometry_seed_k | 8 |
| candidate_union_size | 96 |
| posterior_top_k | 64 |
| radio_lambda | 1.0 |
| minimum_epochs | 12 |

六个 opinion 同属于一个模型：invariant、prototype、learned、marked_radio、floor、geometry；自适应 α=1 arithmetic evidence pooling。one inference checkpoint；no teacher；no distillation；no ensemble；no member axis；no best seed；no prediction averaging。

## TRAINING

搜索完成 25 个 epoch；selected epoch=17；best eligible Energy Score epoch=17，Energy=6.151515 m。只有 epoch≥12 eligible，shortlist 距 eligible 最优 ES 不超过 0.05 m，然后按 MAE、ES、较早 epoch 排序。NLL 未作为主选择标准。

| shortlist epoch | MAE (m) | ES (m) |
| --- | --- | --- |
| 17 | 8.100607 | 6.151515 |

| loss/diagnostic | selection search first→last | selection refit first→last | confirmation refit first→last | final refit first→last |
| --- | --- | --- | --- | --- |
| loss | 13.889477 → 8.355151 | 13.464480 → 8.686642 | 12.950381 → 8.521812 | 12.471633 → 8.235479 |
| joint_geometry_loss | 10.666173 → 7.037581 | 10.351577 → 7.268809 | 9.954275 → 7.110084 | 9.540223 → 6.815572 |
| fused_anchor_nll | 3.295060 → 1.901855 | 3.300367 → 2.020639 | 3.269974 → 2.071378 | 3.258382 → 2.137972 |
| coordinate_nll | 7.371114 → 5.135726 | 7.051210 → 5.248170 | 6.684301 → 5.038705 | 6.281841 → 4.677600 |
| invariant_aux_loss | 3.608394 → 2.059345 | 3.674193 → 2.202843 | 3.768938 → 2.213372 | 3.891029 → 2.245781 |
| reranker_aux_loss | 3.145846 → 1.901972 | 3.141118 → 2.020722 | 3.141160 → 2.072014 | 3.168986 → 2.139046 |
| hierarchy_loss | 0.783594 → 0.050796 | 0.763952 → 0.072880 | 0.738501 → 0.054327 | 0.706305 → 0.039972 |
| coordinate_loss | 18.157919 → 4.673403 | 15.911622 → 4.926366 | 13.470600 → 4.622517 | 11.804260 → 4.355902 |
| ranking_loss | 1.062182 → 0.237403 | 1.028647 → 0.311260 | 1.011749 → 0.353612 | 1.030664 → 0.401051 |
| domain_loss | 2.576643 → 0.888658 | 2.573070 → 0.961151 | 2.492519 → 0.934610 | 2.399492 → 0.916284 |
| invariant_recall_mass | 0.921151 → 0.999036 | 0.921769 → 0.997390 | 0.919543 → 0.996591 | 0.913134 → 0.996039 |
| prototype_recall_mass | 0.890075 → 0.890240 | 0.900055 → 0.900428 | 0.906974 → 0.906721 | 0.905468 → 0.904405 |
| floor_recall_mass | 0.857530 → 0.994607 | 0.850610 → 0.989463 | 0.832239 → 0.987379 | 0.806262 → 0.981792 |
| union_recall_mass | 0.990383 → 0.999833 | 0.989917 → 0.999454 | 0.989657 → 0.999416 | 0.988204 → 0.999234 |
| gate_weight_invariant | 0.216131 → 0.001666 | 0.217426 → 0.002121 | 0.196152 → 0.000960 | 0.142040 → 0.000457 |
| gate_weight_prototype | 0.154517 → 0.000184 | 0.152281 → 0.000339 | 0.142640 → 0.000256 | 0.131073 → 0.000156 |
| gate_weight_learned | 0.234400 → 0.997685 | 0.251364 → 0.996610 | 0.328284 → 0.997914 | 0.432075 → 0.998258 |
| gate_weight_marked_radio | 0.158097 → 0.000171 | 0.152940 → 0.000337 | 0.141498 → 0.000271 | 0.131136 → 0.000213 |
| gate_weight_floor | 0.117166 → 0.000226 | 0.112608 → 0.000461 | 0.097003 → 0.000520 | 0.084072 → 0.000863 |
| gate_weight_geometry | 0.119689 → 0.000067 | 0.113382 → 0.000132 | 0.094422 → 0.000078 | 0.079604 → 0.000053 |

selected early-stop 完整诊断（T=1, scale=1）：

| 指标 | 值 |
| --- | --- |
| coordinate_macro_mae_m | 8.100607 |
| coordinate_macro_energy_score_m | 6.151515 |
| coordinate_macro_mixture_nll | 6.653134 |
| coordinate_macro_mae_se_m | 0.520417 |
| coordinate_macro_energy_score_se_m | 0.394075 |
| coordinate_macro_mixture_nll_se | 0.134276 |
| coordinate_macro_building_error | 0.001659 |
| coordinate_macro_floor_error | 0.005061 |
| mean_anchor_entropy_nats | 1.997020 |
| mean_effective_anchor_count | 7.724104 |
| mean_top1_probability | 0.286259 |
| mean_radial_scale_m | 14.080683 |
| mean_within_component_radial_sd_m | 10.216580 |
| mean_between_anchor_radial_sd_m | 9.475850 |
| mean_posterior_radio_energy | 0.776814 |
| mean_abs_profiled_style_offset | 0.097488 |
| mean_profiled_style_scale | 0.905721 |
| error_radial_scale_correlation | 0.584286 |

| 最终 nearest same building-floor anchor | recall |
| --- | --- |
| recall@8 | 95.708% |
| recall@16 | 99.540% |
| recall@32 | 99.847% |
| recall@64 | 99.949% |

| source support | recall |
| --- | --- |
| invariant | 99.847% |
| prototype | 95.708% |
| floor | 99.540% |
| union | 99.949% |

| evidence source | mean gate weight | normalized source entropy H/log(L) |
| --- | --- | --- |
| invariant | 0.002917 | 0.560912 |
| prototype | 0.000321 | 0.874760 |
| learned | 0.995954 | 0.436075 |
| marked_radio | 0.000299 | 0.907688 |
| floor | 0.000386 | 0.943018 |
| geometry | 0.000122 | 0.877001 |

| nominal coverage | observed coverage | mean predictive radius (m) |
| --- | --- | --- |
| 50% | 53.398% | 7.688489 |
| 80% | 82.882% | 13.838190 |
| 90% | 92.591% | 19.313297 |
| 95% | 97.087% | 25.633635 |

每个搜索 epoch 的全部 TRAIN/EARLY STOP 指标和三个重训的每个 epoch loss 保留在 [JSON 报告](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_2/RUN_REPORT_V18_2.json>) 及 run_audit/*_verification.json。

## CALIBRATION

temperature=0.75；scale multiplier=1.0；temperature boundary=NO；scale boundary=NO。

8×6 固定网格全部 48 点已评价，唯一选择标准为 calibration coordinate-macro mixture NLL；没有更新 backbone/gate/reranker。selection checkpoint SHA 与 calibration 前一致；freeze 固定 registry、selection、calibration SHA、epoch、完整 V182_CONFIG、T 和 scale。

| 指标 | 值 |
| --- | --- |
| coordinate_macro_mae_m | 6.717949 |
| coordinate_macro_energy_score_m | 4.972293 |
| coordinate_macro_mixture_nll | 5.828416 |
| coordinate_macro_mae_se_m | 0.409386 |
| coordinate_macro_energy_score_se_m | 0.309523 |
| coordinate_macro_mixture_nll_se | 0.096459 |
| coordinate_macro_building_error | 0.000993 |
| coordinate_macro_floor_error | 0.011779 |
| mean_anchor_entropy_nats | 1.831425 |
| mean_effective_anchor_count | 6.553940 |
| mean_top1_probability | 0.321273 |
| mean_radial_scale_m | 9.755831 |
| mean_within_component_radial_sd_m | 7.309033 |
| mean_between_anchor_radial_sd_m | 6.231256 |
| mean_posterior_radio_energy | 0.670534 |
| mean_abs_profiled_style_offset | 0.085022 |
| mean_profiled_style_scale | 0.932347 |
| error_radial_scale_correlation | 0.612497 |

| 最终 nearest same building-floor anchor | recall |
| --- | --- |
| recall@8 | 94.011% |
| recall@16 | 98.949% |
| recall@32 | 99.650% |
| recall@64 | 99.873% |

| source support | recall |
| --- | --- |
| invariant | 99.713% |
| prototype | 98.981% |
| floor | 98.566% |
| union | 99.936% |

| evidence source | mean gate weight | normalized source entropy H/log(L) |
| --- | --- | --- |
| invariant | 0.001879 | 0.539104 |
| prototype | 0.000291 | 0.872655 |
| learned | 0.997025 | 0.448328 |
| marked_radio | 0.000290 | 0.895780 |
| floor | 0.000402 | 0.951419 |
| geometry | 0.000112 | 0.870666 |

| nominal coverage | observed coverage | mean predictive radius (m) |
| --- | --- | --- |
| 50% | 53.998% | 5.827146 |
| 80% | 82.351% | 10.172307 |
| 90% | 92.259% | 13.661782 |
| 95% | 97.611% | 17.532488 |

| T / scale | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 6.299435 | 5.952614 | 5.863656 | 5.876722 | 5.938553 | 6.123368 |
| 0.75 | 6.179475 | 5.885382 | 5.828416 | 5.861056 | 5.934843 | 6.132024 |
| 1.0 | 6.141310 | 5.875947 | 5.835312 | 5.877669 | 5.957243 | 6.159861 |
| 1.25 | 6.135839 | 5.891539 | 5.861681 | 5.910115 | 5.993111 | 6.198539 |
| 1.5 | 6.146719 | 5.921208 | 5.899792 | 5.952770 | 6.038192 | 6.245325 |
| 2.0 | 6.195429 | 6.003552 | 5.996122 | 6.056218 | 6.145298 | 6.354646 |
| 2.5 | 6.261564 | 6.096835 | 6.100866 | 6.166637 | 6.258589 | 6.469567 |
| 3.0 | 6.330724 | 6.186726 | 6.199908 | 6.270252 | 6.364515 | 6.576792 |


## STABILITY

五个 seed 均在 member_fit 独立训练冻结 epoch 数，只在 early_stop 使用冻结的 T/scale 评价。no best seed selection / no ensemble / no prediction averaging。

| seed | MAE (m) | ES (m) | NLL | recall@8 | recall@64 |
| --- | --- | --- | --- | --- | --- |
| 18321 | 8.201702 | 6.244987 | 6.680797 | 0.957077 | 0.999489 |
| 18322 | 8.799798 | 6.598817 | 6.808456 | 0.946857 | 1.000000 |
| 18323 | 8.651996 | 6.540244 | 6.795844 | 0.963720 | 0.999489 |
| 18324 | 8.444448 | 6.323932 | 6.780358 | 0.955544 | 0.999489 |
| 18325 | 8.454911 | 6.312218 | 6.750090 | 0.966786 | 0.999489 |

| metric | mean ± sample SD (ddof=1) |
| --- | --- |
| coordinate_macro_energy_score_m | 6.404040 ± 0.155432 |
| coordinate_macro_mae_m | 8.510571 ± 0.227225 |
| coordinate_macro_mixture_nll | 6.763109 ± 0.050917 |
| final_nearest_anchor_recall_at_64 | 0.999591 ± 0.000229 |
| final_nearest_anchor_recall_at_8 | 0.957997 ± 0.007761 |

| seed | invariant | prototype | learned | marked_radio | floor | geometry |
| --- | --- | --- | --- | --- | --- | --- |
| 18321 | 0.002917 | 0.000321 | 0.995954 | 0.000299 | 0.000386 | 0.000122 |
| 18322 | 0.001631 | 0.000251 | 0.997495 | 0.000246 | 0.000286 | 0.000090 |
| 18323 | 0.002638 | 0.000297 | 0.996304 | 0.000289 | 0.000352 | 0.000120 |
| 18324 | 0.003682 | 0.000333 | 0.995054 | 0.000332 | 0.000462 | 0.000136 |
| 18325 | 0.001645 | 0.000252 | 0.997502 | 0.000253 | 0.000262 | 0.000087 |

| source gate weight | mean ± sample SD |
| --- | --- |
| invariant | 0.002503 ± 0.000877 |
| prototype | 0.000291 ± 0.000038 |
| learned | 0.996462 ± 0.001050 |
| marked_radio | 0.000284 ± 0.000035 |
| floor | 0.000350 ± 0.000080 |
| geometry | 0.000111 ± 0.000022 |


## CONFIRMATION

**CONFIRMATION MAE = 6.409965 m**

**CONFIRMATION ENERGY SCORE = 4.579236 m**

**CONFIRMATION NLL = 5.394942**

187 个完整物理坐标、4,133 scans。confirm command count=1。consumption lock 在第一次加载 confirmation shard 前写入；一次性评价后只保留原结果，没有调参或重跑。

| 指标 | 值 |
| --- | --- |
| coordinate_macro_mae_m | 6.409965 |
| coordinate_macro_energy_score_m | 4.579236 |
| coordinate_macro_mixture_nll | 5.394942 |
| coordinate_macro_mae_se_m | 0.534353 |
| coordinate_macro_energy_score_se_m | 0.390297 |
| coordinate_macro_mixture_nll_se | 0.088022 |
| coordinate_macro_building_error | 0.003890 |
| coordinate_macro_floor_error | 0.013255 |
| mean_anchor_entropy_nats | 1.851321 |
| mean_effective_anchor_count | 6.807786 |
| mean_top1_probability | 0.327825 |
| mean_radial_scale_m | 9.784040 |
| mean_within_component_radial_sd_m | 6.997883 |
| mean_between_anchor_radial_sd_m | 6.517941 |
| mean_posterior_radio_energy | 0.592117 |
| mean_abs_profiled_style_offset | 0.080469 |
| mean_profiled_style_scale | 0.940791 |
| error_radial_scale_correlation | 0.760508 |

| 最终 nearest same building-floor anchor | recall |
| --- | --- |
| recall@8 | 92.983% |
| recall@16 | 98.572% |
| recall@32 | 99.589% |
| recall@64 | 99.976% |

| source support | recall |
| --- | --- |
| invariant | 99.710% |
| prototype | 96.226% |
| floor | 98.137% |
| union | 100.000% |

| evidence source | mean gate weight | normalized source entropy H/log(L) |
| --- | --- | --- |
| invariant | 0.000858 | 0.527466 |
| prototype | 0.000223 | 0.883306 |
| learned | 0.998149 | 0.456715 |
| marked_radio | 0.000237 | 0.900596 |
| floor | 0.000467 | 0.953199 |
| geometry | 0.000067 | 0.875153 |

| nominal coverage | observed coverage | mean predictive radius (m) |
| --- | --- | --- |
| 50% | 58.650% | 5.778609 |
| 80% | 86.136% | 9.902122 |
| 90% | 94.217% | 13.367085 |
| 95% | 97.072% | 17.368906 |


## PRE-EXPOSED BENCHMARK

**PRE-EXPOSED DESCRIPTIVE BENCHMARK**

**blind confirmation = NO；pre-exposed benchmark = YES；used_for_model_selection = NO**

**BENCHMARK MAE = 8.083969 m**

**BENCHMARK ENERGY SCORE = 5.776979 m**

**BENCHMARK NLL = 5.984653**

official validationData：1111 rows / 1052 physical coordinates。它已在 V18.1 暴露；本轮只在合法 confirmation 和 final refit 之后评价一次，结果没有用于选择模型或调参。

| 指标 | 值 |
| --- | --- |
| coordinate_macro_mae_m | 8.083969 |
| coordinate_macro_energy_score_m | 5.776979 |
| coordinate_macro_mixture_nll | 5.984653 |
| coordinate_macro_mae_se_m | 0.302241 |
| coordinate_macro_energy_score_se_m | 0.219692 |
| coordinate_macro_mixture_nll_se | 0.066584 |
| coordinate_macro_building_error | 0.000951 |
| coordinate_macro_floor_error | 0.083281 |
| mean_anchor_entropy_nats | 1.921562 |
| mean_effective_anchor_count | 7.151353 |
| mean_top1_probability | 0.326917 |
| mean_radial_scale_m | 10.881011 |
| mean_within_component_radial_sd_m | 7.129646 |
| mean_between_anchor_radial_sd_m | 7.784002 |
| mean_posterior_radio_energy | 0.809423 |
| mean_abs_profiled_style_offset | 0.106142 |
| mean_profiled_style_scale | 0.921375 |
| error_radial_scale_correlation | 0.575632 |

| 最终 nearest same building-floor anchor | recall |
| --- | --- |
| recall@8 | 80.918% |
| recall@16 | 93.429% |
| recall@32 | 97.930% |
| recall@64 | 99.460% |

| source support | recall |
| --- | --- |
| invariant | 97.390% |
| prototype | 88.479% |
| floor | 90.369% |
| union | 99.550% |

| evidence source | mean gate weight | normalized source entropy H/log(L) |
| --- | --- | --- |
| invariant | 0.000372 | 0.565690 |
| prototype | 0.000122 | 0.905060 |
| learned | 0.998579 | 0.484461 |
| marked_radio | 0.000168 | 0.905691 |
| floor | 0.000718 | 0.953025 |
| geometry | 0.000040 | 0.891670 |

| nominal coverage | observed coverage | mean predictive radius (m) |
| --- | --- | --- |
| 50% | 42.484% | 5.616628 |
| 80% | 70.207% | 10.099767 |
| 90% | 82.988% | 14.253744 |
| 95% | 90.729% | 19.061323 |


## COMPUTE

GPU：NVIDIA GeForce RTX 2060（6,144 MiB）；Python：3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；解释器：`D:\miniminiconda\envs\liuyan\python.exe`；PyTorch：2.10.0+cu128；CUDA runtime：12.8。

| 训练 | seconds |
| --- | --- |
| selection search + selection refit | 1030.891822 |
| stability wall seconds | 1839.246176 |
| confirmation refit | 542.455377 |
| final refit | 681.787273 |

| phase | wall seconds | peak CUDA allocated MiB | peak CUDA reserved MiB |
| --- | --- | --- | --- |
| prepare | 7.793467 | 0.000000 | 0.000000 |
| train-selection | 1035.711458 | 710.968750 | 894.000000 |
| calibrate-freeze | 13.929481 | 836.356934 | 902.000000 |
| stability | 1839.246176 | 706.334473 | 890.000000 |
| train-confirmation | 547.148321 | 373.630371 | 474.000000 |
| confirm | 11.463607 | 982.650391 | 1062.000000 |
| train-final | 686.579397 | 375.944824 | 480.000000 |
| benchmark | 8.274162 | 562.218750 | 656.000000 |

显存 peak 来自每个子进程 PyTorch allocator；不是驱动/context/其他进程的全设备峰值。prepare 没有 CUDA allocation。

| phase | strict deterministic | warn_only | flash SDPA | memory-efficient SDPA | math SDPA | cuDNN deterministic |
| --- | --- | --- | --- | --- | --- | --- |
| train-selection | YES | NO | NO | NO | YES | YES |
| calibrate-freeze | NO | NO | YES | YES | YES | NO |
| stability | YES | NO | NO | NO | YES | YES |
| train-confirmation | YES | NO | NO | NO | YES | YES |
| confirm | NO | NO | YES | YES | YES | NO |
| train-final | YES | NO | NO | NO | YES | YES |
| benchmark | NO | NO | YES | YES | YES | NO |

四个训练阶段均严格 deterministic；evaluation 阶段使用原注册 CLI 的默认开关。所有正式子进程 CUBLAS_WORKSPACE_CONFIG=:4096:8、cuDNN benchmark=False。

## AUDIT

| 项目 | YES/NO 或计数 |
| --- | --- |
| one inference checkpoint | YES |
| residual_experts == 1 | YES |
| teacher used | NO |
| distillation used | NO |
| ensemble used | NO |
| best seed selected | NO |
| positive self-anchor target used | NO |
| calibration label 在 freeze 前用于 backbone gradient | NO |
| confirmation 内容在 prepare 后、lock 前访问 | NO |
| prepare confirmation 写入与注册 hash | YES，协议规定 |
| confirm command count | 1 |
| confirmation 用于调参 | NO |
| benchmark 用于调参 | NO |
| benchmark 是新的盲测 | NO |
| 全部 science source/config/freeze 保持不变 | YES |
| calibration 在冻结后用于 outer_refit 梯度训练 | YES，协议规定 |
| confirmation 消费后用于最终 full refit 梯度训练 | YES，协议规定 |

prepare 读取 trainingData 并写入/hash confirmation 是注册步骤；此后未手工加载/查看/评价 confirmation。confirm 中的 SHA 读取和 torch.load 均发生在 lock 已写入之后。consumed lock 从未删除，confirm 和 benchmark 均没有第二次调用。

| phase | training CSV read opens | confirmation shard read opens | benchmark CSV read opens | historical checkpoint reads | denied |
| --- | --- | --- | --- | --- | --- |
| prepare | 2 | 1 | 0 | 0 | 0 |
| train-selection | 0 | 0 | 0 | 0 | 0 |
| calibrate-freeze | 0 | 0 | 0 | 0 | 0 |
| stability | 0 | 0 | 0 | 0 | 0 |
| train-confirmation | 0 | 0 | 0 | 0 | 0 |
| confirm | 0 | 2 | 0 | 0 | 0 |
| train-final | 0 | 2 | 0 | 0 | 0 |
| benchmark | 0 | 0 | 2 | 0 | 0 |

verify-prepare 另做一次 training CSV SHA 校验，之后训练阶段只使用物理 shard。旧 V18.1 outcome 的描述性比较只发生在新 benchmark 完成后的报告阶段，未参与本轮模型/seed/epoch/参数选择。

lock write open time_ns=1790762720338501000；first confirm read time_ns=1790762720344818000。

| HASH | SHA256 |
| --- | --- |
| registry_self_sha256 | `93a442bfe21e378ef87fee3c08442d49d54a41aa3e145cc0bab1dfe55f8a276a` |
| freeze_self_sha256 | `9b7de01b4ca84db703afe5dd930a1bd92147231e893fcc1312ae66fbd1e7b49a` |
| calibration_result_self_sha256 | `b535feaf53a5d82e22f285f489fee9d19c8cae28b9332d823647e6d97b13f6f4` |
| stability_result_self_sha256 | `c00c81b91dfd448a0b163afc310535a5da47dc66e5234689a34815761da5db58` |
| confirmation_lock_self_sha256 | `a9578f7dc94dfd14e01ef85122ccb95ac67fdca44c625b03ab9ca108d2f59ac4` |
| confirmation_lock_file_sha256 | `ca36f083bcc0ba478123cdf24e0bd5e837595c615be6855dd7f1f938f757ffda` |
| confirmation_result_self_sha256 | `1186fa6fd48b464dde69ef1907f2613f173471788a613f0959763cbfac154075` |
| confirmation_result_file_sha256 | `cfeefbd18aa7adb2b076c9cad456a0de16b6bb64d46afc9f7e0f6281ca3d0afe` |
| benchmark_result_self_sha256 | `83c3a0d67ce4da293e81bf8dfb759ffb11a4c19b9ce544f7d69d32354e408b1c` |
| benchmark_result_file_sha256 | `7c0f03af0b00b0a911a84a212f63042b8e12bc850edfb69159c6ecd3634164ae` |
| benchmark_data_sha256 | `5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0` |
| selection_checkpoint_sha256 | `67af4164812ff17a30b84cc552e773e39e36ecdba4f1fcc64bb54cf91aaf7aa7` |
| confirmation_checkpoint_sha256 | `97741796d71dacb37b1ee661cb80b6fabd692dd1fba1156c2f165d373e809e7e` |
| final_checkpoint_sha256 | `b3344cd4ddacbd32faeff13225937efca715d42a7fe3ff60b3db3d95a5c25bd5` |

结果/lock 的 canonical self-hash 与实际文件 byte SHA 分别列出；前者验证字段内容，后者记录当前本机文件字节。

## INTERPRETATION

1. confirmation 的 invariant/prototype nearest-anchor recall 为 99.710%/96.226%。prototype-only rescue 的可识别下界/上界为 0.000%/0.290%；下界为零，当前边际记录不足以确认 prototype 的独立救回数量。没有两源逐 scan 交集，不能把 union 的全部增量归给 prototype。benchmark 的同类范围也列在表中。
2. 96-anchor union 的 confirmation recall 为 100.000%，比 invariant 增加 0.290 个百分点，属于小幅补全，confirmation 候选覆盖已接近饱和。benchmark union 为 99.550%，比 invariant 增加 2.160 个百分点。三类 primary retrieval quota 均保留，因此这些是自然候选覆盖的净增量；没有用强制加入训练目标后的 recall 冒充自然 recall。
3. 同一 PRE-EXPOSED 官方 CSV 上，V18.1/V18.2 final recall@8 为 80.108%/80.918%，差值 +0.810 个百分点；recall@64 差值为 +0.990 个百分点。Top-8 有描述性改善，这回答了窄 posterior 排名问题；训练设计、字典、epoch 和校准不同，不能归因于单一方法修改，也不能升级为新 blind test。
4. confirmation 最大平均 gate 为 learned=99.815%，benchmark 最大为 learned=99.858%；全部六源排序见表。实际融合高度集中在一个来源，不能声称六个来源有均衡的实证贡献。source entropy 是 H/log(L) 的归一化熵，posterior entropy 才以 nats 计。
5. style mismatch 较大时是否自动降低 radio 信任：gate 的无标签输入包含 |style offset| 与 |log gain|，具备这种建模能力；注册结果只保存均值，没有 style-gate 的逐 scan 条件关系，因此不能仅凭平均 gate 与平均 style 声称已经验证。没有追加 confirmation 或 benchmark 推理来补证。confirmation mean |offset|=0.080469，mean scale=0.940791，mean radio gate=0.023655%。这些边际均值无法确定 style mismatch 与 radio 信任的条件关系。
6. floor 使用 soft opinion，并按候选 floor anchor 数分摊 floor mass；condition_building=False，坐标输出没有 hard floor filter。confirmation floor gate 为 0.046682%，floor retrieval recall 为 98.137%，coordinate-macro floor error 为 1.325%。平均 floor opinion 权重较小，没有 floor ablation，不能证明它单独改善结果。floor 分类输出来自 invariant marginal，而不是六源 consensus 的 floor marginal。
7. 同一预暴露 benchmark 的 between-anchor radial SD 从 V18.1 的 9.119794 m 到 V18.2 的 7.784002 m，差值 -1.335792 m，描述性下降。这只描述两个冻结实验；更小的 SD 本身不能证明校准更好，应结合下表的 coverage 和 within-component SD 评价。
8. 值得设计 V18.3：五个独立 seed 的 learned gate 都为 99.505%–99.750%，confirmation 为 99.815%，提示应优先检验证据融合的有效贡献。先预注册两源 retrieval 的联合救回诊断、style 与 radio gate 的逐 scan 条件关系、floor ablation，并结合 coverage 判断不确定性；再决定是否需要方法修改。新增实验必须另行注册并设置合法确认方案，当前 V18.2 保留全部原记录。同一官方 benchmark 相比 V18.1，MAE/ES/NLL 差值为 +0.716018 m / +0.465041 m / +0.246815，本轮三项评分均变差。benchmark 的 50/80/90/95% 实际 coverage 为 42.484%/70.207%/82.988%/90.729%，均低于 nominal，显示该数据上的区间偏窄。V18.3 应关注跨数据的定位评分与 coverage，同时保留候选 ranking 和 gate 诊断。

| 集合 | invariant recall | prototype recall | floor recall | union recall | union−invariant pp | union−prototype pp | prototype-only rescue 下界/上界 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 99.713% | 98.981% | 98.566% | 99.936% | +0.223 | +0.956 | 0.000% / 0.287% |
| confirmation | 99.710% | 96.226% | 98.137% | 100.000% | +0.290 | +3.774 | 0.000% / 0.290% |
| pre_exposed_benchmark | 97.390% | 88.479% | 90.369% | 99.550% | +2.160 | +11.071 | 0.000% / 2.610% |

每个集合按均值从大到小的 gate 权重：

| 集合 | gate ranking |
| --- | --- |
| calibration | learned=0.997025 > invariant=0.001879 > floor=0.000402 > prototype=0.000291 > marked_radio=0.000290 > geometry=0.000112 |
| confirmation | learned=0.998149 > invariant=0.000858 > floor=0.000467 > marked_radio=0.000237 > prototype=0.000223 > geometry=0.000067 |
| pre_exposed_benchmark | learned=0.998579 > floor=0.000718 > invariant=0.000372 > marked_radio=0.000168 > prototype=0.000122 > geometry=0.000040 |

同一 PRE-EXPOSED 官方 CSV 的 V18.1/V18.2 描述性比较（训练设计、字典、校准与 epoch 不同，不能解释为受控因果 ablation）：

| metric | V18.1 | V18.2 | V18.2−V18.1 |
| --- | --- | --- | --- |
| MAE (m) | 7.367951 | 8.083969 | 0.716018 |
| Energy Score (m) | 5.311939 | 5.776979 | 0.465041 |
| mixture NLL | 5.737838 | 5.984653 | 0.246815 |
| floor error | 6.997% | 8.328% | +1.331 pp |
| final recall@8 | 80.108% | 80.918% | +0.810 pp |
| final recall@64 | 98.470% | 99.460% | +0.990 pp |
| between-anchor radial SD (m) | 9.119794 | 7.784002 | -1.335792 |

Top-8 recall 提高；between-anchor SD 下降。这些判断只描述当前两个冻结实验的官方 benchmark。

| 集合 | nominal coverage | observed coverage | difference pp |
| --- | --- | --- | --- |
| confirmation | 50% | 58.650% | +8.650 |
| confirmation | 80% | 86.136% | +6.136 |
| confirmation | 90% | 94.217% | +4.217 |
| confirmation | 95% | 97.072% | +2.072 |
| pre_exposed_benchmark | 50% | 42.484% | -7.516 |
| pre_exposed_benchmark | 80% | 70.207% | -9.793 |
| pre_exposed_benchmark | 90% | 82.988% | -7.012 |
| pre_exposed_benchmark | 95% | 90.729% | -4.271 |

观察覆盖低于 nominal 提示区间偏窄/over-confidence，高于 nominal 提示偏保守；没有根据结果追加校准。V18.3 可预注册 source/union rescue 的联合诊断、style-gate 条件关系和 floor ablation，以检验当前只能描述的机制，并设置新的合法确认方案。

完整原始数值与每轮记录：[RUN_REPORT_V18_2.json](<D:/liuyanlunwen/fhm_dapp_tim/evidence/v18_2/RUN_REPORT_V18_2.json>)。报告生成 UTC：2026-09-30T10:22:48.730344+00:00
