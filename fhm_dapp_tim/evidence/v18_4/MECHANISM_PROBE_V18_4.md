# V18.4 只读 correction × decoder 机制探针


仅在已开放的 UJI trainingData 开发/校准角色作机制诊断；不构成盲测确认，也不改变方法冻结。

已获人工授权：冻结 radio profiler 仅对两个临时 offset/gain 张量做每批 6 步 Adam。网络优化、训练和网络损失反向传播均被禁止。

ES/NLL/MAE 的主报告为坐标组等权均值；其余后验、尾部及 oracle 汇总为行等权。paired correction Δ = correction − none（负值改善）；decoder improvement = mean − alternative（正值改善）。

同场景共用 Student-t draws（seed 184412，64 对独立流抽样）。常量/单候选相关系数按预注册约定记为 0；Spearman 用平均秩。可靠性四分位按稳定排序等数切分，相同可靠性依角色行序分配。Top-8/argmin/MAP 并列依候选顺序；decoder 排名依 mean、MAP、Top-8、GM 顺序破并列。mean 并列最优计为 directly best。KL 的 float32 转换容差为既有门禁 1e-7。

## 1. 场景


| Scenario | Role | Primary | Rows | Groups | T | Scale | epsilon | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | development | True | 2936 | 139 | 1.000000 | 1.000000 | 0.025000 | fit |
| calibration_frozen | calibration | True | 2875 | 139 | 1.500000 | 1.000000 | 0.025000 | selection_refit |
| calibration_T1_control | calibration | False | 2875 | 139 | 1.000000 | 1.000000 | 0.025000 | selection_refit |

## 2. Correction 概率指标


| Scenario | Mode | ES m | NLL | Entropy | Eff anchors | Top-1 | KL | Eff epsilon | rho | r radio | r geometry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | none | 5.721319 | 6.375871 | 2.113564 | 8.634366 | 0.258319 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| development_native | radio_only | 5.729780 | 6.380220 | 2.034389 | 8.012889 | 0.274782 | 0.013868 | 0.013978 | 0.216637 | 0.559114 | 0.000000 |
| development_native | geometry_only | 5.749205 | 6.385936 | 2.033150 | 7.994497 | 0.272516 | 0.008697 | 0.010889 | 0.463735 | 0.000000 | 0.435569 |
| development_native | full | 5.741261 | 6.385063 | 2.014640 | 7.859416 | 0.277759 | 0.015434 | 0.015691 | 0.349987 | 0.559114 | 0.435569 |
| calibration_frozen | none | 5.161616 | 6.073754 | 2.526616 | 12.936291 | 0.184115 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| calibration_frozen | radio_only | 5.083941 | 6.061417 | 2.433750 | 11.854180 | 0.197592 | 0.012356 | 0.012398 | 0.174333 | 0.495919 | 0.000000 |
| calibration_frozen | geometry_only | 5.142991 | 6.064334 | 2.450683 | 12.031998 | 0.193359 | 0.006922 | 0.007353 | 0.273288 | 0.000000 | 0.294134 |
| calibration_frozen | full | 5.078571 | 6.059117 | 2.417957 | 11.667975 | 0.198720 | 0.013245 | 0.013292 | 0.243135 | 0.495919 | 0.294134 |
| calibration_T1_control | none | 5.081118 | 6.075048 | 2.142036 | 8.869059 | 0.252927 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| calibration_T1_control | radio_only | 5.065371 | 6.080925 | 2.064560 | 8.244911 | 0.268427 | 0.014273 | 0.014330 | 0.239395 | 0.573184 | 0.000000 |
| calibration_T1_control | geometry_only | 5.110049 | 6.088435 | 2.061234 | 8.202602 | 0.267230 | 0.008391 | 0.011153 | 0.482373 | 0.000000 | 0.446127 |
| calibration_T1_control | full | 5.078303 | 6.085420 | 2.043887 | 8.077216 | 0.271560 | 0.015676 | 0.015989 | 0.387458 | 0.573184 | 0.446127 |

## 3. Decoder 坐标宏 MAE（m）


| Scenario | Mode | Mean | MAP | Top-8 | GM | Ranking |
| --- | --- | --- | --- | --- | --- | --- |
| development_native | none | 7.689194 | 8.741661 | 7.849082 | 8.042431 | posterior_mean > top8_mean > geometric_median > map_component |
| development_native | radio_only | 7.728727 | 8.888755 | 7.818798 | 8.077913 | posterior_mean > top8_mean > geometric_median > map_component |
| development_native | geometry_only | 7.752503 | 8.713979 | 7.893614 | 8.089436 | posterior_mean > top8_mean > geometric_median > map_component |
| development_native | full | 7.749320 | 8.880742 | 7.851357 | 8.096495 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_frozen | none | 7.136363 | 8.215835 | 7.033182 | 7.164617 | top8_mean > posterior_mean > geometric_median > map_component |
| calibration_frozen | radio_only | 6.928125 | 8.065806 | 6.962786 | 7.132363 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_frozen | geometry_only | 7.075714 | 8.222771 | 7.052995 | 7.202775 | top8_mean > posterior_mean > geometric_median > map_component |
| calibration_frozen | full | 6.942809 | 8.101132 | 6.976303 | 7.149718 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_T1_control | none | 6.886110 | 8.215835 | 7.059563 | 7.361079 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_T1_control | radio_only | 6.826117 | 8.095446 | 6.997759 | 7.337602 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_T1_control | geometry_only | 6.922345 | 8.223214 | 7.102943 | 7.408233 | posterior_mean > top8_mean > geometric_median > map_component |
| calibration_T1_control | full | 6.862605 | 8.126381 | 7.025515 | 7.364429 | posterior_mean > top8_mean > geometric_median > map_component |

## 4. Paired correction Δ 对 none


| Scenario | Mode | Metric | Mean delta | SE | 95% lower | 95% upper | Groups improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | radio_only | energy_score_m | 0.008461 | 0.019530 | -0.029818 | 0.046739 | 0.496403 |
| development_native | radio_only | nll | 0.004349 | 0.008141 | -0.011608 | 0.020306 | 0.597122 |
| development_native | radio_only | posterior_mean_mae_m | 0.039533 | 0.028830 | -0.016974 | 0.096040 | 0.431655 |
| development_native | geometry_only | energy_score_m | 0.027885 | 0.015286 | -0.002076 | 0.057846 | 0.446043 |
| development_native | geometry_only | nll | 0.010065 | 0.007413 | -0.004465 | 0.024595 | 0.640288 |
| development_native | geometry_only | posterior_mean_mae_m | 0.063309 | 0.027698 | 0.009020 | 0.117597 | 0.402878 |
| development_native | full | energy_score_m | 0.019942 | 0.020173 | -0.019597 | 0.059480 | 0.482014 |
| development_native | full | nll | 0.009192 | 0.009391 | -0.009216 | 0.027599 | 0.597122 |
| development_native | full | posterior_mean_mae_m | 0.060126 | 0.032367 | -0.003313 | 0.123564 | 0.410072 |
| calibration_frozen | radio_only | energy_score_m | -0.077675 | 0.019546 | -0.115985 | -0.039364 | 0.669065 |
| calibration_frozen | radio_only | nll | -0.012337 | 0.004719 | -0.021587 | -0.003087 | 0.683453 |
| calibration_frozen | radio_only | posterior_mean_mae_m | -0.208238 | 0.046916 | -0.300194 | -0.116282 | 0.597122 |
| calibration_frozen | geometry_only | energy_score_m | -0.018625 | 0.011986 | -0.042117 | 0.004867 | 0.604317 |
| calibration_frozen | geometry_only | nll | -0.009420 | 0.002696 | -0.014705 | -0.004136 | 0.741007 |
| calibration_frozen | geometry_only | posterior_mean_mae_m | -0.060649 | 0.031922 | -0.123217 | 0.001918 | 0.503597 |
| calibration_frozen | full | energy_score_m | -0.083045 | 0.020410 | -0.123050 | -0.043041 | 0.726619 |
| calibration_frozen | full | nll | -0.014637 | 0.004670 | -0.023790 | -0.005484 | 0.719424 |
| calibration_frozen | full | posterior_mean_mae_m | -0.193554 | 0.050235 | -0.292015 | -0.095092 | 0.589928 |
| calibration_T1_control | radio_only | energy_score_m | -0.015746 | 0.013811 | -0.042815 | 0.011322 | 0.503597 |
| calibration_T1_control | radio_only | nll | 0.005877 | 0.005891 | -0.005670 | 0.017424 | 0.568345 |
| calibration_T1_control | radio_only | posterior_mean_mae_m | -0.059993 | 0.027483 | -0.113859 | -0.006127 | 0.489209 |
| calibration_T1_control | geometry_only | energy_score_m | 0.028932 | 0.012289 | 0.004845 | 0.053018 | 0.438849 |
| calibration_T1_control | geometry_only | nll | 0.013387 | 0.004743 | 0.004090 | 0.022684 | 0.561151 |
| calibration_T1_control | geometry_only | posterior_mean_mae_m | 0.036235 | 0.023245 | -0.009324 | 0.081795 | 0.395683 |
| calibration_T1_control | full | energy_score_m | -0.002815 | 0.014811 | -0.031845 | 0.026215 | 0.525180 |
| calibration_T1_control | full | nll | 0.010372 | 0.006411 | -0.002194 | 0.022938 | 0.561151 |
| calibration_T1_control | full | posterior_mean_mae_m | -0.023505 | 0.030514 | -0.083312 | 0.036301 | 0.467626 |

## 5. None 下 decoder 对 mean 的改善


| Scenario | Decoder | Improvement m | SE | 95% lower | 95% upper | Groups improved | Relative improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | map_component | -1.052467 | 0.243946 | -1.530600 | -0.574333 | 0.258993 | -0.136876 |
| development_native | top8_mean | -0.159888 | 0.118341 | -0.391836 | 0.072059 | 0.424460 | -0.020794 |
| development_native | geometric_median | -0.353237 | 0.139423 | -0.626506 | -0.079967 | 0.359712 | -0.045939 |
| calibration_frozen | map_component | -1.079472 | 0.268375 | -1.605487 | -0.553457 | 0.338129 | -0.151264 |
| calibration_frozen | top8_mean | 0.103180 | 0.156767 | -0.204083 | 0.410444 | 0.510791 | 0.014458 |
| calibration_frozen | geometric_median | -0.028254 | 0.199458 | -0.419193 | 0.362684 | 0.496403 | -0.003959 |
| calibration_T1_control | map_component | -1.329725 | 0.198621 | -1.719023 | -0.940427 | 0.237410 | -0.193103 |
| calibration_T1_control | top8_mean | -0.173453 | 0.081717 | -0.333618 | -0.013289 | 0.446043 | -0.025189 |
| calibration_T1_control | geometric_median | -0.474969 | 0.142517 | -0.754302 | -0.195636 | 0.338129 | -0.068975 |

## 6. Source energy → truth


| Primary scenario | Source | Mean rho | Median rho | P10 | P90 | rho > 0 | rho < 0 | Argmin error m | q_T MAP error m | Argmin beats MAP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | radio | 0.581587 | 0.625581 | 0.289217 | 0.809018 | 0.975817 | 0.024183 | 9.795466 | 8.582106 | 0.407357 |
| development_native | geometry | 0.576031 | 0.637307 | 0.144118 | 0.899016 | 0.946526 | 0.053474 | 59.748684 | 8.582106 | 0.222071 |
| calibration_frozen | radio | 0.557590 | 0.587766 | 0.263208 | 0.799876 | 0.988522 | 0.011478 | 7.920492 | 7.960187 | 0.416348 |
| calibration_frozen | geometry | 0.606623 | 0.665086 | 0.246420 | 0.897000 | 0.976000 | 0.024000 | 67.285657 | 7.960187 | 0.246957 |

## 7. radio reliability 四分位


| Primary scenario | Quartile | Rows | Mean reliability | Mean gain m | Median gain m | Gain > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| development_native | Q1 | 734 | 0.143935 | 0.336116 | 0.241489 | 0.914169 |
| development_native | Q2 | 734 | 0.445505 | 0.394627 | 0.305895 | 0.884196 |
| development_native | Q3 | 734 | 0.716719 | 0.422983 | 0.298317 | 0.846049 |
| development_native | Q4 | 734 | 0.930296 | 0.417283 | 0.358841 | 0.889646 |
| calibration_frozen | Q1 | 719 | 0.102577 | 0.517666 | 0.379763 | 0.977747 |
| calibration_frozen | Q2 | 719 | 0.346451 | 0.730971 | 0.572428 | 0.958275 |
| calibration_frozen | Q3 | 719 | 0.632333 | 0.908075 | 0.598875 | 0.966620 |
| calibration_frozen | Q4 | 718 | 0.902880 | 0.833223 | 0.633781 | 0.974930 |

development_native: oracle gain 均值 0.392752 m；中位数 0.294891 m；正/负比例 0.883515/0.116485；reliability→gain Spearman 0.107335，Pearson 0.072140。

calibration_frozen: oracle gain 均值 0.747454 m；中位数 0.525306 m；正/负比例 0.969391/0.030609；reliability→gain Spearman 0.205892，Pearson 0.163989。

## 8. geometry reliability 四分位


| Primary scenario | Quartile | Rows | Mean reliability | Mean gain m | Median gain m | Gain > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| development_native | Q1 | 734 | 0.111659 | 0.406953 | 0.388057 | 0.912807 |
| development_native | Q2 | 734 | 0.325061 | 0.463050 | 0.473862 | 0.891008 |
| development_native | Q3 | 734 | 0.532196 | 0.434651 | 0.425028 | 0.919619 |
| development_native | Q4 | 734 | 0.773361 | 0.203513 | 0.162549 | 0.881471 |
| calibration_frozen | Q1 | 719 | 0.055929 | 0.355359 | 0.288332 | 0.954103 |
| calibration_frozen | Q2 | 719 | 0.180675 | 0.644060 | 0.550147 | 0.981919 |
| calibration_frozen | Q3 | 719 | 0.338891 | 0.597055 | 0.581045 | 0.958275 |
| calibration_frozen | Q4 | 718 | 0.601466 | 0.418316 | 0.367702 | 0.952646 |

development_native: oracle gain 均值 0.377042 m；中位数 0.362186 m；正/负比例 0.901226/0.098774；reliability→gain Spearman -0.248712，Pearson -0.213828。

calibration_frozen: oracle gain 均值 0.503727 m；中位数 0.450035 m；正/负比例 0.961739/0.038261；reliability→gain Spearman 0.063453，Pearson -0.033382。

## 9. Radio-vs-geometry agreement


| Primary scenario | Mean energy rho | Median energy rho | rho > 0 | Same argmin | Argmin distance m | r Pearson | r Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development_native | 0.397818 | 0.442865 | 0.904973 | 0.038828 | 59.748758 | 0.411689 | 0.399686 |
| calibration_frozen | 0.379087 | 0.410647 | 0.921391 | 0.039652 | 67.112984 | 0.429841 | 0.449336 |

## 10. Calibration T=1 vs T=1.5 符号


| Mode | Metric | T1 delta | T1.5 delta | T1 sign | T1.5 sign | Mode conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| radio_only | energy_score_m | -0.015746 | -0.077675 | -1 | -1 | sign_changed |
| radio_only | nll | 0.005877 | -0.012337 | 1 | -1 | sign_changed |
| radio_only | posterior_mean_mae_m | -0.059993 | -0.208238 | -1 | -1 | sign_changed |
| geometry_only | energy_score_m | 0.028932 | -0.018625 | 1 | -1 | sign_changed |
| geometry_only | nll | 0.013387 | -0.009420 | 1 | -1 | sign_changed |
| geometry_only | posterior_mean_mae_m | 0.036235 | -0.060649 | 1 | -1 | sign_changed |
| full | energy_score_m | -0.002815 | -0.083045 | -1 | -1 | sign_changed |
| full | nll | 0.010372 | -0.014637 | 1 | -1 | sign_changed |
| full | posterior_mean_mae_m | -0.023505 | -0.193554 | -1 | -1 | sign_changed |

## 预注册结论


- radio_only: **MIXED_OR_NEUTRAL**

- geometry_only: **MIXED_OR_NEUTRAL**

- full: **MIXED_OR_NEUTRAL**


Overall correction: **correction_evidence_mixed**。

Decoder: **posterior_mean_supported_as_default**；supported alternatives=[]；posterior_mean_directly_best_both_roles=False。

主判定仅使用 A+B；C 仅检查温度交互。decoder ≥10% / ≤0.25m 门槛和 correction 规则按指令执行，无事后调整。全部 SE/CI、行分位数、四种 mode 的 paired decoder effects 及尾部诊断见 JSON。冻结 decoder 保持原状，下一步仅科学审阅。

Result canonical SHA256: `75bd468302f3defdcb845d27bdc176716267ba0edb3d73d04bcd76d33ccfc1aa`。训练、模型修改/梯度、官方验证数据、外部评价、V18.5 变更均为 false。
