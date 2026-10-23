# ITER3 TUJI 执行中止报告

状态：STOPPED_ENGINEERING_FAILURE；完成 0/9。保留原目录、失败日志和冻结身份；未修改正式源码或测试，也未重跑正式评分。

## 环境与范围

- 输出：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3`。
- 分支：`v17-dataset-adaptive-spatial-joint-fit-fix-v1`；提交 `c247d547893f2abb8c4fffaa531578a86e291bdc`。
- Python：`D:\miniminiconda\envs\liuyan\python.exe`；Python 3.12.12；Torch 2.10.0+cu128；CUDA 12.8；RTX 2060 6GB。
- 编译退出 0；正式原单测 30 passed；FIT 几何预检退出 0，三域九组全部通过。只启动过一次 TUJI 正式命令；UTS/UJI 定位评分未运行。
- 新训练、optimizer 更新、内部 TEST 和官方 TEST 评分均为 0。真实物理训练容器由原适配器读取；TUJI 只请求 results_train，不声称 MAT 容器所有字节隔离。

## 失败与独立定位

- TUJI / seed17301 / backbone C 在第二个几何配置 `fit_quantile_geometry_01` 中止，正式命令退出 1，耗时 12.436673 秒。
- 基础边数 2946；R_max_S1 = R_max_S2 = 1.330907885325485 m；45% 最低要求 1326 条。
- 预检 float64 统计为 2209 条，运行 float32 筛选为 2216 条，多纳入 7 条边界边。报错虽写 below validated minimum，实际触发的是 `len(s2) != validated_count`；没有低于 45%。
- 预检/支持验证先把 FIT 存储坐标转为 float64；run 和 _make_probability 路径原来直接用 float32 做距离/截断，产生半径边界不一致。独立 FIT-only 复算全部 12 配置，float64 计数全部与预检一致。详见 `run_logs/S2_SUPPORT_FAILURE_DIAGNOSIS.json`，含具体锚点 ID 和边长。
- 第一个几何的 CAL-fit 网格已在内存计算；第二几何的计数检查失败。尚未写出 CAL_FIT_SELECTION、CAL_EVAL_REPORT 或完整 CANDIDATE_METRICS。不能把未落盘的中间计算当成完整实验结果。

## 可审阅最小修复（尚未应用）

- 仅在 run 与 _make_probability 两处，将计算边长使用的 NumPy 坐标显式转为 float64，与预检一致。原 float32 存储坐标值逐值不变；连续分量/尺度张量保持原状。
- 不更改半径/分位数、温度、45% 支持要求、1e-6m MAE 门禁、目标函数、系数、随机种子、数据角色或 checkpoint。
- 两项纯合成回归分别直接覆盖运行路径和评分构造路径；原正式源码上均失败，临时预览补丁加原 30 项单测共 32 passed。未在预览源码上执行真实定位评分。
- 最新提案：`run_logs/PROPOSED_PRECISION_FIX_V2.patch`；正式源码和新预览源码身份见 `PROPOSED_PRECISION_FIX.json`。初版夹具、改进夹具和全部负向验证日志均保留。
- 辅助验证中的两项准备问题已记录于 STOPPED_INTEGRITY_REVIEW.json：文本替换助手的匹配数断言，以及首版合成夹具的近似重复尺度；均未修改正式源码/研究参数。

## 命令退出码与耗时

| 步骤 | 退出码 | 秒 |
|---|---:|---:|
| 01_compile | 0 | 0.112801 |
| 02_pytest | 0 | 7.697810 |
| 03_preflight | 0 | 28.820736 |
| 04_inspect_preflight | 0 | 0.219984 |
| 05_run_tuji | 1 | 12.436673 |
| 06_diagnose_support_fit_only | 0 | 6.626510 |
| 07_regression_original_source | 1 | 5.368156 |
| 08_regression_original_source_revised_fixture | 1 | 5.341396 |
| 09_validate_proposed_precision_fix | 0 | 6.737803 |

07/08 是预期的负向回归验证：07 的首版运行夹具尚未到达预期错误，08 的修正夹具精确复现两个截断路径错误；09 是临时修复预览验证。它们不是额外真实数据评分。

## FIT 几何预检

| 数据域/seed | FIT锚点 | 基础边 | S1保留范围 | S2保留范围 | 配置数 | 无距离衰减对照 |
|---|---:|---:|---:|---:|---:|---:|
| tuji_seed17301 | 646 | 2946 | [2209, 2946] | [1473, 2946] | 12 | 4 |
| tuji_seed17302 | 646 | 2946 | [2209, 2946] | [1473, 2946] | 12 | 4 |
| tuji_seed17303 | 646 | 2946 | [2209, 2946] | [1473, 2946] | 12 | 4 |
| uji_seed17301 | 672 | 3232 | [2424, 3232] | [1617, 3232] | 12 | 4 |
| uji_seed17302 | 672 | 3232 | [2424, 3232] | [1617, 3232] | 12 | 4 |
| uji_seed17303 | 672 | 3232 | [2424, 3232] | [1617, 3232] | 12 | 4 |
| uts_seed17301 | 1052 | 5000 | [3750, 5000] | [2500, 5000] | 12 | 4 |
| uts_seed17302 | 1052 | 5000 | [3750, 5000] | [2500, 5000] | 12 | 4 |
| uts_seed17303 | 1052 | 5000 | [3750, 5000] | [2500, 5000] | 12 | 4 |

## 尚未运行/不能报告

- TUJI 完整 9 组、每组 52,488 候选加一行参考：NOT COMPLETED。
- TUJI 逐位置及配对 CAL-eval MAE、ES/W1、NLL、BF accuracy、BF-CE、P95/P99/最坏站点：NOT RUN。
- UTS/UJI 定位运行、TEST 评分、新训练及 Git 提交/上传：NOT RUN。
- 已核验 138 份源码/测试/协议、9 份冻结 TUJI checkpoint、7 份原训练物理文件及三份预检产物在本次执行后身份未变。
- 原目录现已有预检与失败证据。按 ITER3 第 1 节 existing outputs 应停止并上报；如用户批准最小修复后继续，应保留该目录，使用新的唯一输出目录重新编译、测试、九组预检，再从 seed17301/C 完整执行 TUJI，不与失败尝试混合。
- 历史 CAL-eval 已暴露；任何后续完成结果只作回顾性报告，不代表独立盲测。此处尚无可解释的 ITER3 定位收益。
