# Iter2 本地完整执行报告

TUJI → UTS → UJI 已按顺序完成，每域 3 seeds × C/D/E，共 27 个 checkpoint。只进行冻结 backbone 的 CAL-fit 联合参数搜索与 CAL-eval 报告；训练和 optimizer step 均为 0。

## 环境与失败修复

- Python：3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；解释器：`D:\miniminiconda\envs\liuyan\python.exe`。
- Torch：2.10.0+cu128；CUDA runtime：12.8；GPU：NVIDIA GeForce RTX 2060，总显存 6.00 GiB。
- NumPy 2.4.4，SciPy 1.17.1，pandas 3.0.1，pytest 9.0.3。
- 初始编译和 22 项测试通过；首轮 TUJI 因行分块索引错误退出，未写出任何选择或 CAL-eval 结果。新增合成回归先复现错误，随后只修改一行站点索引切片。修复后编译、23 项测试和预检全部通过；随后 UTS 暴露准确率归约问题，用户授权第二项最小修复并在新目录完整重启，正式完整测试 27 项全部通过。详见 `FIX_RECORD.md` 与保留在前一 iter2 目录的原始失败日志。
- 之后三域使用完全相同的冻结源码与规则。修复前/后 hash 及 diff 全部保留，不修改历史科学模块、历史证据或模型。

## 实际命令与耗时

| 命令 | 退出码 | 耗时秒 |
|---|---:|---:|
| `D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 0.110 |
| `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 6.430 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 preflight --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2` | 0 | 29.116 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset tuji --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2 --device cuda` | 1 | 8.121 |
| `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v17_dataset_adaptive_joint_fit_v1.py::test_calibration_row_chunks_preserve_site_aggregation_for_all_betas` | 1 | 5.681 |
| `D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 0.105 |
| `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 5.311 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 preflight --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2` | 0 | 28.374 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset tuji --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2 --device cuda` | 0 | 612.795 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_readonly_audit.py --dataset tuji` | 0 | 8.359 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uts --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2 --device cuda` | 1 | 94.869 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_diagnose_uts_accuracy.py` | 1 | 2.675 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_diagnose_uts_accuracy.py` | 0 | 8.932 |
| `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2/run_logs/test_fix02_proposed.py` | 1 | 6.018 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_fix02/validate_preview.py` | 0 | 5.783 |
| `D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/fhm_dapp/msep_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 0.108 |
| `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_v17_dataset_adaptive_joint_fit_v1.py` | 0 | 6.126 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 preflight --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart` | 0 | 30.161 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset tuji --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart --device cuda` | 0 | 616.124 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_readonly_audit.py --dataset tuji` | 0 | 8.454 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uts --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart --device cuda` | 0 | 1312.007 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_readonly_audit.py --dataset uts` | 0 | 17.090 |
| `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uji --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart --device cuda` | 0 | 1847.503 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_readonly_audit.py --dataset uji` | 0 | 16.305 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_readonly_audit.py --dataset all --final` | 0 | 40.335 |
| `D:\miniminiconda\envs\liuyan\python.exe tmp/v17_iter2_root_counts.py` | 0 | 14.262 |

以上各命令的用时相加为 4731.2 秒，包含失败与回归复现。部分只读审计与主命令重叠，因此该累计数不是总墙钟或纯 GPU 时间。逐命令 UTC 起止时间、stdout/stderr SHA256 见 `run_logs/`。

## FIT 几何与候选覆盖

| 域 | FIT 锚点 | CAL-fit 行 | CAL-eval 行 | k1 P50 m | k8 P50 m | 几何配置 | 无限 tau_virt 对照 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuji | 646 | 215 | 234 | 0.595000 | 1.332029 | 12 | 6 |
| uts | 1052 | 444 | 414 | 2.698278 | 8.153282 | 12 | 6 |
| uji | 672 | 959 | 919 | 3.769962 | 19.009120 | 12 | 6 |

所有有限空间尺度均可追溯到本域 FIT k=1/2/4/8 或 union-kNN 边长分位数乘以固定无量纲倍率，没有混入跨域米制列表。各 seed 的详细分位数及逐建筑楼层统计见原始预检文件；每个最终尺度的全部匹配来源见 `SELECTED_PARAMETER_SUMMARY.csv`。全部配置满足 R_max_S2 ≤ R_max_S1。

名义候选总数为 1,417,176。由于部分 FIT 分位数半径没有任何 S2 边，源码原有的正确性约束排除了正 beta、空虚拟支持的 279,936 个无定义组合；可定义并已完成数值评价的候选为 1,137,240，另有 27 行比较基线。这不代表全部通过 BF 护栏；筛选后的可行数另见覆盖表和只读核验 JSON。没有放宽半径、补造边或用固定米数回退。每组几何的名义/已评价/可行/排除数与原因见 `CANDIDATE_COVERAGE.csv`，完整指标见每 checkpoint 及根目录的 `CANDIDATE_METRICS.csv`。

每个 checkpoint 的 T/theta 精确中心、实际三值邻域与来源见 `CALIBRATION_GRID_SUMMARY.csv`；两种 lambda 的全部最终参数见 `SELECTED_PARAMETER_SUMMARY.csv`。几何与校准在同一候选循环中选择；未采用 sequential freezing。

BF-CE 归一化使用 max(bfbase,0.05)，护栏为 candidate BF-CE ≤ 1.02*bfbase+0.005，并保留 BF accuracy 不下降约束。每个非 beta 参数块及行块的九种 beta 共享 prior/observation 各一次 sparse.mm，完成图乘法后才广播 beta；原始计数回归测试仍通过。

## CAL-eval 三种子描述性汇总

以下各行是同域同 backbone 的三种子均值，不从 CAL-eval 选择 lambda 或部署方案。完整逐种子数值、BF-CE、站点配对向量与所有单结构/旧组合对照见 `CAL_EVAL_REPORT.md` 和各 `CAL_EVAL_REPORT.json`。

| 域 | Backbone | 方法 | MAE m | ES m | Joint NLL | BF 准确率 | P95 m | P99 m | Worst-site m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tuji | C | joint_calibrated_base | 1.865847 | 1.316840 | 3.963951 | 1.000000 | 2.820292 | 3.330369 | 3.406210 |
| tuji | C | joint_lambda_0.02 | 1.874044 | 1.347751 | 3.958098 | 1.000000 | 3.041642 | 3.409244 | 3.552153 |
| tuji | C | joint_lambda_0.05 | 1.874661 | 1.337827 | 3.809544 | 1.000000 | 3.019805 | 3.379943 | 3.519938 |
| tuji | D | joint_calibrated_base | 1.814602 | 1.289308 | 3.818353 | 1.000000 | 2.642150 | 3.224211 | 3.535970 |
| tuji | D | joint_lambda_0.02 | 1.821194 | 1.310769 | 3.812724 | 1.000000 | 2.649972 | 3.306558 | 3.619665 |
| tuji | D | joint_lambda_0.05 | 1.822575 | 1.302546 | 3.769620 | 1.000000 | 2.656302 | 3.307232 | 3.612616 |
| tuji | E | joint_calibrated_base | 1.794441 | 1.270289 | 3.764256 | 1.000000 | 2.762531 | 3.142886 | 3.302837 |
| tuji | E | joint_lambda_0.02 | 1.796894 | 1.288807 | 3.803972 | 1.000000 | 2.759821 | 3.239914 | 3.416626 |
| tuji | E | joint_lambda_0.05 | 1.799224 | 1.282024 | 3.578053 | 1.000000 | 2.775557 | 3.216053 | 3.387438 |
| uts | C | joint_calibrated_base | 3.984811 | 3.316599 | 10.870311 | 0.995169 | 9.320024 | 16.606908 | 19.420777 |
| uts | C | joint_lambda_0.02 | 4.020535 | 3.020864 | 5.976526 | 0.990338 | 9.046867 | 17.816484 | 20.453249 |
| uts | C | joint_lambda_0.05 | 4.020535 | 3.020864 | 5.976526 | 0.990338 | 9.046867 | 17.816484 | 20.453249 |
| uts | D | joint_calibrated_base | 4.012157 | 3.361293 | 10.848249 | 0.995169 | 9.221376 | 14.414030 | 17.434537 |
| uts | D | joint_lambda_0.02 | 4.065869 | 3.151025 | 6.500266 | 1.000000 | 8.994134 | 15.113735 | 18.337839 |
| uts | D | joint_lambda_0.05 | 4.066825 | 3.130155 | 6.408002 | 1.000000 | 8.922293 | 15.124771 | 18.307133 |
| uts | E | joint_calibrated_base | 3.968466 | 3.323526 | 10.849422 | 1.000000 | 8.924485 | 14.534329 | 16.816946 |
| uts | E | joint_lambda_0.02 | 4.032471 | 3.237766 | 7.824875 | 1.000000 | 9.134772 | 15.168152 | 18.256849 |
| uts | E | joint_lambda_0.05 | 4.030502 | 3.199635 | 6.600242 | 1.000000 | 9.085411 | 15.239654 | 18.267547 |
| uji | C | joint_calibrated_base | 7.407989 | 6.249493 | 14.481042 | 0.963020 | 16.745717 | 22.012816 | 25.237769 |
| uji | C | joint_lambda_0.02 | 7.413224 | 5.616314 | 10.615061 | 0.962873 | 16.233922 | 20.814334 | 23.180927 |
| uji | C | joint_lambda_0.05 | 7.413224 | 5.616314 | 10.615061 | 0.962873 | 16.233922 | 20.814334 | 23.180927 |
| uji | D | joint_calibrated_base | 7.106555 | 6.069481 | 14.501097 | 0.961042 | 16.134727 | 19.088859 | 19.753939 |
| uji | D | joint_lambda_0.02 | 7.053564 | 5.461430 | 10.699062 | 0.966514 | 15.506189 | 19.123153 | 19.890389 |
| uji | D | joint_lambda_0.05 | 7.066031 | 5.461981 | 10.695990 | 0.966324 | 15.511006 | 19.146795 | 19.926979 |
| uji | E | joint_calibrated_base | 7.412981 | 6.309456 | 14.484261 | 0.965483 | 16.951221 | 25.144593 | 29.679303 |
| uji | E | joint_lambda_0.02 | 7.298350 | 5.584265 | 10.583739 | 0.968913 | 16.558727 | 22.897138 | 26.055097 |
| uji | E | joint_lambda_0.05 | 7.298350 | 5.584265 | 10.583739 | 0.968913 | 16.558727 | 22.897138 | 26.055097 |

## 相对 clean 的配对变化与结构是否被采用

| 域 | Backbone | lambda | ΔMAE m | ΔES m | ΔNLL | ΔBF 准确率 | ΔP95 m | ΔP99 m | ΔWorst-site m | S1 active /3 | beta>0 /3 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tuji | C | lambda_0.02 | +0.008197 | +0.030911 | -0.005853 | +0.000000 | +0.221349 | +0.078874 | +0.145943 | 3 | 0 |
| tuji | C | lambda_0.05 | +0.008814 | +0.020987 | -0.154407 | +0.000000 | +0.199513 | +0.049574 | +0.113728 | 3 | 1 |
| tuji | D | lambda_0.02 | +0.006592 | +0.021461 | -0.005629 | +0.000000 | +0.007822 | +0.082346 | +0.083695 | 2 | 1 |
| tuji | D | lambda_0.05 | +0.007972 | +0.013238 | -0.048734 | +0.000000 | +0.014152 | +0.083020 | +0.076646 | 2 | 1 |
| tuji | E | lambda_0.02 | +0.002453 | +0.018518 | +0.039716 | +0.000000 | -0.002710 | +0.097028 | +0.113789 | 3 | 0 |
| tuji | E | lambda_0.05 | +0.004783 | +0.011735 | -0.186204 | +0.000000 | +0.013026 | +0.073167 | +0.084601 | 1 | 2 |
| uts | C | lambda_0.02 | +0.035725 | -0.295735 | -4.893786 | -0.004831 | -0.273157 | +1.209577 | +1.032472 | 0 | 3 |
| uts | C | lambda_0.05 | +0.035725 | -0.295735 | -4.893786 | -0.004831 | -0.273157 | +1.209577 | +1.032472 | 0 | 3 |
| uts | D | lambda_0.02 | +0.053712 | -0.210269 | -4.347983 | +0.004831 | -0.227242 | +0.699705 | +0.903302 | 0 | 3 |
| uts | D | lambda_0.05 | +0.054668 | -0.231138 | -4.440247 | +0.004831 | -0.299084 | +0.710741 | +0.872595 | 0 | 3 |
| uts | E | lambda_0.02 | +0.064005 | -0.085760 | -3.024546 | +0.000000 | +0.210287 | +0.633823 | +1.439903 | 1 | 2 |
| uts | E | lambda_0.05 | +0.062036 | -0.123891 | -4.249179 | +0.000000 | +0.160926 | +0.705325 | +1.450601 | 1 | 3 |
| uji | C | lambda_0.02 | +0.005235 | -0.633178 | -3.865981 | -0.000147 | -0.511795 | -1.198482 | -2.056842 | 3 | 3 |
| uji | C | lambda_0.05 | +0.005235 | -0.633178 | -3.865981 | -0.000147 | -0.511795 | -1.198482 | -2.056842 | 3 | 3 |
| uji | D | lambda_0.02 | -0.052990 | -0.608052 | -3.802034 | +0.005471 | -0.628538 | +0.034294 | +0.136450 | 2 | 3 |
| uji | D | lambda_0.05 | -0.040524 | -0.607500 | -3.805107 | +0.005282 | -0.623721 | +0.057936 | +0.173040 | 2 | 3 |
| uji | E | lambda_0.02 | -0.114631 | -0.725191 | -3.900522 | +0.003430 | -0.392494 | -2.247455 | -3.624206 | 2 | 3 |
| uji | E | lambda_0.05 | -0.114631 | -0.725191 | -3.900522 | +0.003430 | -0.392494 | -2.247455 | -3.624206 | 2 | 3 |

负的 MAE/ES/NLL/尾部差值表示改善，正的 BF accuracy 差值表示改善。最差站点列是三种子各自 worst-site 的均值，不能把它当作三种子共同的同一个站点。beta=0 时没有采用虚拟场；alpha 两项均为 0 时没有采用推断图扩散。此时相应空间参数只是边界/破平局标识，不能宣称其已被数据精确辨识。

## 独立只读核验与科学限制

- 27 个完整 checkpoint 结果均通过独立 CSV 目标函数/可行性/选择复算；每个方法站点键一致，站点扫描数汇总成立。最大站点→宏平均差 1.11775624312e-06；clean 对历史数值最大偏差 0。
- 208 个执行源码/测试/协议 SHA256、27 个 checkpoint SHA256、7 份训练物理文件 SHA256，以及 141 份旧组合产物 SHA256 核验通过。源 hash 与输入 hash 分别见 `EXECUTION_FROZEN.json`、`INPUT_IDENTITY.json`。
- CAL-fit/CAL-eval 站点角色由原冻结重建路径核对且无交叉；TEST 的角色键只用于泄漏防护，不构造 TEST 预测包。TUJI 物理 MAT 容器同时含测试变量，原加载器只请求 results_train；完整容器会被打开和哈希，不能声称测试容器所有字节从未接触。
- internal_test_rows_scored=0；official_test_rows_scored=0。没有训练 backbone，没有 optimizer 更新，没有 Git 操作。
- 所有候选评价指标有限；JSON 中字符串 Infinity 仅表示预声明无界参考/无限虚拟带宽，不是数值失效。
- CAL-eval 属于此前已暴露的训练文件内部角色，只能视作回顾性开发证据。三个 seeds 共享物理站点与参考资源，不是三次独立数据采集；本报告不制造新的独立显著性结论。
- ES 使用原连续 Student-t 分布、原固定随机流与每流 32 次抽样，是 Monte Carlo 近似。小幅 ES 差异不能脱离抽样不确定性解释；站点间空间与设备/会话相关仍是局限。
- 保留全部不利结果和两种 lambda，不从本轮 CAL-eval 改选候选或补调网格。Pareto 与近优敏感性仅来自 CAL-fit，见原始 CSV。完成实验不等于三结构已修复或取得独立泛化证明。


## 描述性结果观察（不构成新的晋级门槛）

保留两个预声明 lambda。下列范围覆盖本域 C/D/E × 两个 lambda 的三种子均值，共六个描述性对照；没有根据 CAL-eval 改选赢家。

- TUJI：ΔMAE [+0.002453, +0.008814] m；ΔES [+0.011735, +0.030911] m；二者同时下降 0/6。
- UTS：ΔMAE [+0.035725, +0.064005] m；ΔES [-0.295735, -0.085760] m；二者同时下降 0/6。
- UJI：ΔMAE [-0.114631, +0.005235] m；ΔES [-0.725191, -0.607500] m；二者同时下降 4/6。

本轮未形成跨三域一致的 MAE/ES 改善。TUJI 未获总体收益；UTS 的 ES 改善伴随 MAE 上升；UJI 的 D/E 均值改善，但 C 的 MAE 略升，且 D 的 P99/最坏位置存在小幅恶化。所有正反结果完整保留。按 BF 护栏筛选后的可行候选数见 `CANDIDATE_COVERAGE.csv` 与 `READ_ONLY_VERIFICATION.json`，不能把已评价数量当成可行数量。
