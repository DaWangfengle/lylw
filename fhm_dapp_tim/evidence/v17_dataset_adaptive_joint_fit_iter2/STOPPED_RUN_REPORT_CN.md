# Iter2 停止与待授权修复报告

## 当前结论

本轮尚未完整完成。TUJI 9/9 已完成并通过独立复核；UTS 首个 checkpoint 的 CAL-fit 选择失败，尚未生成 UTS 选择/评价结果；UJI NOT RUN。不得将现有 9/27 结果写成三域成功。

## 已执行及原始记录

- 初始编译成功，初始单测 22 passed；首次 TUJI 在行分块站点索引处 RuntimeError。仅修复一行索引切片并增加回归：新回归修复前 1 failed，修复后完整 23 passed；重新编译与预检通过，三份几何预检产物逐字节不变。见 FIX_RECORD.md 和 run_logs/01 至 08。
- 修复后 TUJI 完整运行退出码 0，612.795 秒；独立只读核算退出码 0，9/9 PASS。clean 历史复现最大偏差 0，站点宏平均复算最大差 4.050308379799361e-7。几何、CAL-fit 全候选、Pareto、近优敏感性和 CAL-eval 已保存。
- UTS 正式 run 退出码 1，94.869 秒：no feasible candidate for lambda=0.02。完整 traceback 见 run_logs/11_run_uts.stderr.log。入口只在选择成功后写候选表，因此这次 UTS 没有可冒充完成结果的 CSV/JSON。
- 之后只对 UTS CAL-fit 的原 clean 边界做一次小规模诊断，未评分 CAL-eval。见 run_logs/13_diagnose_uts_accuracy.stdout.log；退出码 0。
- 首次诊断辅助脚本因 sys.path 缺少仓库根而退出，调整的只是 tmp 辅助脚本；原始错误仍留在 run_logs/12_diagnose_uts_accuracy.stderr.log。
- 新合成回归在冻结实现下 4 failed，证明准确率归约错误；预览补丁只装入合成测试进程，现有 23 项加 4 项新回归共 27 passed。冻结科学源码尚未修改。见 run_logs/14、15。
- 第14步日志辅助程序在已保存原始 stdout/stderr、退出码和 hash 后，显示控制台片段遇到 GBK/UTF-8 编码异常；没有丢失原始测试失败。仅将辅助程序控制台编码错误处理改为 backslashreplace，不影响实验命令、原始日志或模型。

## UTS 的实际数值证据

| 同一原 clean 参数 | baseline 路径 | 候选循环路径 |
|---|---:|---:|
| BF accuracy | 1.0 | 0.9999999403953552 |
| BF-CE | 0.010027704760432243 | 0.010027793236076832 |
| MAE m | 3.971050262451172 | 3.971050500869751 |
| Joint NLL | 10.846961118172908 | 10.846961975097656 |

参数完全相同：T=0.9805950440880841，theta_v=theta_s=1.5，alpha_prior=alpha_obs=beta=0。CAL-fit 为 69 站点 / 444 行。BF-CE 护栏通过；唯一误拒来自 CUDA float32 加权分块求和的准确率舍入。即使所有查询分类都正确，原归约也可得到小于 1 的值。因此这里不能通过扩大 BF accuracy 容差或放松门槛解决。

## 已准备但未应用的最小补丁

`run_logs/FIX_02_PROPOSED.patch`：对每个站点以 int64 累加分类正确行数，除以原站点行数后，用 float64 计算站点宏平均；维持原 BF accuracy 不下降规则和 1e-12 比较完全不变。预测概率、模型、空间候选、T/theta 网格、BF-CE 目标、MC 流与其他指标公式不变。

`run_logs/test_fix02_proposed.py`：CPU/GPU、全对/一条错误四种纯合成情形；九种 beta 均检查精确站点宏准确率，包含不等扫描数的 69 站点 / 444 行。旧实现四项均失败；仅内存/临时模块预览补丁后 27 项通过。这不是正式 UTS 重跑，也没有选择参数或评分 CAL-eval。

## TUJI 已完成结果（三种子均值，仅描述性）

| Backbone | 方法 | MAE m | ES m |
|---|---|---:|---:|
| C | joint_calibrated_base | 1.865847 | 1.316840 |
| C | joint_lambda_0.02 | 1.874044 | 1.347751 |
| C | joint_lambda_0.05 | 1.874661 | 1.337827 |
| D | joint_calibrated_base | 1.814602 | 1.289308 |
| D | joint_lambda_0.02 | 1.821194 | 1.310769 |
| D | joint_lambda_0.05 | 1.822575 | 1.302546 |
| E | joint_calibrated_base | 1.794441 | 1.270289 |
| E | joint_lambda_0.02 | 1.796894 | 1.288807 |
| E | joint_lambda_0.05 | 1.799224 | 1.282024 |

两个 lambda 在 TUJI 的三个 backbone 平均 MAE/ES 均未优于 clean，负结果原样保留；不能据此改网格。名义 472,392 个 TUJI 组合中，279,936 个正 beta/空 S2 边组合按原实现判为无定义并排除，剩余有效 192,456 个，另加9行比较基线。

## 为什么现在需要明确授权

用户附件 [V17_DATASET_ADAPTIVE_JOINT_FIT_CODEX.txt](D:/liulanqi/V17_DATASET_ADAPTIVE_JOINT_FIT_CODEX.txt) 写明：“不得在三个数据集之间临时改代码或选择规则。”TUJI 已正式完成，因此不能自行修改冻结源码再继续 UTS。

建议的可审阅例外：完整保留本 iter2 所有结果、失败日志和冻结清单；授权上述最小准确率汇总修复及四项回归；在全新输出目录重跑编译、完整测试、预检并从 TUJI → UTS → UJI 完整重启，确保最终三个域使用同一修订源码。不改候选、护栏、目标、随机数或模型，不覆盖本轮任何不利结果。

## 数据与证据边界

冻结208份源码/测试/协议hash、27份原checkpoint hash、141份旧组合产物hash仍全部匹配。无backbone训练，无optimizer更新；internal_test_rows_scored=0，official_test_rows_scored=0。CAL-eval 是历史已暴露的训练文件内部角色，非独立盲测；不把三个共享站点的种子视为独立数据采集。没有执行 Git 命令。
