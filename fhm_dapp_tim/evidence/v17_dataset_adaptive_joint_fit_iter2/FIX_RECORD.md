# Iter2 最小实现修复记录

## 01：CAL-fit 行分块的站点索引未同步切片

- 首次 TUJI 启动在 `evaluate_calibration_grid` 的 `scatter_add_` 抛出 RuntimeError：站点索引为 `[18,215]`，当前分块误差为 `[18,128]`。完整原始 traceback 见 `run_logs/04_run_tuji.stderr.log`；退出码为 1。
- 原因：误差按行分块计算，站点索引却仍使用整份 CAL-fit 数据。原 22 项测试都使用单个行块，未覆盖这个分支。
- 最小源码改动只有一行：`site_ids[None,:]` 改为 `site_ids[sl][None,:]`。物理站点分组、宏平均公式、几何候选、网格、目标、护栏、随机数和模型均未改变。
- 新增纯合成回归：5 条查询、2 个重复扫描站点、全部 9 个 beta；比较完整行块与不等长的 2/2/1 行块，8 项指标误差不超过 1e-6。修复前该回归复现相同 RuntimeError；修复后全部 23 项测试通过，重新编译通过。
- 首次失败没有写出任何 `CAL_FIT_SELECTION.json`、`CAL_EVAL_REPORT.json` 或候选结果表；尚未进行 CAL-eval 评分。保留失败日志和空的运行目录，随后从 TUJI 开始完整执行修复后的实验，不混用失败结果。
- 修改前后 SHA256 及差异见 `run_logs/FIX_01.json`、`run_logs/FIX_01.patch`。历史科学模块、历史证据及 checkpoint 不作修改。

本修复属于张量索引正确性修复，不改变科学定义。修复后的同一源码用于 TUJI、UTS、UJI，数据集之间不更换实现或规则。

## 02：UTS float32 准确率汇总导致 clean 边界误判（仅提案，尚未应用）

TUJI 9/9 完成后，UTS seed17301/C 在 CAL-fit 选择阶段因所有候选不可行而退出。同一 clean 参数的 BF accuracy 在基线为 1.0、在 CUDA 分块候选统计中为 0.9999999403953552；BF-CE 护栏通过。原始 traceback 与数值诊断分别见 `run_logs/11_run_uts.stderr.log`、`run_logs/13_diagnose_uts_accuracy.stdout.log`。

已准备 `run_logs/FIX_02_PROPOSED.patch`：按站点 int64 正确行数 / 站点行数再做 float64 宏平均，不修改准确率门槛、容差、任何预测或参数。四项新增 CPU/GPU 合成回归在旧实现下均失败；只在临时预览模块中应用此补丁后，原23项加新4项共27项测试通过。冻结科学源码仍保持第一项修复后的 SHA256 `cef5c13027e38b1fef8c29f64f8ca1c0aa7ba173507b251987dd0a8c48c1f146`。

因用户原指令禁止在三个数据集之间临时改源码，第二项修复等待明确例外授权；UTS 未评分 CAL-eval，UJI 未启动。建议保留本目录全部证据，在新目录按同一修订源码从 TUJI 开始完整重启，避免三个域混用版本。详见 `STOPPED_RUN_REPORT_CN.md`。
