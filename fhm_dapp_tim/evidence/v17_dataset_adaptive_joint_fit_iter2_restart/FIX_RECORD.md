# Iter2 完整重启：实现修复与用户授权

本目录是用户授权后的完整重启，不覆盖前一目录 `../v17_dataset_adaptive_joint_fit_iter2/` 的任何结果。前一目录已保留 TUJI 9 个 checkpoint 的全部正负结果、UTS 失败、原始日志、修复前失败回归和冻结清单。

## 第一项：行分块索引修复

首轮 TUJI 发现站点索引未按误差行块切片：全215行索引对应当前128行误差，引发 `scatter_add_` RuntimeError。仅将 `site_ids[None,:]` 改为 `site_ids[sl][None,:]`。新合成回归在修复前失败，修复后23项通过；几何预检产物逐字节不变。此前尚无选择或CAL-eval结果。

## 第二项：按站点整数正确次数计算 BF accuracy

前一目录的 TUJI 完成后，UTS 首个 checkpoint 报“没有可行候选”。CAL-fit-only 数值诊断显示，同一 clean 参数：baseline BF accuracy=1.0，候选循环的 CUDA float32 加权分块结果=0.9999999403953552，BF-CE 护栏通过。错误来自准确率归约，不是预测变差。

最小修复：以 int64 按站点累加分类正确行数，除以各站点行数，再用 float64 求站点宏平均。保持 BF accuracy 不下降的原定义与 1e-12 比较不变，不扩大容差。预测概率、连续分量、T/theta/alpha/beta/几何候选、目标、BF-CE规则及随机数均不改。

四项新增 CPU/CUDA、全部正确/单行错误的合成回归在旧实现下全部失败；临时预览修复后原23项加新4项共27项通过。收到用户“允许最小数值修复，新目录完整重启（推荐）”的明确授权后，才正式应用第二项补丁并从头运行编译、27项完整测试、预检、TUJI→UTS→UJI。

完整授权在前一目录的 `USER_AUTHORIZATION_FIX02.json`；原始错误与数值诊断在其 `run_logs/11_run_uts.stderr.log`、`run_logs/13_diagnose_uts_accuracy.stdout.log`；提案和回归记录为 `run_logs/FIX_02_PROPOSED.patch`、`run_logs/test_fix02_proposed.py`、`run_logs/14_*`、`run_logs/15_*`。第一项修复 diff 与 hashes 同样保留在前一目录。

本次完整重启的三域统一使用修复后的同一冻结源码。历史科学模块、旧证据与原checkpoint保持不变。修复解决实现正确性问题，不保证任何科学性能改善；所有不利结果照实保留。
