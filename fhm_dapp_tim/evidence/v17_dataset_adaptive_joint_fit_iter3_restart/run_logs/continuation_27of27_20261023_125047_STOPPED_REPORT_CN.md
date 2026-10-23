# ITER3 UTS→UJI 续跑：UTS 门禁失败并停止

状态：FAILED / STOPPED；没有完成三域 27/27。未修改源码、参数、容差、角色或 checkpoint；未重试，UJI 未启动。

## 实际执行

```powershell
D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_dataset_adaptive_joint_fit_v1 run --dataset uts --output fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart --device cuda
```

- 开始：2026-10-23T12:51:27.527232+08:00；结束：2026-10-23T12:54:18.068291+08:00；耗时：170.541060 秒；退出码：1。
- Python：D:\miniminiconda\envs\liuyan\python.exe；Torch：2.10.0+cu128；CUDA：12.8；GPU：NVIDIA GeForce RTX 2060。
- 此次 UTS 正式启动 1 次；UJI 启动 0 次。先前 TUJI 9/9 结果保留，未重跑。

## 失败证据

首个 UTS seed17301/C 在 `_assert_clean_anchor_feasible` 失败（run:1141，门禁错误:829）。日志显示前三个 clean anchor 的 MAE 各比 reference 高 1.430511474609375e-6 m，超过冻结的 1e-6 m；BF accuracy 与 BF-CE 护栏均为 true。
程序先完成 CAL-fit 网格计算再检查此门禁，随后才会保存选参并构造 CAL-eval pack。因此本次没有生成 UTS 完整选参、CAL-eval 或候选 CSV。原错误处理只打印前三个失败项，不能据此声称已知全部失败项数量。
批量计算与 reference 使用不同的浮点计算/归约路径，数值差异是待验证的可能原因；本轮没有运行合成复现、修复或放宽阈值，不能把猜测当成已证实根因。

## 完整性与未运行项目

- TUJI 9/9，UTS 0/9（FAILED），UJI 0/9（NOT RUN），总计 9/27。
- 标准报告：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\CAL_EVAL_REPORT.md`，实际仅含 TUJI；原 INCOMPLETE 9/27 状态未更改。
- 138 份源码、27 份 checkpoint、7 份训练输入、3 份预检、36 份 TUJI 原始文件均与续跑前 SHA256 相同。
- 之前交付清单中的 72 份文件、首次失败目录 45 份文件、根目录 7 份标准汇总全部原样；汇总前备份的 SHA256 也一致。
- internal_test_rows_scored=0、official_test_rows_scored=0；依据原程序角色守卫及输出元数据核验，没有安装系统调用级文件读取跟踪。
- BF-CE 单侧惩罚及选择规则源码未变，但 UTS 未写出完整候选 CSV，因此不能独立复算本次全部候选的目标值。
- 没有启动额外聚合命令，没有手工改写标准科学报告，没有 Git 提交/推送操作。

## 各域原冻结预检支持范围（不冒充完成的运行审计）

| 数据域 | FIT锚点 | 基础边 | S1范围 | S2范围 | S2最低要求 |
|---|---:|---:|---|---|---:|
| tuji | [646] | [2946] | [2209, 2946] | [1473, 2946] | [1326] |
| uts | [1052] | [5000] | [3750, 5000] | [2500, 5000] | [2250] |
| uji | [672] | [3232] | [2424, 3232] | [1617, 3232] | [1455] |

## 已有 TUJI 对照（3 seeds 均值，未重新评分）

| Backbone | λ | ΔMAE | ΔES | ΔW1 | ΔNLL | ΔBF-CE | ΔP95 | ΔP99 | Δworst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 0.02 | +0.0109301805 | +0.0292764336 | -0.142871751 | -0.0137957291 | -1.10491849e-07 | +0.218490742 | +0.110282247 | +0.190337576 |
| C | 0.05 | +0.0109301805 | +0.0292764336 | -0.142871751 | -0.0137957291 | -1.14681484e-07 | +0.218490742 | +0.110282247 | +0.190337576 |
| D | 0.02 | +0.00236455599 | +0.00387042523 | -0.0842289773 | +0.0170555054 | -1.247523e-07 | -0.0167117779 | +0.0658166952 | +0.0906731023 |
| D | 0.05 | +0.00285259883 | -0.00426317773 | -0.0851714399 | -0.00955513681 | +1.96630812e-07 | -0.0137465691 | +0.066300391 | +0.0868046946 |
| E | 0.02 | +0.00588210424 | +0.027429366 | -0.124580367 | +0.0337401836 | -1.81376767e-07 | -0.0128992458 | +0.0803995352 | +0.0984949933 |
| E | 0.05 | +0.00632031759 | +0.0207117595 | -0.125624867 | +0.0058041284 | +1.20649171e-07 | -0.0129591048 | +0.0826916034 | +0.103132261 |

以上差值均为 joint − 同 checkpoint 的 clean baseline。UTS、UJI 的全部对应比较为 NOT AVAILABLE，不从旧实验复制或合成数值。旧组合及 TUJI 更完整比较仍见原标准报告与 RUN_REPORT_ITER3_TUJI_CN.md。

历史 CAL-eval 已暴露；保留的 TUJI 数值仅为回顾性描述性结果，不是新的独立验证。

## 原始日志与核验

- stdout：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_UTS.stdout.log`（空文件，SHA256 已核验）。
- stderr：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_UTS.stderr.log`（完整 traceback，SHA256 已核验）。
- 命令/退出码/耗时：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_UTS.json`。
- 独立停机核验：`D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_dataset_adaptive_joint_fit_iter3_restart\run_logs\continuation_27of27_20261023_125047_STOPPED_VERIFICATION.json`。
