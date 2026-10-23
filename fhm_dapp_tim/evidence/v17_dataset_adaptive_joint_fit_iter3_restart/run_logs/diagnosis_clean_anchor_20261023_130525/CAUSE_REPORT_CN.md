# UTS clean-anchor 门禁：实证根因诊断

结论：与 TUJI 先前问题同属数值实现不一致，但不是同一处 float64/float32 距离截断错误。这次主要是两条 float32 点期望计算路径不同，导致数学恒等的 clean 对照产生不同浮点结果。

## 证据

- 原冻结 UTS seed17301/C checkpoint，仅重建原 CAL-fit 的 444 条扫描；physical=1052，首个几何虚拟分量=2500。没有运行正式 run 命令、没有 CAL-eval/TEST，也没有选新参数。
- 为保持原 CUDA 批量形状，孤立取出包含 clean 行的原 2 组参数×9 beta，共 18 行，只分析原 clean 行。没有在这些行间重新选参。
- 真实诊断逐位复现原报错：reference=3.971050262451172，grid=3.9710516929626465，差 1.430511474609375e-6 m。
- clean 的 alpha_prior=alpha_obs=beta=0；两路径 parent logits 完全相同，虚拟分量总质量严格为 0。不是边数变化、模型变化或真实引入了新校正。

| 隔离操作 | MAE m | 与原基准之差 m | 原 1e-6 门禁 |
|---|---:|---:|---|
| 原始基准：物理分量乘法后求和 | 3.971050262451172 | +0 | 通过 |
| 冻结候选：einsum | 3.971051692962646 | +1.430511474609375e-06 | 失败 |
| 候选概率/分量不变，仅点求和对齐 | 3.971050024032593 | -2.384185791015625e-07 | 通过 |
| 原候选误差不变，仅改用基准宏平均 | 3.971051692962646 | +1.430511474609375e-06 | 失败 |
| 原候选误差不变，仅最后宏平均升 float64 | 3.971051733947829 | +1.471496656968441e-06 | 失败 |

## 定位到的运算差异

1. 主因：候选 `evaluate_calibration_grid` 使用 `torch.einsum("ckbm,bmd->ckbd", ...)` 计算点位置；历史基准 `spatial_metrics` 使用 `(q[...,None] * xy).sum(1)`。float32 的乘加归约顺序不同；同一 q 也能产生不同点预测。实际数据同 q 时最大点差为 6.675763142993674e-05 m。
2. 次要差异：beta=0 的候选仍向 softmax 拼入 2500 个 -inf；基准直接在 1052 个物理分量上 softmax。数学上相同，但 CUDA 浮点归约形状不同，真实最大概率差为 4.172325134277344e-07。虚拟概率仍严格为零。
3. 候选用 GPU float32 扫描权重/分块累加；基准用 NumPy 站点内平均再宏平均。这是另一处实现差异，但本例单独替换它不改变失败结果。仅把最后的平均升成 float64 仍失败，所以不能把“末端累加改 double”直接当作修复。

## 合成复现与边界

- synthetic_cpu_torch.float32：原路径 MAE 差 +1.9073486328125e-06 m，固定 q 点计算最大差 9.1552734375e-05 m。
- synthetic_cuda_torch.float32：原路径 MAE 差 +1.9073486328125e-06 m，固定 q 点计算最大差 9.163217328023165e-05 m。
- synthetic_cuda_torch.float64：原路径 MAE 差 +1.963511095937065e-07 m，固定 q 点计算最大差 1.694908809479978e-13 m。
- CPU 合成例中两路径 q 完全相同，仍出现 1.9073486328125e-6 m 的 MAE 差，直接证明点求和实现自身即可触发门禁。
- 合成 double 例中点计算差降至约 1.7e-13 m；其宏平均仍有约 1.96e-7 m 差，是因为样本权重先以 float32 构造再转成 double，仍与整数计数定义的站点平均有区别。该例不能宣传为“全链路 double 已完全恒等”。

## 与 TUJI 旧错误的区别

- TUJI：预检 float64 距离与运行 float32 距离在半径边界保留了不同的 S2 边；先前两行修复已统一几何距离为 float64。
- 本次 UTS：几何门禁已通过，失败发生在 clean MAE 对照。原几何修复仍在，不能再用“边数精度修复”解释此次失败。

## 修复方向（未实施）

应让候选和 reference 共享点期望与站点宏平均的数值定义；beta=0 还应遵循原 reference 的物理分量 shortcut。需要用同一 q、零质量扩展、非整齐站点扫描数、CPU/CUDA 和真实失败 clean 行的回归证明恒等边界。
本次仅证明点求和对齐能解除当前 seed17301/C 的 clean 行误触，尚未证明一个补丁可覆盖其余 UTS/UJI checkpoint。没有应用补丁，没有放宽 1e-6 门禁，没有用 reference 数值覆盖候选分数，也没有恢复正式 run。

## 保全与文件

- 138 份源码、27 份 checkpoint、7 份训练输入、3 份预检及 117 份先前科学产物 SHA256 均未改变；模型 grad 均为 None。
- 旧 STOPPED_REPORT 保留当时“根因未证实”的记录；本报告补充后续已实证结果，不回写历史报告。
- synthetic_RESULT.json：CPU/CUDA/float64 合成隔离及逐扫描误差差值。
- isolated-real_RESULT.json：真实 clean 行隔离、逐扫描差值、角色与 checkpoint 身份。
- diagnose_iter3_clean_mae.py：两次实际执行的诊断脚本原样副本。
- DIAGNOSIS_INTEGRITY.json：只读保全与核心结论。
- 完整 stdout/stderr/命令/退出码仍在上一级 run_logs 的 diagnosis_clean_anchor_20261023_130525_SYNTHETIC.* 和 *_ISOLATED_CAL_FIT.*。
