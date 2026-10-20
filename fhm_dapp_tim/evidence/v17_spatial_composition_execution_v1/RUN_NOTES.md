# V17 Spatial Composition 本地执行说明

执行日期：2026-10-20（Asia/Shanghai）。分支通过只读 `.git/HEAD` 核对为 `v17-spatial-composition-v1`；未执行任何 Git 命令。

正式流水线首次且仅一次调用，退出码 0；27/27 组、每组 13 种方案全部完成。研究是历史内部 CAL-eval 的回顾性比较，不是独立盲测或正式晋级。

## 环境与耗时

Python：`D:\miniminiconda\envs\liuyan\python.exe`；3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]。
Torch 2.10.0+cu128；CUDA 12.8；NVIDIA GeForce RTX 2060，6 GiB；NumPy 2.4.4；SciPy 1.17.1。
正式 all 命令耗时 494.567 秒；全部四条留档命令累计 529.669 秒。
CUDA 确定性算法未启用。ES 是原有连续 Student-t df4、每流 32 次、两流 Monte Carlo 近似；同组沿用 seed+900 的共同随机数，不宣称跨设备/块大小逐位复现。

## 命令与日志

|阶段|退出码|耗时秒|stdout / stderr|
|---|---:|---:|---|
|environment|0|14.479|`environment.stdout.log` / `environment.stderr.log`|
|pre_run_tests|0|16.494|`pre_run_tests.stdout.log` / `pre_run_tests.stderr.log`|
|composition_all|0|494.567|`composition_all.stdout.log` / `composition_all.stderr.log`|
|closeout_read_only|0|4.130|`closeout_read_only.stdout.log` / `closeout_read_only.stderr.log`|

完整 argv、起止时间和全部日志 SHA256 保存在 `EXECUTION_RECORD.json`。指定的三个测试文件共 36 项通过。

实际命令：

- `D:\miniminiconda\envs\liuyan\python.exe fhm_dapp_tim/evidence/v17_spatial_composition_execution_v1/audit_environment.py`
- `D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_msep_v17_spatial_operators_v1.py tests/test_msep_v17_spatial_structure_v1.py tests/test_msep_v17_spatial_composition_v1.py`
- `D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_spatial_composition_v1 all --input-root fhm_dapp_tim/evidence/v17_two_stage_freeze_v1 --checkpoint-root fhm_dapp_tim/evidence/v17_spatial_structure_v1 --output fhm_dapp_tim/evidence/v17_spatial_composition_v1 --device cuda`
- `D:\miniminiconda\envs\liuyan\python.exe fhm_dapp_tim/evidence/v17_spatial_composition_execution_v1/closeout_read_only.py`

## 冻结输入与只读复核

27 份 C/D/E 检查点匹配原 Phase T、RESULT 和选择记录；两个历史根目录的 91 份模型/fixture 文件清单与字节哈希均未变化，也没有在本轮输出下新写模型。
200 项旧源码/协议和 500 项旧产物均保持原哈希。训练输入 7 份 CSV/MAT 容器哈希匹配。TUJI 仍只解码 results_train；不宣称含测试变量的 MAT 容器字节从未打开。
所有 CAL-fit 选参文件与 CAL-eval 锁的 SHA、检查点、来源、角色键、行数对应一致。冻结清单包含完整源码与划分哈希。未构造任何 TEST 预测/评分包；内部与官方 TEST 评分均为 0。
从保存的完整网格独立重选 argmin，核验 27 组 T/theta、S1 alpha、独立及条件 S2 beta/gamma 与确定性破平局一致。未再次前向、抽样、训练或调参。
逐站点重新汇总 MAE/ES/NLL/BF/运输风险并核验尾部，最大站点到宏平均差 1.36345675372e-06，沿用协议 2e-5 容差。
四份 CSV 全部核对原 JSON、三种子均值/样本 SD 和逐站点差异；351 行方案指标、117 行汇总、1053 组配对、55809 行站点配对。JSON 全部可解析且无 NaN/Inf。详见 `CLOSEOUT_REVIEW.json`。

|域（各三种子一致）|CAL-fit 扫描|CAL-eval 扫描|CAL-eval 站点|物理锚点|唯一边|边P50/P95/max m|>30m边数/比例|
|---|---:|---:|---:|---:|---:|---|---|
|tuji|215|234|46|646|2946|1.189500/1.787000/2.977001|0 / 0.00000000|
|uts|444|414|69|1052|5000|6.061220/12.072623/21.267714|0 / 0.00000000|
|uji|959|919|44|672|3232|12.984447/41.445354/77.537079|384 / 0.11881188|

原图 k=8、8m 核、对称 Sinkhorn、无距离截断保持不变。每检查点精确参数与完整网格见对应 `CAL_FIT_SELECTION.json`；正式报告含 27 组参数表和完整指标。E 原训练图层/保存 alpha 保留，后处理 S1 是额外推断操作。

## 异常、修复与边界

科学命令没有失败，全部 stderr 为空；本轮没有修改科学源码、测试、算子、模型、种子、网格或科学常数，没有需要实施的 bug 修复。
最初请求的带“(1)”附件路径不存在；当时读取了仓库协议。用户随后提供了不带“(1)”的实际 TXT，已在测试及正式运行前确认其正文与仓库协议相同，两个原始文件 SHA 均在执行记录中保留。
用户中断前的环境审计已完成，正式实验尚未启动；恢复时复用该审计并核验源码未变，没有重跑已评分角色。一次辅助 WMI 进程清单读取被系统拒绝，此操作未读数据/模型、未更改进程，不属于实验失败；记录原样保留。
原 spatial-structure 报告披露的跨阶段数值差异保留原状，本轮不修改或豁免旧结论。原始数据加载为重建固定角色读取训练容器与规范元数据；未把 TEST 行打包、预测或用于选参。
所有负结果、P95/P99、最差站点、种子波动及有害站点差异已完整保留。此说明不依据 CAL-eval 改选部署方法，也不追加新实验。

主报告：`../v17_spatial_composition_v1/RUN_REPORT_V17_SPATIAL_COMPOSITION.md`；全部原始结果及汇总在同目录。原始 stdout/stderr、环境、来源快照与独立只读复核在本执行目录。
