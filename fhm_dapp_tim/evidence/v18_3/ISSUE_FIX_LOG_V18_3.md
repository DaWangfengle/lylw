# V18.3 问题、修复与后续处理记录

记录日期：2026-10-01。阶段日志中的时间采用 UTC。本文件配套 `ISSUE_FIX_LOG_V18_3.json` 保存证据文件 SHA-256、修复状态和数据访问核验。

当前状态：Stage 1–4 已完成，五个 stability seed 均完成 17 轮；Stage 5 在容量检查处中止，尚未开始梯度训练；Stage 6–8 未运行。确认集尚未消费，`confirm` 命令次数为 0，benchmark 命令次数为 0。

分支：`v18-3-structured-consensus`。当前正式实验源码 SHA：`342b73cf77222bd93d13eeb094858d569354f9ad`。该修复已本地提交，当前分支领先远端 1 个提交，尚未推送。本次记录更新没有修改科学源码或冻结配置。

## 1. 评价统计调用 NumPy 失败——已修复并完整重启

首次正式实验使用源码 `d7500d3c7ee563d0c7b201e5a3ac4ee4dc21d1fe`。prepare 成功，train-selection 第一轮训练完成，但 early-stop 评价在原 `uji_v18_3_experiment.py:934` 失败：

```text
RuntimeError: Can't call numpy() on Tensor that requires grad.
```

原因：marked-radio profiler 在预测过程中使用梯度，返回的部分诊断张量仍携带 autograd 图；评价统计直接将其转为 NumPy，触发 API 错误。首次测试和合成 CUDA 预检覆盖了预测、训练和 checkpoint 重建，但没有覆盖完整 `evaluate_shard` 统计路径。

修复：在评价统计边界对预测结果调用 `detach()`，共新增三行，包含两行说明。实际代码位于当前 `uji_v18_3_experiment.py:704`：

```python
result = {name: value.detach() for name, value in result.items()}
```

同时新增 `test_v183_evaluate_shard_handles_autograd_diagnostics`，覆盖真实多 batch 评价、两项 NumPy 相关系数、统计有限值和调用前后梯度状态。没有修改模型结构、预测数值、损失、随机种子、轮数选择规则或校准网格。

验证与执行结果：

- CPU/CUDA 合成验证复现原错误；修复后评价完成，所有统计有限，预测值逐位一致。
- 用户明确授权“同意最小修复并完整重启（推荐）”；授权范围是保留失败记录、提交 detach 修复与回归测试、重跑检查、从 Stage 1 重启。
- 本地提交 `342b73cf77222bd93d13eeb094858d569354f9ad`，提交主题 `Fix V18.3 evaluation of autograd diagnostics`。
- 三个指定 `py_compile` 全通过；指定四个 pytest 文件 `43 passed in 5.92s`，没有删除、skip、xfail 测试。
- 扩展 CUDA 预检加入完整评价统计路径并通过。
- 全部旧登记、shard、日志、失败堆栈、补丁验证及授权记录保存在 `failed_attempts/d7500d3-20261001T081726Z/`；历史记录中的“补丁未应用”描述对应当时状态，保留原文。
- 以新 SHA 从 Stage 1 完整重启，之后完成 prepare、train-selection、calibrate-freeze 和五 seed stability。

证据入口：旧目录 `run_audit/failure_record.json`、`train-selection.log`、`implementation_fix_proposal_verification.json`、`restart_authorization.json`；当前 `run_audit/pytest.log`、`preflight.json`、`source_snapshot.json`。

## 2. outer_refit 容量检查误计零权重填充位——已诊断，补丁尚未应用

Stage 5 `train-confirmation --device cuda` 于 `2026-10-01T09:28:19.739558Z` 开始，`09:28:27.104528Z` 以 exit code 1 中止，耗时 7.355575 秒。错误发生在 `v18_3_single.py:259` 的模型构建容量检查：

```text
ValueError: candidate_union_size must simultaneously contain all registered primary source quotas and the widest Delaunay training target
```

原因：代码使用 `neighbour_indices.shape[1]`，把零权重 self/填充位也算作必须容纳的训练目标。合同要求的是非 self、正权重 Delaunay target。

| 训练角色 | 主检索配额 | 张量填充宽度 | 最宽正权重非 self 目标数 | 旧检查所需容量 | 正目标所需容量 | 注册 L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| member_fit | 80 | 14 | 13 | 94 | 93 | 96 |
| selection_refit | 80 | 14 | 13 | 94 | 93 | 96 |
| outer_refit | 80 | 17 | 16 | 97 | 96 | 96 |

准备的最小补丁改为逐行统计 `(neighbour_weights > 0).sum(...)` 的最大值，再与 80 个主检索配额相加。不会删除正目标、删减配额、修改 L/K 或改变预测计算。补丁保存在 `run_audit/capacity_fix_proposal.patch`，尚未应用、尚未提交。

合成验证已完成：CPU/CUDA 上模型 state、目标索引、权重和预测均逐位一致；16 个正目标加 1 个零权重填充位时 `80+16=96` 通过；17 个正目标时 `80+17=97` 仍拒绝，保持真实容量不足时 fail-closed。

注意：审计辅助脚本 `run_audit/sitecustomize.py:205` 也使用了相同的填充宽度判断；若后续获准修复，须同步修正该断言并记录辅助脚本新 SHA，保留原始阶段日志与原 SHA。不能只修改主模型 guard 后绕过审计检查。

本次失败没有产生 confirmation checkpoint，没有开始梯度训练，没有预测或读取确认集/benchmark。证据：`run_audit/train-confirmation.log`、`train-confirmation_execution.json`、`train-confirmation_runtime.json`、`capacity_diagnosis.json`、`capacity_fix_proposal_verification.json`。

## 3. full refit 可能存在真实容量不足——待精确验证，尚不能下最终结论

仅使用 registry 已登记的坐标支持元数据建立几何图，初步得到 933 个坐标中最宽正目标数为 18，所需容量 `80+18=98`，超过冻结的 `L=96`。该检查没有打开 confirmation shard、RSS 或 benchmark，没有训练或评价模型。

这只是风险提示。正式 final refit 会经历 float32 坐标归一化/反归一化、outer_refit 后接 confirmation 的行序、按扫描行数加权的原点和每个 anchor 坐标均值；这些细节可能影响 Delaunay 图。元数据检查没有完整复现这些细节，因此不能把 98 当作已经确认的正式容量需求。

精确静态预检脚本已准备在 `run_audit/check_full_capacity.py`，尚未执行。拟执行内容：

1. 校验原训练 CSV SHA；仅解析 `LONGITUDE/LATITUDE/BUILDINGID/FLOOR` 四列。
2. 复现正式 float32 坐标处理；与现有 outer_refit shard 的坐标逐位比较。
3. 复现 final refit 的拼接行序、原点与 anchor 坐标均值，仅构建几何目标并计算真实容量。
4. 保存精确容量结果；不构建定位模型、不训练、不评价、不打开 confirmation `.pt`、不读取 benchmark。

该操作仍会在正常确认流程前解析含确认行的坐标标签，因此需要明确协议例外。当前不存在 `capacity_preflight_authorization.json`，没有因准备脚本而提前读取这些坐标列。证据：`run_audit/full_support_metadata_capacity.json`、`check_full_capacity.py`。

## 4. 建议的下一步与协议边界

建议先做上述静态容量预检，再决定 V18.3 能否完成：

- 若精确检查证明 full refit 的最宽正目标数不超过 16：申请/记录冻结后最小实现修复的例外；应用容量计数补丁，补充“边界通过/真实溢出拒绝”回归测试，提交修复，重跑三个 `py_compile`、全部指定 pytest 和 CUDA 预检。按原指令默认从 Stage 1 完整重启，归档当前尝试及全部失败记录。
- 若真实目标数超过 16：当前固定配额与 L=96 的合同无法满足全量训练容量要求。保存 V18.3 失败结论，保持确认集未消费；改变 L、检索配额或目标定义需要另立 V18.4 合同与授权，不能当作本次计数修复的一部分。

之前提出的“保留前四阶段、从 Stage 5 续跑”仅是另一个待授权例外方案，尚未执行；它需要额外允许偏离“修复后从 Stage 1 重启”的原规定，并清楚记录各阶段不同源码 SHA。当前建议仍优先遵循原指令完整重启。

权限依据来自用户提供的正式执行指令：第六节“正式 prepare 前仅允许修复明确实现 bug”，并要求修复后 commit、重跑检查、记录新 SHA、“从 Stage 1 重新开始”；第十节“freeze 后正式模型不可修改”；第十二节 Stage 5“不访问 confirmation/benchmark labels”。首次 detach 的授权有明确范围，不自动扩展为本次坐标预检、容量修复或续跑授权。

## 5. 记录与复现原则

完整阶段结果、轨迹与缺失项在 `RUN_STATUS_V18_3.md/json`。本文件集中记录异常触发位置、根因、修复范围、验证证据、提交、授权与重启方式。配套 JSON 记录本次所核验证据的 SHA-256。

截至本记录更新：当前源码与 `source_snapshot.json` 一致；注册配置没有变化；`confirmation_consumed.lock`、确认结果、confirmation/final 模型和 benchmark 结果均不存在。Stage 5 的 `train-confirmation_invocation.lock` 是失败 CLI 的审计记录，区别于确认集消费锁，予以保留。

后续每次实际操作须追加记录：授权原文、处理前后 SHA、变更内容、回归验证、运行起止时间、阶段状态和数据访问。保持“已应用修复”“已验证但未应用补丁”“尚待确认的风险”三者状态准确；不覆盖原失败证据，不填写未运行指标。
