# V9 + V17 历史强基线复现运行报告

完成时间（UTC）：2026-10-07T12:28:27.943403+00:00。本轮完整执行指定 V9/V17 流程，结束状态为 **FAIL**。

| 门槛对象 | outer coordinate-macro 2D MAE（m） | 严格 <6.0 m |
| --- | ---: | --- |
| V9 frozen system | 6.3555593491 | **FAIL** |
| V17 selected E16/L192/K192/alpha=1 | 5.7259850344 | **PASS** |

`next_stage_authorized` = **false**。未放宽、绕过或 waiver 门槛；本阶段结束，不实现后继模型。

## 1. 环境与输入

- Python：`D:\miniminiconda\envs\liuyan\python.exe`；版本 `3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]`。
- PyTorch：`2.10.0+cu128`；CUDA runtime：`12.8`；CUDA available：True。
- GPU：NVIDIA GeForce RTX 2060，显存 6442123264 bytes；系统：`Windows-10-10.0.19045-SP0`。
- 分支：`historical-v9-v17-reproduction`；HEAD：`be4ace44d38e92c665de8646337caf3e2398f098`，与启动时一致。
- 唯一原始数据输入：`D:\liuyanlunwen\data\UJIndoorLoc\trainingData.csv`。
- trainingData.csv SHA256：`45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`。
- 本阶段新产物位于 `fhm_dapp_tim/evidence/historical_reproduction/` 和 `fhm_dapp_tim/models/historical_reproduction/`。
- V17 outer_refit：15871 rows / 746 coordinate anchors；outer-test：4066 rows / 187 coordinate groups。训练均来自 trainingData.csv 的注册划分。

## 2. 预检与执行

正式训练前编译通过，指定三个测试文件为 **17 passed**。训练后诊断修复的最终结果为 **19 passed、0 failed、0 skipped、0 xfailed**（16 个原测试、1 个真实 registry 集成回归案例、2 个 retained-mass 回归案例）。修复后 CUDA preflight 为 ready，固定 seed panel、6.0 m gate、official/external authorization=false 均再次核验通过。

| 阶段 | attempt | exit code | 耗时（min） | 完整记录 |
| --- | ---: | ---: | ---: | --- |
| `compile` | 5 | 0 | 0.00 | [日志](run_audit/compile_attempt05.log) |
| `pytest` | 6 | 0 | 0.22 | [日志](run_audit/pytest_attempt06.log) |
| `preflight` | 2 | 0 | 0.07 | [日志](run_audit/preflight_attempt02.log) |
| `prepare` | 1 | 0 | 0.07 | [日志](run_audit/prepare_attempt01.log) |
| `run-v9` | 1 | 0 | 55.17 | [日志](run_audit/run-v9_attempt01.log) |
| `prepare-v17` | 1 | 0 | 0.18 | [日志](run_audit/prepare-v17_attempt01.log) |
| `run-v17` | 1 | 1 | 241.10 | [日志](run_audit/run-v17_attempt01.log) |
| `check-v17-members` | 1 | 0 | 0.11 | [日志](run_audit/check-v17-members_attempt01.log) |
| `evaluate-v17` | 1 | 0 | 1.05 | [日志](run_audit/evaluate-v17_attempt01.log) |
| `summarize` | 1 | 0 | 0.07 | [日志](run_audit/summarize_attempt01.log) |

初始失败记录保留，未用最终通过记录覆盖。详见 [训练前环境与预检审计](run_audit/ENVIRONMENT_AND_STATIC_PRECHECK.json)、[训练后修复预检](run_audit/POST_TRAINING_IMPLEMENTATION_PRECHECK.json) 和下文修复记录。run-v17 完成全部 25 名成员并处理 4066 条 outer rows 后，在 retained-mass 诊断拼接失败（exit code=1）；修复后单独执行 evaluate-v17 完成同一冻结评价，未重跑任何成员训练。失败时未生成正式 V17 结果，也未读取其数值用于选择或调参。

## 3. V9 原流程结果

- 候选与网格：geometry_invariant / historical_affine，seed 9311..9322，K=[8,16,24,32,48]，lambda=[1/64,1/32,1/16,1/8,1/4,1/2,1,2]，保持原样。
- 原始 E=8、E=10 判定均要求增加 2 名成员，最终两候选各完成 E=12 的训练与 grid。
- 冻结配置：candidate=`geometry_invariant`，E=12，K=24，lambda=0.5。
- development coordinate-macro MAE：6.4253573418 m；Energy Score：4.5756859779 m。
- 冻结依据：`budget_censored_development_risk_minimum`；saturation_claim=false。
- E=12 仍未证明信息饱和。使用运行前已存在于原 V9 源码的 V9-A1-terminal-selection 条款，在预注册预算上限选择 development proper-risk 全局最小值，并明确不宣称饱和。本轮未修改该条款。
- 12 名 final member 均用自己的原始 early-stop 所选 epoch，在 outer_refit 上从头重训。
- 冻结前 outer-test accesses=0；最终 outer-test access number=1。
- outer coordinate-macro MAE：**6.3555593491 m**；Energy Score：4.5328063965 m。
- 辅助 row MAE：5.9159393311 m。严格门槛以 coordinate-macro MAE 判定，V9 为 **FAIL**。

证据：[V9 结果汇总](v9/REPRODUCTION_V9.json)、[正式 outer-test 结果](v9/outer_test_result_v9_uji.json)、[配置冻结](v9/selection_freeze_v9_uji.json)。

## 4. V17 25-member 冻结 R3 结果

每个 seed 独立调用未修改的原 V17-R0 _fit_schedule，在 member_fit/early_stop 上按 coordinate-macro MAE 选择 epoch；随后以同 seed 调用 _fit_refit，在 outer_refit 上从头重训 selected_epoch 轮。最大 80 epoch、patience=10 沿用原定义，不统一固定 epoch，不依据 outer outcome 选 epoch。

官方 check-v17-members：complete_members=25，all_complete=true。全部成员 identity、seed、registry、phase=r3_outer_refit、anchor_count=746 及训练未消费 outer-test/official labels 检查通过。checkpoint 文件 SHA256 已与正式记录逐一核对。

| member | seed | selected_epoch |
| ---: | ---: | ---: |
| 01 | 17301 | 65 |
| 02 | 17302 | 69 |
| 03 | 17303 | 56 |
| 04 | 17304 | 67 |
| 05 | 17305 | 62 |
| 06 | 17306 | 60 |
| 07 | 17307 | 52 |
| 08 | 17308 | 74 |
| 09 | 17309 | 79 |
| 10 | 17310 | 62 |
| 11 | 17311 | 65 |
| 12 | 17312 | 57 |
| 13 | 17313 | 64 |
| 14 | 17314 | 61 |
| 15 | 17315 | 50 |
| 16 | 17316 | 74 |
| 17 | 17317 | 59 |
| 18 | 17318 | 73 |
| 19 | 17319 | 60 |
| 20 | 17320 | 68 |
| 21 | 17321 | 52 |
| 22 | 17322 | 58 |
| 23 | 17323 | 53 |
| 24 | 17324 | 61 |
| 25 | 17325 | 56 |

采用原 R3 balanced cyclic subset estimand：全部 25 seed，每个配置 25 个子集，generator=7，成员各出现 E 次。未选择最优子集，未重选 E/L/K/alpha。Student-t df=4，每 stream 64 predictive draws，预测 seed=17931。

| 配置 | 正式名称 | coordinate-macro MAE（m） | coordinate-macro Energy Score（m） | 辅助 row MAE（m） |
| --- | --- | ---: | ---: | ---: |
| selected E16 | `selected_E16_L192_K192_A1` | 5.7259850344 | 4.3805908810 | 5.2111208572 |
| historical E5 | `historical_E05_L024_K024_A0p25` | 5.8535630576 | 4.6977074527 | 5.3113249991 |

### 配对 familywise 比较

方向：`selected_E16_minus_historical_E5`。按 187 个 coordinate groups 共同重采样，studentized max-T，confidence=0.95，family_size=2，bootstrap=5000 次，seed=17932，critical max-T=1.7846846644。

| endpoint | E16−E5 均值差（m） | 标准误（m） | paired percentile CI95（m） | familywise simultaneous upper bound（m） |
| --- | ---: | ---: | --- | ---: |
| `energy_score_2d_m` | -0.3171165717 | 0.0715213061 | [-0.4654083225, -0.1908520828] | -0.1894735936 |
| `exact_mixture_mean_mae_m` | -0.1275780232 | 0.0411666689 | [-0.2128844034, -0.0516300864] | -0.0541085005 |

差值为负表示 selected E16 的该指标较低；simultaneous upper bound <0 才支持该 endpoint 的 familywise 改善。配对比较不替代严格 sub-6m gate。

### Retained-mass diagnostics

| 配置 | mean | q10 | median | q90 |
| --- | ---: | ---: | ---: | ---: |
| selected E16 | 0.9974737348 | 0.9953150630 | 0.9991931915 | 0.9998739958 |
| historical E5 | 0.9731095328 | 0.9398263693 | 0.9871985912 | 0.9975807369 |

V17 selected 的 sub-6m 结果为 **PASS**。全部数值及逐子集 realization records 保存在 [正式 V17 结果](v17/outer_result_v17_reproduction.json)。

## 5. 问题、最小修复与限制

数据路径和测试 fixture 修复在正式 prepare 与训练前完成；retained-mass 拼接修复在 25 名成员训练完成后执行。两阶段均只修改 reproduction harness/tests，未修改 V9/V17 原始模块、训练损失、seed、split、评分公式、选择规则或 gate。

| 问题 | 原因与处理 | 证据 |
| --- | --- | --- |
| sandbox exec 启动失败 | helper_unknown_error/setup refresh 发生在命令启动前；通过经审查的本地执行继续，不修改科学规则 | [初始审计](run_audit/INITIAL_EXECUTION_AUDIT.json) |
| pytest tmp_path 权限与日志编码失败 | 默认 Windows temp 父目录不可读；改用获准 model root 下 fixture temp；logger/child 使用 UTF-8，保留首份部分日志 | [运行环境修复](run_audit/PRECHECK_OPERATIONAL_REPAIR.json) |
| PYTEST_ADDOPTS 中 Windows 反斜杠被 shlex 吞掉 | basetemp 改为 as_posix；错误路径经绝对路径边界核验后移动到 model root，保留而未删除 | [路径与 fixture 修复](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| 两个原 registry 测试引用不存在的旧 design | 测试显式绑定本轮已提供的 DESIGN_V9_REPRODUCTION.md 与同一 TRAIN_DATA；原 registry hash/tamper/no-validation 断言保留；新增真实 registry 集成测试 | [测试修复](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| reproduction harness 的训练 CSV 路径错误 | TRAIN_DATA 从 package ROOT/data 改为 repository ROOT.parent/data，只改这一输入绑定；集成测试先失败再通过；删除该 assignment 后的全文件 AST 逐项一致 | [输入路径修复与 AST 证明](run_audit/PRECHECK_TRAINING_INPUT_PATH_REPAIR.json) |
| apply_patch 报 reparse point，以及审计路径键格式不一致 | 原生路径检查确认目标为 workspace 内普通文件，改用带边界核验的文件写入；失败键查询未发生写入，随后仅规范化审计路径键 | [同一修复记录](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| V17 retained-mass 最后短批拼接失败 | 原统计段抽取为私有 helper；逐批展开全部 member-row scalar 后拼接，保持 mean/q10/median/q90 定义不变。4066=127×32+2 回归先复现失败，再与完整 25×4066 参考值精确一致；非诊断部分全文件 AST 不变；全部 checkpoint 和 registry 字节哈希不变，只重跑冻结评价 | [失败与修复计划](run_audit/V17_RETAINED_MASS_FAILURE_AND_REPAIR_PLAN.json)、[实现修复证明](run_audit/V17_RETAINED_MASS_IMPLEMENTATION_REPAIR.json) |
| 原 V17 CUDA attention backward 非确定性提示 | 按原运行设置执行并保留 warning，未通过改算法或训练设置进行修复；固定 seed 不能据此保证 GPU 训练逐位复现 | [V17 完整日志](run_audit/run-v17_attempt01.log) |

pytest 中间结果依次包括临时目录/日志失败、2 failed/14 passed（旧 design）、3 failed/14 passed（训练路径），训练前为 17 passed。训练后的新回归先得到 1 failed/18 passed（最后 width-2 batch），修复后为 19 passed。完整 attempt 日志和 JSON 均保留于 run_audit。

最终仅 3 个 tracked 文件包含本轮最小实现/测试修复：`fhm_dapp_tim/fhm_dapp/historical_reproduction.py`、`tests/test_historical_reproduction.py`、`tests/test_uji_v9_experiment.py`。其余 18 个启动时登记文件哈希不变，7 个注册的 V9/V17 科学模块仍保持原哈希。原 registry 保留注册时 runner 快照 `e09c66606316336c9297cd7194936ce6cae498121e5ffa984d15c014af66a269`；诊断修复后的 runner 为 `99cdeade06a6a59235c87ff9d094993e2c9fe31cf034a43049739a58ca8edfee`，旧/新哈希与 AST 证明通过追加审计明确关联，未手改 registry。最终明细见 [完整核验](run_audit/FINAL_EXECUTION_VERIFICATION.json)。

## 6. 结论与边界确认

正式总结果见 [REPRODUCTION_SUMMARY.json](REPRODUCTION_SUMMARY.json)：V9=FAIL，V17 selected=PASS，both_historical_baselines_reproduced=false，next_stage_authorized=false。

- 本阶段没有运行 V18/V19 模型，也没有使用其 outcome 调参。
- 没有读取或预测 data/UJIndoorLoc/validationData.csv，official validation predictions=0。
- 没有使用外部数据集，external predictions=0。
- 没有根据 outer outcome 调参、重选 seed、epoch 或 frozen law。
- 没有放宽、绕过或 waiver <6.0 m gate；未沿用 V19-R0 的任何 gate waiver。
- 没有设计或实现下一阶段融合模型。
- 没有执行 git commit/push/pull/merge/rebase，也没有创建或删除分支；branch/HEAD 与启动时一致。
- 所有新产物限定在本轮 evidence/model roots，失败证据和 checkpoint 均保留。
- 本任务到此结束；即使某个基线通过，也不继续实施后继阶段。
