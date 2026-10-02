# V18.4 Decoder Diagnostic Gate：运行与问题记录

记录日期：2026-10-02，Asia/Shanghai。原始执行日志采用 UTC。

状态：**尚未完成。已有 radio 推理过程与本次禁止 backward/optimizer step 的指令冲突，等待明确例外授权。** 没有生成、覆盖或上传两个 probe 结果文件，没有得到任何 decoder 指标或机制结论。

## 已完成的检查

- 分支 `v18-4-trust-region-posterior`；正式源码 SHA `109613f77078ce3e19302b24ac85feb90a281f7b`。
- 使用 `D:\miniminiconda\envs\liuyan\python.exe`。
- 已阅读附件、probe 源码、设计说明、测试和仓库执行指令。
- 五个必需输入存在；confirmation lock/result 与两个 probe 输出在运行前均不存在。
- 两个指定 `py_compile` 通过；指定 pytest **25 passed in 10.54s**，未删、skip、xfail 测试。
- CUDA 可用：RTX 2060，PyTorch 2.10.0+cu128，CUDA 12.8。
- 全部已核验的科学源码、V18.3 checkpoint、registry、freeze 和允许角色输入的文件 SHA 均未改变。

## 问题 1：工具进程启动失败——已恢复

首次读取附件的工具进程报 `CreateProcessWithLogonW failed: 1056`，未执行读取命令。改为顺序重试后成功。该问题发生在科学运行之前，没有修改文件或接触数据。记录在 `run_audit/preflight.json`。

## 问题 2：新增审计代码不兼容 bytes 路径——已修复

第一次 CLI 启动在 checkpoint 重建的 SciPy Delaunay 阶段中止。审计辅助函数直接使用 `os.fspath(path)`，SciPy 临时文件路径为 bytes，随后与 str 前缀比较触发 TypeError。

修复仅在新增辅助代码：改为 `os.fsdecode(os.fspath(path))` 再做路径归一化。合成 bytes 临时文件与四点 Delaunay 检查通过，没有读取科学输入或产生预测。

原始 helper、执行记录、runtime、堆栈日志和 invocation marker 全部保留。第一次运行未加载任何角色 shard，成功预测次数为 0。证据：`run_audit/probe_execution.json`、`probe_runtime.json`、`failed_startup_sitecustomize.py`、`audit_path_fix_verification.json`。

## 问题 3：审计把重建时恢复 train 标记误判为训练——已修复

第二次 CLI 启动在 `model.initialize_anchor_keys` 中止。原代码使用 `torch.no_grad` 初始化 anchor keys，并恢复新建模型原来的 train 标记；之后才加载冻结 checkpoint state 并设为 eval。这不涉及训练更新，但新增审计代码过早禁止了该模式恢复。

修复仅在新增辅助代码：对已经完整加载的冻结模型及其子模块禁止 `train(True)`。随后单独做完整 CUDA checkpoint 重建检查：epoch=17、eval=True、模型 state 与 checkpoint 逐位一致、所有参数的 `.grad` 均为 None；没有加载角色 shard 或执行预测。

第二次运行未加载任何角色 shard，成功预测次数为 0。原 helper 与记录保留。证据：`run_audit/probe_02_execution.json`、`probe_02_runtime.json`、`second_startup_sitecustomize.py`、`checkpoint_reconstruction_verification.json`。

## 问题 4：冻结 radio 推理实际使用 backward/Adam step——待用户明确授权

第三次 CLI 完成 checkpoint 重建与 epoch 核验，并加载了允许访问的 early_stop shard。首次调用 `v183_predict` 时，审计在创建 radio nuisance optimizer 前拦截，随后退出；没有返回完整预测或计算 decoder 指标。

实际调用链：

```text
v18_4_decoder_probe.evaluate_role
  -> v18_3_single.v183_predict
  -> v18_3_single.structured_evidence_opinions
  -> coherent_marked_v9.profile_shared_affine_marked_energy
```

源码位置：`v18_3_single.py:847` 调用 profiler；`coherent_marked_v9.py:354` 创建 Adam，`:381` 调用 `objective.mean().backward()`，`:382` 调用 `optimizer.step()`。

优化变量是每个 batch 新建的 `raw_offset` 与 `raw_log_scale`，各为扫描行数长度的叶子张量。现有预测调用对 query、anchor fields、prior 等使用 detach，因此这属于拟合扫描层 radio nuisance 的既有推理过程，并不是更新 checkpoint 网络权重。注册配置固定每 batch 6 步、学习率 0.2。

但用户附件第 7 节明确要求“脚本不得：train、backward、optimizer step、修改 checkpoint”。直接放行原 profiler 就会执行被明确禁止的 API；移除 profiler 或改变其求解过程则可能改变冻结 posterior，并涉及禁止修改的 V18.3 方法。当前没有采取这两种操作。

拟申请的例外仅限原 profiler 的两个临时 nuisance 张量，保留全部冻结计算、四个 decoder、K=8、modified Weiszfeld、10%/0.25 m 阈值及两个角色要求。须审计 nuisance tensor 身份与 checkpoint 参数/缓冲区分离，禁止网络参数梯度和更新，逐位核验完整模型 state 前后相同，并继续限制数据访问。

具体可审阅范围在 `run_audit/RADIO_PROFILING_EXCEPTION_PROPOSAL.json`，状态为未授权、未应用。没有自动扩大旧 V18.3 修复授权。第三次证据：`run_audit/probe_03_execution.json`、`probe_03_runtime.json`、`probe_03.log`。

## 运行次数与数据边界

共三次 CLI 启动，全部失败：前两次在模型重建时停止，第三次在首次 radio optimizer 创建前停止。**完成的科学 probe 为 0，返回成功的预测为 0，decoder 指标为 0。** 记录保留实际 CLI 次数，不把失败启动隐去或称为一次成功运行。

第三次 early_stop shard 已加载；calibration 仅在预检读取文件字节核验 SHA，没有加载或评价。confirmation shard 从未打开，official validation 从未打开。没有创建或删除 confirmation 消费锁。第三次加载的模型完整 state 在退出时逐位不变，磁盘 checkpoint SHA 未变；没有执行 backward 或 optimizer step。

两个正式输出仍不存在。没有选出任何固定机制结论，也没有据结果修改 decoder、阈值或方法。无结果可上传；下一次数据评价必须先取得上述明确例外授权。原始 `.log` 保存在本地，遵循既有 `.gitignore`。

配套 `RUN_STATUS_DECODER_GATE_V18_4.json` 保存检查结果、问题处理、全部阶段时间和 runtime 证据。当前没有修改 V18.3 或 V18.4 科学源码。
