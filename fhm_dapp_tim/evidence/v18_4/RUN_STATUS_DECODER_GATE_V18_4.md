# V18.4 Decoder Diagnostic Gate 完成状态

已完成唯一一次完整科学 probe。固定结论：**loss_aligned_geometric_median_supported**。

几何中位数 coordinate-macro MAE：early_stop 4.805945 m，相比 posterior_mean 改善 55.06%；calibration 7.475349 m，改善 41.13%。两个角色均达到原注册的 10% 改善要求，均未恶化。

所有 decoder 的 MAE、SE、行分位数、配对改善、posterior 诊断及问题修复见 [完整报告](DECODER_PROBE_REPORT_V18_4.md)。原脚本输出 JSON/Markdown 保留原样。

两项编译通过；25 passed in 10.54s。三次失败启动的实际次数、堆栈、原始审计辅助源码、修复验证与用户授权完整保留；第四次 CLI 成功，完成的科学 probe 总数为 1。没有隐去失败、重跑已完成的评价或修改阈值。

用户明确授权仅原 radio profiler 的两个临时参数做注册的 6 步 Adam。CPU/CUDA 合成检查 33 个输出张量逐位不变；正式推理 42 个 batch、5203 scans、252 次 nuisance step。网络参数梯度为 0，checkpoint 模型完整 state 逐位不变，磁盘输入和科学源码 SHA 不变。

没有访问 confirmation 或 official validation，没有消费、创建或删除 confirmation lock。仅使用注册的四种 decoder、两个允许角色和固定判定规则。

初始停止记录原样保存在 run_audit/PRE_AUTHORIZATION_RUN_STATUS_DECODER_GATE_V18_4.md/json；授权在 run_audit/radio_profile_authorization.json；验证在 run_audit/result_verification.json。

本状态记录形成时，结果与记录尚待推送；后续用户明确授权上传本轮全部产物，授权与首次推送审批拒绝原因见 [发布授权记录](run_audit/publication_authorization.json)。实际发布提交以 Git 历史和远端分支核验为准。推送后停止本任务，不继续 V18.3 或实现正式 V18.4 模型。
