# G2 check 失败与未应用修复方案

唯一一次 analyze 成功，正式 JSON 和 Markdown 已生成。随后 check 退出码为 1。严格保留失败，不把独立复核通过替代正式 check。

根因：JSON 写入器按键排序；make_report 遍历 associations 的插入顺序。初次生成和 JSON 重载后，表格六行顺序不同。全部报告行的多重集完全一致，没有指标、数字或结论的差异。差异见 REPORT_RENDER_ORDER_FAILURE.diff。

用户指令明确写“如……已经产生部分结果，必须先停止，如实上报”。因此当前未修改新 G2 源码或测试、未改写正式 JSON/报告，也未重跑 analyze 或 check。

已另存两个未应用的最小补丁：PROPOSED_RENDER_ORDER_SOURCE_FIX.diff 和 PROPOSED_RENDER_ORDER_REGRESSION_TEST.diff。源码补丁只固定原有六代理变量的报告遍历顺序；合成数据回归检查 JSON 排序/反序重载后报告不变。隔离函数验证已确认补丁渲染可逐字复现现有正式报告。它不涉及任何研究指标、公式、角色、G1 文件或训练权重。

若用户另行明确授权此 analyze 后的非科学性修复例外，可应用补丁、补回归、重跑编译/测试/preflight，然后仅重跑 check；不需要且不允许重跑 analyze 或重写现有结果。当前仍为正式 check FAIL / 执行 STOP。

另有数值解释限制：本楼宇楼层训练锚点总数完全由楼宇楼层决定，在控制楼宇楼层后残差为零，其 partial rank correlation 不可识别。原始 JSON 的两个微小非零数来自浮点投影残差；在独立解释中呈现 N/A，不覆盖原始数值。详见 CORRELATION_RESIDUAL_REVIEW.json。

审计脚本创建曾发生一次工具写入失败；已在 INITIAL_EXECUTION_AUDIT.json 记录并改用工作区内的审查执行创建审计工具，没有科学文件变动。
