# ITER3 三域 27/27 产物发布

本目录记录实验完成后，用户另行授权的本地提交和 GitHub 推送打包。打包没有重跑实验、评分或选参，没有修改科学结果。

- TUJI、UTS、UJI 各 9/9，共 27/27；原生汇总和独立只读核算均通过。完整结果与限制见 [中文报告](../RUN_REPORT_ITER3_27OF27_CN.md)、[原生三域汇总](../CAL_EVAL_REPORT.md) 和 [独立核算 JSON](../CONTINUATION_READ_ONLY_VERIFICATION.json)。
- 发布新增的 UTS/UJI 逐 checkpoint 选择与评价 JSON、三域汇总、近优敏感性 CSV、原始 stdout/stderr、首次 UTS 失败、根因诊断、全部修复版本、最终补丁和回归测试。相关证据位于 `../runs/` 和 `../run_logs/`。
- TUJI 九组原始结果保持原字节；UTS/UJI 使用有授权记录的数值修订。源码版本差异和研究常量未变的证明见报告与 `../run_logs/numerical_revision_20261023_133252/REVISION_FREEZE.json`。27/27 是完成状态，不代表三域均取得性能改善或正式方法晋级。
- 按用户此前要求，所有大体积 `CANDIDATE_METRICS.csv` 及其 gzip 文件继续仅保留本地，模型权重、缓存和临时辅助文件也不上传。`PUBLICATION_MANIFEST.json` 记录这些本地候选文件的路径、大小和 SHA256，不把它们列为已发布内容。
- 本目录的 `PARETO_FRONTIER.csv.gz` 是本次三域完整 Pareto 表的压缩副本，已解压并验证大小及 SHA256 与原始 CSV 完全一致。原始 CSV 保留本地，压缩副本约 3.54 MB；具体身份以清单为准。
- 原 `../publication/`、`RUN_REPORT_ITER3_TUJI_CN.md`、`READ_ONLY_VERIFICATION.json`、`DELIVERY_INTEGRITY.json` 和先前归档保持为 TUJI 9/27 阶段快照。旧归档及恢复脚本针对旧阶段，不能代替本目录的三域清单。
- `.gitattributes` 保留本轮证据原字节，避免 Windows 换行转换破坏科学哈希。报告中的“未执行 Git 操作”指科学执行阶段；本次打包和上传由用户随后单独授权。

本轮结果仍限于回顾性的 CAL-eval 描述性比较。全部负结果和尾部风险均保留；internal TEST 与 official TEST 的评分计数均为 0。
