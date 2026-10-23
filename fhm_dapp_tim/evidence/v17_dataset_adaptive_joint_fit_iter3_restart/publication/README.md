# ITER3 TUJI 轻量产物发布

本目录只记录实验完成后，用户另行要求本地提交并推送 GitHub 时进行的打包。没有重新训练、重新评分、改选参数或改写原科学产物。

- `v17_dataset_adaptive_joint_fit_iter3/` 保留首次失败的 45 份产物，包括 float32/float64 截断边界不一致的诊断、原始日志、修复前源码、补丁及用户授权。
- `v17_dataset_adaptive_joint_fit_iter3_restart/` 是经授权两行精度修复后，在新目录完整重启的 TUJI 9/9 组结果。主要报告为 [RUN_REPORT_ITER3_TUJI_CN.md](../RUN_REPORT_ITER3_TUJI_CN.md)，修复说明为 [FIX_RECORD.md](../FIX_RECORD.md)。32 项测试、九组 FIT 预检和独立范围审计通过；UTS/UJI 没有定位评分，原生聚合器的 `INCOMPLETE 9/27` 状态原样保留。
- 用户在大文件推送过程中明确要求“那个大文件不传了，忽略他”。原推送已停止且远端未接收该提交；全部 10 份 `CANDIDATE_METRICS.csv` 和它们的 15 份 gzip 归档仅保留本地，并用本目录的精确忽略规则排除。没有删除任何本地原件。
- 发布其余 JSON、Markdown、小型 CSV、原始 stdout/stderr、失败与修复证据，以及约 1.36 MB 的 `PARETO_FRONTIER.csv.gz`。模型权重、缓存和临时辅助脚本留在本地。
- `ARCHIVE_MANIFEST.json` 的 `csv_files` 只列出本次实际发布的 Pareto 归档，`local_only_csv_files` 列出未上传候选表的路径、大小与 SHA256，避免把未上传内容说成已发布。归档已解压核验，原交付清单中的 72 份重启产物及 45 份失败产物也已在本地核验。原 `DELIVERY_INTEGRITY.json` 和其引用的科学文件均未修改。
- `.gitattributes` 为这两个产物目录保留原始字节，避免换行转换破坏冻结哈希。报告所述“未执行 Git 操作”指科学运行阶段；本次发布由用户随后单独授权。

从仓库根目录只读验证已发布的 Pareto 压缩归档及原文件身份：

```powershell
python fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart/publication/restore_csv_archives.py
```

下载仓库后恢复缺失的 Pareto CSV：

```powershell
python fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart/publication/restore_csv_archives.py --restore
```

已有且身份一致的 Pareto CSV 会保留；已有内容冲突则停止，不覆盖。大体积候选表无法仅凭本次 GitHub 产物恢复，需要本地原件；原交付清单仍保留这些文件的身份记录。全部不利科学结论和失败证据保留，发布范围的调整不改变研究结果。
