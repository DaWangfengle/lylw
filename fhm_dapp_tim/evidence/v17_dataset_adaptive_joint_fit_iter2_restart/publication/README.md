# 本轮产物的无损压缩发布说明

本目录只记录实验完成后、用户要求提交并推送 GitHub 时进行的文件打包。没有重新训练、重新评分、改选参数或改写科学结果。

- `v17_dataset_adaptive_joint_fit_iter2_restart/` 是经用户授权最小修复后，从 TUJI → UTS → UJI 完整重启并完成的 27 组结果。主要结论见其 `EXECUTION_SUMMARY_CN.md`。
- `v17_dataset_adaptive_joint_fit_iter2/` 保留此前中止状态、9 组 TUJI 结果、全部失败日志、补丁及用户授权，不能与完整重启结果混作同一次实验。
- 两个目录的全部 `CANDIDATE_METRICS.csv` 和根目录 `PARETO_FRONTIER.csv` 以 gzip 无损压缩存档。较大的原文件按 64 MiB 原始字节分段，再逐段独立压缩；这些分段不保证落在 CSV 行边界上。按清单顺序解压并连接即可恢复原文件。
- 本地原 CSV 原样保留，只在这两个精确目录内忽略其 Git 上传；压缩归档、其他 JSON/Markdown/小型 CSV、stdout/stderr 原始日志与修复记录一并提交。模型权重和运行缓存继续保留本地。
- `ARCHIVE_MANIFEST.json` 记录每个原文件及每段压缩文件的大小、SHA256、顺序和全部解压核验结果。原 `DELIVERY_INTEGRITY.json` 及其 139 项原始文件身份未被更改；打包前全部逐字节核验通过。
- `.gitattributes` 对本轮源码、测试及这两个产物目录保留原始字节，避免换行转换破坏冻结哈希。实验报告中“没有 Git 操作”描述的是科学执行阶段；本次发布是实验完成后用户另行授权的操作。

从仓库根目录使用现有 Python 只读校验全部压缩文件及解压结果：

```powershell
python fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart/publication/restore_csv_archives.py
```

在新的下载目录恢复缺失的原 CSV（已有且哈希相同的文件保留；已有内容冲突则停止）：

```powershell
python fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter2_restart/publication/restore_csv_archives.py --restore
```

恢复后即可直接使用原始 CSV 路径复核冻结清单。正反结果及所有原始失败证据均保留；打包不改变研究结论，也不构成新的独立验证。
