**V18.5 本轮产物提交与远端历史整合说明**

记录时间（上海）：2026-10-06T16:53:58.552274+08:00

本记录产生于合并检查通过后、合并提交和第二次正常推送之前；最终发布以 GitHub 分支头和合并提交为准。

用户授权：将本轮的产物本地提交并推送到 GitHub。

分支：v18-5-joint-radio-calibration

本轮产物提交：fdd02b6d66c76482384861305b2cc190f3ce4379

本轮实际实验源码：2e84fac20b8a98bcaaeff6783ca1e3b63a8a5407

远端合并父提交：5b4051aa8f1e8ef656dde047cbb8aa2f4a94f187

**产物与历史完整性**

本轮原始产物共 19 份 JSON/Markdown，包括预检、registry、四个 lambda 的开发汇总、完整 192 单元校准、method freeze、执行报告、失败记录和修复记录。原始正式审计 PASS、104 项预检测试通过，均对应实际运行的 2e84fac 版本。

为保留记录中的文件 SHA256，.gitattributes 对这些正式结果和 run_audit 的 JSON/Markdown 禁用换行转换。暂存时 19 份产物均与本地原始字节一致；模型权重、原始日志、缓存和临时文件沿用现有忽略规则。

首次正常 push 被 Git 拒绝，因为远端自共同基点新增了 21 个提交。获取后，采用保留双方历史的 merge，原始运行源码 2e84fac、产物提交 fdd02b6 和远端全部提交均保留。自动合并无冲突，Windows 测试路径的 as_posix() 修复也被保留。

**新旧 schema 的适用范围**

远端新增代码主要涉及环境指纹、确定性 smoke、验收文件的原子写出和字节溯源。它们未参与本轮已完成的实验。

本轮原始 acceptance 缺少后来新增的 environment、environment_sha256；原始 registry 缺少 pre_run_acceptance_file_sha256、pre_run_environment、pre_run_environment_sha256。最新审计新增了这些必需字段，所以本轮旧 schema 的历史 PASS 应按其记录的 2e84fac 源码解释。未来使用新代码产生的产物应满足新 schema。

本轮产物保留原始字段与原始哈希，本次 Git 发布没有补写环境字段，也没有重跑正式预检、prepare、develop、calibrate-freeze 或正式 artifact audit。合并后的仓库版本与本轮实际运行版本已经在此区分说明。

**合并单元检查及临时目录问题**

针对合并后的 tests/test_v18_5_pre_run_acceptance.py 执行单元检查。首次 12 passed、4 fixture setup errors：指定的临时目录父目录 D:/liuyanlunwen/tmp 尚不存在，触发 WinError 3。

修复操作仅为创建该父目录，并使用经过 workspace 边界核验的新临时子目录。重新运行得到 16 passed in 5.76s，无需修改测试源码。这些检查不是正式实验或 benchmark 评价。

完整 push 错误、临时目录错误和单元检查输出、原始产物文件 SHA256 清单收录于 GIT_PUBLICATION_PREPARATION_V18_5.json。

