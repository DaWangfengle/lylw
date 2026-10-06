# V18.5 Windows 验收路径修复

用户授权：修复并继续跑。

此前完整验收结果 103 passed、1 failed；失败记录保持原字节。
唯一修改是 `_test_targets()` 将 `str(path.relative_to(REPO_ROOT))` 替换为
`path.relative_to(REPO_ROOT).as_posix()`，使 Windows 路径符合现有测试的跨平台序列化契约。
现有测试、测试发现集合、科学方法、网格、种子和选择器均未改变。

- 既有定向回归：1 passed in 3.76s，无失败/skip/xfail。
- 原提交：`bab4c496ea1394451f94a8a85e52f6c6b03a1b0a`
- 修复后的干净本地提交：`2e84fac20b8a98bcaaeff6783ca1e3b63a8a5407`
- 方法契约 SHA256 保持不变：`745824e4a8a9ca1972bdf225f2d97ebd84830751a79160576b5834645b142b05`
- 验证契约更新为：`52a6650a17908ebb6b37870d51c89f1f3fb0b5634dcf539577810a54baa35a6f`
- 本地提交是为了满足正式验收及注册表的干净源码/提交绑定要求；没有推送 GitHub。
- 将从完整统一验收重新启动，通过后才进入 prepare/develop/calibrate-freeze/artifact-audit。
- 保留前次原始日志、阶段 JSON、执行失败 JSON/MD；没有删除、覆盖或修复任何正式产物。
