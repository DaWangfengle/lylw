# V18.5 首次正式 UJI 执行记录：预运行验收失败

状态：**FAIL CLOSED AT STAGE 0**。正式实验尚未开始。

- 分支：`v18-5-joint-radio-calibration`
- 实际源码提交：`bab4c496ea1394451f94a8a85e52f6c6b03a1b0a`
- Python：`D:\miniminiconda\envs\liuyan\python.exe`
- 命令：`python -m fhm_dapp_tim.fhm_dapp.v18_5_pre_run_acceptance --require-cuda`
- 验收总耗时：23.659560 秒；退出码 1。
- 测试实际结果：**103 passed、1 failed**，共 104 项，10.94 秒；无 skip/xfail/xpass。
- 19 个注册模块编译已完成；CPU/CUDA smoke、训练源联合审计未执行到。
- 预先核验 CUDA 可用，设备 RTX 2060；这不等同于 CUDA smoke PASS。
- 当前方法契约 SHA256：`745824e4a8a9ca1972bdf225f2d97ebd84830751a79160576b5834645b142b05`
- 当前验证契约 SHA256：`643613e3445d23de8197ef37053cec8d8a73dbe3f62088adf60bafdd031bcb1c`
- Acceptance SHA256：未生成；两个正式验收文件均不存在。

## 失败原因

`tests/test_v18_5_pre_run_acceptance.py:34` 中的测试
`test_test_target_discovery_includes_external_and_uji_contracts` 要求
`tests/test_uji_v18_5_development.py` 属于发现列表。
验收 `_test_targets()` 使用 `str(path.relative_to(REPO_ROOT))`，Windows 返回反斜杠路径
`tests\\test_uji_v18_5_development.py`。文件已被发现，字符串格式未满足跨平台契约。

## 处理与后续修复建议

依据执行指令第 5、14 节，发现测试失败后停止，不重试，不进入 prepare。
未修改源码、测试、方法、种子、网格或选择器；原始失败日志和命令记录完整保留。
建议的最小修复是在新的授权任务中将该验收路径序列化改为
`path.relative_to(REPO_ROOT).as_posix()`，维持现有测试，不修改科学方法。
**本轮未实施修复。** 修复后须在新的源码版本上从完整统一验收重新开始。

## 阶段和科学边界

- Acceptance FAIL；prepare/develop/calibrate-freeze/artifact-audit 均未运行。
- 所有 11 个规定验收/正式结果/检查点路径及其正式输出未生成。
- 无正式模型训练，无 UJI validationData 访问，无外部测试性能访问或正式外部评测。
- 无 post-hoc probe，无 Git 提交/推送/上传操作。
- 既有 tracked 文件保持干净；新增文件仅为运行日志和执行失败记录。

## 保留的失败证据

- 日志：`fhm_dapp_tim/evidence/v18_5/run_audit/FIRST_FORMAL_UJI_stage0_acceptance_20261006T055942748916Z.log`
- 阶段记录：`fhm_dapp_tim/evidence/v18_5/run_audit/FIRST_FORMAL_UJI_stage0_acceptance_20261006T055942748916Z.json`
- JSON canonical execution_record_sha256：`dd2c416601c6936a4e8829295f06fc9ae45dcb50161038b90b9dbf7789df2569`
