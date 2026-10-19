# 冻结前数值修复记录

新增完整运行器和合成测试按本轮指令实现；未修改任何历史 V17/G1–G21、上一轮双阶段源码/测试/结果或检查点。

## F1：CUDA 追加虚拟候选影响父特征的逐位恒等

- `preflight_initial` 退出 1：TUJI FIT-only smoke 的 `virtual evidence altered parent features` 断言失败。正式 F/T 实验均未开始，EXECUTION_MANIFEST 尚未产生。
- 独立 FIT 数值诊断：646 父锚点、2946 虚拟边、120 模型 AP，8 条 FIT 扫描。visibility 无差；strength 最大差 `4.76837158203125e-7`，1938 个元素逐位不同。没有优化器更新。
- 原因：一次计算 `[B,N+E,AP]` 后切父前缀时，CUDA 归约/向量化布局随总候选宽度变化。即使只用父候选均值/RMS，父原始强度评分的舍入也可能改变。
- 初始通用合成形状 N+E=3646 未复现（通过日志保留）；改用实际触发的 N+E=3592 后，纯合成 CUDA 回归在修复前确实失败（`cuda_field_regression_shape_before_fix`）。
- 仅修改本轮新增空间算子 `g13_continuous_field_features`：以固定 `[B,N,AP]` 布局计算父原始两通道；虚拟通道另算，然后共用原父均值/RMS。不是分别归一化虚拟候选。公式、缺失阈值、df、方差、稳定地板和逐位恒等断言均未改。
- 原始算子副本、失败运行器副本、原始失败日志和精确 diff 保存在 `run_logs/`。完整 CPU/数学/CUDA 回归修复后通过；实际通过数及后续预检以 EXECUTION_RECORD.json 和原始 stdout 为准。

另修正新增合成测试将 requires_grad 张量转标量时的 PyTorch 提示，测试取值先 detach；没有改变模型计算。

生成 diff 的辅助命令首次使用系统默认 GBK 读 UTF-8 运行器，发生 UnicodeDecodeError；改为显式 UTF-8 后成功。未读取或修改任何科学结果。

## F2：用户授权的 ES 零质量分量数值修复与继续（2026-10-19）

- 用户明确回复“允许最小数值修复并继续（推荐）”。完整授权范围、原 manifest SHA 和原有33份F产物 SHA 见 USER_AUTHORIZATION_20261019.json。
- 不改历史 ES helper；只在本轮运行器的 ES 抽样入口排除全部查询均为零概率的组件。输入仍保留完整 N+E 混合，所有正权重、连续中心/尺度、随机流及概率法则不变。beta=0 追加零尾列会改变 CUDA cumsum 归约布局，从而偶发改变离散抽样 ID；合成原路径确实失败，修复后父/扩展 ES 逐位一致。
- 原冻结 manifest 不重写，单独追加 ENGINEERING_REVISION.json 绑定授权、唯一修改的运行器前/后 SHA、新合成测试 SHA 和精确 diff。修订门禁仍校验全部原输入、历史来源及33份原产物。
- 新的显式授权继续路径跳过已完成TUJI六组；失败UTS严格读取已有 CAL_FIT_SELECTION.json，不再调用拟合/参数选择。原锁保留，另写 AUTHORIZED_RESUME.lock。纯合成回归验证“无重新拟合、只进入 CAL-eval、原文件逐字不变”。
- 首次修订回归：49通过、1个fixture设置错误，原因是pytest basetemp父目录不存在。仅创建工作区内临时父目录，无源码/公式修改。完整重跑50通过（6.47s），编译退出0；全部原失败stdout/stderr保留。
- 授权恢复的UTS17301/raw_neighbors已通过 beta=0 NLL/ES 数值门禁。其原CAL-fit选择文件SHA保持 2b4db10a47dfe653be8fe62939768f485b638311d58f7cdde206ac3955aacaaf，无重新选参。

## 已知非阻塞警告

Phase T 首个E检查点保存时，`float(model.raw_alpha.sigmoid())`在记录alpha元数据时触发PyTorch“requires_grad tensor转标量”UserWarning。该值只用于JSON/检查点元数据，实际训练反向传播、优化器、保存的raw_alpha权重与推断均不依赖这个Python标量；不会修改梯度或模型。完整警告在phase_T.stderr.log。按不修改冻结科学源码的约束保留警告并继续，不为消除提示另改实现。

## 附加跨阶段数值复核（非科学源码修复）

额外只读比较F/T的18组相同O/A检查点，新增的“逐站点MAE/ES差均<2e-5”断言未成立。该断言不是用户协议的同pack beta=0门禁。完整差异另存CROSS_PHASE_OA_NUMERICAL_REVIEW.json，原失败与审计脚本修订diff保留。最大站点MAE差0.0006201863m，ES差0.09826636m；MAE宏差均<2e-5m，ES宏差最大0.001526876m，BF准确率逐站点相同。F推断块8、T推断块32，CUDA浮点前向/MC CDF路径并非逐位恒等；不新跑前向去追数值一致。科学代码、强制测试、容差、权重和所有已见结果不变。独立收尾审计明确标为PASS_WITH_DISCLOSED_CROSS_PHASE_DIFFERENCES，附加诊断标NOT_IDENTICAL，不宣称全部附加诊断通过。
