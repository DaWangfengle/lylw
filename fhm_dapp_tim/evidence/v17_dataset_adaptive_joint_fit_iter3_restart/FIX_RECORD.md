# ITER3 距离截断精度修复与完整重启

原始指令：`D:/liulanqi/V17_DATASET_ADAPTIVE_JOINT_FIT_ITER3_CODEX.txt`。

首次运行保留在 `../v17_dataset_adaptive_joint_fit_iter3/`。原始 30 项单测与九组 FIT 几何预检通过，但 TUJI seed17301/C 在 `fit_quantile_geometry_01` 的 S2 计数检查停止，退出码 1。基础边数 2946，R_max_S1=R_max_S2=1.330907885325485m，最低要求 1326 条。预检计算保留 2209 条，运行时筛选保留 2216 条；7 条边界边因计算精度差异被额外纳入。异常文案为“below validated minimum”，实际触发条件是保留数与验证值不一致，并非低于 45%。

根因是预检及 `_validate_geometry_support` 将存储坐标转为 float64 计算距离，而 `run` 和 `_make_probability` 使用了原 float32 NumPy 坐标直接求范数/比较半径。

用户已明确授权：“允许最小修复，新目录重跑 TUJI（推荐）”。授权原文、时间及修复前后 SHA256 见原失败目录的 `USER_AUTHORIZATION_PRECISION_FIX.json`；实际补丁见其 `run_logs/APPLIED_PRECISION_FIX.patch`。

实际源码只改两行：在 `run` 和 `_make_probability` 取得用于边长计算的 NumPy 坐标时添加 `.astype(np.float64,copy=False)`，与预检一致。存储坐标的数值、网络参数、连续分量中心与尺度张量不变；候选分位数/半径、温度、45% 支持要求、1e-6m MAE 门禁、两个 lambda、目标函数、随机流和站点角色均未调整。

新增两项纯合成回归：一项直接覆盖评分概率构造的候选截断和原分量不变性，另一项覆盖实际 `run` 路径的候选截断并在真实评分前主动结束。两项修订后回归在原正式源码上均失败，在临时修复预览上与原单测合计 32 passed；应用修复后，正式新测试文件也完整 32 passed。没有用真实 CAL-eval 指标挑选修复。

- 修复前源码 SHA256：`125fd47b2e4d778ead948120f3e47cd5e1ed36cb1562cdb7ff75e4085ec05489`
- 修复后源码 SHA256：`2895555096b253f66a7eccf445510dccc5de027e0429960bdae337bf826fe498`
- 修复后正式测试 SHA256：`2adc3e21ca71dc9c975e808c634900b94e3895d15c91f6ba2777795590801dc5`

首次辅助预览助手的匹配数断言，以及首版合成运行夹具的重复尺度问题，均保留在旧目录 STOPPED 报告、完整日志与夹具版本中。它们只发生于辅助验证，不改变正式科学源码或参数；最终修订夹具保留实际几何与支持门禁并准确复现两个路径的错误。

原始失败目录、冻结记录、修复前源文件与所有日志均保留。本目录是用户批准的全新重启目录；重新执行编译、正式单测、九组 FIT 预检，再从 TUJI seed17301/C 开始完整运行。三份预检输出与修复前对应文件逐字节相同。UTS/UJI 仅参加 FIT 几何预检，不启动其定位评分；内部与官方 TEST 均不评分，不训练网络，不进行 Git 上传操作。
