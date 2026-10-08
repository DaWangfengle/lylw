# V17-G1 本地执行报告

生成时间（UTC）：2026-10-08T07:55:26.317099+00:00

**最终决定：STOP_GEOMETRY_DENSITY_MECHANISM；冻结 gamma=0.7。**

本轮是用户独立授权的 development 几何支持机制探索。历史 V17 selected 的 outer coordinate-macro MAE 为 5.725985 m，V9 为 6.355559 m，历史 next_stage_authorized=false 原样保留。本轮不构成历史实验 PASS、waiver 或独立 outer-test 验证。

## 环境、耗时和执行顺序

| 项目 | 记录 |
| --- | --- |
| Python 解释器 | D:\miniminiconda\envs\liuyan\python.exe |
| Python | 3.12.12 \| packaged by Anaconda, Inc. \| (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)] |
| 操作系统 | Windows-10-10.0.19045-SP0 |
| Torch / CUDA runtime | 2.10.0+cu128 / 12.8 |
| GPU | NVIDIA GeForce RTX 2060，6442123264 bytes |
| 25 成员训练秒数合计 | 6033.700559800 |
| 已知日志阶段运行秒数合计 | 6436.354414000 |
| 中断阶段耗时无法恢复数 | 1 |
| 从初始审计至报告的 wall-clock 秒数（含中断） | 7856.953805000 |
| 最终测试 | 37 passed，0 failed / skipped / xfailed |
| 确认命令执行次数 | 1 |

执行顺序为编译、完整测试、CUDA 预检、prepare、成员 1..25 串行训练、check-members、calibrate 与 freeze、唯一一次 confirm。每条阶段命令的 stdout/stderr 与日志 SHA256 均独立保存在 run_audit；已完成阶段记录退出码、起止 UTC 和运行秒数，中断阶段无法恢复的退出码与耗时明确标为 N/A。

## 注册数据、源码和不可变记录

| 文件或绑定 | SHA256 |
| --- | --- |
| trainingData.csv | 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168 |
| 历史 V17 registry（self） | 15a42989c7fb3f4c92906cebefd08d4fa359d80bffe238ec3bdb9a8e268b450b |
| 历史 V17 registry（原始文件） | b667027ff121806dd7448a22b933b24d42a4a15c4a7831af5a861bccc834abab |
| G1 DESIGN | 6a11a8093b9a8f2619f21a5cdc140bff43e04ffad8cba3ab65ba98a34a3dac01 |
| G1 registry（self） | bfd44edf4f5f05f08dea2bc2c9972138b361fff529467c166c3e7ab333ebccf6 |
| G1 registry（原始文件） | c01ec1df8916f7c980c3640b012e2b4b3600781c74a26bb967276952e267db49 |
| selection_refit_v17_g1.csv | 1ead6c5d635f7da35fb10e509ddb09dac7af448086c763aa48e4d1f42fd2e949 |
| G1 测试文件 | 1473244a9f176adc0c2a88a58806dbc7c2b547e7d0d3804884c2cf7d2ea2c6f5 |
| g1_execution | a93888793b4ef91355c8480d9d14737073fb1aba71b650a3ef38edbeda8dfca0 |
| g1_protocol | 43c19ff5db2b3cc4e6c3487ef999aa73f0458738d89fef3537bf2f7c2a53521b |
| r0_training | 9b8afdbdd5475d0a94cb8e3cacfc66a62796b72964c3f5414eb7bf524c050b4d |
| r3_scoring | 4e17632041fe1364e253a6e8275772f45479b6cca454ce8351b6211e87139082 |

716 项受保护历史资产核验未变。历史文本源码和证据按 SHA256 检查；历史二进制仅检查文件大小及 mtime，未读取或复用历史模型权重。新 G1 生产模块及设计文件保留初始字节，registry 冻结后未修改这些源码、registry 或模型。保存文件的 checkpoint 哈希在训练完成、calibration 和 confirmation 之间一致。

## 坐标切分与无重叠证明

| 角色 | RSS 行数 | 物理坐标数 | 坐标键 SHA256 | 原始行索引 SHA256 |
| --- | --- | --- | --- | --- |
| selection_refit | 12717 | 595 | dee73a63ed8c3eddcc789454852c34b8042baa4886c8ba944f6c11cd64b28eb6 | 1c4d7c6e1da6339ee8fb6642ef6d2957c37a4822d388786b64d25a8f712b14f2 |
| calibration | 1529 | 72 | 31457ee770b9fc1ec4c19a5cb79a065141552788160bd476fc156f09dc1f4108 | 16aac6406d751cfecd59aefa306cfd7428938e35c110a8099f52b845bc28af1e |
| confirmation | 1625 | 79 | d3c44cb9443b1c04038629c3277319655ab74f437b4a57b0bf501baa38e32183 | 20c46fe771129a5314e2f733951c515759931ed295ba65bbb1e6db42b5d48d06 |

development 共 151 个物理坐标 / 3,154 行，按 building/floor 与固定 hash seed 18111 分为 calibration 72 组和 confirmation 79 组。同一物理坐标的全部 RSS 行属于同一侧。两侧坐标交集和行索引交集均为 0，并且并集等于 development；两侧与 selection_refit、outer_test 坐标无重叠。切分在任何 G1 预测前固定，证明见 run_audit/PREPARE_GATE.json。

## 25 名成员训练来源及 checkpoint

所有成员使用原始 V17 R0 网络、损失、AdamW、增强、数据变换与固定 epoch，仅重新训练 selection_refit 12,717 行 / 595 anchors。25 个 seeds 为 17301..17325；epoch 源自历史 member_fit/early_stop 决策。本轮未重新选择 epoch，未复用历史 outer_refit checkpoint，也未用 development / confirmation / outer / official 标签拟合或提前停止。

| 成员 | seed | epoch | 训练秒数 | checkpoint SHA256 |
| --- | --- | --- | --- | --- |
| 1 | 17301 | 65 | 250.412793900 | ddd8e4acb213ddba370308cb967ff2c95d6f25bccb436ccd1911cfd77be031d0 |
| 2 | 17302 | 69 | 264.424151400 | e8ba0f7b928ca013a2e2f8153c17ea85ff3be70a03b68273f271742e2724e79a |
| 3 | 17303 | 56 | 215.668915100 | f65f8a2de4468835d751b3e5791973e206d654810c1ba38f5b8b61edfe730e2e |
| 4 | 17304 | 67 | 263.032766200 | 74fc2ad61143630df3722adc3f00c637503595c54feb050c5402e1fb9bd6df0d |
| 5 | 17305 | 62 | 239.677075700 | 96550e8ec9be9bc45e08b56b59898e3a57f422c5fb9e84843ac162af1effce7d |
| 6 | 17306 | 60 | 232.879511000 | 477bdb623fabfffd5c18c2c9f90fcf5c4611155e2e2e5327e846e2cebb3b1309 |
| 7 | 17307 | 52 | 203.345787500 | 1434f77e0b2aa4f93099c33c72d8d471c2431136c07cff8e4f579b89950222fd |
| 8 | 17308 | 74 | 286.704998200 | 9f816b1a416235c44cd7ad7281742d61ac6e4c897c21de9da699fcf2aed6b74f |
| 9 | 17309 | 79 | 307.935510500 | a46292d394b7fb4e8337b55d7e8e7f9d5248085703772120c9b323936c2b2949 |
| 10 | 17310 | 62 | 241.707952000 | 23f75367036055832bc2db97e95916c0218a9f9720ae247ea95a83e607739924 |
| 11 | 17311 | 65 | 249.282728800 | 8f37475a5fa3ca8b23864c0507077cf0da3ef35b12a736bdda45fe53386a4bac |
| 12 | 17312 | 57 | 222.407883000 | 4372b446e7087ea262da303400f0feb81e7e53ac6b8f02cbb2c01fe37f8b65e3 |
| 13 | 17313 | 64 | 248.870084400 | ad86ea197408256744ee37b4161c51a4b68e86f99889c5ee5c0cfb55280e91a6 |
| 14 | 17314 | 61 | 232.661692800 | 168158d5d2f03178b1993f3ebf66501a32431e93bc8b194a6f79abff16889613 |
| 15 | 17315 | 50 | 194.258823000 | d5915803a023871d0075663aa4412aab39b7761bce957d890f484409bb7d93b9 |
| 16 | 17316 | 74 | 286.004969300 | 8624efc21503e620bdcbcd420f12a7f220d2d34fa0b81144558272001b4c3c8f |
| 17 | 17317 | 59 | 228.878507600 | a9f417376a38545e74210b5f8538ed95d67c283092b454a59b5f7f01abc81a5c |
| 18 | 17318 | 73 | 278.278008400 | af740808061139d63d31ec4d83bebad13a14cddd8baf0fb50e53b5da156893b6 |
| 19 | 17319 | 60 | 234.890485100 | c1dba1d7ce4a74954df3dde73050e2526ebde3157c8ad01f08031dffcac1ff95 |
| 20 | 17320 | 68 | 264.051395600 | 1e5cbe7664d24a27331f77b9680d5527807fd485e0ba63c0f582066cbc82ea54 |
| 21 | 17321 | 52 | 200.813407800 | a3b05b8d9802207110d5dfcad5f6f77e2ec57dcd97a3f26003f0b7a7035f6e74 |
| 22 | 17322 | 58 | 227.302801700 | fd7c4bd3bcb06e11f61979bb3226606f9cb52a1a076f938b34a982da481b5070 |
| 23 | 17323 | 53 | 206.119405700 | 69401f192bc30cbce834c6a70fd4843fbfa8962ab6cdca3bb03f512f86fdaeed |
| 24 | 17324 | 61 | 237.177496000 | 30aa4ffab96ee9d5536f081178e50acd79dbde673318819f77d9ed9941274ef8 |
| 25 | 17325 | 56 | 216.913409100 | 2877cef9d1546ab6258e9c42b4892a38f733fa5e9020415edb0b1fc831904d8e |

## Calibration、ES 约束与冻结选择

候选 gamma 固定为 0 / 0.35 / 0.70，邻域半径固定 10m。选择规则为 calibration coordinate-macro MAE 最小，且 ES ≤ gamma=0 ES+0.02m；并列时选择较小 gamma。本轮冻结 gamma=0.7。

| gamma | coordinate-macro MAE (m) | coordinate-macro ES (m) | row MAE (m) | ES guard |
| --- | --- | --- | --- | --- |
| 0.0 | 6.194178462 | 5.134415288 | 5.480589981 | True |
| 0.35 | 6.110030377 | 5.118244124 | 5.449656424 | True |
| 0.7 | 6.072321895 | 5.124763227 | 5.467208217 | True |

ES guard 上限：5.154415288 m。

| 冻结绑定 | SHA256 |
| --- | --- |
| calibration（self） | 0729b512f74de0d607afb4f4717919a2f139c1110f62e798cd5c0c443045f86d |
| calibration（原始文件） | 76a5379d43b93742587daea701948222014946fb1aaa2a7a5e8b5a01c9079f6e |
| freeze（self） | 2bf7e709997f27ad83898e57f06cd63370dab508b622ea2577930b94050e75fe |
| freeze（原始文件） | 24e9a1599bce2e1bf76a4b4da1e6707d0c8967777502a316d7d0fa07a508c43c |

calibration 和 freeze 完成后才执行 confirmation。冻结记录的 confirmation_metrics_accessed=false，冻结链和命令时间顺序均已核验。confirmation 未触发重新选择 gamma。

## 唯一一次 Confirmation

baseline 为 gamma=0，selected 为冻结 gamma，placebo 使用相同 gamma 并在训练 building/floor 内以 seed 18112 置乱 log-density。使用原始 R3 E16/L192/K192/alpha1，t4 分布，64 个共同 draws，固定 scoring seed 18114。

| 方案 | coordinate-macro MAE (m) | coordinate-macro ES (m) | row MAE (m) | row ES (m) |
| --- | --- | --- | --- | --- |
| baseline | 5.495018476 | 4.401050694 | 5.456216177 | 4.468596188 |
| selected | 5.351248714 | 4.277971940 | 5.181528242 | 4.265384289 |
| placebo | 5.566040141 | 4.383771822 | 5.419465211 | 4.382451472 |

confirmation 文件 self hash：a54a0a8ff452adf3245bd5145afa52b56f9a2c8eedec650e3e29923b57a9b76c；原始文件 SHA256：48806835eb64051a707fd52cd7e594baa1f7c68fda7a8f9db1f6b03bd0888a9d。

统计量为固定 25 个平衡 E16 子集的平均损失：每成员恰好出现 16 次，子集互相重叠。先对每个子集计算损失，再平均；不把平均预测分布当作该估计量，也不把 25 个子集当作 25 次独立重复实验或一个固定 16 成员部署模型。

### 25 子集分布

| 方案 | 指标 | mean | sample SD | min | q10 | median | q90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | MAE | 5.495018476 | 0.053289792 | 5.414834690 | 5.437261487 | 5.477440117 | 5.568673794 | 5.579155865 |
| baseline | ES | 4.401050694 | 0.029272261 | 4.346569547 | 4.368414336 | 4.400205669 | 4.443503179 | 4.454772809 |
| selected | MAE | 5.351248714 | 0.047270797 | 5.294150155 | 5.301711389 | 5.344476422 | 5.421021782 | 5.444627228 |
| selected | ES | 4.277971940 | 0.021162511 | 4.235862682 | 4.257108519 | 4.275646406 | 4.309152277 | 4.322566978 |
| placebo | MAE | 5.566040141 | 0.067816067 | 5.490895621 | 5.499824285 | 5.542807660 | 5.668860601 | 5.699466617 |
| placebo | ES | 4.383771822 | 0.029266801 | 4.337008418 | 4.349630407 | 4.375751681 | 4.417185567 | 4.447470149 |

上表 SD 是重叠子集结果的描述统计，不能解释为 25 个独立重复的标准误。

| 子集 | baseline MAE | selected MAE | placebo MAE | baseline ES | selected ES | placebo ES |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.544769572 | 5.397445276 | 5.639433691 | 4.441566373 | 4.282210348 | 4.418771951 |
| 2 | 5.471939017 | 5.353371436 | 5.552272206 | 4.400205669 | 4.312738648 | 4.392963667 |
| 3 | 5.445632313 | 5.308879535 | 5.505919671 | 4.383298963 | 4.276277004 | 4.360258217 |
| 4 | 5.465065861 | 5.296469026 | 5.504598294 | 4.379841977 | 4.257538188 | 4.375751681 |
| 5 | 5.570564402 | 5.424493242 | 5.673364050 | 4.444794383 | 4.288397514 | 4.435775245 |
| 6 | 5.439668287 | 5.306846904 | 5.499798397 | 4.346569547 | 4.266528629 | 4.366661392 |
| 7 | 5.463436859 | 5.344476422 | 5.542807660 | 4.386391829 | 4.275646406 | 4.373314733 |
| 8 | 5.565837882 | 5.390439624 | 5.636656116 | 4.411905188 | 4.258867958 | 4.406825646 |
| 9 | 5.502706464 | 5.360283585 | 5.568366505 | 4.416223987 | 4.322566978 | 4.414805991 |
| 10 | 5.435656954 | 5.304370626 | 5.501636490 | 4.380403449 | 4.269476619 | 4.337008418 |
| 11 | 5.477440117 | 5.324147215 | 5.518998471 | 4.414323768 | 4.279465513 | 4.380509947 |
| 12 | 5.564264043 | 5.415814592 | 5.664701797 | 4.418624109 | 4.317387903 | 4.403517943 |
| 13 | 5.464001375 | 5.304745965 | 5.499863116 | 4.399213699 | 4.288593930 | 4.387449380 |
| 14 | 5.466429852 | 5.326016062 | 5.524318518 | 4.367466295 | 4.249139158 | 4.352521605 |
| 15 | 5.550960524 | 5.375795302 | 5.615411152 | 4.402496078 | 4.261883374 | 4.367627870 |
| 16 | 5.578967208 | 5.441985661 | 5.671633137 | 4.454772809 | 4.303772720 | 4.447470149 |
| 17 | 5.422743040 | 5.299938564 | 5.497377246 | 4.378865997 | 4.262373488 | 4.347103065 |
| 18 | 5.471974740 | 5.329819171 | 5.529126917 | 4.390100062 | 4.272257463 | 4.355569045 |
| 19 | 5.552023593 | 5.392118795 | 5.642183597 | 4.445035227 | 4.289136894 | 4.414060426 |
| 20 | 5.477814377 | 5.312629224 | 5.516326126 | 4.427838615 | 4.287735331 | 4.396710321 |
| 21 | 5.444081251 | 5.325942889 | 5.523655952 | 4.370179889 | 4.269978932 | 4.373906269 |
| 22 | 5.523502735 | 5.359412344 | 5.583095630 | 4.369836398 | 4.256822073 | 4.357258036 |
| 23 | 5.579155865 | 5.444627228 | 5.699466617 | 4.430609832 | 4.292009514 | 4.409355320 |
| 24 | 5.414834690 | 5.294150155 | 5.490895621 | 4.359890683 | 4.235862682 | 4.347702942 |
| 25 | 5.481990881 | 5.346998999 | 5.549096538 | 4.405812536 | 4.272631220 | 4.371396286 |

## 配对推断、familywise bounds 与 gate

按 confirmation 物理坐标组进行 5,000 次 paired studentized max-T，seed 18113；同时覆盖 MAE 和 Energy Score 两个端点，95% familywise simultaneous upper bounds。推断条件于本轮固定模型、子集和确认坐标样本。

| 端点 | selected − baseline (m) | 95% familywise simultaneous upper (m) |
| --- | --- | --- |
| exact_mixture_mean_mae_m | -0.143769762 | 0.012352838 |
| energy_score_2d_m | -0.123078755 | -0.018468910 |

placebo − baseline 的 coordinate-macro MAE：0.071021664 m。

| GO 条件 | 结果 |
| --- | --- |
| energy_familywise_upper_not_positive | True |
| geometry_better_than_permuted_density | True |
| mae_familywise_upper_below_zero | False |
| mae_gain_at_least_0p15m | False |

全部条件同时通过：False；原始决定：STOP_GEOMETRY_DENSITY_MECHANISM。

逐坐标差值、max-T 完整数值及所有子集成员列表见 confirmation_v17_g1.json；没有对不利结果豁免、回头重选、增加确认次数或实现后续模型。

## 失败、修复和 CUDA 确定性

首轮测试：34 passed / 1 failed。旧 R3 测试引用的 DESIGN_V17_R3_OUTER_CONFIRMATION.md 在本地不存在。仅在新 tests/test_uji_v17_g1.py 中增加 session 隔离 fixture，文档明确标为 G1 测试依赖，并引用未修改的原始 R3 评分源码；未宣称其为恢复的原始预注册设计。旧 R3 测试的数值断言及历史文件保持原样。另加独立相反预测的期望损失检验，以及 selection_refit 序列化与已验证 resume 的接口回归。

第二轮测试：36 passed / 1 failed。新增回归使用了错误的嵌套 mae 字段名，产生 KeyError；修正为原始 R3 返回的 exact_mixture_mean_mae_coordinate_macro_m。第三轮完整测试：37 passed / 0 failed，无 skipped / xfailed。G1 生产模块、参数、切分、损失、优化器和 gate 均未改动，修复发生在 prepare 前。

原始 stdout/stderr 和失败退出码均保留。差异与修复证明见 PRECHECK_FIXTURE_REPAIR.json/.diff、PRECHECK_REGRESSION_FIELD_REPAIR.json/.diff；两次失败的 pytest 临时目录也留存于新 G1 模型目录。初始进程启动器失败通过同范围审查执行恢复；第一次修复工具发生的字符串解析错误在执行前中止，无文件变动，已记入修复审计。

| 阶段 | attempt | 退出码 | 秒数 | stdout SHA256 | stderr SHA256 |
| --- | --- | --- | --- | --- | --- |
| calibrate | 1 | 0 | 39.686485200 | f4be74abca7e5ab312a5a97732d364ace2b20a29501f821cd784a6cd00330afa | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| check-members | 1 | 0 | 4.930556400 | 20bce1ec6c708d24b24711017c46551eaa53f3c97e7c256a306b5a4596a12fd8 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| compile | 1 | 0 | 0.105330200 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| compile | 2 | 0 | 0.078985300 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| compile | 3 | 0 | 0.082233100 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| confirm | 1 | 0 | 40.951029900 | ef1939fdfa50c69eb9b3e1b0e24cf7532ee8e7f3319603d74c97b437df136312 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| preflight | 1 | 0 | 4.162801900 | dadb1ac223164ee7408f5d029b863ef2d49ce463fd690403b3ac53b451972c4d | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| prepare | 1 | 0 | 8.152436400 | c4016a744d3905d3303e9624ba041e75773c98eea2508133648a61dae191ef27 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| pytest | 1 | 1 | 9.663473300 | 7d3ef4af4ab93acc126c6f0687fc218b3613c93d2a8d4434f81881712630421d | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| pytest | 2 | 1 | 7.839892700 | b61cc0d559040d4911a60a96dbed1dabf5e693a0f57542d9f2451095b04f4523 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| pytest | 3 | 0 | 7.747464400 | e0cb6c1437cfce142c6318294eeb4bb44228eb4e7f7b51ca7996e37108fbe439 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| train-member-01 | 1 | 0 | 261.065008300 | 8fa1a776a30bdcd99bd95e372d59f3d96e27c57ce27de6c02c5e3ca726be40d1 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-02 | 1 | 0 | 275.470241400 | a9e98b8ee2cd44058feadf556fd22c454ed100e7d20e05029d4a2cdb2ad5a0bc | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-03 | 1 | 0 | 226.849989100 | c9647ef9cc50a618578b40c8c3df66b49f91ca44ed72d07d8713bbdb075a7b04 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-04 | 1 | 0 | 273.977497800 | bc8cb157f1ff67b866c997a3d0251965c739168824b5db01a4932384029f74e8 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-05 | 1 | 0 | 250.775221500 | 330b5177f941c4ad42b47d7b2d981aa5abc18256f80636f89a8d50e12c216df0 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-06 | 1 | 0 | 243.975788200 | f108f8bbd579814b5f4e279340b073568c031b904cf2f8ca2a9e7088f5488ff7 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-07 | 1 | N/A | N/A | a3be4bca76340f13449cf7217b572313c9ec5748770d6434f0b7985976be6221 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-07 | 2 | 0 | 214.274754300 | f6e68dfabd2be5f3288a8e776f254620ab98d5fa4aad1aab45cae0f9f9805f97 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-08 | 1 | 0 | 297.744801900 | 80f93f30916ccd0455bd86db5c5e2450d8bd2a3fa5ed3cc11349fa405572c292 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-09 | 1 | 0 | 319.055649500 | 90f7fba9a96292c05be5700cf85577affd6c335be03b8aec126ff47e9e44ced0 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-10 | 1 | 0 | 254.335947500 | 9554019cae22fa0a518b72b248fa080959e527647b5b1412de05f5fafdf5b0ee | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-11 | 1 | 0 | 260.530210000 | 7419a1aff4aa9cd8c5eb437d6e7857a3bee262109337d8c255b6897569aba320 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-12 | 1 | 0 | 233.541001800 | a0c2d73e9c2a437c31b1910c5443642b2d0d2fe417f35ca374c8e4484b0f6122 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-13 | 1 | 0 | 260.050653400 | 00ef72fd0c2f4769de3f4c7d40502497d3639ecdc92fcbd9d30eb69ae23e9667 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-14 | 1 | 0 | 243.816424700 | ce9ebee022549ef3963a60a64bbc13ec8b8adb56a88e288693a5a58ae430d2f1 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-15 | 1 | 0 | 205.397069700 | 9d2154d1e5010bfb6a24ae36dd5f977769f14fe9282d9fda5c852758ebfaf5f6 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-16 | 1 | 0 | 297.050261300 | 3d001d18f9121690f2552037b1f3f5969909cda9288b61dac9cd99bd3a67e4aa | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-17 | 1 | 0 | 240.068940600 | a96f3efda5cb1e8e73c8e9816a2ee59125871a559c730505074bbadea40d5396 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-18 | 1 | 0 | 289.589288600 | 31fdfc974ca57d704ce95135e3e3e2c8d5b8f5bbc2a81e975dfe060770aeae5d | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-19 | 1 | 0 | 246.096105200 | 17bf7296470ce2ce165b229cf1a58b5c2aec6e1b5c7e728d2098f8008a831e64 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-20 | 1 | 0 | 275.200295900 | 5f2a19bda32235a2a4743817c983560d270de9c037f585507b240ecb91747e7d | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-21 | 1 | 0 | 211.861892200 | fe6483bf7a2b13afc154c9bb7ad62739cc8f03312b35d196b15a76df73b173e6 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-22 | 1 | 0 | 238.476840400 | e246d61800e2179c9356a2cc0627249638e110fb40467f9861b294d7dd84f8e6 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-23 | 1 | 0 | 217.254977300 | 919387550641c913e2058e786d0ce093bacfc30373f1f6d3e83804445537234c | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-24 | 1 | 0 | 248.381743400 | 2a5ab63296e66c6c298950d9b7654f74be9a30ac2ecb4c88b27147b91d171f48 | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |
| train-member-25 | 1 | 0 | 228.113121200 | 74a45d97c5e6bab5003b4df8151842371782f8722a7908f8392bbe0f02cc108c | d99e0ded63790f79cf66c3492a84c2a2f574455ca6f5e4f8deed1d1f3e693048 |

stderr 中保存 26 条警告记录。原始 CUDA Memory Efficient attention backward 警告说明该内核默认非确定；本轮保留原训练配置和精度，固定 seed 不能承诺 GPU 训练逐位相同。完整警告见 run_audit/FINAL_EXECUTION_VERIFICATION.json 与各成员 stderr 日志。

训练期间发生一次 turn 中断：成员 1..6 已完成，成员 7 第一次尝试停在已记录的 epoch 15/52，未生成 checkpoint。先核验前六名的 seed、epoch、registry、selection_refit CSV 和 checkpoint 哈希，保留成员 7 两份原始日志；其退出码与精确耗时无法恢复，记为 N/A。成员 7 以同一 seed 和完整 52 epochs 重新训练为 attempt02，后续仍按原编号执行。详见 INTERRUPTION_RESUME_VERIFICATION_01.json。该恢复未修改冻结 G1 生产源码、registry、设计或已完成模型；报告工具另行记录修订审计。

## 证据范围

本轮只在 V17 development 内探索和确认几何密度机制。595-anchor selection_refit 模型及本轮 confirmation cohort 与历史 746-anchor outer_refit / outer-test 不同，不能把这里的结果与旧 5.725985 m 当作同一测试集比较。

official validation、outer-test 和外部数据预测次数均为 0；未进行外部泛化验证，未完成 Q1 完整证据链，没有部署或最终模型结论。历史 next_stage_authorized=false 未修改。本轮所有操作在本地执行，Git 操作由用户自行负责。

报告排版修订仅转义 Python 版本中的表格竖线、统一中断字段的 N/A 显示并澄清耗时记录；原始报告、原始最终审计及差异均保存在 run_audit/REPORT_FORMATTING_AMENDMENT.json 所列路径中。科学结果及冻结文件未改动。
