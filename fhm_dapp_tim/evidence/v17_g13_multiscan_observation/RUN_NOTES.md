# V17-G13 运行记录

最终交付：唯一正式 CUDA run 已完成；用户授权的内存渲染检查 VERIFIED。原 CLI 最终 check 因显示顺序失败（exit 1），失败原文保留。数值屏幕为 PROMISING_RETROSPECTIVE_EVIDENCE，仅回顾性描述性信号。
环境：D:\miniminiconda\envs\liuyan\python.exe；Python3.12.12 / Torch2.10.0+cu128 / CUDA12.8；RTX2060 6GB。全流程 15.96 分钟，正式 run 45.513 秒。

命令 exit：compile 0；初测 1（25 passed/3 个 tmp_path 错误）；创建临时父目录 0；重测 0（28 passed in 5.47s）；preflight 0；FIT-only CUDA/公式 0；prepare 0；冻结 check 0；run 0；原最终 check 1；只读审计 0；授权内存检查 0（6 个纯合成报告 roundtrip 回归通过）。每命令耗时、stdout/stderr、退出码和输出 SHA256 见 RUN_STDOUT.log / RUN_STDERR.log。
问题1：辅助命令 --basetemp 缺少父目录；创建本轮临时父目录后重测，无源码/测试修改。
问题2：_report 直接显示系数字典，运行时 visibility,strength，JSON 按键排序后重渲染为 strength,visibility，只有第24行变化。先 STOP 并保存差异，随后用户明确授权仅在内存固定这两个字段的显示顺序。授权检查禁用训练/推断入口；原文件不改、原失败不抹除、未重跑 run。原 STOP 版记录全文及 SHA 也保存在 stdout。

FIT/CAL/CONFIRM（站点/行）：TUJI 643/3199、137/695、139/701；UTS 945/5791、258/1625、263/1692。角色两两坐标交集0；原 G11 锚点一一映射通过。
FIT 扫描分布（次数:点数）：TUJI 1:1; 2:2; 3:22; 4:6; 5:568; 6:44；UTS 1:11; 2:41; 3:205; 4:143; 5:99; 6:80; 7:70; 8:71; 9:76; 10:37; 11:35; 12:28; 13:23; 14:10; 15:9; 16:1; 17:3; 18:1; 20:1; 24:1。两域中位数5，单次扫描站点分别1与11，不假设扫描独立。
原 seen AP：TUJI115/259、UTS557/589；五成员 AP mask/原型/锚点一致。NumPy 独立公式复算最大差<1.44e-7；FIT 零系数概率及MAE/ES逐位一致；FIT CUDA smoke 峰值约71/92MiB（非正式运行峰值）。

十份原 G6/G4 checkpoint SHA 全匹配且未变，完整逐成员 SHA 在 manifest.original_E5_panel_provenance；面板列表 SHA：TUJI 5cc16ab002a4f01264d415ceb6a63624df754e825aac69662072527be2ca34e9；UTS 5f064251a176ee04aac226cc9447b1ca754fea1eba8e786f22b285b2357376c3。
371 个历史源码/测试/证据/协议及 G13 源码、测试、设计、根协议 SHA 未变；全部输入/输出 SHA、角色键 SHA 与授权前后核验见日志/manifest。大模型仍在本地，未执行 Git。

一次 CAL 联合拟合：站点等权、两域各1/2；batch24 累积全损失，每epoch Adam一步，共24步，无早停/选择。θ_visibility=0.837480605，θ_strength=0.946925223；单因素使用同次系数。

| CONFIRM | Anchor MAE/ES m | Joint MAE/ES m | ΔMAE/ΔES m | 改善/恶化点 |
|---|---:|---:|---:|---:|
| tuji | 2.657774/1.867334 | 2.425471/1.670775 | -0.232303/-0.196559 | 114/25 |
| uts | 5.107528/3.937673 | 4.579336/3.689665 | -0.528193/-0.248008 | 175/88 |

tuji P95：4.466831→4.152300m；joint 最差 5.172195m；最大恶化 0.421228m，键 [0, 0, 10.129, 15.501]。
uts P95：12.221708→11.819741m；joint 最差 22.199463m；最大恶化 16.909778m，键 [0, 15, 68.9735, 7.0808]。
五方法全部逐站点 MAE/ES、P90/P95/最差点和配对向量保存在 evaluation；两域各 MAE改善≥0.15m 且 ES不恶化，但 UTS 最差点恶化须保留。

数据：仅 UTS_training 与五个TUJI MAT results_train；物理容器含 results_test 字节且被打开/哈希，但限定变量未解码/返回 results_test。FIT smoke 使用数据打开白名单；正式冻结加载路径未调用其他test/Tampere/UJI。全训练文件坐标用于角色重建，CONFIRM未进入 FIT统计或系数拟合；系数先冻结，再提取/评分 CONFIRM。
无 CUDA确定性 warning、NaN/Inf/OOM。RSS=-1无法区分漏检与地板裁剪；AP独立 composite likelihood、相关扫描、设备差异、ES Monte Carlo及既有历史确认标签限制结论。所有新证据仍来自相同 RSS，不是独立传感器；不能宣称新盲测、正式晋级或外部泛化，UJI等待独立协议。
