# MSEP-Loc 单成员锚点后验三阶段实验实测报告

生成时间：2026-10-16T21:46:34.269267+08:00。

三域真实训练与一次正式 TEST 评分已完成。所有八组、负结果及逐位置伤害均保留。没有使用官方提供测试文件，没有执行 Git 操作；本报告不选择部署策略，不执行后续模型或集成试验。

这是当前四角色站点留出的新实验，原始训练语料和部分物理位置此前有历史研究暴露。不是全新外部盲测，不是空间块留出；同域近邻及扫描/设备相关性可能降低有效独立样本数。不能与历史 E5、5.7m/4.58m 的不同 TEST 拆分直接比较，也不能证明 JCR Q1 外部泛化。

## 环境、冻结与命令

分支：`ref: refs/heads/msep-single-anchor-posterior-v1`；起始源码引用：`8845d1ffc6bf32bc8c005dd960016c94d0184b86`。Python `D:\miniminiconda\envs\liuyan\python.exe`，3.12.12；PyTorch 2.10.0+cu128，CUDA 12.8，NVIDIA GeForce RTX 2060（6.00 GiB）；NumPy 2.4.4，SciPy 1.17.1，pandas 3.0.1，pytest 9.0.3。

正式方法于 2026-10-16T21:25:18.048009+08:00 冻结，`METHOD_FREEZE.json` SHA256 `a1ebb1ec7e7352c0b007842f2ee4e06e45f57b3ff82b218fec24164da6767e12`。三域均 seed=17301、split_seed=20261016、全 FIT 锚点、batch=64、base最多80轮（DEV早停）、joint固定10轮、ES两流各32次。没有降低 batch、延长训练、温度扩网格或选择 TEST 赢家。

每域 `TEST_INVOCATION.lock` 在所有权重/校准参数保存之后、首次 TEST 评分之前排他创建，绑定源代码和 checkpoint SHA。每种方法每域仅一次正式评分。smoke为1轮base+1轮joint、batch32、两流各8次ES，全部只评价DEV；其checkpoint无TEST站点键，RESULT无test_metrics。

|命令记录标签|退出码|耗时(s)|
|---|---:|---:|
|environment|0|6.889|
|compile_initial|0|0.114|
|pytest_initial|0|6.737|
|pytest_loader_regression_before_fix|1|5.423|
|pytest_loader_fixed|0|5.314|
|compile_safe_loader|1|0.096|
|compile_pre_smoke|0|0.095|
|pytest_pre_smoke|0|5.166|
|safe_study_help|0|4.290|
|data_audit|0|10.434|
|smoke_tuji|0|21.805|
|smoke_uts|0|40.814|
|smoke_uji|0|62.353|
|smoke_result_audit|0|2.922|
|full_tuji|0|145.578|
|full_uts|0|414.728|
|full_uji|0|523.364|
|full_result_audit|0|2.938|

所有实际参数、开始结束时间、stdout/stderr 路径及日志 SHA256 见 `EXECUTION_RECORD.json`。原始日志保留，不把回归演示的失败删除。命令前缀均为指定 Python：

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset tuji --device auto --base-epochs 1 --joint-epochs 1 --smoke --batch 32 --es-draws 8 --output fhm_dapp_tim/evidence/msep_single_smoke_tuji
```

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset uts --device auto --base-epochs 1 --joint-epochs 1 --smoke --batch 32 --es-draws 8 --output fhm_dapp_tim/evidence/msep_single_smoke_uts
```

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset uji --device auto --base-epochs 1 --joint-epochs 1 --smoke --batch 32 --es-draws 8 --output fhm_dapp_tim/evidence/msep_single_smoke_uji
```

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset tuji --device auto --base-epochs 80 --joint-epochs 10 --batch 64 --es-draws 32 --seed 17301 --split-seed 20261016 --support all --output fhm_dapp_tim/evidence/msep_single_tuji_full_v1
```

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset uts --device auto --base-epochs 80 --joint-epochs 10 --batch 64 --es-draws 32 --seed 17301 --split-seed 20261016 --support all --output fhm_dapp_tim/evidence/msep_single_uts_full_v1
```

```powershell
& "D:\miniminiconda\envs\liuyan\python.exe" -m fhm_dapp_tim.fhm_dapp.msep_single_study --dataset uji --device auto --base-epochs 80 --joint-epochs 10 --batch 64 --es-draws 32 --seed 17301 --split-seed 20261016 --support all --output fhm_dapp_tim/evidence/msep_single_uji_full_v1
```

**明确的命令例外**：没有执行旧 `v18_4_datasets --help`。该入口忽略参数而实际加载四域训练数据，TUJI默认还解码测试变量；已只读检查入口并运行安全的 MSEP CLI `--help`，exit=0。不能把被替代的命令写成已通过。

原指定测试9 passed；限制变量的合成回归在修复前真实产生2 failed/10 passed，修复后最终12 passed。日志扩展引入的一处花括号编译错误也保留；修复后编译/测试以及三域冒烟均成功。只改新MSEP study和新测试，未改后验数学算子、方法常数、历史科学源码或数据。完整原始失败及源码diff见 `FIX_RECORD.md`。

## 数据和四角色隔离

按(building,floor,raw XY四位)分组，各楼层固定PRNG划分约70% FIT/10% DEV/10% CAL/10% TEST（逐层floor取整，剩余全部FIT）。每个物理位置全部扫描只属于一角色。此处原始坐标来自canonical float32适配器；UJI投影坐标已有float32量化，未声称厘米以下几何精度。

|域|FIT站点/扫描|DEV站点/扫描|CAL站点/扫描|TEST站点/扫描|AP总数/FIT模型AP|
|---|---:|---:|---:|---:|---:|
|TUJI|646/3241|91/448|91/453|91/453|259/116|
|UTS|1052/6519|138/846|138/862|138/881|589/556|
|UJI|672/14398|87/1792|87/1867|87/1880|520/324|

三域角色交集均为空，原始/居中坐标分组均一一对应，FIT站点与anchor ID一一对应；非FIT sample_anchor全部为-1。G13观测memory仅调用原FIT RSS与FIT anchor IDs构建，model AP mask来自FIT原型，设备/用户类别不进入后验门控。真实XY/楼层仅用于训练/校准损失、DEV早停和评分诊断，推断API不接收query真值。

TUJI五个MAT物理容器被打开及哈希，但MSEP调用限定 `variable_names=("results_train",)`，返回字典拒绝results_test；不能声称容器所有测试字节从未被打开。没有加载UTS_test、UJI validationData或Tampere。

|域|FIT每站扫描分布(count:站点数)|至少2扫描站点|
|---|---|---:|
|TUJI|`{"3": 16, "4": 5, "5": 577, "6": 48}`|646|
|UTS|`{"1": 15, "2": 48, "3": 215, "4": 170, "5": 119, "6": 87, "7": 63, "8": 73, "9": 80, "10": 49, "11": 42, "12": 29, "13": 27, "14": 13, "15": 9, "16": 3, "17": 3, "18": 2, "19": 1, "20": 1, "22": 2, "35": 1}`|1037|
|UJI|`{"2": 2, "3": 1, "5": 1, "6": 1, "7": 2, "8": 2, "9": 9, "10": 105, "11": 1, "12": 3, "14": 4, "15": 2, "16": 1, "17": 4, "18": 9, "19": 48, "20": 307, "21": 1, "22": 1, "23": 1, "24": 2, "25": 3, "26": 2, "27": 13, "28": 8, "29": 21, "30": 57, "31": 1, "34": 1, "35": 2, "36": 2, "37": 6, "38": 10, "39": 5, "40": 11, "42": 2, "44": 2, "47": 2, "48": 1, "49": 5, "50": 4, "58": 1, "60": 3, "62": 1, "70": 1, "80": 1}`|672|

RSS>-0.999才算模型检测；-1可能同时表示缺失或裁剪至-110dBm的有效弱信号，无法完美区分。重复扫描不等于独立采样；FIT统计没有估计设备/会话独立性。

输入完整SHA和每建筑楼层、设备的角色计数及role-key SHA见 `DATA_AUDIT.json`；三域正式结果的角色SHA已与该独立预检匹配。输入SHA：

- TUJI `data\TUJI1\Formatted_data_A12.mat`：`73c59c6a8a12c45c4b6bf9b93b45cb89d0cfb7bb147260dc763f0d5f5af314a5`
- TUJI `data\TUJI1\Formatted_data_POCO.mat`：`9973384ac04365a55a2e6b33b16b61724cf41869e828b1544c7c4bc61c9d2495`
- TUJI `data\TUJI1\Formatted_data_S20.mat`：`e3e327243852ba38087c4e8f07c4d94c7b8acc0987e961d5dc6e8fcf23dd354e`
- TUJI `data\TUJI1\Formatted_data_S7.mat`：`c33ac65f9b61f6b24272b256b091c54747d5bce3f186a8a96e83cbc3e9f9fdab`
- TUJI `data\TUJI1\Formatted_data_tabS7.mat`：`f8a9bf0a94367f2255a3c94b2bd4b633d73d7000a8f1a0cf5c25e90d231b0e23`
- UTS `data\UTSIndoorLoc\UTS_training.csv`：`a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b`
- UJI `data\UJIndoorLoc\trainingData.csv`：`45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`

## 八组方法和训练条件

|简称|JSON策略键|训练/干预类型|
|---|---|---|
|Raw|`v17_single_raw`|原基线骨干|
|T=.6|`v17_single_temp_t06`|固定骨干/固定温度|
|T_CAL|`v17_single_temp_fit`|固定骨干/CAL 1参数|
|G13_hist|`v17_single_g13_historical_cal`|固定骨干/CAL原代理2系数|
|G13_NLL|`v17_single_g13`|固定骨干/CAL连续NLL2系数|
|Unified_frozen|`v17_single_unified_frozen`|固定骨干/CAL门控|
|Joint_no_ev|`unified_joint_no_evidence`|联合骨干/推理关证据|
|Joint|`unified_joint`|FIT真实联合训练|

前六组使用同一base骨干、所有相同FIT候选、同一连续分量中心和Student-t尺度；只改锚点概率。统一冻结头的全局系数与G13_NLL完全相同，只在CAL学习4→2线性可靠性门控（10参数）。G13_hist是单域单成员、原24轮Dirac ES+.3点风险+.1KL代理，不是原E5双域pooled的历史系数；G13_NLL改用连续Student-t NLL+.3点风险+.05KL，须作为不同校准目标解释。

Joint从同一选中base开始，在FIT对query自身anchor排除后，实际更新编码器、检索keys、连续Student-t头和可靠性后验，门控/系数均在FIT训练。原G13观测统计固定。**Joint_no_ev不是独立训练的原始模型**：它仅在同一共同训练骨干上关闭观测证据，保留T=.6。比较Joint与Joint_no_ev隔离的是该共同骨干的推理证据开关。Joint与固定骨干方法的对比同时含有骨干与分量变化，不能只归因概率修正。

原V17同建筑邻居损失保留，仍可能跨楼层；domain adversarial权重固定为0。单成员R3原子评分neural_scale=1、prototype_weight=32/3，默认全FIT支持；并非历史R0 top24/.75配置，也不是原E5 Top192实验的严格数值复现。

Joint的leave-own-anchor-out不等于完全未见站点meta-learning：预训练base和全FIT观测统计原本已见这些FIT查询。真正的四角色站点隔离用于DEV/CAL/TEST，不将FIT内部自排除夸大为端到端独立外部评价。

CAL温度优化90步、lr=.045、T∈[.25,3]；G13_hist 24轮、Adam .05、每轮累积所有CAL站点等权梯度、批块12；G13_NLL 100步、Adam .04；frozen gate 100步、Adam .025、.01门控惩罚。Joint原定10轮、骨干lr3e-5/头lr1e-3、KL .02、门控惩罚 .002。全部保持冻结源码，不按TEST调参。

|域|base实际轮数/选中轮|选中DEV MAE(m)|joint轮数|CAL T|G13_hist VIS/RSS|G13_NLL VIS/RSS|
|---|---:|---:|---:|---:|---|---|
|TUJI|80/80|2.567775|10|0.520972|0.043511/1.010196|0.685154/1.500000|
|UTS|80/77|5.518287|10|1.140179|0.018354/0.607657|0.000000/0.000000|
|UJI|54/42|7.336136|10|1.241687|0.000000/0.000000|0.000000/0.000000|

最终联合后验全局theta（直接只读冻结checkpoint）：TUJI VIS/RSS=[0.7628957033157349, 1.3589577674865723]；UTS VIS/RSS=[0.4167191684246063, 0.5264749526977539]；UJI VIS/RSS=[0.0, 0.0]。完整theta、gate weight/bias及checkpoint SHA见 `PARAMETER_REVIEW.json`，未对原始RESULT/权重作补写。


base遵循原实现：需比记录best改善>.01m才更新best，在epoch≥20且距best≥patience12时早停；否则到80轮上限，不因曲线继续下降追加预算。FIT基本训练和joint按扫描DataLoader，CAL目标按站点等权。只有一个训练seed，不能声称已估计训练随机性。

## 一次TEST：完整性能表

MAE、ES、P90/P95均先站点内RSS扫描均值，再站点等权。下表楼宇楼层joint accuracy按RESULT原字段是**扫描加权**；下一表的全部诊断是独立重算的**站点等权**，两者不混用。所有距离单位m。

|域|方法|类型|站点宏MAE|站点宏ES|站点P90|站点P95|楼宇楼层准确率(扫描%)|
|---|---|---|---:|---:|---:|---:|---:|
|TUJI|Raw|原基线骨干|2.315522|1.687391|3.659558|3.911735|100.000|
|TUJI|T=.6|固定骨干/固定温度|1.973313|1.474512|2.790162|3.029442|100.000|
|TUJI|T_CAL|固定骨干/CAL 1参数|1.965860|1.430057|2.811753|2.977649|100.000|
|TUJI|G13_hist|固定骨干/CAL原代理2系数|1.877677|1.362875|2.735890|2.964256|100.000|
|TUJI|G13_NLL|固定骨干/CAL连续NLL2系数|1.835707|1.335541|2.708012|2.896191|100.000|
|TUJI|Unified_frozen|固定骨干/CAL门控|1.773260|1.280926|2.609368|2.755892|100.000|
|TUJI|Joint_no_ev|联合骨干/推理关证据|1.938391|1.409494|2.919898|3.015121|100.000|
|TUJI|Joint|FIT真实联合训练|1.737678|1.258624|2.578335|2.774739|100.000|
|UTS|Raw|原基线骨干|4.598887|3.554238|9.280729|10.918188|99.319|
|UTS|T=.6|固定骨干/固定温度|4.663843|4.021636|9.310876|11.286570|97.730|
|UTS|T_CAL|固定骨干/CAL 1参数|5.017315|3.676980|9.900229|11.184461|99.659|
|UTS|G13_hist|固定骨干/CAL原代理2系数|4.566318|4.043317|9.117913|11.258201|99.319|
|UTS|G13_NLL|固定骨干/CAL连续NLL2系数|4.663843|4.021636|9.310876|11.286570|97.730|
|UTS|Unified_frozen|固定骨干/CAL门控|4.663843|4.021636|9.310876|11.286570|97.730|
|UTS|Joint_no_ev|联合骨干/推理关证据|4.887938|4.137782|9.516077|10.962014|86.720|
|UTS|Joint|FIT真实联合训练|4.831482|4.092682|9.438212|10.737503|90.806|
|UJI|Raw|原基线骨干|6.759699|5.573776|11.345235|14.465719|96.383|
|UJI|T=.6|固定骨干/固定温度|7.259512|6.335494|12.945335|14.469185|96.277|
|UJI|T_CAL|固定骨干/CAL 1参数|6.979737|5.236836|11.809805|13.768648|96.277|
|UJI|G13_hist|固定骨干/CAL原代理2系数|7.259512|6.335494|12.945335|14.469185|96.277|
|UJI|G13_NLL|固定骨干/CAL连续NLL2系数|7.259512|6.335494|12.945335|14.469185|96.277|
|UJI|Unified_frozen|固定骨干/CAL门控|7.259512|6.335494|12.945335|14.469185|96.277|
|UJI|Joint_no_ev|联合骨干/推理关证据|6.843320|5.582931|11.966121|13.981235|56.489|
|UJI|Joint|FIT真实联合训练|6.843320|5.582931|11.966121|13.981235|56.489|

## 概率/候选诊断（均为事后真值辅助）

点预测对所有建筑/楼层的连续分量求混合均值，类别准确率由建筑-楼层概率边缘的argmax另算；都未用真楼层筛选。TUJI canonical数据只有一个建筑/楼层类别，100%类别准确率在该域是平凡结果。

正确邻域=同建筑同楼层且锚点XY距离≤5m。所有策略全FIT支持，所以覆盖不会因概率修正而改变。Top1/Top10是概率排序后的邻域命中；温度正变换不改变候选排序，三种温度控制逐位置Top1/Top10已验证完全相同。

|域|方法|站点joint准确率%|近邻覆盖%|近邻质量|Top1近邻%|Top10近邻%|熵(nats)|最近连续中心oracle(m)|真楼层概率|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|TUJI|Raw|100.000|100.000|0.653427|87.582|99.560|4.806990|0.537504|1.000000|
|TUJI|T=.6|100.000|100.000|0.792756|87.582|99.560|3.692030|0.537504|1.000000|
|TUJI|T_CAL|100.000|100.000|0.816764|87.582|99.560|3.365712|0.537504|1.000000|
|TUJI|G13_hist|100.000|100.000|0.826746|90.623|100.000|3.495096|0.537504|1.000000|
|TUJI|G13_NLL|100.000|100.000|0.841791|91.502|100.000|3.354546|0.537504|1.000000|
|TUJI|Unified_frozen|100.000|100.000|0.861058|92.161|100.000|3.195068|0.537504|1.000000|
|TUJI|Joint_no_ev|100.000|100.000|0.795881|86.630|99.341|3.654063|0.278545|1.000000|
|TUJI|Joint|100.000|100.000|0.857720|89.890|99.780|3.223035|0.278545|1.000000|
|UTS|Raw|99.078|97.826|0.421644|68.940|89.372|3.059285|0.650580|0.605403|
|UTS|T=.6|97.343|97.826|0.599975|68.940|89.372|1.333572|0.650580|0.834735|
|UTS|T_CAL|99.275|97.826|0.350746|68.940|89.372|3.684549|0.650580|0.520435|
|UTS|G13_hist|99.078|97.826|0.645159|71.223|90.097|1.117841|0.650580|0.883239|
|UTS|G13_NLL|97.343|97.826|0.599975|68.940|89.372|1.333572|0.650580|0.834735|
|UTS|Unified_frozen|97.343|97.826|0.599975|68.940|89.372|1.333572|0.650580|0.834735|
|UTS|Joint_no_ev|88.705|97.826|0.523740|61.716|85.483|1.595743|0.666087|0.716974|
|UTS|Joint|91.516|97.826|0.553027|61.716|86.751|1.463861|0.666087|0.755659|
|UJI|Raw|96.048|64.368|0.272998|36.130|60.299|1.692936|1.136918|0.800668|
|UJI|T=.6|95.911|64.368|0.332257|36.130|60.299|0.856637|1.136918|0.910915|
|UJI|T_CAL|95.866|64.368|0.231380|36.130|60.299|2.254879|1.136918|0.716940|
|UJI|G13_hist|95.911|64.368|0.332257|36.130|60.299|0.856637|1.136918|0.910915|
|UJI|G13_NLL|95.911|64.368|0.332257|36.130|60.299|0.856637|1.136918|0.910915|
|UJI|Unified_frozen|95.911|64.368|0.332257|36.130|60.299|0.856637|1.136918|0.910915|
|UJI|Joint_no_ev|59.105|64.368|0.203774|23.147|53.065|1.352179|1.149154|0.529748|
|UJI|Joint|59.105|64.368|0.203774|23.147|53.065|1.352179|1.149154|0.529748|

最近连续中心误差是用真实XY在实际Student-t分量中心中取最小，代码不要求该分量所属楼层正确。这是不可部署的oracle诊断，既不是任意凸混合点决策误差下界，也不证明概率已给正确空间模式足够质量。邻域质量、覆盖、Top-k、真实楼层概率均不能作为部署门控输入。

## 逐位置配对和伤害

`PAIRED_SITE_ANALYSIS.json`保存每域全部28对策略的同序site keys、逐位置MAE/ES差值、改善/恶化数、最大伤害与改善。这里没有新增p值或选择晋级门槛，均为描述性结果。差值=候选−参照，负数改善。

|域|候选−参照|宏MAE差|宏ES差|改善/恶化/相同站点|
|---|---|---:|---:|---:|
|TUJI|G13_hist−T_CAL|-0.088183|-0.067182|68/23/0|
|TUJI|G13_NLL−T_CAL|-0.130153|-0.094516|71/20/0|
|TUJI|Unified_frozen−G13_NLL|-0.062448|-0.054615|66/25/0|
|TUJI|Joint−T_CAL|-0.228182|-0.171433|65/26/0|
|TUJI|Joint−Unified_frozen|-0.035581|-0.022302|47/44/0|
|TUJI|Joint−Joint_no_ev|-0.200712|-0.150870|65/26/0|
|UTS|G13_hist−T_CAL|-0.450997|+0.366336|79/59/0|
|UTS|G13_NLL−T_CAL|-0.353472|+0.344656|77/61/0|
|UTS|Unified_frozen−G13_NLL|+0.000000|+0.000000|0/0/138|
|UTS|Joint−T_CAL|-0.185833|+0.415701|76/62/0|
|UTS|Joint−Unified_frozen|+0.167638|+0.071045|72/66/0|
|UTS|Joint−Joint_no_ev|-0.056456|-0.045100|81/57/0|
|UJI|G13_hist−T_CAL|+0.279776|+1.098657|34/53/0|
|UJI|G13_NLL−T_CAL|+0.279776|+1.098657|34/53/0|
|UJI|Unified_frozen−G13_NLL|+0.000000|+0.000000|0/0/87|
|UJI|Joint−T_CAL|-0.136417|+0.346094|44/43/0|
|UJI|Joint−Unified_frozen|-0.416193|-0.752563|51/36/0|
|UJI|Joint−Joint_no_ev|+0.000000|+0.000000|0/0/87|

主要新候选Joint相对冻结Unified_frozen的最大伤害（每域保留前三，不只报告最好位置）：

|域|物理站点|参照MAE|Joint MAE|MAE差|ES差|
|---|---|---:|---:|---:|---:|
|TUJI|`[0, 0, 8.942, 29.214]`|2.094429|2.774154|+0.679725|+0.327301|
|TUJI|`[0, 0, 8.346, 0.597]`|2.049744|2.646151|+0.596407|+0.426512|
|TUJI|`[0, 0, 14.898, 22.657]`|1.092704|1.659468|+0.566764|+0.323057|
|UTS|`[0, 10, 76.7581, 21.9741]`|3.365505|10.805808|+7.440303|+4.801657|
|UTS|`[0, 8, 76.3467, 12.0527]`|5.514354|10.725450|+5.211095|+2.056762|
|UTS|`[0, 14, 20.6914, 24.1952]`|5.627345|10.629171|+5.001826|-2.485888|
|UJI|`[2, 0, -7351.981, 4864843.5]`|13.478556|18.063280|+4.584724|+3.234860|
|UJI|`[1, 1, -7560.3774, 4864859.5]`|10.027673|13.852818|+3.825145|+1.496470|
|UJI|`[1, 2, -7496.9473, 4864853.0]`|7.016920|9.184776|+2.167857|+0.790419|

## 第一性原理机制判断

### TUJI

**候选支持与排序**：全FIT同层5m邻域的站点宏覆盖为100.000%，Raw Top1/Top10命中87.582%/99.560%，邻域质量0.653427。因本轮支持=all，不能归咎Top192排除；覆盖不足指真实FIT几何不足，而不是连续模型必然无法定位。Raw最近连续中心oracle为0.537504m，部署均值MAE为2.315522m；两者差距描述概率/点决策与可用中心之间的空间错配，不是因果识别。

Top10−Top1相差11.978个百分点，表示这部分扫描的5m近邻在Top10但没有成为Top1（先站点内平均）。这是有可用近邻仍排序不优的直接描述，不能等同于所有混合均值误差由Top1引起。

**固定骨干观测的增量**：T_CAL MAE/ES=1.965860/1.430057，G13_hist=1.877678/1.362875，G13_NLL=1.835707/1.335541，Unified_frozen=1.773260/1.280926。相同连续分量下的变化只能来自概率动作；但G13统计和神经网络使用同一RSS，不能称独立新传感信息。CAL目标和自由参数量不同，无法只凭这组结果证明门控机制或真实隐含anchor概率校准。

**联合学习与连续分量**：Joint MAE/ES=1.737678/1.258624，P95=2.774739；同骨干关闭证据后=1.938391/1.409494。Joint最近连续中心oracle=0.278545m，相比固定base=0.537504m，骨干/头变化与概率变化并存；不据此宣称连续Student-t分量足够或不足已经得到唯一证实。

Joint相对Unified_frozen：MAE和ES均下降；47个位置改善、44个位置恶化。即使宏平均有利，也必须结合上表伤害和尾部，不视为所有位置可靠性改善。

尾部变化（Joint−Unified_frozen）：P95 +0.018847m，P99 -0.099219m，最坏站点 -0.183525m。正值就是对应尾部恶化，不能因平均改善予以豁免。

### UTS

**候选支持与排序**：全FIT同层5m邻域的站点宏覆盖为97.826%，Raw Top1/Top10命中68.940%/89.372%，邻域质量0.421644。因本轮支持=all，不能归咎Top192排除；覆盖不足指真实FIT几何不足，而不是连续模型必然无法定位。Raw最近连续中心oracle为0.650580m，部署均值MAE为4.598887m；两者差距描述概率/点决策与可用中心之间的空间错配，不是因果识别。

Top10−Top1相差20.431个百分点，表示这部分扫描的5m近邻在Top10但没有成为Top1（先站点内平均）。这是有可用近邻仍排序不优的直接描述，不能等同于所有混合均值误差由Top1引起。

**固定骨干观测的增量**：T_CAL MAE/ES=5.017315/3.676980，G13_hist=4.566318/4.043317，G13_NLL=4.663843/4.021636，Unified_frozen=4.663843/4.021636。相同连续分量下的变化只能来自概率动作；但G13统计和神经网络使用同一RSS，不能称独立新传感信息。CAL目标和自由参数量不同，无法只凭这组结果证明门控机制或真实隐含anchor概率校准。

**联合学习与连续分量**：Joint MAE/ES=4.831482/4.092682，P95=10.737503；同骨干关闭证据后=4.887938/4.137782。Joint最近连续中心oracle=0.666087m，相比固定base=0.650580m，骨干/头变化与概率变化并存；不据此宣称连续Student-t分量足够或不足已经得到唯一证实。

Joint相对Unified_frozen：MAE未改善，不能将其称为超越冻结可靠性校准；72个位置改善、66个位置恶化。即使宏平均有利，也必须结合上表伤害和尾部，不视为所有位置可靠性改善。

尾部变化（Joint−Unified_frozen）：P95 -0.549066m，P99 -2.452464m，最坏站点 -2.032089m。正值就是对应尾部恶化，不能因平均改善予以豁免。

**冻结门控的结构性退化**：CAL-NLL将两个全局theta都拟合为0。因为eta=theta×(1+.75 tanh(gate))且frozen阶段theta固定，任意门控都无法产生证据修正，其损失对gate的证据梯度为0。G13_NLL、Unified_frozen、T=.6的全部保存指标精确相同。不能把该对照当作一个实际生效的动态门控，也不能由此证明AP观测特征没有信息。参数为0是本次预设优化/约束的合法结果，本轮不通过重置系数或改目标修复科学负结果。

相对Raw，Joint宏MAE差+0.232594m、ES差+0.538444m；扫描加权楼宇楼层准确率差-8.513个百分点。存在不利指标，不能称为无代价提高。

### UJI

**候选支持与排序**：全FIT同层5m邻域的站点宏覆盖为64.368%，Raw Top1/Top10命中36.130%/60.299%，邻域质量0.272998。因本轮支持=all，不能归咎Top192排除；覆盖不足指真实FIT几何不足，而不是连续模型必然无法定位。Raw最近连续中心oracle为1.136918m，部署均值MAE为6.759699m；两者差距描述概率/点决策与可用中心之间的空间错配，不是因果识别。

Top10−Top1相差24.169个百分点，表示这部分扫描的5m近邻在Top10但没有成为Top1（先站点内平均）。这是有可用近邻仍排序不优的直接描述，不能等同于所有混合均值误差由Top1引起。

**固定骨干观测的增量**：T_CAL MAE/ES=6.979737/5.236837，G13_hist=7.259512/6.335494，G13_NLL=7.259512/6.335494，Unified_frozen=7.259512/6.335494。相同连续分量下的变化只能来自概率动作；但G13统计和神经网络使用同一RSS，不能称独立新传感信息。CAL目标和自由参数量不同，无法只凭这组结果证明门控机制或真实隐含anchor概率校准。

**联合学习与连续分量**：Joint MAE/ES=6.843319/5.582931，P95=13.981234；同骨干关闭证据后=6.843319/5.582931。Joint最近连续中心oracle=1.149154m，相比固定base=1.136918m，骨干/头变化与概率变化并存；不据此宣称连续Student-t分量足够或不足已经得到唯一证实。

Joint相对Unified_frozen：MAE和ES均下降；51个位置改善、36个位置恶化。即使宏平均有利，也必须结合上表伤害和尾部，不视为所有位置可靠性改善。

尾部变化（Joint−Unified_frozen）：P95 -0.487951m，P99 -3.861919m，最坏站点 -8.108948m。正值就是对应尾部恶化，不能因平均改善予以豁免。

**冻结门控的结构性退化**：CAL-NLL将两个全局theta都拟合为0。因为eta=theta×(1+.75 tanh(gate))且frozen阶段theta固定，任意门控都无法产生证据修正，其损失对gate的证据梯度为0。G13_NLL、Unified_frozen、T=.6的全部保存指标精确相同。不能把该对照当作一个实际生效的动态门控，也不能由此证明AP观测特征没有信息。参数为0是本次预设优化/约束的合法结果，本轮不通过重置系数或改目标修复科学负结果。

**联合后验也退化为无观测证据**：最终FIT联合训练theta=(0,0)，即使gate权重本身非零，也被全局零系数完全关闭。Joint与Joint_no_ev的每项TEST指标精确相同。因此该域的任何骨干变化都不能称为最终观测证据改善；完整参数已另存只读提取记录。

相对Raw，Joint宏MAE差+0.083620m、ES差+0.009156m；扫描加权楼宇楼层准确率差-39.894个百分点。存在不利指标，不能称为无代价提高。

**本轮能与不能识别的内容**：

- 全支持控制和同分量概率修正可检验排序/分配是否有可利用空间，但不能把低oracle中心误差直接变为可部署性能，也不能由宏平均将少数严重错误掩盖。
- 纯温度保持Top1/Top10顺序，MAE/ES变化是质量集中程度与混合均值的变化；G13改变顺序才涉及额外FIT统计证据的重排。观测增益不等于新增独立信息。
- Joint同时更新表示、检索与连续分量，缺少独立训练的no-evidence骨干、固定联合中心的额外反事实，无法唯一分离表示改善、中心改善与观测门控贡献。本轮不补做这些新模型。
- 联合阶段的mixture_loss只监督二维XY，没有继续使用base training_loss中的建筑/楼层分类项；同XY跨楼层模式的权重可能在XY目标下不可辨识。UTS类别准确率下降、UJI从96.383%降至56.489%与该目标缺少楼层约束一致，但没有单因素对照，不能声称已证明唯一原因。二维MAE接近不代表楼层可靠性保持。
- 位置均值用于二维平方风险更自然，非欧氏MAE的理论最优动作；所有方法共享同类混合均值动作，未在TEST挑选中位数等解码器。
- ES为df4独立坐标Student-t连续混合的MC近似，种子由seed+900+batch_start固定，两流各32次，方法间共用随机数组；小ES差异没有MC误差条，不能当独立显著性。G13_hist训练的离散代理ES不是最终连续ES。
- 未证明真实隐含锚点概率校准；单seed、每域不足140个TEST站点、随机站点而非空间块隔离、历史语料暴露和扫描相关性都限制结论。

不启动新模型、集成扩员、温度/门控搜索或任何官方外部测试。后续若考虑进入核心机制，应预注册独立位置/采集与公平训练预算，在新未观察评估中直接对照T_CAL及固定骨干G13；本轮结果本身不授予部署或正式晋级。

就本轮实测而言，UTS已经构成Joint相对Raw宏MAE、ES及类别准确率恶化的反例，所以不能把单成员统一联合后验描述为三域一致可靠的替代。固定骨干CAL校准、联合训练和最终点决策是不同干预，必须按完整表格分别判断；不据TEST数值回头更改任何一种方法。

如后续只选一个机制做独立验证，优先检验“联合微调时保留FIT建筑/楼层监督”这一单因素，其余后验、预算和组件固定；当前仅指出目标可辨识性问题，不在已观察TEST上补跑或挑权重。

## 只读核验与交付

|域|站点/扫描|JSON重汇总最大绝对差|checkpoint SHA256|
|---|---:|---:|---|
|TUJI|91/453|8.54e-07|`c9d727a5e90540ac852e4304bf04a3f6c51376a3beda598c89c04a5aeab3eac5`|
|UTS|138/881|6.18e-07|`a3125b6614740f40b9087aba611e1980b3665986482a973db48d7f347a0f1b07`|
|UJI|87/1880|7.7e-07|`0d28302141f3bc2e1b552517f4607c1fbb02df6e387e034945ea3137b81eb5c9`|

独立审计只读RESULT/SITE_DIAGNOSTICS/checkpoint，没有再次前向TEST：四角色与预检SHA匹配；八策略站点序列/扫描数相同；保存的站点平均重新计算宏MAE/ES、P90/P95及扫描加权诊断均匹配（浮点32/64汇总差≤2e-5m）。SITE文件没有逐扫描原始向量，因此没有声称独立从逐扫描重算。所有JSON可解析且无NaN/Inf。

同base六策略的oracle中心向量完全相同，同joint两策略也相同；源码确认每组内复用同一次提取的完整center/scale张量。joint checkpoint的encoder/anchor_keys/residual_head均实际变化，固定anchor几何/原型/AP mask保持不变。冻结头全局theta与G13_NLL相同。原126个历史科学模块及全部输入文件在最终完整性核验中与冻结前哈希比较。

没有强制所有CUDA内核确定性，固定seed不能保证跨机器/运行逐位相同。成功科学命令stderr是否为空及全部退出码以原日志为准；没有把回归演示失败或编译错误当成通过。耗时为本机命令墙钟，不是纯GPU活跃算力。

三域正式目录（均含RESULT.json、SITE_DIAGNOSTICS.json、CHECKPOINTS.pt和TEST_INVOCATION.lock）：

- TUJI：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_tuji_full_v1`；checkpoint：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_tuji_full_v1\CHECKPOINTS.pt`。
- UTS：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_uts_full_v1`；checkpoint：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_uts_full_v1\CHECKPOINTS.pt`。
- UJI：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_uji_full_v1`；checkpoint：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_uji_full_v1\CHECKPOINTS.pt`。

公共执行目录：`D:\liuyanlunwen\fhm_dapp_tim\evidence\msep_single_execution_v1`。包含METHOD_FREEZE、DATA_AUDIT、SMOKE/FULL_RESULT_AUDIT、PAIRED_SITE_ANALYSIS、FIX_RECORD、完整执行记录与日志。所有权重留本地，用户另行安排产物上传。
