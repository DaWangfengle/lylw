# V17 双阶段冻结校准实验实测报告

本轮为内部 CAL-eval 收尾比较：3 数据域 × 3 配对种子，18 条单成员训练轨迹，均完成 80 轮，共 1440 轮。没有 Joint 微调，未评估内部 TEST 或官方测试。

三种检查点是同一训练轨迹的选择敏感性比较；唯一预声明的 Stage 2 主检查点为 `best_dev_epoch_21_80`。以下均值 ± 样本标准差描述相同评价站点上三个种子的波动，不是三个独立数据集的显著性检验。

## 执行与数据边界

Python `D:\miniminiconda\envs\liuyan\python.exe`；3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]。Torch 2.10.0+cu128，CUDA 12.8，NVIDIA GeForce RTX 2060（6.00 GiB）；NumPy 2.4.4，SciPy 1.17.1，pandas 3.0.1，pytest 9.0.3。

指定编译与 13 项单测通过。全部命令及 stdout/stderr、退出码、墙钟时长、日志 SHA256 保存在 EXECUTION_RECORD.json 和 run_logs/。命令为指定 Python：compileall；两个指定 pytest 文件；按 tuji/uts/uji、17301/17302/17303 串行执行 run（batch64、threads4、ES draws32）；最后 aggregate。

|域|FIT 站点/扫描|DEV|CAL-fit|CAL-eval|TEST（仅角色元数据）|FIT 锚点/AP|每骨干更新次数|
|---|---:|---:|---:|---:|---:|---:|---:|
|tuji|646/3236|91/453|45/215|46/234|91/457|646/120|4080|
|uts|1052/6534|138/873|69/444|69/414|138/843|1052/557|8240|
|uji|672/14453|87/1828|43/959|44/919|87/1778|672/332|18080|

正式 9 次作业累计墙钟 111.14 分钟；审计前已完成命令累计 111.94 分钟。每个 paired job 包含两次完整训练及校准/评分，未分别计时单个骨干 GPU 核函数。

物理站点所有扫描角色不交叉；原坐标与居中 float32 四位分组一一对应。G13 原型/检出率/条件均值方差、AP seen mask 只由 FIT 得到。TUJI 五个 MAT 容器被打开和哈希，但仅请求 results_train，results_test 未反序列化；UTS/UJI 只读 training CSV。

## Stage 1：完整训练与检查点敏感性

两方案在每个 dataset/seed 从相同初始化复制，重置相同训练与 DataLoader 种子，AdamW/损失/增强/扫描/批量/80 轮预算相同，唯一训练改动为邻居标签限制建筑和楼层。排除 1–20 轮仅影响主检查点选择，没有增加学习率 warmup。旧规则实现为 score < best−0.01、epoch≥20 且 epoch−best_epoch≥12，模拟停止不影响实际 80 轮执行。

|域|方案|旧选中轮/模拟停止轮（三种子）|全80最优轮|21–80最优轮|
|---|---|---|---|---|
|tuji|raw_neighbors|78/None; 79/None; 80/None|80,79,80|80,79,80|
|tuji|same_floor_neighbors|76/None; 80/None; 80/None|76,80,80|76,80,80|
|uts|raw_neighbors|2/20; 80/None; 1/20|79,80,79|79,80,79|
|uts|same_floor_neighbors|80/None; 79/None; 1/20|80,79,79|80,79,79|
|uji|raw_neighbors|32/44; 39/51; 53/65|57,39,53|57,39,53|
|uji|same_floor_neighbors|38/50; 40/52; 42/54|38,40,42|38,40,42|

各轮 FIT 总损失、全部 training_loss 分量和 DEV 五项指标完整保存在每个 EPOCH_HISTORY.csv/RESULT.json；没有按 CAL-eval 选 checkpoint。

|域|方案|检查点|CAL-eval MAE(m)|连续 ES(m)|BF准确率|预测BF条件MAE(m)|联合NLL|
|---|---|---|---:|---:|---:|---:|---:|
|tuji|raw_neighbors|best_dev_epoch_1_80|2.3137 ± 0.0907|1.7159 ± 0.1059|1.0000 ± 0.0000|2.3137 ± 0.0907|4.4253 ± 0.1559|
|tuji|raw_neighbors|best_dev_epoch_21_80|2.3137 ± 0.0907|1.7159 ± 0.1059|1.0000 ± 0.0000|2.3137 ± 0.0907|4.4253 ± 0.1559|
|tuji|raw_neighbors|legacy_early_stop|2.2934 ± 0.0904|1.6878 ± 0.0941|1.0000 ± 0.0000|2.2934 ± 0.0904|4.4726 ± 0.0880|
|tuji|same_floor_neighbors|best_dev_epoch_1_80|2.2920 ± 0.0293|1.6589 ± 0.0478|1.0000 ± 0.0000|2.2920 ± 0.0293|4.4495 ± 0.0448|
|tuji|same_floor_neighbors|best_dev_epoch_21_80|2.2920 ± 0.0293|1.6589 ± 0.0478|1.0000 ± 0.0000|2.2920 ± 0.0293|4.4495 ± 0.0448|
|tuji|same_floor_neighbors|legacy_early_stop|2.2920 ± 0.0293|1.6589 ± 0.0478|1.0000 ± 0.0000|2.2920 ± 0.0293|4.4495 ± 0.0448|
|uji|raw_neighbors|best_dev_epoch_1_80|7.9642 ± 0.3421|6.3890 ± 0.3083|0.9416 ± 0.0027|7.9486 ± 0.1947|14.8211 ± 0.0495|
|uji|raw_neighbors|best_dev_epoch_21_80|7.9642 ± 0.3421|6.3890 ± 0.3083|0.9416 ± 0.0027|7.9486 ± 0.1947|14.8211 ± 0.0495|
|uji|raw_neighbors|legacy_early_stop|7.7581 ± 0.2189|6.1744 ± 0.2342|0.9450 ± 0.0083|7.7828 ± 0.2229|14.7261 ± 0.1757|
|uji|same_floor_neighbors|best_dev_epoch_1_80|7.6181 ± 0.0556|6.2091 ± 0.1251|0.9627 ± 0.0048|7.8294 ± 0.1179|14.4353 ± 0.0301|
|uji|same_floor_neighbors|best_dev_epoch_21_80|7.6181 ± 0.0556|6.2091 ± 0.1251|0.9627 ± 0.0048|7.8294 ± 0.1179|14.4353 ± 0.0301|
|uji|same_floor_neighbors|legacy_early_stop|7.6181 ± 0.0556|6.2091 ± 0.1251|0.9627 ± 0.0048|7.8294 ± 0.1179|14.4353 ± 0.0301|
|uts|raw_neighbors|best_dev_epoch_1_80|4.9055 ± 0.1368|3.6213 ± 0.1557|0.9327 ± 0.0333|4.5761 ± 0.1656|11.6729 ± 0.1069|
|uts|raw_neighbors|best_dev_epoch_21_80|4.9055 ± 0.1368|3.6213 ± 0.1557|0.9327 ± 0.0333|4.5761 ± 0.1656|11.6729 ± 0.1069|
|uts|raw_neighbors|legacy_early_stop|8.1898 ± 2.9250|5.4589 ± 1.7974|0.9334 ± 0.0329|6.6584 ± 1.9565|10.1026 ± 2.2619|
|uts|same_floor_neighbors|best_dev_epoch_1_80|4.4869 ± 0.3154|3.3906 ± 0.2489|0.9912 ± 0.0077|4.4440 ± 0.2449|10.8866 ± 0.0879|
|uts|same_floor_neighbors|best_dev_epoch_21_80|4.4869 ± 0.3154|3.3906 ± 0.2489|0.9912 ± 0.0077|4.4440 ± 0.2449|10.8866 ± 0.0879|
|uts|same_floor_neighbors|legacy_early_stop|6.1654 ± 2.7202|4.3405 ± 1.5000|0.9903 ± 0.0084|5.5452 ± 1.8284|9.7601 ± 2.0163|

同楼层减原邻居的配对差（负 MAE/ES 为改善，正 BF 为改善）：

|域|检查点|ΔMAE(m)|ΔES(m)|ΔBF准确率|MAE改善种子数|
|---|---|---:|---:|---:|---:|
|tuji|legacy_early_stop|-0.0014 ± 0.0631|-0.0290 ± 0.0506|0.0000 ± 0.0000|1/3|
|tuji|best_dev_epoch_1_80|-0.0216 ± 0.0615|-0.0571 ± 0.0594|0.0000 ± 0.0000|2/3|
|tuji|best_dev_epoch_21_80|-0.0216 ± 0.0615|-0.0571 ± 0.0594|0.0000 ± 0.0000|2/3|
|uts|legacy_early_stop|-2.0244 ± 3.0493|-1.1184 ± 1.8233|0.0570 ± 0.0385|3/3|
|uts|best_dev_epoch_1_80|-0.4186 ± 0.1806|-0.2307 ± 0.2639|0.0585 ± 0.0373|3/3|
|uts|best_dev_epoch_21_80|-0.4186 ± 0.1806|-0.2307 ± 0.2639|0.0585 ± 0.0373|3/3|
|uji|legacy_early_stop|-0.1400 ± 0.2488|0.0347 ± 0.1825|0.0178 ± 0.0131|2/3|
|uji|best_dev_epoch_1_80|-0.3461 ± 0.3024|-0.1798 ± 0.2174|0.0211 ± 0.0073|3/3|
|uji|best_dev_epoch_21_80|-0.3461 ± 0.3024|-0.1798 ± 0.2174|0.0211 ± 0.0073|3/3|

## Stage 2：冻结主检查点后的四方案

每个骨干共用同一 CAL-fit/CAL-eval pack，候选 ID、连续分量中心/尺度均不变；只有 T 和两项 G13 系数改变。校准只用 CAL-fit，连续 Student-t NLL + 0.30 二维点风险，站点等权，Adam lr0.035、110 次更新；T∈[.25,3]、theta∈[0,1.5]。所有 ES 使用相同 seed+900、两流各32次抽样；比较跨种子波动包含模型和 MC 随机流差异。

|域|方案|校准|MAE(m)|ES(m)|BF准确率|预测BF条件MAE(m)|联合NLL|
|---|---|---|---:|---:|---:|---:|---:|
|tuji|raw_neighbors|g13_fixed_t06|1.9824 ± 0.0435|1.4120 ± 0.0359|1.0000 ± 0.0000|1.9824 ± 0.0435|4.0456 ± 0.0857|
|tuji|raw_neighbors|raw|2.3137 ± 0.0907|1.7159 ± 0.1059|1.0000 ± 0.0000|2.3137 ± 0.0907|4.4253 ± 0.1559|
|tuji|raw_neighbors|temperature_only|2.0591 ± 0.0466|1.4895 ± 0.0189|1.0000 ± 0.0000|2.0591 ± 0.0466|4.2010 ± 0.0977|
|tuji|raw_neighbors|temperature_plus_g13|1.9818 ± 0.0487|1.4161 ± 0.0168|1.0000 ± 0.0000|1.9818 ± 0.0487|4.0303 ± 0.1022|
|tuji|same_floor_neighbors|g13_fixed_t06|1.9714 ± 0.0295|1.3805 ± 0.0222|1.0000 ± 0.0000|1.9714 ± 0.0295|4.0580 ± 0.0615|
|tuji|same_floor_neighbors|raw|2.2920 ± 0.0293|1.6589 ± 0.0478|1.0000 ± 0.0000|2.2920 ± 0.0293|4.4495 ± 0.0448|
|tuji|same_floor_neighbors|temperature_only|2.0428 ± 0.0209|1.4576 ± 0.0205|1.0000 ± 0.0000|2.0428 ± 0.0209|4.2196 ± 0.0623|
|tuji|same_floor_neighbors|temperature_plus_g13|1.9758 ± 0.0291|1.3903 ± 0.0169|1.0000 ± 0.0000|1.9758 ± 0.0291|4.0507 ± 0.0601|
|uji|raw_neighbors|g13_fixed_t06|8.0480 ± 0.3858|7.0180 ± 0.3664|0.9403 ± 0.0011|8.1409 ± 0.2740|15.2159 ± 0.0488|
|uji|raw_neighbors|raw|7.9642 ± 0.3421|6.3890 ± 0.3083|0.9416 ± 0.0027|7.9486 ± 0.1947|14.8211 ± 0.0495|
|uji|raw_neighbors|temperature_only|8.4384 ± 0.1983|6.2898 ± 0.5107|0.9397 ± 0.0016|7.8998 ± 0.2242|14.8033 ± 0.0498|
|uji|raw_neighbors|temperature_plus_g13|8.3151 ± 0.1354|6.0799 ± 0.3878|0.9473 ± 0.0031|7.8366 ± 0.2499|14.6778 ± 0.0330|
|uji|same_floor_neighbors|g13_fixed_t06|7.9242 ± 0.1394|6.8643 ± 0.1396|0.9615 ± 0.0056|8.0430 ± 0.1219|14.8478 ± 0.0533|
|uji|same_floor_neighbors|raw|7.6181 ± 0.0556|6.2091 ± 0.1251|0.9627 ± 0.0048|7.8294 ± 0.1179|14.4353 ± 0.0301|
|uji|same_floor_neighbors|temperature_only|8.7222 ± 0.3247|6.1938 ± 0.2488|0.9643 ± 0.0065|7.8042 ± 0.1885|14.3979 ± 0.0233|
|uji|same_floor_neighbors|temperature_plus_g13|8.8516 ± 0.1406|6.2482 ± 0.2077|0.9652 ± 0.0062|7.9373 ± 0.2354|14.3895 ± 0.0196|
|uts|raw_neighbors|g13_fixed_t06|4.4922 ± 0.1032|3.7737 ± 0.0646|0.9279 ± 0.0250|4.5197 ± 0.2330|11.6495 ± 0.1698|
|uts|raw_neighbors|raw|4.9055 ± 0.1368|3.6213 ± 0.1557|0.9327 ± 0.0333|4.5761 ± 0.1656|11.6729 ± 0.1069|
|uts|raw_neighbors|temperature_only|5.4561 ± 0.1795|3.8193 ± 0.1822|0.9366 ± 0.0320|4.7101 ± 0.1481|11.8186 ± 0.0914|
|uts|raw_neighbors|temperature_plus_g13|5.4549 ± 0.1800|3.7907 ± 0.1577|0.9366 ± 0.0320|4.7098 ± 0.1478|11.8183 ± 0.0913|
|uts|same_floor_neighbors|g13_fixed_t06|4.2056 ± 0.2108|3.4774 ± 0.0873|0.9912 ± 0.0077|4.2540 ± 0.1893|10.8998 ± 0.1376|
|uts|same_floor_neighbors|raw|4.4869 ± 0.3154|3.3906 ± 0.2489|0.9912 ± 0.0077|4.4440 ± 0.2449|10.8866 ± 0.0879|
|uts|same_floor_neighbors|temperature_only|5.3938 ± 0.2692|3.6338 ± 0.2675|0.9912 ± 0.0077|4.7455 ± 0.2701|11.0527 ± 0.0580|
|uts|same_floor_neighbors|temperature_plus_g13|5.1153 ± 0.1921|3.4352 ± 0.2665|0.9952 ± 0.0084|4.3423 ± 0.2192|10.9814 ± 0.0465|

校准参数均来自 CAL-fit，不能从下表或 CAL-eval 再选一个方案作为部署结论：

|域/种子|骨干|校准|T|theta visibility|theta strength|
|---|---|---|---:|---:|---:|
|tuji/17301|raw_neighbors|temperature_only|0.611003|0.000000|0.000000|
|tuji/17301|raw_neighbors|g13_fixed_t06|0.600000|1.133952|1.500000|
|tuji/17301|raw_neighbors|temperature_plus_g13|0.732851|1.500000|1.500000|
|tuji/17301|same_floor_neighbors|temperature_only|0.560463|0.000000|0.000000|
|tuji/17301|same_floor_neighbors|g13_fixed_t06|0.600000|1.212460|1.500000|
|tuji/17301|same_floor_neighbors|temperature_plus_g13|0.664058|1.500000|1.500000|
|tuji/17302|raw_neighbors|temperature_only|0.484181|0.000000|0.000000|
|tuji/17302|raw_neighbors|g13_fixed_t06|0.600000|1.500000|1.500000|
|tuji/17302|raw_neighbors|temperature_plus_g13|0.565759|1.436285|1.500000|
|tuji/17302|same_floor_neighbors|temperature_only|0.548419|0.000000|0.000000|
|tuji/17302|same_floor_neighbors|g13_fixed_t06|0.600000|0.539023|1.500000|
|tuji/17302|same_floor_neighbors|temperature_plus_g13|0.647892|1.011961|1.500000|
|tuji/17303|raw_neighbors|temperature_only|0.549851|0.000000|0.000000|
|tuji/17303|raw_neighbors|g13_fixed_t06|0.600000|1.364551|1.500000|
|tuji/17303|raw_neighbors|temperature_plus_g13|0.654186|1.500000|1.500000|
|tuji/17303|same_floor_neighbors|temperature_only|0.533267|0.000000|0.000000|
|tuji/17303|same_floor_neighbors|g13_fixed_t06|0.600000|1.500000|1.500000|
|tuji/17303|same_floor_neighbors|temperature_plus_g13|0.626277|1.500000|1.500000|
|uts/17301|raw_neighbors|temperature_only|1.146873|0.000000|0.000000|
|uts/17301|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17301|raw_neighbors|temperature_plus_g13|1.146364|0.000000|0.000000|
|uts/17301|same_floor_neighbors|temperature_only|1.258638|0.000000|0.000000|
|uts/17301|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17301|same_floor_neighbors|temperature_plus_g13|1.657428|1.178598|0.817887|
|uts/17302|raw_neighbors|temperature_only|1.204996|0.000000|0.000000|
|uts/17302|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17302|raw_neighbors|temperature_plus_g13|1.205046|0.000000|0.000000|
|uts/17302|same_floor_neighbors|temperature_only|1.221169|0.000000|0.000000|
|uts/17302|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17302|same_floor_neighbors|temperature_plus_g13|1.540001|1.210498|0.647705|
|uts/17303|raw_neighbors|temperature_only|1.129210|0.000000|0.000000|
|uts/17303|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17303|raw_neighbors|temperature_plus_g13|1.128845|0.000000|0.000000|
|uts/17303|same_floor_neighbors|temperature_only|1.315708|0.000000|0.000000|
|uts/17303|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uts/17303|same_floor_neighbors|temperature_plus_g13|1.469874|0.029162|0.624092|
|uji/17301|raw_neighbors|temperature_only|1.178603|0.000000|0.000000|
|uji/17301|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17301|raw_neighbors|temperature_plus_g13|1.441342|1.500000|0.000000|
|uji/17301|same_floor_neighbors|temperature_only|1.347362|0.000000|0.000000|
|uji/17301|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17301|same_floor_neighbors|temperature_plus_g13|1.672646|1.500000|0.000000|
|uji/17302|raw_neighbors|temperature_only|1.263717|0.000000|0.000000|
|uji/17302|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17302|raw_neighbors|temperature_plus_g13|1.514330|1.475086|0.000000|
|uji/17302|same_floor_neighbors|temperature_only|1.301691|0.000000|0.000000|
|uji/17302|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17302|same_floor_neighbors|temperature_plus_g13|1.618422|1.500000|0.000000|
|uji/17303|raw_neighbors|temperature_only|1.233187|0.000000|0.000000|
|uji/17303|raw_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17303|raw_neighbors|temperature_plus_g13|1.480488|1.500000|0.000000|
|uji/17303|same_floor_neighbors|temperature_only|1.315541|0.000000|0.000000|
|uji/17303|same_floor_neighbors|g13_fixed_t06|0.600000|0.000000|0.000000|
|uji/17303|same_floor_neighbors|temperature_plus_g13|1.620394|1.500000|0.000000|

## 温度之外的额外证据收益

下面固定比较 temperature_plus_g13 − temperature_only，避免把温度本身的收益归给 G13。没有据此替换预声明方案或重训。

|域|骨干|ΔMAE(m)|ΔES(m)|ΔBF准确率|MAE/ES同时严格降低的种子数|
|---|---|---:|---:|---:|---:|
|tuji|raw_neighbors|-0.0773 ± 0.0075|-0.0734 ± 0.0178|0.0000 ± 0.0000|3/3|
|tuji|same_floor_neighbors|-0.0670 ± 0.0115|-0.0673 ± 0.0083|0.0000 ± 0.0000|3/3|
|uts|raw_neighbors|-0.0012 ± 0.0013|-0.0287 ± 0.0279|0.0000 ± 0.0000|2/3|
|uts|same_floor_neighbors|-0.2785 ± 0.0810|-0.1986 ± 0.0132|0.0040 ± 0.0068|3/3|
|uji|raw_neighbors|-0.1233 ± 0.1741|-0.2099 ± 0.1600|0.0076 ± 0.0047|3/3|
|uji|same_floor_neighbors|0.1295 ± 0.1907|0.0544 ± 0.2066|0.0009 ± 0.0008|0/3|

## 尾部与逐站点伤害

以下是站点内扫描均值的 P95/P99/最坏站点误差；不是扫描误差分位数。害处计数以同骨干同主检查点的 raw 为参照，按 seed17301/17302/17303 顺序列出。原始全站点向量和配对差在 SITE_DIAGNOSTICS.json 与 SITE_DELTAS_ALL_RUNS.csv；不能作为推断门控。

|域|骨干|校准|P95(m)|P99(m)|最坏站点MAE(m)|较raw恶化站点数（三种子）|
|---|---|---|---:|---:|---:|---|
|tuji|raw_neighbors|raw|4.0035 ± 0.2708|4.6867 ± 0.1712|4.8633 ± 0.1211|0,0,0|
|tuji|raw_neighbors|temperature_only|3.1631 ± 0.0134|3.6680 ± 0.3166|3.9661 ± 0.5435|18,17,21|
|tuji|raw_neighbors|g13_fixed_t06|3.0764 ± 0.1623|3.4039 ± 0.1033|3.5193 ± 0.1925|19,13,20|
|tuji|raw_neighbors|temperature_plus_g13|3.0495 ± 0.0977|3.3821 ± 0.0950|3.5011 ± 0.1895|20,15,19|
|tuji|same_floor_neighbors|raw|3.9213 ± 0.1455|4.5435 ± 0.2481|4.6929 ± 0.1611|0,0,0|
|tuji|same_floor_neighbors|temperature_only|3.0658 ± 0.1010|3.5426 ± 0.4096|3.7710 ± 0.5884|17,17,19|
|tuji|same_floor_neighbors|g13_fixed_t06|2.9961 ± 0.1158|3.1481 ± 0.1866|3.2253 ± 0.2294|17,15,18|
|tuji|same_floor_neighbors|temperature_plus_g13|3.0142 ± 0.1466|3.1400 ± 0.1813|3.2019 ± 0.2183|17,15,18|
|uts|raw_neighbors|raw|10.0946 ± 0.1875|16.1548 ± 0.9547|19.0744 ± 1.6527|0,0,0|
|uts|raw_neighbors|temperature_only|11.4874 ± 1.3260|16.1804 ± 1.2379|19.4694 ± 3.2451|52,49,49|
|uts|raw_neighbors|g13_fixed_t06|10.5315 ± 0.9104|15.9179 ± 0.5191|18.2595 ± 1.8185|30,31,30|
|uts|raw_neighbors|temperature_plus_g13|11.4826 ± 1.3248|16.1793 ± 1.2369|19.4692 ± 3.2404|52,49,49|
|uts|same_floor_neighbors|raw|10.7276 ± 1.6065|16.6669 ± 3.1094|19.8729 ± 1.7266|0,0,0|
|uts|same_floor_neighbors|temperature_only|12.5251 ± 1.7834|19.0612 ± 2.6593|20.9822 ± 1.3492|55,52,54|
|uts|same_floor_neighbors|g13_fixed_t06|9.3021 ± 1.0403|16.9640 ± 1.4348|19.3552 ± 2.1878|27,26,30|
|uts|same_floor_neighbors|temperature_plus_g13|11.7152 ± 1.3732|18.5155 ± 1.4253|20.8758 ± 1.1125|45,41,48|
|uji|raw_neighbors|raw|16.6493 ± 0.9102|28.7954 ± 7.6418|36.4907 ± 12.3217|0,0,0|
|uji|raw_neighbors|temperature_only|18.0730 ± 1.8498|30.6331 ± 5.0924|38.3659 ± 9.4584|25,30,24|
|uji|raw_neighbors|g13_fixed_t06|16.9001 ± 1.4839|29.0428 ± 8.1546|36.4171 ± 12.7330|28,25,25|
|uji|raw_neighbors|temperature_plus_g13|17.2517 ± 2.3665|30.0775 ± 5.3000|37.0950 ± 10.3464|23,29,23|
|uji|same_floor_neighbors|raw|16.3426 ± 0.9013|19.4951 ± 1.0768|20.2571 ± 1.5779|0,0,0|
|uji|same_floor_neighbors|temperature_only|18.8299 ± 1.2716|23.9377 ± 2.7808|26.3321 ± 3.3641|28,26,33|
|uji|same_floor_neighbors|g13_fixed_t06|17.6751 ± 1.2891|20.5694 ± 1.9239|21.9066 ± 2.7846|23,31,26|
|uji|same_floor_neighbors|temperature_plus_g13|21.9419 ± 2.5691|25.0131 ± 1.4989|26.0492 ± 1.1869|24,30,33|

每个域和校准方法最严重的负迁移例子（跨两骨干/三种子，逐站点相对其 raw）：

|域|校准|骨干/种子|位置(B,F,X,Y)|站点ΔMAE(m)|
|---|---|---|---|---:|
|tuji|temperature_only|raw_neighbors/17303|[0, 0, 4.17, 5.366]|+1.203109|
|tuji|g13_fixed_t06|raw_neighbors/17303|[0, 0, 11.321, 10.134]|+2.180746|
|tuji|temperature_plus_g13|raw_neighbors/17303|[0, 0, 11.321, 10.134]|+2.122481|
|uts|temperature_only|same_floor_neighbors/17303|[0, 10, 15.6568, 22.8553]|+5.488109|
|uts|g13_fixed_t06|raw_neighbors/17303|[0, 12, 98.8942, 7.1231]|+4.131096|
|uts|temperature_plus_g13|same_floor_neighbors/17301|[0, 12, 18.4289, 13.9411]|+5.578021|
|uji|temperature_only|same_floor_neighbors/17301|[2, 2, -7313.0654, 4864792.0]|+11.544045|
|uji|g13_fixed_t06|same_floor_neighbors/17303|[2, 2, -7408.3389, 4864811.5]|+10.042359|
|uji|temperature_plus_g13|same_floor_neighbors/17301|[2, 2, -7313.0654, 4864792.0]|+16.549335|

## 实测判断与局限

- tuji：主检查点同楼层修复 ΔMAE -0.0216 ± 0.0615 m、ΔES -0.0571 ± 0.0594 m、ΔBF 0.0000 ± 0.0000；MAE 严格改善 2/3 个种子。不同指标的正负结果均保留，不能仅以分类准确率或平均 MAE 宣称全面成功。
- uts：主检查点同楼层修复 ΔMAE -0.4186 ± 0.1806 m、ΔES -0.2307 ± 0.2639 m、ΔBF 0.0585 ± 0.0373；MAE 严格改善 3/3 个种子。不同指标的正负结果均保留，不能仅以分类准确率或平均 MAE 宣称全面成功。
- uji：主检查点同楼层修复 ΔMAE -0.3461 ± 0.3024 m、ΔES -0.1798 ± 0.2174 m、ΔBF 0.0211 ± 0.0073；MAE 严格改善 3/3 个种子。不同指标的正负结果均保留，不能仅以分类准确率或平均 MAE 宣称全面成功。

TUJI 单建筑楼层的邻居 ID 完全相同，权重通过原程序 atol=rtol=1e-7 等价检查，但不是逐位相同：最大权重差 2.980232239e-08，2512/5168 项不同。配对轨迹逐轮 DEV MAE 最大绝对差（三种子）为 [0.3234727382659912, 0.31040382385253906, 1.285637617111206] m。没有强制全 CUDA 确定性，不能唯一分离浮点归一化与GPU随机执行的影响，不能把 TUJI 的差异归因于排除跨楼层邻居。

T=1 与 T=.6 是本轮四方案定义的一部分。g13_fixed_t06 与 raw 的差包含温度变化，不能全部归因于观测证据；temperature_plus_g13 与 temperature_only 才是本轮可报告的额外证据对照，但二者来自各自 CAL-fit 优化，不是识别物理因果。参数触边和种子差异须连同校准表解释，不能扩网格追结果。

所有骨干为单成员且同一划分下公平重训；不能直接与旧 E5/G13 的不同预算和划分绝对 MAE 作改进或显著性比较。三个种子共享 CAL-eval 站点，站点也可能有空间/设备/会话相关性。CAL-eval 来自此前已研究的训练语料内部位置，非独立外部盲测。连续 ES 是有限 MC 近似；BF-XY NLL 的概率对象与仅 XY NLL 不同，不能跨定义排优劣。

本轮未加入新网络、门控、Joint 微调或 TEST 选择。是否推进后续机制应另行预注册，不能用本轮 CAL-eval 再挑部署策略；当前收尾结果本身不构成正式外部泛化证明。

## 完整性、异常及文件

独立核验 PASS：9 次作业，18×80 轮，54 个检查点与9个共享参考 bundle，126 条 run-level、42 条聚合、81 条 seed 配对、5247 条逐站点配对记录。194 个源码/测试/指令文件及7个训练输入 SHA256 全部保持不变。检查点选择由独立程序回放；Stage1 主检查点 raw 与 Stage2 raw 逐站点 JSON 完全一致。

从保存站点向量复算总体指标的最大绝对差为 2.330139358e-06，来自原程序 float32 reduction 与序列化后 double 归并的舍入差；审计允许2e-5。这里没有更改任何科学门禁。源码未输出原始逐扫描误差，故不声称独立重建 scan→site；本次独立复核的是 saved site→macro/CSV/paired deltas。

随机种子固定，未额外开启原脚本未设定的全 CUDA 确定性内核。未新增学习率调度、改变 batch/epochs、重新选划分或改模型源代码。

结果审计/报告辅助脚本首次运行：数值交叉核验完成后，Markdown 生成时因 append_interpretation 漏传 execution 参数发生 NameError。补齐函数调用/签名的参数后，仅重跑只读文件审计和报告生成。9 次正式科学作业及 aggregate 均为 exit 0；没有重训或重新评分。原始 helper、stdout/stderr 与部分数值审计保留，修复详见 FIX_RECORD.md。

已完成科学命令 stderr 均为空；未发生训练/评分失败、重试、OOM 或科学代码修复。

源文件和输入逐项 SHA 在 BEFORE_RUN_AUDIT.json；各输出/54权重 SHA 和独立数值复核在 INTEGRITY_REVIEW.json；每作业 RUN_MANIFEST.json 绑定 RESULT/SITE_DIAGNOSTICS/REFERENCE_STATE 和核心源文件。模型 .pt 仅在本地保存。本轮未执行任何 Git 命令或上传。

核心科学命令及辅助编译记录（全部命令含失败见 EXECUTION_RECORD.json）：

```text
exit=0 wall_s=0.317: D:\miniminiconda\envs\liuyan\python.exe -m compileall -q fhm_dapp_tim/fhm_dapp/msep_v17_two_stage_freeze_v1.py tests/test_msep_v17_two_stage_freeze_v1.py
exit=0 wall_s=26.528: D:\miniminiconda\envs\liuyan\python.exe -m pytest -q tests/test_msep_v17_two_stage_freeze_v1.py tests/test_msep_joint_ablation_v1.py
exit=0 wall_s=16.753: D:\miniminiconda\envs\liuyan\python.exe fhm_dapp_tim/evidence/v17_two_stage_freeze_v1/audit_before_run.py
exit=0 wall_s=248.817: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset tuji --seed 17301 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\tuji_seed17301 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=254.516: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset tuji --seed 17302 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\tuji_seed17302 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=265.848: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset tuji --seed 17303 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\tuji_seed17303 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=665.374: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uts --seed 17301 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uts_seed17301 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=692.764: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uts --seed 17302 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uts_seed17302 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=675.486: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uts --seed 17303 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uts_seed17303 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=1243.663: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uji --seed 17301 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uji_seed17301 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=1353.689: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uji --seed 17302 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uji_seed17302 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=1268.243: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 run --dataset uji --seed 17303 --output D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1\uji_seed17303 --device auto --batch 64 --threads 4 --es-draws 32
exit=0 wall_s=4.549: D:\miniminiconda\envs\liuyan\python.exe -m fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1 aggregate --root D:\liuyanlunwen\fhm_dapp_tim\evidence\v17_two_stage_freeze_v1
exit=0 wall_s=0.132: D:\miniminiconda\envs\liuyan\python.exe -m py_compile fhm_dapp_tim/evidence/v17_two_stage_freeze_v1/audit_and_report.py
```


## 收尾观察与辅助脚本最终记录

18 条轨迹中，全 1–80 轮最优 DEV 与第 21–80 轮最优 DEV 均选中同一轮，且对应 checkpoint 张量一致；本轮排除前 20 轮没有进一步改变最终主检查点。旧早停回放则在 UTS 的部分种子停于尚未出现后期收益的阶段。

主检查点下，同楼层方案在 UTS 三个种子的 MAE 和 ES 都更低；在 UJI 三个种子的 MAE 都更低，但 seed17302 的 ES 增加 0.011892m。不能把平均收益写成所有指标、所有种子都改善。TUJI 目标仅为数值等价对照，其轨迹差异的限制已单列披露。

四种冻结后验校准具有明显域差异：TUJI 的三个校准策略平均 MAE/ES 均低于各自 raw；UTS 的固定 T=.6+G13 平均 MAE 较低但平均 ES 更高；UJI 的所有三种校准策略平均 MAE 均高于各自 raw。自由 T 与 G13 联合拟合也没有形成三域一致收益，不能从 CAL-eval 再挑赢家宣称可靠泛化。

报告辅助脚本首次 NameError 的原始 stdout/stderr、失败源码副本和部分数值审计均保留。实际修复差异另存 run_logs/audit_report_helper_fix.diff；科学源码与正式结果未改动。辅助脚本的最终执行记录：

|命令标签|退出码|墙钟秒|
|---|---:|---:|
|audit_report|1|6.745716|
|compile_audit_helper|0|0.131649|
|audit_report_fixed|0|5.965822|

最终只读数值审计 PASS。记录辅助报告失败不代表正式模型训练/评分重试；九次科学运行均仅一次，aggregate 仅一次。
