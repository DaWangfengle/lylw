> 本轮已完整执行：Phase F 18组、Phase T 45组记录（27条新训练×80轮、18条历史O/A复用），50项回归通过。以下均为历史内部CAL-eval的回顾性描述结果。协议门禁通过；附加跨阶段逐站点恒等诊断不成立，已明确披露。

## 主要结果导读

D增加固定0.30分量运输风险后，相对C的平均MAE在三域均下降，但UJI的ES改善几乎为零且存在种子间反转。E完整检索logit图扩散只带来TUJI/UTS很小均值变化，在UJI明显恶化MAE、ES和尾部；不支持将E视为跨域通用改进。

三种子坐标宏MAE均值（m）：

|域|O|A|C|D|E|
|---|---:|---:|---:|---:|---:|
|tuji|2.313652|2.292017|1.994047|1.864494|1.849955|
|uts|4.905488|4.486917|4.355353|4.190409|4.181283|
|uji|7.964195|7.618143|7.570520|7.308497|7.636376|

三种子连续Student-t ES均值（m）：

|域|O|A|C|D|E|
|---|---:|---:|---:|---:|---:|
|tuji|1.639962|1.658908|1.439864|1.319390|1.305097|
|uts|3.645662|3.393100|3.357089|3.322494|3.313246|
|uji|6.397606|6.232526|6.136653|6.133760|6.390956|

O为原邻居；A为同建筑同楼层邻居；C替换全锚点BF-XY联合NLL；D加0.30运输风险；E加可训练完整检索logit图扩散。完整样本SD、预测BF条件MAE、NLL、尾部与逐站点数据见下文及CSV。
# V17 空间可辨识性实测报告

本轮是相同内部 CAL-eval 物理站点上的回顾性描述比较；不是独立盲测或正式晋级。内部 TEST/官方测试评分行数均为 0。

## 环境、协议与来源

Python 3.12.12 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 20:05:38) [MSC v.1929 64 bit (AMD64)]；Torch 2.10.0+cu128；CUDA 12.8；NVIDIA GeForce RTX 2060。
来源、训练 CSV/MAT 容器、冻结模型及角色键 SHA256 见 EXECUTION_MANIFEST.json。原有源码、历史产物和数据没有修改。
TUJI MAT 仅反序列化 results_train；物理容器包含测试变量，不声称全部测试字节未打开。
CAL-fit 先完成新联合校准、空间参数选择与参数文件冻结，然后才构建 CAL-eval 评分包；所有预声明方案完整报告。
Phase F 不训练；Phase T O/A 在源代码、配置、划分、选中轮次匹配后复用原 80 轮检查点，其余 C/D/E 各训练完整80轮，按21–80轮最低DEV MAE选中。
C 用联合 BF-XY NLL 替换旧 NLL；D 固定增加0.30分量运输风险，保留0.25坐标损失；E对完整检索logit施加可微图扩散，训练和推断使用同一保存alpha。
ES为完整独立轴df4 Student-t混合的2×32样本近似，各方案同seed+900随机流；规范锚点顺序不同于历史Top-K排序，因此历史ES随机采样实现的逐次取样不要求逐位相同。
三种子均值±样本标准差仅描述同一评价集上的训练波动，不能当作独立站点显著性。

## 完整指标汇总

|阶段|域|骨干|方案|MAE m|ES m|联合NLL|BF准确率|P95 m|P99 m|最差站点 m|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|F|tuji|raw_neighbors|1A_bf_mass|1.99777 ± 0.05468|1.39738 ± 0.03395|3.99911 ± 0.10149|1.00000 ± 0.00000|3.05624 ± 0.05529|3.45794 ± 0.13355|3.67106 ± 0.27993|
|F|tuji|raw_neighbors|1A_global|1.99777 ± 0.05468|1.39708 ± 0.03480|3.99911 ± 0.10149|1.00000 ± 0.00000|3.05624 ± 0.05529|3.45794 ± 0.13355|3.67106 ± 0.27993|
|F|tuji|raw_neighbors|1B_bf_mass|1.97677 ± 0.04653|1.42881 ± 0.02428|4.06177 ± 0.10667|1.00000 ± 0.00000|3.01926 ± 0.15461|3.39063 ± 0.10714|3.54316 ± 0.21333|
|F|tuji|raw_neighbors|1B_global|1.97677 ± 0.04653|1.42884 ± 0.02422|4.06177 ± 0.10667|1.00000 ± 0.00000|3.01926 ± 0.15461|3.39063 ± 0.10714|3.54316 ± 0.21333|
|F|tuji|raw_neighbors|1C_bf_mass|2.00378 ± 0.05334|1.40146 ± 0.02493|4.03349 ± 0.09865|1.00000 ± 0.00000|3.05875 ± 0.06459|3.44907 ± 0.11230|3.64245 ± 0.26479|
|F|tuji|raw_neighbors|1C_global|2.00378 ± 0.05334|1.40148 ± 0.02490|4.03349 ± 0.09865|1.00000 ± 0.00000|3.05875 ± 0.06459|3.44907 ± 0.11230|3.64245 ± 0.26479|
|F|tuji|raw_neighbors|1D_bf_mass|1.99777 ± 0.05468|1.39755 ± 0.03402|3.99911 ± 0.10149|1.00000 ± 0.00000|3.05624 ± 0.05529|3.45794 ± 0.13355|3.67106 ± 0.27993|
|F|tuji|raw_neighbors|1D_global|1.99777 ± 0.05468|1.39708 ± 0.03480|3.99911 ± 0.10149|1.00000 ± 0.00000|3.05624 ± 0.05529|3.45794 ± 0.13355|3.67106 ± 0.27993|
|F|tuji|raw_neighbors|2A|1.98472 ± 0.04548|1.38887 ± 0.03143|3.57160 ± 0.01859|1.00000 ± 0.00000|3.03951 ± 0.10547|3.38671 ± 0.09433|3.56518 ± 0.21817|
|F|tuji|raw_neighbors|2A_beta_only_control|1.98472 ± 0.04548|1.38887 ± 0.03143|3.57160 ± 0.01859|1.00000 ± 0.00000|3.03951 ± 0.10547|3.38671 ± 0.09433|3.56518 ± 0.21817|
|F|tuji|raw_neighbors|2A_beta_only_control_geometric_diagnostic|1.97747 ± 0.02953|1.38254 ± 0.02286|3.51169 ± 0.03127|1.00000 ± 0.00000|3.01573 ± 0.05179|3.39604 ± 0.12495|3.54433 ± 0.27155|
|F|tuji|raw_neighbors|2A_geometric_diagnostic|1.97747 ± 0.02953|1.38254 ± 0.02286|3.51169 ± 0.03127|1.00000 ± 0.00000|3.01573 ± 0.05179|3.39604 ± 0.12495|3.54433 ± 0.27155|
|F|tuji|raw_neighbors|2B|1.98472 ± 0.04548|1.38887 ± 0.03143|3.57160 ± 0.01859|1.00000 ± 0.00000|3.03951 ± 0.10547|3.38671 ± 0.09433|3.56518 ± 0.21817|
|F|tuji|raw_neighbors|2B_geometric_diagnostic|1.97747 ± 0.02953|1.38254 ± 0.02286|3.51169 ± 0.03127|1.00000 ± 0.00000|3.01573 ± 0.05179|3.39604 ± 0.12495|3.54433 ± 0.27155|
|F|tuji|raw_neighbors|historical_g13_fixed_t06|1.98240 ± 0.04349|1.40170 ± 0.04079|4.04560 ± 0.08572|1.00000 ± 0.00000|3.07638 ± 0.16234|3.40392 ± 0.10327|3.51926 ± 0.19250|
|F|tuji|raw_neighbors|historical_raw|2.31365 ± 0.09070|1.63954 ± 0.04530|4.42528 ± 0.15588|1.00000 ± 0.00000|4.00350 ± 0.27077|4.68672 ± 0.17117|4.86328 ± 0.12110|
|F|tuji|raw_neighbors|historical_temperature_only|2.05910 ± 0.04662|1.44807 ± 0.03586|4.20096 ± 0.09766|1.00000 ± 0.00000|3.16306 ± 0.01336|3.66803 ± 0.31660|3.96607 ± 0.54353|
|F|tuji|raw_neighbors|historical_temperature_plus_g13|1.98183 ± 0.04868|1.39093 ± 0.02991|4.03026 ± 0.10219|1.00000 ± 0.00000|3.04950 ± 0.09771|3.38209 ± 0.09496|3.50111 ± 0.18951|
|F|tuji|raw_neighbors|joint_calibrated_base|1.97677 ± 0.04653|1.42884 ± 0.02422|4.06177 ± 0.10667|1.00000 ± 0.00000|3.01926 ± 0.15461|3.39063 ± 0.10714|3.54316 ± 0.21333|
|F|tuji|raw_neighbors|joint_calibrated_base_bf_mass|1.97677 ± 0.04653|1.42881 ± 0.02428|4.06177 ± 0.10667|1.00000 ± 0.00000|3.01926 ± 0.15461|3.39063 ± 0.10714|3.54316 ± 0.21333|
|F|tuji|same_floor_neighbors|1A_bf_mass|2.00119 ± 0.02753|1.38487 ± 0.01060|4.02285 ± 0.04405|1.00000 ± 0.00000|3.00418 ± 0.10967|3.30737 ± 0.10234|3.50122 ± 0.13206|
|F|tuji|same_floor_neighbors|1A_global|2.00119 ± 0.02753|1.38487 ± 0.01060|4.02285 ± 0.04405|1.00000 ± 0.00000|3.00418 ± 0.10967|3.30737 ± 0.10234|3.50122 ± 0.13206|
|F|tuji|same_floor_neighbors|1B_bf_mass|1.96849 ± 0.02944|1.40258 ± 0.01537|4.07689 ± 0.06688|1.00000 ± 0.00000|3.00071 ± 0.10140|3.16918 ± 0.13707|3.22970 ± 0.18048|
|F|tuji|same_floor_neighbors|1B_global|1.96849 ± 0.02944|1.40258 ± 0.01537|4.07689 ± 0.06688|1.00000 ± 0.00000|3.00071 ± 0.10140|3.16918 ± 0.13707|3.22970 ± 0.18048|
|F|tuji|same_floor_neighbors|1C_bf_mass|2.00789 ± 0.02805|1.39678 ± 0.02064|4.06124 ± 0.04450|1.00000 ± 0.00000|2.99979 ± 0.14585|3.28629 ± 0.11212|3.46194 ± 0.15758|
|F|tuji|same_floor_neighbors|1C_global|2.00789 ± 0.02805|1.39677 ± 0.02063|4.06124 ± 0.04450|1.00000 ± 0.00000|2.99979 ± 0.14585|3.28629 ± 0.11212|3.46194 ± 0.15758|
|F|tuji|same_floor_neighbors|1D_bf_mass|2.00119 ± 0.02753|1.38486 ± 0.01059|4.02285 ± 0.04405|1.00000 ± 0.00000|3.00418 ± 0.10967|3.30737 ± 0.10234|3.50122 ± 0.13206|
|F|tuji|same_floor_neighbors|1D_global|2.00119 ± 0.02753|1.38487 ± 0.01060|4.02285 ± 0.04405|1.00000 ± 0.00000|3.00418 ± 0.10967|3.30737 ± 0.10234|3.50122 ± 0.13206|
|F|tuji|same_floor_neighbors|2A|1.97976 ± 0.02695|1.39389 ± 0.02760|3.56508 ± 0.03234|1.00000 ± 0.00000|2.95435 ± 0.16469|3.19101 ± 0.11015|3.30714 ± 0.14643|
|F|tuji|same_floor_neighbors|2A_beta_only_control|1.97976 ± 0.02695|1.39389 ± 0.02760|3.56508 ± 0.03234|1.00000 ± 0.00000|2.95435 ± 0.16469|3.19101 ± 0.11015|3.30714 ± 0.14643|
|F|tuji|same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|1.96993 ± 0.02623|1.38721 ± 0.02987|3.50934 ± 0.02464|1.00000 ± 0.00000|2.95240 ± 0.15197|3.17607 ± 0.15644|3.28499 ± 0.22823|
|F|tuji|same_floor_neighbors|2A_geometric_diagnostic|1.96993 ± 0.02623|1.38721 ± 0.02987|3.50934 ± 0.02464|1.00000 ± 0.00000|2.95240 ± 0.15197|3.17607 ± 0.15644|3.28499 ± 0.22823|
|F|tuji|same_floor_neighbors|2B|1.97976 ± 0.02695|1.39389 ± 0.02760|3.56508 ± 0.03234|1.00000 ± 0.00000|2.95435 ± 0.16469|3.19101 ± 0.11015|3.30714 ± 0.14643|
|F|tuji|same_floor_neighbors|2B_geometric_diagnostic|1.96993 ± 0.02623|1.38721 ± 0.02987|3.50934 ± 0.02464|1.00000 ± 0.00000|2.95240 ± 0.15197|3.17607 ± 0.15644|3.28499 ± 0.22823|
|F|tuji|same_floor_neighbors|historical_g13_fixed_t06|1.97137 ± 0.02946|1.38792 ± 0.01573|4.05800 ± 0.06151|1.00000 ± 0.00000|2.99608 ± 0.11582|3.14806 ± 0.18655|3.22535 ± 0.22942|
|F|tuji|same_floor_neighbors|historical_raw|2.29202 ± 0.02935|1.65894 ± 0.05481|4.44946 ± 0.04479|1.00000 ± 0.00000|3.92130 ± 0.14547|4.54349 ± 0.24808|4.69294 ± 0.16113|
|F|tuji|same_floor_neighbors|historical_temperature_only|2.04284 ± 0.02086|1.45877 ± 0.01723|4.21957 ± 0.06230|1.00000 ± 0.00000|3.06578 ± 0.10096|3.54258 ± 0.40956|3.77099 ± 0.58836|
|F|tuji|same_floor_neighbors|historical_temperature_plus_g13|1.97582 ± 0.02906|1.37698 ± 0.02878|4.05074 ± 0.06013|1.00000 ± 0.00000|3.01420 ± 0.14658|3.13998 ± 0.18130|3.20186 ± 0.21829|
|F|tuji|same_floor_neighbors|joint_calibrated_base|1.96849 ± 0.02944|1.40258 ± 0.01537|4.07689 ± 0.06688|1.00000 ± 0.00000|3.00071 ± 0.10140|3.16918 ± 0.13707|3.22970 ± 0.18048|
|F|tuji|same_floor_neighbors|joint_calibrated_base_bf_mass|1.96849 ± 0.02944|1.40258 ± 0.01537|4.07689 ± 0.06688|1.00000 ± 0.00000|3.00071 ± 0.10140|3.16918 ± 0.13707|3.22970 ± 0.18048|
|F|uji|raw_neighbors|1A_bf_mass|7.68904 ± 0.52693|5.91669 ± 0.39260|14.57439 ± 0.06278|0.94160 ± 0.00274|15.78010 ± 1.04133|27.46385 ± 7.84878|34.56270 ± 12.22304|
|F|uji|raw_neighbors|1A_global|7.58804 ± 0.50918|6.15744 ± 0.46413|14.41432 ± 0.05647|0.94845 ± 0.00255|15.82912 ± 1.81684|27.11990 ± 8.87017|33.91781 ± 12.97212|
|F|uji|raw_neighbors|1B_bf_mass|7.85680 ± 0.46745|6.28643 ± 0.43869|14.83718 ± 0.08206|0.94160 ± 0.00274|16.09394 ± 0.78504|28.23270 ± 7.78756|35.73419 ± 12.31002|
|F|uji|raw_neighbors|1B_global|7.70994 ± 0.37322|6.46313 ± 0.46247|14.65962 ± 0.03270|0.94786 ± 0.00393|16.17323 ± 1.63458|27.65718 ± 8.44161|34.85458 ± 12.62746|
|F|uji|raw_neighbors|1C_bf_mass|7.74276 ± 0.50758|5.97180 ± 0.46773|14.58915 ± 0.08379|0.94160 ± 0.00274|15.51880 ± 1.43601|27.56133 ± 7.95241|34.79869 ± 12.52445|
|F|uji|raw_neighbors|1C_global|7.63259 ± 0.47907|6.22277 ± 0.43955|14.45305 ± 0.05440|0.94693 ± 0.00101|15.94730 ± 1.78915|27.15161 ± 8.80266|34.08278 ± 12.96175|
|F|uji|raw_neighbors|1D_bf_mass|7.73939 ± 0.52416|5.97564 ± 0.46346|14.58903 ± 0.08429|0.94160 ± 0.00274|15.53287 ± 1.41207|27.56854 ± 7.94944|34.81986 ± 12.51981|
|F|uji|raw_neighbors|1D_global|7.58804 ± 0.50918|6.15744 ± 0.46413|14.41432 ± 0.05647|0.94845 ± 0.00255|15.82912 ± 1.81684|27.11990 ± 8.87017|33.91781 ± 12.97212|
|F|uji|raw_neighbors|2A|7.50113 ± 0.37360|5.64145 ± 0.36111|10.72496 ± 0.11568|0.94846 ± 0.00342|15.08435 ± 1.35964|26.62427 ± 8.28671|33.11966 ± 11.89509|
|F|uji|raw_neighbors|2A_beta_only_control|7.50113 ± 0.37360|5.64145 ± 0.36111|10.72496 ± 0.11568|0.94846 ± 0.00342|15.08435 ± 1.35964|26.62427 ± 8.28671|33.11966 ± 11.89509|
|F|uji|raw_neighbors|2A_beta_only_control_geometric_diagnostic|7.50101 ± 0.37324|5.64134 ± 0.36081|10.72460 ± 0.11459|0.94846 ± 0.00342|15.07996 ± 1.35821|26.62568 ± 8.28696|33.12255 ± 11.89346|
|F|uji|raw_neighbors|2A_geometric_diagnostic|7.50101 ± 0.37324|5.64134 ± 0.36081|10.72460 ± 0.11459|0.94846 ± 0.00342|15.07996 ± 1.35821|26.62568 ± 8.28696|33.12255 ± 11.89346|
|F|uji|raw_neighbors|2B|7.50113 ± 0.37360|5.64145 ± 0.36111|10.72496 ± 0.11568|0.94846 ± 0.00342|15.08435 ± 1.35964|26.62427 ± 8.28671|33.11966 ± 11.89509|
|F|uji|raw_neighbors|2B_geometric_diagnostic|7.50101 ± 0.37324|5.64134 ± 0.36081|10.72460 ± 0.11459|0.94846 ± 0.00342|15.07996 ± 1.35821|26.62568 ± 8.28696|33.12255 ± 11.89346|
|F|uji|raw_neighbors|historical_g13_fixed_t06|8.04800 ± 0.38580|7.00227 ± 0.35797|15.21591 ± 0.04881|0.94028 ± 0.00111|16.90007 ± 1.48388|29.04281 ± 8.15457|36.41708 ± 12.73298|
|F|uji|raw_neighbors|historical_raw|7.96419 ± 0.34212|6.39762 ± 0.33222|14.82109 ± 0.04948|0.94160 ± 0.00274|16.64933 ± 0.91021|28.79534 ± 7.64178|36.49057 ± 12.32160|
|F|uji|raw_neighbors|historical_temperature_only|8.43836 ± 0.19830|6.29531 ± 0.35311|14.80333 ± 0.04983|0.93966 ± 0.00164|18.07303 ± 1.84982|30.63304 ± 5.09235|38.36576 ± 9.45825|
|F|uji|raw_neighbors|historical_temperature_plus_g13|8.31507 ± 0.13538|6.08913 ± 0.33717|14.67776 ± 0.03297|0.94729 ± 0.00308|17.25166 ± 2.36646|30.07736 ± 5.29999|37.09481 ± 10.34630|
|F|uji|raw_neighbors|joint_calibrated_base|7.65695 ± 0.37791|6.45042 ± 0.42406|14.68408 ± 0.02715|0.94632 ± 0.00470|16.38188 ± 1.73053|27.87912 ± 8.22909|34.92360 ± 12.46555|
|F|uji|raw_neighbors|joint_calibrated_base_bf_mass|7.79846 ± 0.46701|6.28886 ± 0.41094|14.87527 ± 0.08068|0.94160 ± 0.00274|16.62803 ± 0.91674|28.37764 ± 7.74046|35.83022 ± 12.16275|
|F|uji|same_floor_neighbors|1A_bf_mass|7.42891 ± 0.06775|5.95960 ± 0.17163|14.30827 ± 0.05735|0.96275 ± 0.00477|16.26090 ± 0.44503|18.39694 ± 1.48092|19.12111 ± 2.55784|
|F|uji|same_floor_neighbors|1A_global|7.34598 ± 0.05285|6.07928 ± 0.04390|14.30407 ± 0.05019|0.96813 ± 0.00399|16.72037 ± 0.51446|18.35991 ± 0.54340|18.99639 ± 1.23909|
|F|uji|same_floor_neighbors|1B_bf_mass|7.52731 ± 0.05442|6.19385 ± 0.19417|14.49289 ± 0.06696|0.96275 ± 0.00477|16.23827 ± 0.62973|18.55458 ± 1.72419|19.55149 ± 2.87311|
|F|uji|same_floor_neighbors|1B_global|7.44539 ± 0.09969|6.28741 ± 0.09108|14.49310 ± 0.06707|0.96515 ± 0.00427|16.68315 ± 0.75996|18.16375 ± 0.25418|18.47349 ± 0.38423|
|F|uji|same_floor_neighbors|1C_bf_mass|7.44542 ± 0.05466|5.97424 ± 0.15113|14.32181 ± 0.04201|0.96275 ± 0.00477|16.24010 ± 0.49325|18.41149 ± 1.50768|19.12568 ± 2.65715|
|F|uji|same_floor_neighbors|1C_global|7.37184 ± 0.06265|6.12336 ± 0.08887|14.34015 ± 0.06279|0.96820 ± 0.00438|16.81653 ± 0.44024|18.35581 ± 0.50336|18.99930 ± 1.08610|
|F|uji|same_floor_neighbors|1D_bf_mass|7.44307 ± 0.05708|5.95854 ± 0.17120|14.31232 ± 0.04987|0.96275 ± 0.00477|16.25240 ± 0.47680|18.38801 ± 1.47751|19.08163 ± 2.58457|
|F|uji|same_floor_neighbors|1D_global|7.34598 ± 0.05285|6.07928 ± 0.04390|14.30407 ± 0.05019|0.96813 ± 0.00399|16.72037 ± 0.51446|18.35991 ± 0.54340|18.99639 ± 1.23909|
|F|uji|same_floor_neighbors|2A|7.39020 ± 0.04844|5.60806 ± 0.11531|10.55672 ± 0.05234|0.96809 ± 0.00374|16.13262 ± 0.95985|17.79001 ± 0.73547|18.59044 ± 1.21796|
|F|uji|same_floor_neighbors|2A_beta_only_control|7.39020 ± 0.04844|5.60806 ± 0.11531|10.55672 ± 0.05234|0.96809 ± 0.00374|16.13262 ± 0.95985|17.79001 ± 0.73547|18.59044 ± 1.21796|
|F|uji|same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|7.38880 ± 0.04676|5.60691 ± 0.11530|10.54381 ± 0.03808|0.96809 ± 0.00374|16.12120 ± 0.96552|17.79749 ± 0.72880|18.60327 ± 1.22361|
|F|uji|same_floor_neighbors|2A_geometric_diagnostic|7.38880 ± 0.04676|5.60691 ± 0.11530|10.54381 ± 0.03808|0.96809 ± 0.00374|16.12120 ± 0.96552|17.79749 ± 0.72880|18.60327 ± 1.22361|
|F|uji|same_floor_neighbors|2B|7.39701 ± 0.05446|5.60180 ± 0.10712|10.55725 ± 0.05277|0.96790 ± 0.00407|16.24071 ± 1.09981|17.79517 ± 0.74174|18.59922 ± 1.22225|
|F|uji|same_floor_neighbors|2B_geometric_diagnostic|7.39561 ± 0.05270|5.60065 ± 0.10717|10.54436 ± 0.03868|0.96790 ± 0.00407|16.22939 ± 1.10617|17.80266 ± 0.73501|18.61209 ± 1.22758|
|F|uji|same_floor_neighbors|historical_g13_fixed_t06|7.92417 ± 0.13943|6.85294 ± 0.13452|14.84784 ± 0.05327|0.96148 ± 0.00562|17.67519 ± 1.28917|20.56940 ± 1.92395|21.90662 ± 2.78460|
|F|uji|same_floor_neighbors|historical_raw|7.61814 ± 0.05557|6.23290 ± 0.16248|14.43528 ± 0.03012|0.96275 ± 0.00477|16.34259 ± 0.90135|19.49501 ± 1.07663|20.25707 ± 1.57795|
|F|uji|same_floor_neighbors|historical_temperature_only|8.72216 ± 0.32468|6.24559 ± 0.10554|14.39786 ± 0.02333|0.96426 ± 0.00655|18.82995 ± 1.27161|23.93757 ± 2.78065|26.33210 ± 3.36416|
|F|uji|same_floor_neighbors|historical_temperature_plus_g13|8.85163 ± 0.14056|6.20821 ± 0.21369|14.38952 ± 0.01961|0.96519 ± 0.00616|21.94184 ± 2.56907|25.01301 ± 1.49885|26.04912 ± 1.18698|
|F|uji|same_floor_neighbors|joint_calibrated_base|7.42044 ± 0.06929|6.31473 ± 0.09208|14.53049 ± 0.06547|0.96590 ± 0.00320|16.83472 ± 0.77698|17.96806 ± 0.70798|18.21243 ± 0.82877|
|F|uji|same_floor_neighbors|joint_calibrated_base_bf_mass|7.50116 ± 0.07979|6.24368 ± 0.17138|14.53476 ± 0.05627|0.96275 ± 0.00477|16.65139 ± 0.92364|18.56604 ± 1.72091|19.52300 ± 2.79792|
|F|uts|raw_neighbors|1A_bf_mass|4.55871 ± 0.19520|3.42784 ± 0.04115|11.49059 ± 0.06933|0.93273 ± 0.03330|9.80833 ± 0.80301|15.12638 ± 1.23887|16.97527 ± 0.57934|
|F|uts|raw_neighbors|1A_global|4.10304 ± 0.09113|3.30058 ± 0.05973|11.05987 ± 0.06727|0.98309 ± 0.00754|8.34849 ± 0.89991|14.40516 ± 1.39252|16.55437 ± 1.54748|
|F|uts|raw_neighbors|1B_bf_mass|4.64624 ± 0.13607|3.53770 ± 0.03025|11.65113 ± 0.10472|0.93273 ± 0.03330|9.09005 ± 0.57489|15.85238 ± 1.36447|18.22814 ± 0.59372|
|F|uts|raw_neighbors|1B_global|4.09764 ± 0.05228|3.43485 ± 0.09543|11.25208 ± 0.09676|0.98309 ± 0.01464|8.51185 ± 0.35145|13.32810 ± 0.09432|17.41280 ± 1.82587|
|F|uts|raw_neighbors|1C_bf_mass|4.61302 ± 0.18825|3.47404 ± 0.02772|11.47146 ± 0.07607|0.93273 ± 0.03330|9.67220 ± 0.57491|15.56315 ± 1.26895|18.06826 ± 1.03074|
|F|uts|raw_neighbors|1C_global|4.15240 ± 0.09056|3.36528 ± 0.03662|11.12497 ± 0.04322|0.97947 ± 0.00837|8.41411 ± 0.52698|14.91067 ± 0.86528|16.84532 ± 1.46980|
|F|uts|raw_neighbors|1D_bf_mass|4.61794 ± 0.19641|3.47108 ± 0.02297|11.48446 ± 0.09850|0.93273 ± 0.03330|9.44280 ± 0.20959|15.66934 ± 1.09901|18.07987 ± 1.02395|
|F|uts|raw_neighbors|1D_global|4.10304 ± 0.09113|3.30058 ± 0.05973|11.05987 ± 0.06727|0.98309 ± 0.00754|8.34849 ± 0.89991|14.40516 ± 1.39252|16.55437 ± 1.54748|
|F|uts|raw_neighbors|2A|3.98422 ± 0.04485|2.99038 ± 0.07568|6.36519 ± 0.07522|0.98309 ± 0.01464|8.23502 ± 0.59525|13.62206 ± 0.74113|16.75939 ± 1.49658|
|F|uts|raw_neighbors|2A_beta_only_control|3.98422 ± 0.04485|2.99038 ± 0.07568|6.36519 ± 0.07522|0.98309 ± 0.01464|8.23502 ± 0.59525|13.62206 ± 0.74113|16.75939 ± 1.49658|
|F|uts|raw_neighbors|2A_beta_only_control_geometric_diagnostic|3.98437 ± 0.04589|2.98987 ± 0.07645|6.36588 ± 0.07341|0.98309 ± 0.01464|8.23294 ± 0.58970|13.61942 ± 0.73756|16.75908 ± 1.49986|
|F|uts|raw_neighbors|2A_geometric_diagnostic|3.98437 ± 0.04589|2.98987 ± 0.07645|6.36588 ± 0.07341|0.98309 ± 0.01464|8.23294 ± 0.58970|13.61942 ± 0.73756|16.75908 ± 1.49986|
|F|uts|raw_neighbors|2B|3.98422 ± 0.04485|2.99038 ± 0.07568|6.36519 ± 0.07522|0.98309 ± 0.01464|8.23502 ± 0.59525|13.62206 ± 0.74113|16.75939 ± 1.49658|
|F|uts|raw_neighbors|2B_geometric_diagnostic|3.98437 ± 0.04589|2.98987 ± 0.07645|6.36588 ± 0.07341|0.98309 ± 0.01464|8.23294 ± 0.58970|13.61942 ± 0.73756|16.75908 ± 1.49986|
|F|uts|raw_neighbors|historical_g13_fixed_t06|4.49215 ± 0.10318|3.76343 ± 0.10828|11.64954 ± 0.16978|0.92790 ± 0.02498|10.53145 ± 0.91041|15.91792 ± 0.51906|18.25946 ± 1.81851|
|F|uts|raw_neighbors|historical_raw|4.90549 ± 0.13680|3.64475 ± 0.07196|11.67287 ± 0.10689|0.93273 ± 0.03330|10.09458 ± 0.18749|16.15483 ± 0.95465|19.07441 ± 1.65274|
|F|uts|raw_neighbors|historical_temperature_only|5.45609 ± 0.17947|3.65809 ± 0.10753|11.81859 ± 0.09141|0.93663 ± 0.03201|11.48736 ± 1.32594|16.18037 ± 1.23792|19.46941 ± 3.24513|
|F|uts|raw_neighbors|historical_temperature_plus_g13|5.45488 ± 0.17998|3.65747 ± 0.11114|11.81831 ± 0.09135|0.93663 ± 0.03201|11.48258 ± 1.32474|16.17926 ± 1.23688|19.46915 ± 3.24038|
|F|uts|raw_neighbors|joint_calibrated_base|4.09764 ± 0.05228|3.43485 ± 0.09543|11.25208 ± 0.09676|0.98309 ± 0.01464|8.51185 ± 0.35145|13.32810 ± 0.09432|17.41280 ± 1.82587|
|F|uts|raw_neighbors|joint_calibrated_base_bf_mass|4.52691 ± 0.12715|3.48051 ± 0.02126|11.74494 ± 0.10891|0.93273 ± 0.03330|9.17536 ± 0.19270|14.99061 ± 0.70331|16.85063 ± 1.10823|
|F|uts|same_floor_neighbors|1A_bf_mass|4.01259 ± 0.19267|3.29620 ± 0.17086|10.85290 ± 0.07176|0.99122 ± 0.00772|9.67263 ± 0.89499|16.48373 ± 1.24491|19.85879 ± 2.27064|
|F|uts|same_floor_neighbors|1A_global|3.94558 ± 0.16203|3.25914 ± 0.12622|10.82055 ± 0.09350|0.99517 ± 0.00837|8.86886 ± 0.56762|16.81589 ± 1.21533|18.66308 ± 1.30817|
|F|uts|same_floor_neighbors|1B_bf_mass|4.06565 ± 0.23268|3.28040 ± 0.16929|10.85027 ± 0.07295|0.99122 ± 0.00772|9.78815 ± 1.40504|16.60372 ± 1.45522|20.71154 ± 2.73849|
|F|uts|same_floor_neighbors|1B_global|3.99347 ± 0.17771|3.31940 ± 0.09261|10.81388 ± 0.08951|0.99517 ± 0.00837|8.75036 ± 0.63558|16.93726 ± 1.35319|19.49642 ± 1.49063|
|F|uts|same_floor_neighbors|1C_bf_mass|4.02431 ± 0.20746|3.27683 ± 0.16285|10.84653 ± 0.07426|0.99122 ± 0.00772|9.70539 ± 1.07956|16.52153 ± 1.33405|20.20257 ± 2.42726|
|F|uts|same_floor_neighbors|1C_global|3.96413 ± 0.18077|3.28111 ± 0.11754|10.80616 ± 0.08260|0.99517 ± 0.00837|8.75539 ± 0.67374|16.89399 ± 1.33219|19.15126 ± 1.61380|
|F|uts|same_floor_neighbors|1D_bf_mass|4.03522 ± 0.20097|3.27696 ± 0.16058|10.84122 ± 0.07800|0.99122 ± 0.00772|9.67361 ± 1.10005|16.55819 ± 1.29572|20.32607 ± 2.31557|
|F|uts|same_floor_neighbors|1D_global|3.95457 ± 0.17076|3.28381 ± 0.13857|10.81174 ± 0.08866|0.99517 ± 0.00837|8.78256 ± 0.63715|16.87909 ± 1.26909|18.96463 ± 1.40905|
|F|uts|same_floor_neighbors|2A|4.07264 ± 0.18483|2.95318 ± 0.12190|5.78592 ± 0.09903|0.99517 ± 0.00837|8.75576 ± 0.63013|16.72685 ± 1.39949|18.60743 ± 2.18845|
|F|uts|same_floor_neighbors|2A_beta_only_control|4.07264 ± 0.18483|2.95318 ± 0.12190|5.78592 ± 0.09903|0.99517 ± 0.00837|8.75576 ± 0.63013|16.72685 ± 1.39949|18.60743 ± 2.18845|
|F|uts|same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|4.07162 ± 0.18257|2.95230 ± 0.12050|5.78020 ± 0.09350|0.99517 ± 0.00837|8.73479 ± 0.61783|16.73172 ± 1.40780|18.58743 ± 2.17608|
|F|uts|same_floor_neighbors|2A_geometric_diagnostic|4.07162 ± 0.18257|2.95230 ± 0.12050|5.78020 ± 0.09350|0.99517 ± 0.00837|8.73479 ± 0.61783|16.73172 ± 1.40780|18.58743 ± 2.17608|
|F|uts|same_floor_neighbors|2B|4.07264 ± 0.18483|2.95318 ± 0.12190|5.78592 ± 0.09903|0.99517 ± 0.00837|8.75576 ± 0.63013|16.72685 ± 1.39949|18.60743 ± 2.18845|
|F|uts|same_floor_neighbors|2B_geometric_diagnostic|4.07162 ± 0.18257|2.95230 ± 0.12050|5.78020 ± 0.09350|0.99517 ± 0.00837|8.73479 ± 0.61783|16.73172 ± 1.40780|18.58743 ± 2.17608|
|F|uts|same_floor_neighbors|historical_g13_fixed_t06|4.20561 ± 0.21083|3.47765 ± 0.13412|10.89976 ± 0.13757|0.99122 ± 0.00772|9.30211 ± 1.04036|16.96402 ± 1.43477|19.35516 ± 2.18782|
|F|uts|same_floor_neighbors|historical_raw|4.48692 ± 0.31536|3.39319 ± 0.19523|10.88657 ± 0.08792|0.99122 ± 0.00772|10.72765 ± 1.60653|16.66694 ± 3.10942|19.87292 ± 1.72664|
|F|uts|same_floor_neighbors|historical_temperature_only|5.39385 ± 0.26921|3.53253 ± 0.18474|11.05271 ± 0.05801|0.99122 ± 0.00772|12.52510 ± 1.78343|19.06119 ± 2.65931|20.98224 ± 1.34916|
|F|uts|same_floor_neighbors|historical_temperature_plus_g13|5.11533 ± 0.19213|3.45620 ± 0.24911|10.98139 ± 0.04654|0.99517 ± 0.00837|11.71522 ± 1.37316|18.51545 ± 1.42526|20.87576 ± 1.11246|
|F|uts|same_floor_neighbors|joint_calibrated_base|3.94772 ± 0.14175|3.31896 ± 0.06705|10.86034 ± 0.09016|0.99517 ± 0.00837|9.05796 ± 0.50689|16.86534 ± 1.17564|18.59143 ± 0.91915|
|F|uts|same_floor_neighbors|joint_calibrated_base_bf_mass|4.00057 ± 0.18443|3.32882 ± 0.17811|10.90156 ± 0.07895|0.99122 ± 0.00772|9.77758 ± 0.87846|16.49668 ± 1.28547|19.79120 ± 2.33405|
|T|tuji|A|A|2.29202 ± 0.02935|1.65891 ± 0.05480|4.44946 ± 0.04479|1.00000 ± 0.00000|3.92130 ± 0.14547|4.54349 ± 0.24808|4.69294 ± 0.16113|
|T|tuji|C|C|1.99405 ± 0.01908|1.43986 ± 0.02290|4.17918 ± 0.09523|1.00000 ± 0.00000|3.16976 ± 0.16526|4.03480 ± 0.23268|4.43442 ± 0.37300|
|T|tuji|D|D|1.86449 ± 0.02009|1.31939 ± 0.00612|3.93557 ± 0.03421|1.00000 ± 0.00000|2.85659 ± 0.09841|3.14449 ± 0.06056|3.23765 ± 0.08146|
|T|tuji|E|E|1.84996 ± 0.04829|1.30510 ± 0.02127|3.87845 ± 0.02233|1.00000 ± 0.00000|2.86052 ± 0.11457|3.21831 ± 0.26445|3.35805 ± 0.41212|
|T|tuji|O|O|2.31365 ± 0.09070|1.63996 ± 0.04461|4.42528 ± 0.15588|1.00000 ± 0.00000|4.00350 ± 0.27077|4.68672 ± 0.17117|4.86328 ± 0.12110|
|T|uji|A|A|7.61814 ± 0.05558|6.23253 ± 0.16244|14.43528 ± 0.03012|0.96275 ± 0.00477|16.34260 ± 0.90134|19.49509 ± 1.07679|20.25707 ± 1.57795|
|T|uji|C|C|7.57052 ± 0.20205|6.13665 ± 0.23547|14.40565 ± 0.05931|0.96167 ± 0.00538|16.82251 ± 0.08959|22.40673 ± 6.57056|26.03542 ± 11.49991|
|T|uji|D|D|7.30850 ± 0.20956|6.13376 ± 0.12120|14.49981 ± 0.05750|0.95770 ± 0.00149|16.46466 ± 0.56757|20.53933 ± 2.68805|21.51262 ± 2.80951|
|T|uji|E|E|7.63638 ± 0.46224|6.39096 ± 0.58423|14.48532 ± 0.20544|0.96656 ± 0.00579|16.91049 ± 1.09250|26.60913 ± 7.06305|32.25489 ± 10.94964|
|T|uji|O|O|7.96420 ± 0.34212|6.39761 ± 0.33111|14.82108 ± 0.04948|0.94160 ± 0.00274|16.64932 ± 0.91018|28.79540 ± 7.64181|36.49067 ± 12.32168|
|T|uts|A|A|4.48692 ± 0.31536|3.39310 ± 0.19506|10.88657 ± 0.08792|0.99122 ± 0.00772|10.72765 ± 1.60651|16.66694 ± 3.10943|19.87291 ± 1.72666|
|T|uts|C|C|4.35535 ± 0.07794|3.35709 ± 0.11463|10.87517 ± 0.05715|0.98639 ± 0.00152|9.82930 ± 0.34826|15.33542 ± 1.24463|18.09626 ± 2.12562|
|T|uts|D|D|4.19041 ± 0.11014|3.32249 ± 0.03074|10.82849 ± 0.06776|1.00000 ± 0.00000|9.67921 ± 0.63694|14.51027 ± 1.48358|15.68739 ± 1.60999|
|T|uts|E|E|4.18128 ± 0.10957|3.31325 ± 0.10716|10.83900 ± 0.09779|1.00000 ± 0.00000|9.21230 ± 0.34236|13.07521 ± 1.62239|13.64231 ± 1.17998|
|T|uts|O|O|4.90549 ± 0.13680|3.64566 ± 0.07178|11.67287 ± 0.10689|0.93273 ± 0.03330|10.09458 ± 0.18748|16.15483 ± 0.95465|19.07443 ± 1.65275|

## 配对结果、负结果与尾部

下表负MAE/ES差表示改善；同一比较保留三个种子。完整站点向量与伤害见SITE_DELTAS_ALL_RUNS.csv；不得根据此表重选方法。

|阶段/域/骨干|对照→候选|ΔMAE 均值±SD|ΔES 均值±SD|单站点最大伤害 m|
|---|---|---:|---:|---:|
|F/tuji/raw_neighbors|2A→2B|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/raw_neighbors|historical_raw→historical_g13_fixed_t06|-0.33126 ± 0.08407|-0.23784 ± 0.08148|2.18075|
|F/tuji/raw_neighbors|historical_raw→historical_temperature_only|-0.25456 ± 0.09516|-0.19147 ± 0.05706|1.20311|
|F/tuji/raw_neighbors|historical_raw→historical_temperature_plus_g13|-0.33182 ± 0.08887|-0.24860 ± 0.07234|2.12248|
|F/tuji/raw_neighbors|joint_calibrated_base→1A_global|+0.02100 ± 0.01729|-0.03176 ± 0.01071|0.45407|
|F/tuji/raw_neighbors|joint_calibrated_base→1B_global|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/raw_neighbors|joint_calibrated_base→1C_global|+0.02701 ± 0.01373|-0.02735 ± 0.00077|0.37902|
|F/tuji/raw_neighbors|joint_calibrated_base→1D_global|+0.02100 ± 0.01729|-0.03176 ± 0.01071|0.45407|
|F/tuji/raw_neighbors|joint_calibrated_base→2A|+0.00795 ± 0.01249|-0.03997 ± 0.03253|0.23831|
|F/tuji/raw_neighbors|joint_calibrated_base→2A_beta_only_control|+0.00795 ± 0.01249|-0.03997 ± 0.03253|0.23831|
|F/tuji/raw_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|+0.00070 ± 0.02262|-0.04630 ± 0.03257|0.28662|
|F/tuji/raw_neighbors|joint_calibrated_base→2A_geometric_diagnostic|+0.00070 ± 0.02262|-0.04630 ± 0.03257|0.28662|
|F/tuji/raw_neighbors|joint_calibrated_base→2B|+0.00795 ± 0.01249|-0.03997 ± 0.03253|0.23831|
|F/tuji/raw_neighbors|joint_calibrated_base→2B_geometric_diagnostic|+0.00070 ± 0.02262|-0.04630 ± 0.03257|0.28662|
|F/tuji/raw_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|-0.00000 ± 0.00000|-0.00003 ± 0.00007|0.00000|
|F/tuji/raw_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|+0.02100 ± 0.01729|-0.03143 ± 0.00974|0.45407|
|F/tuji/raw_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/raw_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|+0.02701 ± 0.01373|-0.02735 ± 0.00071|0.37902|
|F/tuji/raw_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|+0.02100 ± 0.01729|-0.03126 ± 0.00984|0.45407|
|F/tuji/same_floor_neighbors|2A→2B|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/same_floor_neighbors|historical_raw→historical_g13_fixed_t06|-0.32065 ± 0.04842|-0.27103 ± 0.06931|1.94048|
|F/tuji/same_floor_neighbors|historical_raw→historical_temperature_only|-0.24917 ± 0.03563|-0.20017 ± 0.07048|1.08694|
|F/tuji/same_floor_neighbors|historical_raw→historical_temperature_plus_g13|-0.31620 ± 0.04717|-0.28197 ± 0.08354|1.92943|
|F/tuji/same_floor_neighbors|joint_calibrated_base→1A_global|+0.03270 ± 0.01183|-0.01771 ± 0.00497|0.34320|
|F/tuji/same_floor_neighbors|joint_calibrated_base→1B_global|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/same_floor_neighbors|joint_calibrated_base→1C_global|+0.03940 ± 0.01232|-0.00582 ± 0.01485|0.29930|
|F/tuji/same_floor_neighbors|joint_calibrated_base→1D_global|+0.03270 ± 0.01183|-0.01771 ± 0.00497|0.34320|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2A|+0.01127 ± 0.00453|-0.00869 ± 0.01951|0.19603|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control|+0.01127 ± 0.00453|-0.00869 ± 0.01951|0.19603|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|+0.00145 ± 0.01165|-0.01537 ± 0.02208|0.24553|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2A_geometric_diagnostic|+0.00145 ± 0.01165|-0.01537 ± 0.02208|0.24553|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2B|+0.01127 ± 0.00453|-0.00869 ± 0.01951|0.19603|
|F/tuji/same_floor_neighbors|joint_calibrated_base→2B_geometric_diagnostic|+0.00145 ± 0.01165|-0.01537 ± 0.02208|0.24553|
|F/tuji/same_floor_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|-0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/same_floor_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|+0.03270 ± 0.01183|-0.01771 ± 0.00497|0.34320|
|F/tuji/same_floor_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/tuji/same_floor_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|+0.03940 ± 0.01232|-0.00581 ± 0.01481|0.29930|
|F/tuji/same_floor_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|+0.03270 ± 0.01183|-0.01772 ± 0.00498|0.34320|
|F/uji/raw_neighbors|2A→2B|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/uji/raw_neighbors|historical_raw→historical_g13_fixed_t06|+0.08381 ± 0.06295|+0.60466 ± 0.10581|2.84008|
|F/uji/raw_neighbors|historical_raw→historical_temperature_only|+0.47417 ± 0.14414|-0.10230 ± 0.03306|5.18110|
|F/uji/raw_neighbors|historical_raw→historical_temperature_plus_g13|+0.35088 ± 0.29197|-0.30848 ± 0.00559|8.01320|
|F/uji/raw_neighbors|joint_calibrated_base→1A_global|-0.06890 ± 0.20235|-0.29298 ± 0.15236|1.66367|
|F/uji/raw_neighbors|joint_calibrated_base→1B_global|+0.05299 ± 0.04100|+0.01271 ± 0.06570|1.00471|
|F/uji/raw_neighbors|joint_calibrated_base→1C_global|-0.02436 ± 0.18708|-0.22765 ± 0.16995|1.74351|
|F/uji/raw_neighbors|joint_calibrated_base→1D_global|-0.06890 ± 0.20235|-0.29298 ± 0.15236|1.66367|
|F/uji/raw_neighbors|joint_calibrated_base→2A|-0.15582 ± 0.09145|-0.80897 ± 0.06653|2.41418|
|F/uji/raw_neighbors|joint_calibrated_base→2A_beta_only_control|-0.15582 ± 0.09145|-0.80897 ± 0.06653|2.41418|
|F/uji/raw_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|-0.15594 ± 0.09088|-0.80908 ± 0.06666|2.41446|
|F/uji/raw_neighbors|joint_calibrated_base→2A_geometric_diagnostic|-0.15594 ± 0.09088|-0.80908 ± 0.06666|2.41446|
|F/uji/raw_neighbors|joint_calibrated_base→2B|-0.15582 ± 0.09145|-0.80897 ± 0.06653|2.41418|
|F/uji/raw_neighbors|joint_calibrated_base→2B_geometric_diagnostic|-0.15594 ± 0.09088|-0.80908 ± 0.06666|2.41446|
|F/uji/raw_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|+0.14151 ± 0.16653|-0.16156 ± 0.02649|9.41137|
|F/uji/raw_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|-0.10942 ± 0.10178|-0.37218 ± 0.04349|2.02617|
|F/uji/raw_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.05833 ± 0.02393|-0.00243 ± 0.04395|1.31273|
|F/uji/raw_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|-0.05570 ± 0.10209|-0.31706 ± 0.09362|2.14782|
|F/uji/raw_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|-0.05907 ± 0.10983|-0.31323 ± 0.08959|2.14782|
|F/uji/same_floor_neighbors|2A→2B|+0.00681 ± 0.01179|-0.00625 ± 0.01083|0.38215|
|F/uji/same_floor_neighbors|historical_raw→historical_g13_fixed_t06|+0.30603 ± 0.19379|+0.62005 ± 0.03124|10.04240|
|F/uji/same_floor_neighbors|historical_raw→historical_temperature_only|+1.10402 ± 0.27979|+0.01270 ± 0.06729|11.54403|
|F/uji/same_floor_neighbors|historical_raw→historical_temperature_plus_g13|+1.23349 ± 0.09024|-0.02468 ± 0.10849|16.54933|
|F/uji/same_floor_neighbors|joint_calibrated_base→1A_global|-0.07445 ± 0.03883|-0.23545 ± 0.04875|3.23463|
|F/uji/same_floor_neighbors|joint_calibrated_base→1B_global|+0.02495 ± 0.03206|-0.02731 ± 0.02649|1.68037|
|F/uji/same_floor_neighbors|joint_calibrated_base→1C_global|-0.04860 ± 0.02578|-0.19137 ± 0.01129|2.95362|
|F/uji/same_floor_neighbors|joint_calibrated_base→1D_global|-0.07445 ± 0.03883|-0.23545 ± 0.04875|3.23463|
|F/uji/same_floor_neighbors|joint_calibrated_base→2A|-0.03023 ± 0.06722|-0.70667 ± 0.20739|3.89905|
|F/uji/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control|-0.03023 ± 0.06722|-0.70667 ± 0.20739|3.89905|
|F/uji/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|-0.03164 ± 0.06621|-0.70782 ± 0.20738|3.89448|
|F/uji/same_floor_neighbors|joint_calibrated_base→2A_geometric_diagnostic|-0.03164 ± 0.06621|-0.70782 ± 0.20738|3.89448|
|F/uji/same_floor_neighbors|joint_calibrated_base→2B|-0.02342 ± 0.07895|-0.71292 ± 0.19912|3.92782|
|F/uji/same_floor_neighbors|joint_calibrated_base→2B_geometric_diagnostic|-0.02483 ± 0.07791|-0.71408 ± 0.19913|3.92331|
|F/uji/same_floor_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|+0.08072 ± 0.11153|-0.07105 ± 0.11476|5.24754|
|F/uji/same_floor_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|-0.07225 ± 0.03248|-0.28409 ± 0.02648|1.59917|
|F/uji/same_floor_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.02615 ± 0.03246|-0.04983 ± 0.02703|1.10780|
|F/uji/same_floor_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|-0.05574 ± 0.03380|-0.26944 ± 0.02709|1.50966|
|F/uji/same_floor_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|-0.05808 ± 0.03691|-0.28514 ± 0.02152|1.64942|
|F/uts/raw_neighbors|2A→2B|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/uts/raw_neighbors|historical_raw→historical_g13_fixed_t06|-0.41333 ± 0.11841|+0.11868 ± 0.03669|4.13110|
|F/uts/raw_neighbors|historical_raw→historical_temperature_only|+0.55061 ± 0.13251|+0.01334 ± 0.03558|4.17048|
|F/uts/raw_neighbors|historical_raw→historical_temperature_plus_g13|+0.54939 ± 0.13373|+0.01273 ± 0.03929|4.15528|
|F/uts/raw_neighbors|joint_calibrated_base→1A_global|+0.00540 ± 0.04255|-0.13427 ± 0.04548|6.64918|
|F/uts/raw_neighbors|joint_calibrated_base→1B_global|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/uts/raw_neighbors|joint_calibrated_base→1C_global|+0.05476 ± 0.04007|-0.06957 ± 0.08549|6.22065|
|F/uts/raw_neighbors|joint_calibrated_base→1D_global|+0.00540 ± 0.04255|-0.13427 ± 0.04548|6.64918|
|F/uts/raw_neighbors|joint_calibrated_base→2A|-0.11342 ± 0.01790|-0.44447 ± 0.02709|3.06742|
|F/uts/raw_neighbors|joint_calibrated_base→2A_beta_only_control|-0.11342 ± 0.01790|-0.44447 ± 0.02709|3.06742|
|F/uts/raw_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|-0.11327 ± 0.01742|-0.44499 ± 0.02648|3.05363|
|F/uts/raw_neighbors|joint_calibrated_base→2A_geometric_diagnostic|-0.11327 ± 0.01742|-0.44499 ± 0.02648|3.05363|
|F/uts/raw_neighbors|joint_calibrated_base→2B|-0.11342 ± 0.01790|-0.44447 ± 0.02709|3.06742|
|F/uts/raw_neighbors|joint_calibrated_base→2B_geometric_diagnostic|-0.11327 ± 0.01742|-0.44499 ± 0.02648|3.05363|
|F/uts/raw_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|+0.42927 ± 0.08684|+0.04566 ± 0.08744|10.45455|
|F/uts/raw_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|+0.03180 ± 0.07087|-0.05268 ± 0.02081|4.78910|
|F/uts/raw_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.11934 ± 0.03548|+0.05719 ± 0.01230|3.89787|
|F/uts/raw_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|+0.08612 ± 0.07624|-0.00648 ± 0.00760|4.23072|
|F/uts/raw_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|+0.09104 ± 0.08239|-0.00944 ± 0.00259|4.23072|
|F/uts/same_floor_neighbors|2A→2B|+0.00000 ± 0.00000|+0.00000 ± 0.00000|0.00000|
|F/uts/same_floor_neighbors|historical_raw→historical_g13_fixed_t06|-0.28131 ± 0.12490|+0.08446 ± 0.11026|3.74382|
|F/uts/same_floor_neighbors|historical_raw→historical_temperature_only|+0.90694 ± 0.10633|+0.13934 ± 0.08026|5.48811|
|F/uts/same_floor_neighbors|historical_raw→historical_temperature_plus_g13|+0.62841 ± 0.13463|+0.06301 ± 0.15698|5.57804|
|F/uts/same_floor_neighbors|joint_calibrated_base→1A_global|-0.00215 ± 0.02054|-0.05982 ± 0.07746|1.16634|
|F/uts/same_floor_neighbors|joint_calibrated_base→1B_global|+0.04574 ± 0.03773|+0.00044 ± 0.07459|1.62350|
|F/uts/same_floor_neighbors|joint_calibrated_base→1C_global|+0.01641 ± 0.04037|-0.03785 ± 0.08185|1.40752|
|F/uts/same_floor_neighbors|joint_calibrated_base→1D_global|+0.00685 ± 0.02914|-0.03515 ± 0.10209|0.99200|
|F/uts/same_floor_neighbors|joint_calibrated_base→2A|+0.12492 ± 0.06123|-0.36579 ± 0.08200|2.68614|
|F/uts/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control|+0.12492 ± 0.06123|-0.36579 ± 0.08200|2.68614|
|F/uts/same_floor_neighbors|joint_calibrated_base→2A_beta_only_control_geometric_diagnostic|+0.12390 ± 0.05936|-0.36666 ± 0.08061|2.64148|
|F/uts/same_floor_neighbors|joint_calibrated_base→2A_geometric_diagnostic|+0.12390 ± 0.05936|-0.36666 ± 0.08061|2.64148|
|F/uts/same_floor_neighbors|joint_calibrated_base→2B|+0.12492 ± 0.06123|-0.36579 ± 0.08200|2.68614|
|F/uts/same_floor_neighbors|joint_calibrated_base→2B_geometric_diagnostic|+0.12390 ± 0.05936|-0.36666 ± 0.08061|2.64148|
|F/uts/same_floor_neighbors|joint_calibrated_base→joint_calibrated_base_bf_mass|+0.05285 ± 0.05742|+0.00986 ± 0.13382|2.87778|
|F/uts/same_floor_neighbors|joint_calibrated_base_bf_mass→1A_bf_mass|+0.01203 ± 0.00845|-0.03262 ± 0.00845|1.08298|
|F/uts/same_floor_neighbors|joint_calibrated_base_bf_mass→1B_bf_mass|+0.06508 ± 0.04826|-0.04842 ± 0.02197|1.70154|
|F/uts/same_floor_neighbors|joint_calibrated_base_bf_mass→1C_bf_mass|+0.02375 ± 0.02317|-0.05200 ± 0.01644|0.95470|
|F/uts/same_floor_neighbors|joint_calibrated_base_bf_mass→1D_bf_mass|+0.03465 ± 0.01784|-0.05186 ± 0.01977|1.29172|
|T/tuji/A|O→A|-0.02163 ± 0.06147|+0.01895 ± 0.01946|0.62447|
|T/tuji/C|A→C|-0.29797 ± 0.02458|-0.21904 ± 0.06917|0.72651|
|T/tuji/D|C→D|-0.12955 ± 0.01884|-0.12047 ± 0.02771|1.19812|
|T/tuji/E|D→E|-0.01454 ± 0.04438|-0.01429 ± 0.02634|0.75820|
|T/uji/A|O→A|-0.34605 ± 0.30244|-0.16508 ± 0.39778|13.09551|
|T/uji/C|A→C|-0.04762 ± 0.14811|-0.09587 ± 0.38151|20.00724|
|T/uji/D|C→D|-0.26202 ± 0.39242|-0.00289 ± 0.35197|7.52315|
|T/uji/E|D→E|+0.32788 ± 0.57792|+0.25720 ± 0.53149|28.92124|
|T/uts/A|O→A|-0.41857 ± 0.18059|-0.25256 ± 0.17572|10.19219|
|T/uts/C|A→C|-0.13156 ± 0.25900|-0.03601 ± 0.11254|3.59407|
|T/uts/D|C→D|-0.16494 ± 0.14592|-0.03459 ± 0.14422|8.11234|
|T/uts/E|D→E|-0.00913 ± 0.02191|-0.00925 ± 0.10184|5.07389|

## 每次运行的固定选择

|阶段/域/种子/骨干|方案|epoch|参数|
|---|---|---:|---|
|F/tuji/17301/raw_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17301/raw_neighbors|historical_temperature_only|80|{"temperature": 0.611002504825592, "theta": [0.0, 0.0]}|
|F/tuji/17301/raw_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [1.133952260017395, 1.5]}|
|F/tuji/17301/raw_neighbors|historical_temperature_plus_g13|80|{"temperature": 0.7328513264656067, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|joint_calibrated_base|80|{"temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 5.0765485763549805, "structure": "1A", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1B_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.207306385040283, "structure": "1B", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1C_global|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.28493213653564453, "log_partition_gap_mean": -0.7677780985832214, "normalization": "global", "objective": 5.12201452255249, "structure": "1C", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1D_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 5.0765485763549805, "structure": "1D", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 5.0765485763549805, "structure": "1A", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1B_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.207306385040283, "structure": "1B", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1C_bf_mass|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.28493213653564453, "log_partition_gap_mean": -0.7677780985832214, "normalization": "bf_mass", "objective": 5.122014999389648, "structure": "1C", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|1D_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 5.0765485763549805, "structure": "1D", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.815235614776611, "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.815235614776611, "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.815235614776611, "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/raw_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.815235614776611, "temperature": 0.6217843672075842, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|historical_raw|76|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17301/same_floor_neighbors|historical_temperature_only|76|{"temperature": 0.5604634881019592, "theta": [0.0, 0.0]}|
|F/tuji/17301/same_floor_neighbors|historical_g13_fixed_t06|76|{"temperature": 0.6, "theta": [1.2124595642089844, 1.5]}|
|F/tuji/17301/same_floor_neighbors|historical_temperature_plus_g13|76|{"temperature": 0.6640579104423523, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|joint_calibrated_base|76|{"temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|joint_calibrated_base_bf_mass|76|{"normalization": "bf_mass", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1A_global|76|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 5.11099100112915, "structure": "1A", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1B_global|76|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.217132568359375, "structure": "1B", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1C_global|76|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.2208089828491211, "log_partition_gap_mean": -0.7761206030845642, "normalization": "global", "objective": 5.150447845458984, "structure": "1C", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1D_global|76|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 5.11099100112915, "structure": "1D", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1A_bf_mass|76|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 5.11099100112915, "structure": "1A", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1B_bf_mass|76|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.217132568359375, "structure": "1B", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1C_bf_mass|76|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.2208089828491211, "log_partition_gap_mean": -0.7761206030845642, "normalization": "bf_mass", "objective": 5.150447845458984, "structure": "1C", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|1D_bf_mass|76|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 5.110991477966309, "structure": "1D", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2A|76|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2A_geometric_diagnostic|76|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2B|76|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.748519420623779, "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2B_geometric_diagnostic|76|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.748519420623779, "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2A_beta_only_control|76|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.748519420623779, "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17301/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|76|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.748519420623779, "temperature": 0.5664616358132669, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|historical_raw|79|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17302/raw_neighbors|historical_temperature_only|79|{"temperature": 0.4841809868812561, "theta": [0.0, 0.0]}|
|F/tuji/17302/raw_neighbors|historical_g13_fixed_t06|79|{"temperature": 0.6, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|historical_temperature_plus_g13|79|{"temperature": 0.5657593607902527, "theta": [1.4362850189208984, 1.5]}|
|F/tuji/17302/raw_neighbors|joint_calibrated_base|79|{"temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|joint_calibrated_base_bf_mass|79|{"normalization": "bf_mass", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1A_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 4.995786666870117, "structure": "1A", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1B_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.080052375793457, "structure": "1B", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1C_global|79|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.22317028045654297, "log_partition_gap_mean": -0.6098145842552185, "normalization": "global", "objective": 5.024787902832031, "structure": "1C", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1D_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 4.995786666870117, "structure": "1D", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1A_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 4.995786666870117, "structure": "1A", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1B_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.080052375793457, "structure": "1B", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1C_bf_mass|79|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.22317028045654297, "log_partition_gap_mean": -0.6098145842552185, "normalization": "bf_mass", "objective": 5.024787902832031, "structure": "1C", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|1D_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 4.995786666870117, "structure": "1D", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2A|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2A_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2B|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.614036560058594, "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2B_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.614036560058594, "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2A_beta_only_control|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.614036560058594, "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/raw_neighbors|2A_beta_only_control_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.614036560058594, "temperature": 0.48497759361534737, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17302/same_floor_neighbors|historical_temperature_only|80|{"temperature": 0.5484191179275513, "theta": [0.0, 0.0]}|
|F/tuji/17302/same_floor_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [0.5390226244926453, 1.5]}|
|F/tuji/17302/same_floor_neighbors|historical_temperature_plus_g13|80|{"temperature": 0.6478921175003052, "theta": [1.0119606256484985, 1.5]}|
|F/tuji/17302/same_floor_neighbors|joint_calibrated_base|80|{"temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 4.9207234382629395, "structure": "1A", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1B_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.023388385772705, "structure": "1B", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1C_global|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.26943492889404297, "log_partition_gap_mean": -0.7458484768867493, "normalization": "global", "objective": 4.952006816864014, "structure": "1C", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1D_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 4.9207234382629395, "structure": "1D", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 4.920723915100098, "structure": "1A", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1B_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.023388385772705, "structure": "1B", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1C_bf_mass|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.26943492889404297, "log_partition_gap_mean": -0.7458484768867493, "normalization": "bf_mass", "objective": 4.952006816864014, "structure": "1C", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|1D_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 4.920723915100098, "structure": "1D", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.68287992477417, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.68287992477417, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.68287992477417, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17302/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.68287992477417, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17303/raw_neighbors|historical_temperature_only|80|{"temperature": 0.5498513579368591, "theta": [0.0, 0.0]}|
|F/tuji/17303/raw_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [1.364551305770874, 1.5]}|
|F/tuji/17303/raw_neighbors|historical_temperature_plus_g13|80|{"temperature": 0.6541863679885864, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|joint_calibrated_base|80|{"temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 4.963794708251953, "structure": "1A", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1B_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.053244590759277, "structure": "1B", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1C_global|80|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.2791261672973633, "log_partition_gap_mean": -0.6404188275337219, "normalization": "global", "objective": 4.9951090812683105, "structure": "1C", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1D_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 4.963794708251953, "structure": "1D", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 4.963794708251953, "structure": "1A", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1B_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.053244590759277, "structure": "1B", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1C_bf_mass|80|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.2791261672973633, "log_partition_gap_mean": -0.6404188275337219, "normalization": "bf_mass", "objective": 4.9951090812683105, "structure": "1C", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|1D_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 4.963794708251953, "structure": "1D", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.585483551025391, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.585483551025391, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.585483551025391, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/raw_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.585483551025391, "temperature": 0.5606268759834927, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/tuji/17303/same_floor_neighbors|historical_temperature_only|80|{"temperature": 0.5332674980163574, "theta": [0.0, 0.0]}|
|F/tuji/17303/same_floor_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|historical_temperature_plus_g13|80|{"temperature": 0.6262767314910889, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|joint_calibrated_base|80|{"temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 5.022280216217041, "structure": "1A", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1B_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 5.1130876541137695, "structure": "1B", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1C_global|80|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.24913692474365234, "log_partition_gap_mean": -0.6007514595985413, "normalization": "global", "objective": 5.05515193939209, "structure": "1C", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1D_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 5.022280216217041, "structure": "1D", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 5.022280216217041, "structure": "1A", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1B_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 5.1130876541137695, "structure": "1B", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1C_bf_mass|80|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.24913692474365234, "log_partition_gap_mean": -0.6007514595985413, "normalization": "bf_mass", "objective": 5.05515193939209, "structure": "1C", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|1D_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 5.022280216217041, "structure": "1D", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.627554416656494, "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.627554416656494, "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 4.627554416656494, "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/tuji/17303/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 4.627554416656494, "temperature": 0.5323422930893447, "theta": [1.5, 1.5]}|
|F/uji/17301/raw_neighbors|historical_raw|57|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17301/raw_neighbors|historical_temperature_only|57|{"temperature": 1.1786034107208252, "theta": [0.0, 0.0]}|
|F/uji/17301/raw_neighbors|historical_g13_fixed_t06|57|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17301/raw_neighbors|historical_temperature_plus_g13|57|{"temperature": 1.4413423538208008, "theta": [1.5, 0.0]}|
|F/uji/17301/raw_neighbors|joint_calibrated_base|57|{"temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|joint_calibrated_base_bf_mass|57|{"normalization": "bf_mass", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1A_global|57|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.453275680541992, "structure": "1A", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1B_global|57|{"alpha_obs": 0.8, "alpha_prior": 0.0, "normalization": "global", "objective": 17.590957641601562, "structure": "1B", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1C_global|57|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.4064769744873047, "log_partition_gap_mean": -1.508196473121643, "normalization": "global", "objective": 17.461233139038086, "structure": "1C", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1D_global|57|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.453275680541992, "structure": "1D", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1A_bf_mass|57|{"alpha_obs": 0.0, "alpha_prior": 0.8, "normalization": "bf_mass", "objective": 18.0120849609375, "structure": "1A", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1B_bf_mass|57|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 18.185100555419922, "structure": "1B", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1C_bf_mass|57|{"alpha_obs": 0.65, "alpha_prior": 0.65, "log_partition_gap_max": -0.5199108123779297, "log_partition_gap_mean": -1.908980131149292, "normalization": "bf_mass", "objective": 17.99778175354004, "structure": "1C", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|1D_bf_mass|57|{"alpha_obs": 0.8, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 17.997276306152344, "structure": "1D", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2A|57|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2A_geometric_diagnostic|57|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2B|57|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 13.93349552154541, "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2B_geometric_diagnostic|57|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 13.93349552154541, "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2A_beta_only_control|57|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 13.93349552154541, "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/raw_neighbors|2A_beta_only_control_geometric_diagnostic|57|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 13.93349552154541, "temperature": 0.8660254037844386, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|historical_raw|38|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17301/same_floor_neighbors|historical_temperature_only|38|{"temperature": 1.3473620414733887, "theta": [0.0, 0.0]}|
|F/uji/17301/same_floor_neighbors|historical_g13_fixed_t06|38|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17301/same_floor_neighbors|historical_temperature_plus_g13|38|{"temperature": 1.6726455688476562, "theta": [1.5, 0.0]}|
|F/uji/17301/same_floor_neighbors|joint_calibrated_base|38|{"temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|joint_calibrated_base_bf_mass|38|{"normalization": "bf_mass", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1A_global|38|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.544179916381836, "structure": "1A", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1B_global|38|{"alpha_obs": 0.65, "alpha_prior": 0.0, "normalization": "global", "objective": 17.639686584472656, "structure": "1B", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1C_global|38|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.12853431701660156, "log_partition_gap_mean": -1.1254310607910156, "normalization": "global", "objective": 17.558536529541016, "structure": "1C", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1D_global|38|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.544179916381836, "structure": "1D", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1A_bf_mass|38|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 17.857051849365234, "structure": "1A", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1B_bf_mass|38|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 17.96590805053711, "structure": "1B", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1C_bf_mass|38|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.12853431701660156, "log_partition_gap_mean": -1.1254310607910156, "normalization": "bf_mass", "objective": 17.866445541381836, "structure": "1C", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|1D_bf_mass|38|{"alpha_obs": 0.1, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 17.856643676757812, "structure": "1D", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2A|38|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2A_geometric_diagnostic|38|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2B|38|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.01, "normalization": "global", "objective": 13.823511123657227, "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2B_geometric_diagnostic|38|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.01, "normalization": "global", "objective": 13.823511123657227, "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2A_beta_only_control|38|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 13.824054718017578, "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17301/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|38|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 13.824054718017578, "temperature": 0.8482765367941033, "theta": [1.5, 0.5625]}|
|F/uji/17302/raw_neighbors|historical_raw|39|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17302/raw_neighbors|historical_temperature_only|39|{"temperature": 1.263716697692871, "theta": [0.0, 0.0]}|
|F/uji/17302/raw_neighbors|historical_g13_fixed_t06|39|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17302/raw_neighbors|historical_temperature_plus_g13|39|{"temperature": 1.5143295526504517, "theta": [1.4750863313674927, 0.0]}|
|F/uji/17302/raw_neighbors|joint_calibrated_base|39|{"temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|joint_calibrated_base_bf_mass|39|{"normalization": "bf_mass", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1A_global|39|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.81513214111328, "structure": "1A", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1B_global|39|{"alpha_obs": 0.65, "alpha_prior": 0.0, "normalization": "global", "objective": 17.933557510375977, "structure": "1B", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1C_global|39|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.18100929260253906, "log_partition_gap_mean": -1.2737228870391846, "normalization": "global", "objective": 17.8338680267334, "structure": "1C", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1D_global|39|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.81513214111328, "structure": "1D", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1A_bf_mass|39|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 18.269699096679688, "structure": "1A", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1B_bf_mass|39|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 18.339393615722656, "structure": "1B", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1C_bf_mass|39|{"alpha_obs": 0.65, "alpha_prior": 0.65, "log_partition_gap_max": -0.22289085388183594, "log_partition_gap_mean": -1.6115248203277588, "normalization": "bf_mass", "objective": 18.254852294921875, "structure": "1C", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|1D_bf_mass|39|{"alpha_obs": 0.5, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 18.25351905822754, "structure": "1D", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2A|39|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2A_geometric_diagnostic|39|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2B|39|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.102289199829102, "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2B_geometric_diagnostic|39|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.102289199829102, "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2A_beta_only_control|39|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.102289199829102, "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/raw_neighbors|2A_beta_only_control_geometric_diagnostic|39|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.102289199829102, "temperature": 1.011531611649396, "theta": [1.5, 0.8125]}|
|F/uji/17302/same_floor_neighbors|historical_raw|40|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17302/same_floor_neighbors|historical_temperature_only|40|{"temperature": 1.3016905784606934, "theta": [0.0, 0.0]}|
|F/uji/17302/same_floor_neighbors|historical_g13_fixed_t06|40|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17302/same_floor_neighbors|historical_temperature_plus_g13|40|{"temperature": 1.6184216737747192, "theta": [1.5, 0.0]}|
|F/uji/17302/same_floor_neighbors|joint_calibrated_base|40|{"temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|joint_calibrated_base_bf_mass|40|{"normalization": "bf_mass", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1A_global|40|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 17.527435302734375, "structure": "1A", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1B_global|40|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "global", "objective": 17.580989837646484, "structure": "1B", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1C_global|40|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.05073833465576172, "log_partition_gap_mean": -0.7954939007759094, "normalization": "global", "objective": 17.531139373779297, "structure": "1C", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1D_global|40|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 17.527435302734375, "structure": "1D", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1A_bf_mass|40|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 17.804828643798828, "structure": "1A", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1B_bf_mass|40|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 17.85641098022461, "structure": "1B", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1C_bf_mass|40|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.06960010528564453, "log_partition_gap_mean": -1.1106481552124023, "normalization": "bf_mass", "objective": 17.802143096923828, "structure": "1C", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|1D_bf_mass|40|{"alpha_obs": 0.35, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 17.801376342773438, "structure": "1D", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2A|40|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2A_geometric_diagnostic|40|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2B|40|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 13.907004356384277, "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2B_geometric_diagnostic|40|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 13.907004356384277, "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2A_beta_only_control|40|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 13.907004356384277, "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17302/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|40|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 13.907004356384277, "temperature": 0.8054795374885969, "theta": [1.5, 0.5]}|
|F/uji/17303/raw_neighbors|historical_raw|53|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17303/raw_neighbors|historical_temperature_only|53|{"temperature": 1.2331867218017578, "theta": [0.0, 0.0]}|
|F/uji/17303/raw_neighbors|historical_g13_fixed_t06|53|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17303/raw_neighbors|historical_temperature_plus_g13|53|{"temperature": 1.4804879426956177, "theta": [1.5, 0.0]}|
|F/uji/17303/raw_neighbors|joint_calibrated_base|53|{"temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|joint_calibrated_base_bf_mass|53|{"normalization": "bf_mass", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1A_global|53|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.83376693725586, "structure": "1A", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1B_global|53|{"alpha_obs": 0.8, "alpha_prior": 0.0, "normalization": "global", "objective": 18.005006790161133, "structure": "1B", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1C_global|53|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.22470664978027344, "log_partition_gap_mean": -1.3609061241149902, "normalization": "global", "objective": 17.859521865844727, "structure": "1C", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1D_global|53|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "global", "objective": 17.83376693725586, "structure": "1D", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1A_bf_mass|53|{"alpha_obs": 0.0, "alpha_prior": 0.8, "normalization": "bf_mass", "objective": 18.2161922454834, "structure": "1A", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1B_bf_mass|53|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 18.38486099243164, "structure": "1B", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1C_bf_mass|53|{"alpha_obs": 0.65, "alpha_prior": 0.65, "log_partition_gap_max": -0.28517627716064453, "log_partition_gap_mean": -1.7203065156936646, "normalization": "bf_mass", "objective": 18.207706451416016, "structure": "1C", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|1D_bf_mass|53|{"alpha_obs": 0.65, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 18.207706451416016, "structure": "1D", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2A|53|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2A_geometric_diagnostic|53|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2B|53|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.120077133178711, "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2B_geometric_diagnostic|53|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.120077133178711, "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2A_beta_only_control|53|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.120077133178711, "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/raw_neighbors|2A_beta_only_control_geometric_diagnostic|53|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.120077133178711, "temperature": 0.9908006493089963, "theta": [1.5, 0.625]}|
|F/uji/17303/same_floor_neighbors|historical_raw|42|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uji/17303/same_floor_neighbors|historical_temperature_only|42|{"temperature": 1.3155409097671509, "theta": [0.0, 0.0]}|
|F/uji/17303/same_floor_neighbors|historical_g13_fixed_t06|42|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uji/17303/same_floor_neighbors|historical_temperature_plus_g13|42|{"temperature": 1.6203944683074951, "theta": [1.5, 0.0]}|
|F/uji/17303/same_floor_neighbors|joint_calibrated_base|42|{"temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|joint_calibrated_base_bf_mass|42|{"normalization": "bf_mass", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1A_global|42|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 17.720989227294922, "structure": "1A", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1B_global|42|{"alpha_obs": 0.65, "alpha_prior": 0.0, "normalization": "global", "objective": 17.813385009765625, "structure": "1B", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1C_global|42|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.10655593872070312, "log_partition_gap_mean": -0.7981484532356262, "normalization": "global", "objective": 17.735248565673828, "structure": "1C", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1D_global|42|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 17.720989227294922, "structure": "1D", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1A_bf_mass|42|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 18.085683822631836, "structure": "1A", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1B_bf_mass|42|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 18.131790161132812, "structure": "1B", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1C_bf_mass|42|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.14113378524780273, "log_partition_gap_mean": -1.1136380434036255, "normalization": "bf_mass", "objective": 18.074800491333008, "structure": "1C", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|1D_bf_mass|42|{"alpha_obs": 0.5, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 18.074800491333008, "structure": "1D", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2A|42|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2A_geometric_diagnostic|42|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2B|42|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.052906036376953, "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2B_geometric_diagnostic|42|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.052906036376953, "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2A_beta_only_control|42|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 14.052906036376953, "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uji/17303/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|42|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 14.052906036376953, "temperature": 0.9408130652357323, "theta": [1.5, 0.8125]}|
|F/uts/17301/raw_neighbors|historical_raw|79|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17301/raw_neighbors|historical_temperature_only|79|{"temperature": 1.146872878074646, "theta": [0.0, 0.0]}|
|F/uts/17301/raw_neighbors|historical_g13_fixed_t06|79|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17301/raw_neighbors|historical_temperature_plus_g13|79|{"temperature": 1.1463639736175537, "theta": [0.0, 0.0]}|
|F/uts/17301/raw_neighbors|joint_calibrated_base|79|{"temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|joint_calibrated_base_bf_mass|79|{"normalization": "bf_mass", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1A_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.690949440002441, "structure": "1A", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1B_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 12.802773475646973, "structure": "1B", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1C_global|79|{"alpha_obs": 0.35, "alpha_prior": 0.35, "log_partition_gap_max": -0.7089376449584961, "log_partition_gap_mean": -1.9938836097717285, "normalization": "global", "objective": 12.746253967285156, "structure": "1C", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1D_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.690949440002441, "structure": "1D", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1A_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 13.717389106750488, "structure": "1A", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1B_bf_mass|79|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 13.746565818786621, "structure": "1B", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1C_bf_mass|79|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.9647254943847656, "log_partition_gap_mean": -2.7661690711975098, "normalization": "bf_mass", "objective": 13.685356140136719, "structure": "1C", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|1D_bf_mass|79|{"alpha_obs": 0.8, "alpha_prior": 0.35, "normalization": "bf_mass", "objective": 13.684033393859863, "structure": "1D", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2A|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2A_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2B|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.994174003601074, "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2B_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.994174003601074, "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2A_beta_only_control|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.994174003601074, "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/raw_neighbors|2A_beta_only_control_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.994174003601074, "temperature": 0.9311223502193782, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17301/same_floor_neighbors|historical_temperature_only|80|{"temperature": 1.2586376667022705, "theta": [0.0, 0.0]}|
|F/uts/17301/same_floor_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17301/same_floor_neighbors|historical_temperature_plus_g13|80|{"temperature": 1.6574275493621826, "theta": [1.1785982847213745, 0.8178867101669312]}|
|F/uts/17301/same_floor_neighbors|joint_calibrated_base|80|{"temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.2, "normalization": "global", "objective": 12.350576400756836, "structure": "1A", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1B_global|80|{"alpha_obs": 0.35, "alpha_prior": 0.0, "normalization": "global", "objective": 12.356151580810547, "structure": "1B", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1C_global|80|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2254047393798828, "log_partition_gap_mean": -0.7172291278839111, "normalization": "global", "objective": 12.349296569824219, "structure": "1C", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1D_global|80|{"alpha_obs": 0.1, "alpha_prior": 0.2, "normalization": "global", "objective": 12.348770141601562, "structure": "1D", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.35, "normalization": "bf_mass", "objective": 12.543471336364746, "structure": "1A", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1B_bf_mass|80|{"alpha_obs": 0.5, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 12.549060821533203, "structure": "1B", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1C_bf_mass|80|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2254047393798828, "log_partition_gap_mean": -0.7172291278839111, "normalization": "bf_mass", "objective": 12.540454864501953, "structure": "1C", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|1D_bf_mass|80|{"alpha_obs": 0.2, "alpha_prior": 0.2, "normalization": "bf_mass", "objective": 12.540454864501953, "structure": "1D", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.569751739501953, "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.569751739501953, "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.569751739501953, "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17301/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.569751739501953, "temperature": 0.9704945602939673, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|historical_raw|80|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17302/raw_neighbors|historical_temperature_only|80|{"temperature": 1.2049964666366577, "theta": [0.0, 0.0]}|
|F/uts/17302/raw_neighbors|historical_g13_fixed_t06|80|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17302/raw_neighbors|historical_temperature_plus_g13|80|{"temperature": 1.2050461769104004, "theta": [0.0, 0.0]}|
|F/uts/17302/raw_neighbors|joint_calibrated_base|80|{"temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|joint_calibrated_base_bf_mass|80|{"normalization": "bf_mass", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1A_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.542550086975098, "structure": "1A", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1B_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 12.625219345092773, "structure": "1B", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1C_global|80|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.40630340576171875, "log_partition_gap_mean": -1.205275058746338, "normalization": "global", "objective": 12.582820892333984, "structure": "1C", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1D_global|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.542550086975098, "structure": "1D", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1A_bf_mass|80|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 13.50255298614502, "structure": "1A", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1B_bf_mass|80|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 13.560280799865723, "structure": "1B", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1C_bf_mass|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -0.9626646041870117, "log_partition_gap_mean": -2.8858654499053955, "normalization": "bf_mass", "objective": 13.48222541809082, "structure": "1C", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|1D_bf_mass|80|{"alpha_obs": 0.5, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 13.48222541809082, "structure": "1D", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2A|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2A_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2B|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.752162456512451, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2B_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.752162456512451, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2A_beta_only_control|80|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.752162456512451, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/raw_neighbors|2A_beta_only_control_geometric_diagnostic|80|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.752162456512451, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17302/same_floor_neighbors|historical_raw|79|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17302/same_floor_neighbors|historical_temperature_only|79|{"temperature": 1.221168875694275, "theta": [0.0, 0.0]}|
|F/uts/17302/same_floor_neighbors|historical_g13_fixed_t06|79|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17302/same_floor_neighbors|historical_temperature_plus_g13|79|{"temperature": 1.5400011539459229, "theta": [1.2104976177215576, 0.6477052569389343]}|
|F/uts/17302/same_floor_neighbors|joint_calibrated_base|79|{"temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|joint_calibrated_base_bf_mass|79|{"normalization": "bf_mass", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1A_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.35, "normalization": "global", "objective": 12.244882583618164, "structure": "1A", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1B_global|79|{"alpha_obs": 0.5, "alpha_prior": 0.0, "normalization": "global", "objective": 12.253335952758789, "structure": "1B", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1C_global|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2492809295654297, "log_partition_gap_mean": -0.7674033045768738, "normalization": "global", "objective": 12.243520736694336, "structure": "1C", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1D_global|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "normalization": "global", "objective": 12.243520736694336, "structure": "1D", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1A_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.35, "normalization": "bf_mass", "objective": 12.502650260925293, "structure": "1A", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1B_bf_mass|79|{"alpha_obs": 0.5, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 12.511706352233887, "structure": "1B", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1C_bf_mass|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2492809295654297, "log_partition_gap_mean": -0.7674033045768738, "normalization": "bf_mass", "objective": 12.502062797546387, "structure": "1C", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|1D_bf_mass|79|{"alpha_obs": 0.35, "alpha_prior": 0.2, "normalization": "bf_mass", "objective": 12.500160217285156, "structure": "1D", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2A|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2A_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2B|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.517993927001953, "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2B_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.517993927001953, "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2A_beta_only_control|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.517993927001953, "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17302/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.517993927001953, "temperature": 0.8054795374885969, "theta": [1.5, 1.25]}|
|F/uts/17303/raw_neighbors|historical_raw|79|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17303/raw_neighbors|historical_temperature_only|79|{"temperature": 1.129210114479065, "theta": [0.0, 0.0]}|
|F/uts/17303/raw_neighbors|historical_g13_fixed_t06|79|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17303/raw_neighbors|historical_temperature_plus_g13|79|{"temperature": 1.12884521484375, "theta": [0.0, 0.0]}|
|F/uts/17303/raw_neighbors|joint_calibrated_base|79|{"temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|joint_calibrated_base_bf_mass|79|{"normalization": "bf_mass", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1A_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.667198181152344, "structure": "1A", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1B_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.0, "normalization": "global", "objective": 12.766201972961426, "structure": "1B", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1C_global|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.44199562072753906, "log_partition_gap_mean": -1.1303044557571411, "normalization": "global", "objective": 12.725683212280273, "structure": "1C", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1D_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.5, "normalization": "global", "objective": 12.667198181152344, "structure": "1D", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1A_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.65, "normalization": "bf_mass", "objective": 13.752615928649902, "structure": "1A", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1B_bf_mass|79|{"alpha_obs": 1.0, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 13.822366714477539, "structure": "1B", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1C_bf_mass|79|{"alpha_obs": 0.5, "alpha_prior": 0.5, "log_partition_gap_max": -1.0063953399658203, "log_partition_gap_mean": -2.6774144172668457, "normalization": "bf_mass", "objective": 13.73922348022461, "structure": "1C", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|1D_bf_mass|79|{"alpha_obs": 0.5, "alpha_prior": 0.5, "normalization": "bf_mass", "objective": 13.73922348022461, "structure": "1D", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2A|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2A_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2B|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.8927717208862305, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2B_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.8927717208862305, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2A_beta_only_control|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.8927717208862305, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/raw_neighbors|2A_beta_only_control_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.8927717208862305, "temperature": 0.8933474401087248, "theta": [1.5, 1.5]}|
|F/uts/17303/same_floor_neighbors|historical_raw|79|{"temperature": 1.0, "theta": [0.0, 0.0]}|
|F/uts/17303/same_floor_neighbors|historical_temperature_only|79|{"temperature": 1.3157079219818115, "theta": [0.0, 0.0]}|
|F/uts/17303/same_floor_neighbors|historical_g13_fixed_t06|79|{"temperature": 0.6, "theta": [0.0, 0.0]}|
|F/uts/17303/same_floor_neighbors|historical_temperature_plus_g13|79|{"temperature": 1.4698737859725952, "theta": [0.02916187047958374, 0.624091625213623]}|
|F/uts/17303/same_floor_neighbors|joint_calibrated_base|79|{"temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|joint_calibrated_base_bf_mass|79|{"normalization": "bf_mass", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1A_global|79|{"alpha_obs": 0.0, "alpha_prior": 0.35, "normalization": "global", "objective": 12.232105255126953, "structure": "1A", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1B_global|79|{"alpha_obs": 0.35, "alpha_prior": 0.0, "normalization": "global", "objective": 12.244308471679688, "structure": "1B", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1C_global|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2113800048828125, "log_partition_gap_mean": -0.7088437676429749, "normalization": "global", "objective": 12.23202133178711, "structure": "1C", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1D_global|79|{"alpha_obs": 0.05, "alpha_prior": 0.35, "normalization": "global", "objective": 12.23175048828125, "structure": "1D", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1A_bf_mass|79|{"alpha_obs": 0.0, "alpha_prior": 0.35, "normalization": "bf_mass", "objective": 12.374175071716309, "structure": "1A", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1B_bf_mass|79|{"alpha_obs": 0.35, "alpha_prior": 0.0, "normalization": "bf_mass", "objective": 12.386524200439453, "structure": "1B", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1C_bf_mass|79|{"alpha_obs": 0.2, "alpha_prior": 0.2, "log_partition_gap_max": -0.2113800048828125, "log_partition_gap_mean": -0.7088437676429749, "normalization": "bf_mass", "objective": 12.373516082763672, "structure": "1C", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|1D_bf_mass|79|{"alpha_obs": 0.1, "alpha_prior": 0.35, "normalization": "bf_mass", "objective": 12.37234115600586, "structure": "1D", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2A|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2A_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2B|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.494498252868652, "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2B_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.494498252868652, "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2A_beta_only_control|79|{"beta": 0.8, "center_mode": "residual_interpolated", "gamma_space": 0.0, "normalization": "global", "objective": 7.494498252868652, "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|F/uts/17303/same_floor_neighbors|2A_beta_only_control_geometric_diagnostic|79|{"beta": 0.8, "center_mode": "geometric_diagnostic_only", "gamma_space": 0.0, "normalization": "global", "objective": 7.494498252868652, "temperature": 0.9215314530743932, "theta": [1.25, 1.375]}|
|T/tuji/17301/A|A|76|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17301/C|C|77|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17301/D|D|63|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17301/E|E|62|{"alpha": 0.05443807691335678, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17301/O|O|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17302/A|A|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17302/C|C|78|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17302/D|D|69|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17302/E|E|73|{"alpha": 0.049428004771471024, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17302/O|O|79|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17303/A|A|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17303/C|C|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17303/D|D|77|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17303/E|E|80|{"alpha": 0.04736290127038956, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/tuji/17303/O|O|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17301/A|A|38|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17301/C|C|38|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17301/D|D|33|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17301/E|E|27|{"alpha": 0.05191889405250549, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17301/O|O|57|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17302/A|A|40|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17302/C|C|45|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17302/D|D|55|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17302/E|E|37|{"alpha": 0.03542928025126457, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17302/O|O|39|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17303/A|A|42|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17303/C|C|36|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17303/D|D|56|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17303/E|E|47|{"alpha": 0.03382566198706627, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uji/17303/O|O|53|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17301/A|A|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17301/C|C|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17301/D|D|75|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17301/E|E|79|{"alpha": 0.052165593951940536, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17301/O|O|79|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17302/A|A|79|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17302/C|C|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17302/D|D|77|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17302/E|E|80|{"alpha": 0.05348854884505272, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17302/O|O|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17303/A|A|79|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17303/C|C|80|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17303/D|D|70|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17303/E|E|80|{"alpha": 0.05074629560112953, "temperature": 1.0, "theta": [0.0, 0.0]}|
|T/uts/17303/O|O|79|{"alpha": 0.0, "temperature": 1.0, "theta": [0.0, 0.0]}|

## 数值审计与解释限制

INTEGRITY_REVIEW.json独立从逐站点数据重算宏指标并回放选中epoch，所有硬断言通过；单次命令、原始stdout/stderr、退出码及耗时见EXECUTION_RECORD.json。
完整图按建筑/楼层隔离、对称且双随机；BF保持模式仅保留未缩放原始scores的类别质量。1C逐扫描logZ差记录在RESULT.json，其非正性不保证MAE改善。
虚拟分量为N个唯一父锚点加E条唯一无向边，无填充；beta=0物理证据/后验严格恒等，扩展NLL/ES allclose；q<=0组件严格排除。
中点RSS原型和Key插值后仍调用原神经/原型距离法则，虚拟G13使用物理参考均值/RMS；gamma以归一化RSS平方计量且不额外上截断。
纯几何中点仅为诊断；2A主对照复用2B选中的beta，额外beta-only控制另列，不能从CAL-eval挑选。
TUJI原O/A邻居ID相同但权重有约3e-8舍入差；CUDA确定性算法未开启，小差异不得解释成跨楼层监督修复的因果作用。
已有训练语料与历史内部评价已经被观察，空间相邻、扫描/设备相关、MC误差及仅三个随机种子限制推论。禁止与历史E5不同划分MAE直接作同协议优越性声明。
本报告不实施新模型或自动开启后续阶段；是否继续应结合均值、尾部、BF错误和跨域一致性判断。

## 执行收尾与授权例外

正式 Phase F 共18组骨干成功（含保留的TUJI六组和用户授权恢复的UTS一组）。原UTS失败尝试及锁原样保留，不能宣称所有原始命令都一次成功。
用户于2026-10-19明确允许最小ES数值修复并继续。USER_AUTHORIZATION_20261019.json、ENGINEERING_REVISION.json和run_logs/authorized_es_and_resume.diff保留完整修订链；原EXECUTION_MANIFEST.json与33份既有F产物逐字不变。
ES修复只在内部采样时忽略全零质量列，输入完整N+E混合仍用于概率指标；不改变任何正权重、中心/尺度、抽样种子、网格、容差或分区。历史ES科学代码未改。TUJI已完成六组保留原结果，不重复评分。
失败UTS恢复严格使用原CAL_FIT_SELECTION.json，SHA256为2b4db10a47dfe653be8fe62939768f485b638311d58f7cdde206ac3955aacaaf；没有重新拟合或重选。原失败前部分CAL-eval指标未持久化，未编造或补写。
首次CUDA父G13布局修复、原beta=0 ES故障、失败回归及pytest临时目录问题均见FIX_RECORD.md，原始stdout/stderr从未覆盖。修复后完整50项回归通过；包含默认路径损失/梯度/预测逐位一致和一次编码器前向复用检查。
Phase T完成45条80轮历史/新训练记录：18条O/A严格核验后复用，27条C/D/E为本轮真实新训练，共2160新轮次。不得将复用的1440历史轮次包装成本轮新训练。各臂/种子仅训练一次，未早停、未延长预算。
批量64、AdamW、原增强与配对种子固定。C/D/E全锚点连续头是预注册损失所需计算；不声称与旧Top32训练的FLOPs或真实GPU时间相同。CUDA确定性算法未启用。

|命令标签|状态|退出码|墙钟秒|
|---|---|---:|---:|
|operators_initial|COMPLETE|0|10.557|
|runner_compile_initial|COMPLETE|0|0.102|
|runner_tests_initial|COMPLETE|0|7.209|
|runner_tests_complete|COMPLETE|0|7.230|
|preflight_initial|FAILED|1|11.877|
|cuda_field_regression_before_fix|COMPLETE|0|5.406|
|cuda_field_regression_shape_before_fix|FAILED|1|5.589|
|regression_after_field_fix|COMPLETE|0|7.581|
|final_compile|COMPLETE|0|0.113|
|preflight_fixed|COMPLETE|0|38.540|
|tests_before_frozen|COMPLETE|0|7.557|
|frozen_phase|FAILED|1|68.199|
|beta_zero_candidate_synthetic_tests|COMPLETE|0|5.433|
|stopped_readonly_audit|COMPLETE|0|4.670|
|beta_zero_candidate_full_synthetic_tests|COMPLETE|0|5.655|
|authorized_revision_tests|FAILED|1|11.573|
|authorized_revision_tests_fixed_path|COMPLETE|0|8.062|
|authorized_revision_compile|COMPLETE|0|0.089|
|frozen_authorized_resume|COMPLETE|0|265.736|
|phase_T|COMPLETE|0|13162.992|
|aggregate|COMPLETE|0|9.809|
|closeout_helpers_compile|COMPLETE|0|0.104|
|independent_closeout|COMPLETE|0|7.824|

上述科学阶段命令合计墙钟 225.11 分钟（包括失败F尝试，排除等待用户授权的暂停时间）。

## 数据和站点身份

|域|FIT站点/扫描|DEV站点/扫描|CAL-fit站点/扫描|CAL-eval站点/扫描|TEST站点/扫描（仅原角色元数据）|
|---|---:|---:|---:|---:|---:|
|tuji|646/3236|91/453|45/215|46/234|91/457|
|uts|1052/6534|138/873|69/444|69/414|138/843|
|uji|672/14453|87/1828|43/959|44/919|87/1778|

原训练容器坐标仅用于复原不可变角色键；没有建立TEST推断包、加载官方测试用于实验或计算其性能。TUJI物理MAT容器包含测试变量，但只解码results_train。

7份输入完整SHA256、全部角色键SHA见EXECUTION_MANIFEST.json；45份训练链检查点完整SHA和选中epoch见INDEPENDENT_CLOSEOUT.json的checkpoint_identity。原冻结权重没有被改写。

## 预声明训练链的配对差

以下保留全部三个种子的均值±样本SD；负MAE/ES为改善。用于回顾性机制判断，不按CAL-eval选择新的训练方案。

|域|对照→候选|ΔMAE m|ΔES m|ΔP95 m|ΔP99 m|单站点最大伤害 m|
|---|---|---:|---:|---:|---:|---:|
|tuji|O→A|-0.021635 ± 0.061471|+0.018946 ± 0.019457|-0.082203 ± 0.174085|-0.143229 ± 0.222480|0.624474|
|tuji|A→C|-0.297970 ± 0.024584|-0.219044 ± 0.069172|-0.751544 ± 0.259517|-0.508696 ± 0.275171|0.726507|
|tuji|C→D|-0.129553 ± 0.018844|-0.120474 ± 0.027710|-0.313167 ± 0.088391|-0.890309 ± 0.173084|1.198122|
|tuji|D→E|-0.014538 ± 0.044384|-0.014293 ± 0.026339|+0.003935 ± 0.212916|+0.073819 ± 0.293177|0.758203|
|uts|O→A|-0.418571 ± 0.180594|-0.252561 ± 0.175717|+0.633070 ± 1.427749|+0.512110 ± 4.061470|10.192194|
|uts|A→C|-0.131564 ± 0.258997|-0.036012 ± 0.112536|-0.898345 ± 1.260451|-1.331528 ± 1.888976|3.594070|
|uts|C→D|-0.164944 ± 0.145918|-0.034595 ± 0.144215|-0.150093 ± 0.886366|-0.825141 ± 1.190700|8.112339|
|uts|D→E|-0.009127 ± 0.021908|-0.009248 ± 0.101843|-0.466912 ± 0.973969|-1.435069 ± 2.434347|5.073894|
|uji|O→A|-0.346053 ± 0.302440|-0.165080 ± 0.397775|-0.306725 ± 0.551514|-9.300305 ± 7.192227|13.095507|
|uji|A→C|-0.047623 ± 0.148113|-0.095874 ± 0.381508|+0.479916 ± 0.852464|+2.911634 ± 5.500196|20.007237|
|uji|C→D|-0.262023 ± 0.392415|-0.002893 ± 0.351969|-0.357858 ± 0.631806|-1.867397 ± 8.073496|7.523149|
|uji|D→E|+0.327879 ± 0.577924|+0.257197 ± 0.531489|+0.445838 ± 1.531888|+6.069797 ± 8.993476|28.921239|

## 独立审计的实际范围

独立只读核算全部63组结果、405条运行指标、135条三种子汇总、378条种子配对、20034条站点配对；全部JSON可解析且数值有限。
站点到宏指标最大浮点差为1.72700273016e-06；从C/D/E轮次日志重建损失公式的最大差为3.09880640259e-07。后者为序列化浮点均值审计，未放宽训练的数学不变量。
每条训练历史都是1–80轮，独立回放21–80最优DEV epoch；逐一核验检查点SHA、有限值、E的alpha、全网格确定性选择、beta=0门禁、Jensen界及原始BF质量保持。
194个历史源码/测试、7份原数据及全部原冻结产物仍匹配SHA；单独保存规范图/边ID的SHA256。原始逐扫描指标没有由该运行器输出，因此只声明从已保存站点指标重算宏指标，不能伪称独立从扫描重建站点。
内部TEST评分行数=0；官方测试评分行数=0；未进行任何Git操作。

## 额外跨阶段数值诊断：不完全恒等

独立审计状态为PASS_WITH_DISCLOSED_CROSS_PHASE_DIFFERENCES：协议强制门禁通过，但额外的跨阶段逐站点恒等检查没有成立，不能笼统称所有附加诊断通过。
18组相同O/A检查点的F与T结果：最大逐站点MAE差0.0006201863m（小于1mm），最大逐站点ES差0.09826636m；BF准确率逐站点完全相同。各组MAE宏差绝对值均小于2e-5m，ES宏差绝对值最大0.001526876m。完整向量见CROSS_PHASE_OA_NUMERICAL_REVIEW.json。
F以8条查询块构建包含虚拟场的pack，T以32条查询块构建原后验pack；CUDA非确定性/形状相关浮点前向与CDF舍入会影响离散MC抽样。已有JSON没有保存逐组件q/中心张量，因此只报告观察到的差异与实现事实，不声称已由新前向唯一证明物理原因。
该额外跨阶段断言并非协议要求的同pack beta=0数学不变量；失败原文保留，没有修改科学源码、容差、权重、结果或重跑评分来凑一致。F内beta=0的严格父权重/中心恒等及NLL/ES allclose仍全部通过。分别解释F/T内部配对结果，避免把跨阶段细小ES差当成模型收益。

## 科学判断：观察、反例与不能识别的贡献

1. **干净校准与空间干预应分开解释。** 新joint_calibrated_base同时改变旧校准目标中的BF掩码、分量运输风险，并使用锁定网格替代旧Adam拟合；与历史Stage-2控制的差异不能唯一归因于BF掩码。空间臂均固定该新基线T/theta。
2. **Structure 1未提供跨域一致的MAE收益。** 原骨干1C全局扩散相对干净基线的MAE变化为TUJI +0.027009m、UTS +0.054761m、UJI -0.024360m；同楼层骨干分别+0.039405m、+0.016410m、-0.048598m。Jensen logZ界全部满足，但这并未保证MAE更低。独立两通道或BF质量保持模式的全部负结果也保留。
3. **Structure 2表现为概率/点决策权衡。** 2B相对干净基线，在UTS原骨干平均MAE -0.113419m、ES -0.444474m；同楼层骨干却MAE +0.124916m、ES -0.365785m。TUJI的MAE也略恶化；不能据ES改善宣称连续定位精度已解决。18/18组beta都选中上界0.8；17/18组gamma=0，只有UJI同楼层17301选择0.01。该组在三个种子均值中使2B相对2A的MAE约+0.006807m、ES约-0.006255m，未支持gamma的稳定额外精度收益，也不据此扩展网格。
4. **C的解释不等于单独BF掩码效应。** TUJI只有一个BF类别，BF掩码不能在那里改变类别支持；C仍改善，说明原Top32训练NLL改为全锚点NLL等预注册支持变化也参与了对照。C整体替换方案可以比较，但不能把全部收益归因于消除错误楼层密度。
5. **D有均值信号，但不足以称稳健解决。** D-C的平均MAE在TUJI/UTS/UJI分别为-0.129553m、-0.164944m、-0.262023m；UJI的ES均值只变-0.002893m，三个种子中两个ES恶化、一个MAE恶化，BF宏准确率与联合NLL也有退化。原始每种子配对和站点伤害必须与均值同读。
6. **E的跨域通用性被本轮反例反驳。** E-D在UJI平均MAE +0.327879m、ES +0.257197m、P99约+6.069797m、最差站点MAE均值约+10.742268m。TUJI/UTS的MAE均值改善分别仅0.014538m/0.009127m，且存在种子反转。alpha从初始0.1降到约0.0338–0.0544，不代表该层在已见CAL-eval或未见位置上必然有益。

### E相对D的严重站点伤害

|域/种子|物理键|扫描数|D MAE|E MAE|损害m|D BF准确率|E BF准确率|
|---|---|---:|---:|---:|---:|---:|---:|
|tuji/17301|[0,0,0.0,26.234]|5|2.154110|2.912313|+0.758203|1.000000|1.000000|
|tuji/17302|[0,0,9.534,2.385]|5|3.282900|3.824728|+0.541827|1.000000|1.000000|
|uts/17302|[0,6,33.1754,8.5137]|7|4.593618|9.667512|+5.073894|1.000000|1.000000|
|uts/17301|[0,0,74.0978,8.8102]|4|8.932426|12.332017|+3.399591|1.000000|1.000000|
|uji/17303|[1,3,-7526.4995,4864893.0]|16|15.149511|44.070750|+28.921239|1.000000|1.000000|
|uji/17301|[1,3,-7526.4995,4864893.0]|16|16.536878|30.243596|+13.706717|1.000000|1.000000|

UJI位置[1,3,-7526.4995,4864893.0]在17303中从15.149511m恶化至44.070750m，且D/E的BF准确率均为1；因此该反例并不是“仅错层率上升”就能解释。现有输出没有保存其逐组件q/中心张量，不能从站点误差反推唯一概率排序或残差头因果机制。

后续若继续，只建议在新的物理空间隔离/未观察样本上预注册验证D-C这一固定0.30运输项，完整并报连续ES、BF准确率、P99和单站点伤害；本轮不实施，也不根据这些CAL-eval结果调alpha、gamma或其他网格。
