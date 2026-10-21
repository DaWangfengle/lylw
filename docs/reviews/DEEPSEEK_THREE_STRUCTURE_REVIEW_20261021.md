# DeepSeek 三结构沟通与新增实验复核

复核日期：2026-10-21（Asia/Shanghai）。依据用户提供的 session.v4.jsonl、四张图、本地源码和结果 JSON。此次仅做代码阅读、哈希核对、已保存数值重算和 CPU 合成反例；没有加载 checkpoint、训练、重新评分 CAL/TEST，也没有修改旧实验或 DeepSeek 实验文件。

**结论：原三结构没有被覆盖。DeepSeek 新增了另一条路线，并最终建议放弃其中两项原机制。新增结果有参考价值，但组合分支存在实现错误，若干数学解释与统计结论过度，不能据此宣布三结构被否定、修好或已经定型。**

## 1. 实际改了什么

用户叙述顺序是“训练—图证据—连续场”；仓库编号分别为 S3—S1—S2。

| 层 | 原冻结实现 | DeepSeek 新实验 | 最终推荐 |
|---|---|---|---|
| 训练层 / S3 | 同 BF 近邻、联合 BF-XY NLL；D 加分量中心运输风险，E 另有训练期图层 | 复用 C/D/E，零重训 | 取 D |
| 图证据 / S1 | Sinkhorn 双随机同 BF 图；一步传播；先验与 G13 观测可独立控制，主要用 1A/1D | 新 1G：全通道 resolvent；改 MH 归一化、25m 截断，重新校准 T/θ | 从主模型排除 |
| 连续场 / S2 | 图边中点；插值 key、RSS 原型、G13 统计与连续分量；Cell LSE 分配质量 | 2K 只扩张 Student-t 尺度；2Q 放对称等权节点；另试固定 cell 质量的 2A_mass | 以 2K/2Q 代替 |

最终建议成为“原 D 主干＋保均值的概率密度校准”，没有完成“三个原结构共同改善定位”的目标。2K/2Q 不改善混合均值的 MAE 是构造性质；消除交互项不等于产生 MAE 协同。

新增脚本：[v17_gmrf_bandwidth_v1](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1)。新增结果：[outputs/v17_gmrf_bandwidth_v1](D:/liuyanlunwen/outputs/v17_gmrf_bandwidth_v1)。用户在历史沟通里授权了验证和继续探索，因此不宜称为未经授权运行；授权也不使科研解释自动正确。

按原冻结清单重新计算 SHA256：

| 对象 | 文件数 | 不匹配 |
|---|---:|---:|
| composition 绑定的源码/协议 | 204 | 0 |
| 原 composition 结果与审计 | 142 | 0 |
| 原 spatial_structure 产物 | 500 | 0 |
| 原 C/D/E checkpoint | 27 | 0 |

旧模型和旧结果完整保留。检查 checkpoint 只读取字节计算哈希，没有加载模型评分。

## 2. 关键错误：1G+2A 没有用 1G 父 logits

[run_validation.py:304](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1/run_validation.py:304) 对单独 1G 调用 smooth_channels，随后用传播后的 scores/z 计算概率。

但 [组合入口](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1/run_validation.py:337) 重新生成未经过新 1G 的 vpack，然后：

~~~python
parent_star = s3.complete_logits(vpack, params_star)
delta = parent_star - parent_clean
~~~

complete_logits 只计算 scores/T + θ·z，不会自动图传播。这里把为传播后特征选出的参数用于未传播特征，所谓传到虚拟点的“位移”也因此错误。

CPU 合成复核使用实际 smooth_channels、complete_logits、fast_cell_mixture，并从 AST 提取执行上述 parent_star 赋值。三锚点、一条查询、a=.4：

- 父 logits 与真实 1G 最大差：1.1655555632；
- β=0 时组合概率与单独 1G 最大差：0.0463081634；
- 两个合成预测点相距：0.2268749985。

这只证明接线不一致，不是实际数据的修正误差。β=0 也不能退回单独 1G，因此“1G×2A 交互”不是正确的因子对照。

受影响：RESULTS.json 的 1G+2A、带宽组合、noprop，第一张图，以及 H6/H7 的“+0.28m 交互、+0.10m 传播贡献”。不能据此否定原三结构。单独 1G、2K 和独立 hypotheses 脚本中的 clean 2A_mass 不因这个具体错误自动全部无效。

原 composition [compose_parent_logits](D:/liuyanlunwen/fhm_dapp_tim/fhm_dapp/msep_v17_spatial_composition_v1.py:291) 确实先做 S1 再传播位移，新错误不在旧代码中。

## 3. 数学与模型解释

**γ 不是“接错位置”。** [原设计](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_spatial_structure_v1/DESIGN_V17_SPATIAL_STRUCTURE.md) 明确将 γ_space 加到插值 RSS 观测方差。位置 Student-t 尺度是另一个对象、另一种单位；改为位置尺度扩张属于新干预。γ=0 只表示在该网格与目标下没选中正 γ，不证明观测不确定性不存在。

**ρ=1/√12 漏了 Student-t 方差系数。** df=4、scale=σ 时，每轴 Var=2σ²。若额外均匀元胞误差为 d²/12，采用 σ'²=σ²+(ρd)² 做同族方差匹配，应为 2ρ²d²=d²/12，即 ρ=1/√24≈.2041，不是 .2887。即便如此，也还要假定均匀元胞、误差独立且 d 是元胞宽度；当前 d 是每锚点八近邻距离中位数。Student-t 加均匀噪声一般不仍是同一 Student-t。UJI 细网格 ρ=.25 也不等于“恰好命中 .289”。[PyTorch v2.10 官方方差源码](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributions/studentT.py)。

**1G 不只是纯平滑。** 行随机 W 下，R=(I−aW)⁻¹ 满足 R1=1/(1−a)，放大常数模式；带保留项传播为 (1−a)(I−aW)⁻¹。正全局系数对分类 argmax 可略去，对 softmax 概率不可直接略去。T/θ 可吸收部分缩放，但有界网格下不能将全部变化纯归因于传播阶数。合成常数输入在 a=.4 下变成 1.6667。[原论文公式](https://proceedings.neurips.cc/paper/2506-learning-with-local-and-global-consistency.pdf)。

“GMRF 就是 Kriging”也需明确先验、噪声和协方差模型；换成逆矩阵不能自动证明物理插值正确。a 无量纲，h/(1−a) 未被验证为米制相关长度。

**D 损失不能直接写成 .55×同一均值误差。** Jensen 分解对同一个 q/中心集合成立，但旧坐标项使用候选子集均值，[model.py:1283](D:/liuyanlunwen/fhm_dapp_tim/fhm_dapp/model.py:1283)；新增运输风险使用全锚点 q，[structure.py:312](D:/liuyanlunwen/fhm_dapp_tim/fhm_dapp/msep_v17_spatial_structure_v1.py:312)。两者不能直接合并系数。W1 再除以批次中位数会改变目标与梯度，也不是“数值不变”的重新表述。

**H1 没有识别训练原因。** 移除校准的 .30 项只测试一种校准变化；没有重训对照，不能断言问题来自“训练 top-32 NLL”。C/D/E 已用全锚点联合 NLL 替换原候选 NLL，[structure.py:314](D:/liuyanlunwen/fhm_dapp_tim/fhm_dapp/msep_v17_spatial_structure_v1.py:314)。尺度扩张也可能补偿中心偏差或权重失配，不能唯一归因为离散网格。

共享图不构成“组合必然失败”的定理。不保均值意味着预测可以改变，变化既可能有益也可能有害。原 Cell LSE 总概率仍为 1，应区分总概率归一化、父 cell 相对质量与一阶矩。原模型本来已有连续残差和 Student-t，锚点间距不是固定定位精度下界。

## 4. 统计与执行限制

当前 JSON 有 27 条主表和 27 条 hypotheses 记录，error 均为 0；这不等于全部猜想已被证明。

1. [light_metrics](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1/run_validation.py:203)、rho 选择及 selection_check 使用逐扫描 NLL.mean()；full_metrics 是站点宏 NLL。不能不标注就混用。
2. [run_hypotheses.py:202](D:/liuyanlunwen/fhm_dapp_tim/evidence/v17_gmrf_bandwidth_v1/run_hypotheses.py:202) 的 nll_gain_at_rho_star 实际取 CAL-eval 曲线 min，且用宏 NLL 减行 NLL。UTS 的 CAL-fit rho 均选 .4，CAL-eval 行 NLL 最小均在 .35。H2 宏表和 h3 站点向量另有正确路径，应分别对待。
3. 首轮就对 alpha/rho 网格计算 CAL-eval 全曲线，随后追加细网格、求积与选择检查，多次运行。可作回顾性探索，不能说预先锁定后只看一次 CAL-eval。此次未发现官方 TEST 评分；“零 TEST”不能把 CAL-eval 反复分析变成独立确认。
4. 九个 C/D/E×seed 组合共享评价站点且有配对训练关系，不能当九次独立 Bernoulli 试验。p≈.002 不支持独立显著性，底层组合分支同时有 bug。
5. bootstrap 在各 seed 中独立抽同一批站点，丢失跨 seed 配对。应按物理站点对齐后共同重采样；空间相关仍需单独考虑。
6. A1/A2/A4/A6 多数仅记录差值。A6 最大 5.7220e−6，超过沟通写的 1e−6，仍继续；可能只是 float32 误差，但不是严格按承诺门禁通过。2Q 数学上保均值，保存结果最大宏 MAE 差为 4.7684e−6m，不能写浮点逐位完全一致。
7. MH 用自环补足，合成图对角线为 [.030967,0,.093386]，元数据仍写 zero_diagonal=true。dropped_over_cap=8593 也不是原唯一边删减数；UJI 唯一边实际由 3232 变为 2696。
8. 图的误差棒是三种子 SD，不是独立验证置信区间。NLL 含坐标计量单位造成的加性常数，“NLL 砍半”不应解释成通用概率性能百分比。

## 5. 可保留的探索结果

从 HYPOTHESES.json 独立聚合 D 主干三个 seed；历史 CAL-eval 描述，不进行新选择或晋级。以下 clean 支撑比较不经过错误的 1G+2A 分支。

| 方法 | TUJI MAE | UTS MAE | UJI MAE | TUJI NLL | UTS NLL | UJI NLL |
|---|---:|---:|---:|---:|---:|---:|
| clean D | 1.814602 | 4.012157 | 7.106555 | 3.818353 | 10.848249 | 14.501097 |
| 中点 2A，β=.8 | 1.827312 | 4.041473 | 7.111727 | 3.381214 | 5.802123 | 10.509401 |
| 固定 cell 质量 2A_mass，β=.8 | 1.816364 | 3.958806 | 7.027438 | 3.396468 | 6.061618 | 10.788272 |
| 尺度扩张 2K | 1.814602 | 4.012157 | 7.106555 | 3.390007 | 5.280853 | 6.471234 |
| 对称节点 2Q4 | 1.814602 | 4.012157 | 7.106556 | 3.257384 | 6.454174 | 12.902712 |

2K ES：TUJI 1.289308→1.290728（略差），UTS 3.361293→2.949104，UJI 6.069481→5.438309。

固定父 cell 质量的 UTS/UJI MAE 改善，是更接近“修好原连续场”的描述性线索；TUJI略差，不能说三域获胜。2K 是尺度校准对照，不解决点位置偏差，也不是虚拟观测场的等价实现。

第一张交互图暂缓科学使用；第二张保留纯 base/2K/2A 对比，撤回含 bug 组合及因果贡献比例；第三张保留尺度敏感性，撤回 .289 理论命中；第四张保留探索性取舍，不据此宣布最终模型。

## 6. 原组合的负结果仍然有效

原冻结表中 D 主干：

| 历史 CAL-eval | clean D | 原 1A | 原 2A | 原 1A+2A |
|---|---:|---:|---:|---:|
| TUJI | 1.814602 | 1.824199 | 1.827312 | 1.837789 |
| UTS | 4.012157 | 3.998521 | 4.041473 | 4.157160 |
| UJI | 7.106555 | 7.070120 | 7.111726 | 7.233840 |

发现 DeepSeek 新 bug 不会反向证明旧组合有效。当前组合尚未达成目标，需要局部修正与受控验证；三个名字本身也不证明创新或协同。

## 7. 与用户目标一致的下一步

保留原 S3/S1/S2、冻结证据与全部 DeepSeek 探索结果。后者作为候选对照，不覆盖主模型。先纠正组合接线、指标聚合与数学解释；若重做真实评分，应另存修订实验，不能替换原负结果。

下一轮若只检验一个科学变量，建议：**固定同一 D 主干、同一 S1 父后验、同一虚拟观测和连续分量，只比较 S2 是否重新分配父 cell 总质量。** 配套无 S1 的相同对照，运行前冻结 beta/选择目标/聚合规则，确保 beta=0 退回实际 S1 后验。2K 单列为概率尺度对照。

这个方向来自回顾性线索，尚未验证。先回答“哪个环节移动均值、RSS 证据能否让它向正确方向移动”，再决定端到端训练与独立 TEST。不能承诺修后必涨精度，也不宜现在直接按“最终 D+带宽”消耗独立 TEST。
