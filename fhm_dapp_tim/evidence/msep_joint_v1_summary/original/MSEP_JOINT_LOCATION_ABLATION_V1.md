# MSEP-Loc v1：单成员完整位置联合学习的受控科学实验

## 科学问题（不是预设成功）

现有分支 msep-single-anchor-posterior-v1 的三个 TEST 已暴露（TUJI Raw 2.3155m / UTS 4.5989m / UJI 6.7597m；UJI 原 Joint BF 准确率跌至 56.49%）。这些历史数值只做背景，**不作为本阶段模型/超参数选择依据，不与新拆分直接比较**。

待证伪命题：
1. V17 训练软邻居跨楼层是否损害空间检索（先度量当前 FIT 软标签的跨楼层质量占比）？
2. 原 XY-only 联合微调为何损坏 UJI 建筑—楼层区分？真实 BF-XY 联合概率目标在**相同骨干、相同微调预算**下能否降低伤害？
3. 固定 T=0.6 是否压制 G13 增量？学习 T 及 theta 能否在独立 CAL 子集上改善定位，而不只是改变尖锐程度？
4. 原 V17 训练时的学习型原型权重与 MSEP 推理 R3 常数 32/3，是否带来可观决策差别？仅冻结换评分规律的反事实不能证明重新训练会改进。
5. 与无观测模型同起点、同训练预算的 BF-XY+FIT G13 能否带来增量？负结果也完整记录。

## 方法（严格隔离可解释变量）

全程沿用 V17 单成员 AnchorStudentTModel / RSS adapter / G13 FIT memory / 30m residual / Student-t df=4 / full FIT anchor support，不改历史科学文件。新代码：

- fhm_dapp_tim/fhm_dapp/msep_joint_science_v1.py：同楼层软邻居、精确 BF-XY Student-t 混合 NLL、后验优化、双点决策/概率指标。
- fhm_dapp_tim/fhm_dapp/msep_joint_ablation_v1.py：两种从相同初始化开始的基础训练、六个独立微调及冻结校准、评估与记录。
- tests/test_msep_joint_ablation_v1.py：几何污染、不具唯一最近邻的联合分布、梯度流、数值边界、合法推理。

\[
p_\theta(c,y\mid r)=\sum_{a:c_a=c} q_\theta(a\mid r)\,t_4(y;\mu_a(r),\Sigma_a(r)).
\]

Joint 目标严格是全锚点归一化后验下的 \(-\log p_\theta(c^*,y^*\mid r)\)，另有 **只在训练**按真实 BF 条件化的 0.3 欧氏坐标风险、相同 0.02 KL。真实 BF 从不进入推断；推断同时报告 full-mixture XY 均值与“预测的 BF 内部的 XY 均值”。XY-only 对照用原形式的 marginal NLL+0.3 全混合点风险+0.02 KL。

同楼层邻居只允许同 B/F；若该 BF 仅有一个 FIT 锚点，则把自身作为软标签唯一退化目标；多锚点层内不足 8 个邻居时以 **零质量**填充自身索引，避免跨楼层有效软标签。报告旧/新错误层级正质量。注意这与强制所有预测锚点同层不同：后验仍对所有 BF 竞争。

### 因果表（全部单成员）

|基础训练|不微调|XY-only 微调|BF-XY 微调|BF-XY+训练 G13|
|---|---|---|---|---|
|旧同建筑邻居|raw|raw_xy|raw_joint|raw_joint_with_evidence|
|新同建筑同楼层邻居|floor_neighbors|floor_neighbors_xy|floor_neighbors_joint|floor_neighbors_joint_with_evidence|

所有 refit 从各自 frozen base 独立复制、同随机种子/10 epoch/64 batch 和相同网络 LR=3e-5（证据 θ 单独 1e-3）。即使 joint 优于原 base，也不能仅用不同训练轮数证明 joint 目标有效；**同基座 XY-only 与 BF-XY 的相同预算配对差异**才是主要因果对照。证据 θ 限制 [0,1.5]，无乘法门控；G13 观测表固定来自 FIT。

### 冻结后验 A

仅 raw base，固定全部候选 Student-t 位置及尺度；CAL-fit 同时估计：
raw；T=.6；CAL 学 T；固定 .6 学 θ；CAL 联合学 T/θ。使用与评估风险接近的连续 NLL + 0.3 Euclidean point，位置等权，不存在固定 theta=0 后门控无法恢复的问题。所有在 CAL-eval 上对照，CAL-eval 不用于拟合模型或选择拟合温度。

另对 raw base 计算原学习型 retrieval_logits 与 R3 固定 (neural=1, prototype=32/3) 的排序/决策反事实；两个概率在相同候选中心/尺度上重加权，不额外训练。

### 数据纪律与统计语义

- 默认 split_seed=20261017，对每一 (B,F,XY) 物理站点拆 FIT/DEV/CAL/TEST；所有扫描同角色。与旧 20261016 TEST 不同，但历史整体语料已暴露，**不是独立盲测**。
- 基础模型只用 FIT 训练、DEV 早停。CAL 按物理站点再次分成 CAL-fit/CAL-eval（固定种子，不交叉）。
- CAL-fit 只用于后验概率参数；CAL-eval 用于探索性所有方法对照。无任何 TEST 预测/评分，不使用官方 test/validation 文件，也不评价多成员。
- 主要看 CAL-eval 上等权站点宏：全混合欧氏 MAE、预测 BF 内的 MAE、BF joint accuracy、Energy Score、joint NLL、P95 与逐站点伤害。ES 全混合（非 BF-条件分布）；它与联合 joint NLL 衡量不同的概率对象，应并列解释。
- TUJI 单建筑单楼层：联合类别监督退化为无额外类别信息的负对照，不应期待楼层修复增益。UTS/UJI 则检验多楼层结构。
- 保留所有负结果、站点诊断；不根据 CAL-eval 微调超参数后再次称其为独立 test。
- 最终模型选择之后，需额外未参与决策的空间块或外部评价，且至少检查多个训练随机种子和有效站点独立性。

## 精准限制

本阶段主实验是**从头训练两种基础 V17，然后做相同预算的独立微调**，尚不是 BF-XY 损失从第 1 个 epoch 直接参与的全程端到端新基础训练。先检验上次 UJI 楼层崩塌的因果源，再决定是否值得实现从头 BF-XY。避免同时改网络、残差半径、Top-L 和多成员数造成不可归因。

G13-with-evidence 与 no-evidence 是不同的 refit 训练，θ 进入训练梯度；不是简单在同一已训练模型上推理开关。CAL 温度/证据实验则是另一种明确标注的纯后验校准干预。

## 执行顺序

1. Python compile、pytest、CPU 最小 smoke（base_epochs=1, joint_epochs=1, es_draws=8, skip_evidence_refit），先确认三数据源和代码的基本正确性。
2. CUDA 三域正式探索性运行 base_epochs=80、joint_epochs=10、es_draws=32，保留 RESULT.json、SITE_DIAGNOSTICS.json、stdout/stderr、可选 checkpoints。
3. 逐站点配对比较：每个 base 内 XY vs BF-XY；floor vs raw 的差异；evidence on/off 独立训练。分类准确率只在合法多层域判读。
4. 下轮将真实结果上传 GitHub 后再决定：保留联合目标 / RSS 空间可辨识性 / 局部残差 / 从头联合训练。**不预先承诺 MAE 会提高。**
