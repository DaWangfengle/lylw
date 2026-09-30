V18 单成员实验已完成 register → train-selection → freeze → train-final → outer，outer 只运行一次。角色隔离、freeze gate 和数值检查通过，但共享 loader 在 freeze 前加载全表，因此用户要求的“不得提前读取 outer-test 标签”按物理加载口径未完全满足。本报告明确保留这一审计缺口。

**Outer MAE：9.114772 m**
**Outer Energy Score：6.258374 m**
**Outer NLL：7.001560**

Git 起始 SHA：8e9a50d2bfc89027f97925d695a194db3d6eb8d5
Git 正式实验 SHA：17860cf50b7b32d97854a02c642ffb86a9b2562c
分支：v18-single-member。全部正式阶段的源码哈希保持一致；结果和模型保存在本地，未推送。

**TESTS**

两个 py_compile 均 exit 0。指定 pytest：12 passed in 3.89s；没有删除、跳过、弱化或修改任何测试。首次为 11 passed / 1 failed，原因是 V18 模块说明文字含协议测试禁止的官方验证文件名。

- ed2e5ab50cc1feedf218246b86357c505a554198：仅修改 uji_v18_experiment.py 的说明文字。
- 17860cf50b7b32d97854a02c642ffb86a9b2562c：修正 DEFAULT_TRAIN，使用仓库根目录 data/UJIndoorLoc/trainingData.csv。
- 两处修复均发生在正式 registration 之前。V18_CONFIG、模型定义、种子、目标函数、候选数均未改变。
- 额外合成数据核验：CUDA 前向/反向有限、权重归一化、checkpoint 重建后预测逐元素完全一致；不使用真实数据训练。

**PROTOCOL**

总训练行：19937；物理坐标：933；AP：520。

| Role | 坐标数 | 行数 |
|---|---:|---:|
| member_fit | 506 | 10770 |
| early_stop | 89 | 1895 |
| selection_refit | 595 | 12665 |
| development | 151 | 3139 |
| outer_refit | 746 | 15804 |
| outer_test | 187 | 4133 |

三项 disjoint intersection 全部为 0；三项并集恒等式成立；全部角色均包含 13 个 building-floor strata；三项 hull_missing 检查全部为 0。registry self-hash、training 文件 SHA256 均有效；注册文件未修改。official_validation_in_scope=false；checkpoint_count_at_inference=1；residual_experts=1。

**MODEL**

selection：606,873 参数 / 595 anchors；final：626,201 参数 / 746 anchors。
residual_experts=1；scan_feature_mode=bounded_maximal_invariant；prototype_metric=none；candidate_reranker=true；candidate_reranker_mode=summary；inference_top_k=64；train_candidates=48；posterior_temperature=1.0；Student-t df=4。
selected epoch=1；early-stop 实际运行 29 epochs；selection_refit 和 final_refit 均只运行冻结的 1 epoch。
early-stop 最优 coordinate-macro mixture NLL：9.418310753。

**DEVELOPMENT / OUTER TEST**

| 指标 | Development | Outer test |
|---|---:|---:|
| coordinate-macro MAE (m) | 8.559119 | 9.114772 |
| coordinate-macro mixture NLL | 7.867259 | 7.001560 |
| coordinate-macro Energy Score (m) | 6.313241 | 6.258374 |
| coordinate-macro building error | 0.099338% | 0.655717% |
| coordinate-macro floor error | 16.852526% | 9.963745% |
| mean anchor entropy (nats) | 2.902898 | 2.923346 |
| mean radial predictive scale (m) | 22.524632 | 20.252485 |

MAE / NLL / Energy Score / building error / floor error 按坐标分组宏平均；entropy 与 radial scale 按现有协议为 scan mean。Energy Score 使用 64 次固定随机采样。

**COMPUTE**

selection 训练（epoch selection + refit）：165.210645 秒；final 训练：9.704448 秒。
GPU：NVIDIA GeForce RTX 2060，6,144 MiB；各阶段合计 GPU 总占用采样峰值：2133 MiB。
GPU 以 2 秒间隔用 nvidia-smi 采样，数值含其他进程占用；并非 PyTorch allocator 精确峰值。
解释器：D:\miniminiconda\envs\liuyan\python.exe；Python 3.12.12；PyTorch 2.10.0+cu128。
现有 warn_only 确定性设置下，PyTorch attention backward 报告 nondeterministic kernel warning；未因此修改冻结代码或配置，不能承诺 CUDA 训练逐位可复现。

**AUDIT**

- one inference checkpoint：yes（selection/final 是两个阶段产物，推理仅加载该阶段的一个 checkpoint）。
- residual_experts == 1：yes。
- 正式模型 ensemble / consensus / member weighting：no。
- outer 在 freeze 前参与训练或评分：no；outer 评分只运行一次且在 freeze 与 final 之后。
- outer 标签在 freeze 前是否被加载到内存：yes。共享 _load_uji / load_dataset 读取整张 trainingData；随后用注册 row indices 限定训练和评分。因此不能将“无提前 outer 评分”写成“无提前 outer 标签加载”。
- validationData.csv accessed：no；未访问官方验证文件。
- V9/V17 历史结果 JSON / outer outcomes accessed：no。
- development 或 outer outcome-driven tuning：no；源码和配置自注册起保持一致。
- registry / freeze / development / outer self-hash、checkpoint SHA256 链：全部通过。
- 训练全部 loss trajectory、所有 checkpoint tensors / stats 均无 NaN/Inf。

**运行判断与 V18.1 项目**

1. 数值稳定、训练目标下降：loss 8.990921 → 2.448771；但 early-stop NLL 9.418311 → 11.805462，最优在 epoch 1。存在训练与未见坐标泛化分数的分离，不能称为已达到良好的泛化收敛。
2. 外层均值 radial scale=20.252485 m，entropy=2.923346 nats；entropy 最大参考值 log(64)=4.158883。均值未出现尺度接近零或熵接近 0 / 完全均匀的明显崩溃；仅凭均值不能断言概率已校准。
3. radio reranker：训练各 epoch 的 reranker_loss、相关 checkpoint 参数与最终评估汇总均有限；未发现 NaN/Inf。
4. Delaunay：selection 与 final 的全部 13 strata 使用 rank-2 Delaunay；transform 有限、对称、重建一致，最小 eigenvalue=0.452570856，邻域不跨 building-floor。
5. 下一次 V18.1 需要重新注册并优先处理按 role 的物理数据加载隔离；泛化 NLL 恶化、epoch-1 选择、概率尺度校准、CUDA 严格确定性应作为新实验问题记录。任何架构、loss、候选、温度或 epoch 选择规则修改均属于新实验；本次 outer 不再运行。

**SHA256**

- registry self-hash：f5b9bb72d9d84efbaa0746ce18e17e33ff1e900d3018b96a671c46b2dcf2a2f6
- selection checkpoint：6dcc0ccdb6c4adcadfee75312d756dbc498be180d9144d8102d3cb060e3a0d1f
- development result self-hash：f0843ab86451be75f556c3b6bd154c9a025f5b6b73c8c1ec1d982203b229c896
- freeze self-hash：379deb013d51b9f075e0e57235f3676757e90756f10311ef4e6c8676498e84cc
- final checkpoint：22b92ee307c0b207d7d724aaa75f91cc63eb0183d89a2e843074e1fa22965c54
- outer result self-hash：83d9b7200d7c5089345e7bc0e2add1db6b23b58545199c9ce1fb68dcd3541f52

完整机器审计：RUN_SUMMARY_V18.json；阶段命令、时间戳、GPU 采样、测试日志、源码哈希快照、outer 单次调用锁、training trajectory 均位于 run_audit/。
