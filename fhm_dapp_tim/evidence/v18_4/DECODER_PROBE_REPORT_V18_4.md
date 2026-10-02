# V18.4 Decoder Diagnostic Gate 完整结果与执行记录

本次是已开放开发角色上的诊断，不是新的盲测。原 probe JSON/Markdown 均保留原样。

固定机制结论：**loss_aligned_geometric_median_supported**

分支 `v18-4-trust-region-posterior`；正式源码 `109613f77078ce3e19302b24ac85feb90a281f7b`。

两项 py_compile 通过；指定测试 25 passed in 10.54s。CPU/CUDA 合成验证：审计前后 33 个预测张量逐位一致。

用户已授权仅原 radio profiler 的两个临时 offset/gain 张量执行注册的 6 步 Adam。42 个预测 batch 共 252 次临时优化；网络参数梯度为 0，完整模型 state 逐位不变，磁盘输入与科学源码 SHA 不变。

## 四个固定 decoder

MAE、SE 和分位数的单位均为 m；相对改善以 posterior_mean 为基准。

| 角色 | Decoder | 坐标宏 MAE | MAE SE | 行 MAE | 行 Median | 行 P75 | 行 P90 | 相对改善 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| early_stop | posterior_mean | 10.694250 | 0.648259 | 10.525770 | 8.975503 | 12.765450 | 17.397012 | 0.00% |
| early_stop | map_component | 4.396712 | 0.323476 | 4.265161 | 2.844665 | 5.031493 | 8.439652 | 58.89% |
| early_stop | top8_mean | 4.355324 | 0.360120 | 4.315598 | 3.090020 | 5.521992 | 7.996081 | 59.27% |
| early_stop | geometric_median | 4.805945 | 0.447220 | 4.646433 | 3.196715 | 5.985636 | 8.631082 | 55.06% |
| calibration | posterior_mean | 12.698292 | 0.575772 | 12.187034 | 10.488933 | 15.487467 | 21.370224 | 0.00% |
| calibration | map_component | 8.000909 | 0.514634 | 7.523208 | 5.162756 | 9.016401 | 13.830158 | 36.99% |
| calibration | top8_mean | 7.387286 | 0.449186 | 6.881653 | 4.965751 | 8.300628 | 12.430021 | 41.82% |
| calibration | geometric_median | 7.475349 | 0.450533 | 7.004540 | 5.230586 | 8.545210 | 12.868807 | 41.13% |

## 坐标配对改善

改善为 posterior_mean 误差减候选 decoder 误差；正值表示改善。

| 角色 | Decoder | 平均改善 (m) | 改善 SE (m) | 改善坐标比例 |
|---|---|---:|---:|---:|
| early_stop | map_component | 6.297537 | 0.668529 | 88.76% |
| early_stop | top8_mean | 6.338925 | 0.573035 | 94.38% |
| early_stop | geometric_median | 5.888305 | 0.546262 | 93.26% |
| calibration | map_component | 4.697383 | 0.667981 | 76.16% |
| calibration | top8_mean | 5.311006 | 0.604706 | 80.79% |
| calibration | geometric_median | 5.222943 | 0.551901 | 84.11% |

## Posterior 与 decoder 位移诊断

| 角色 | 平均熵 (nats) | 平均有效 anchors | 平均 top1 概率 | GM 平均位移 (m) | GM P90 位移 (m) |
|---|---:|---:|---:|---:|---:|
| early_stop | 3.749801 | 42.607006 | 0.092574 | 8.860714 | 15.948902 |
| calibration | 3.748924 | 42.575874 | 0.092522 | 9.755023 | 17.340802 |

## 判定与科学边界

按两个角色均至少改善 10%、任一角色不恶化超过 0.25 m 的原规则独立核验通过；没有按数值排名新增 decoder 或修改阈值。

Use weighted geometric median as the registered V18.4 point decoder while keeping the full posterior for NLL/ES/uncertainty.

5203 行：early_stop 89 坐标/2064 行，calibration 151 坐标/3139 行。epoch=17、temperature=1.0、scale=0.75。

只读取允许的五个科学输入；没有打开 confirmation 或 official validation，没有创建/删除 confirmation 消费锁，没有训练或更新网络权重。完整 posterior 保持原计算，未把不同点 decoder 包装为不同 NLL/ES 模型。

## 问题与修复记录

1. 工具附件读取进程启动报 1056；顺序重试恢复，发生在科学运行前。

2. 第一次 CLI 的辅助审计不兼容 SciPy bytes 临时路径；使用 os.fsdecode 修复，合成 bytes/Delaunay 验证通过。未加载角色 shard 或预测。

3. 第二次 CLI 的辅助审计误拒绝 no_grad checkpoint 重建中的 train 标记恢复；守卫限定完整加载后的模型。CUDA 重建验证 state 逐位一致、eval=True、参数梯度为 0。未加载角色 shard 或预测。

4. 第三次 CLI 在原 radio nuisance optimizer 创建前被拦截；early_stop 已加载，但没有返回完整预测或指标。查明原推理有 backward/Adam step，与附件字面禁止条款冲突。

5. 用户明确授权仅临时 radio 参数优化后，加入 optimizer tensor 身份、独立叶子张量、注册函数范围、六步次数、网络零梯度与 state 不变守卫。CPU/CUDA 合成验证 33 个张量逐位一致，随后第四次 CLI 完成唯一一次完整科学 probe。

实际记录：3 次失败启动 + 1 次成功运行；没有隐去失败或声称只有一次 CLI 调用，没有结果驱动重跑。所有原始失败记录、授权与验证保存于 run_audit。

## 计算与来源

CUDA RTX 2060；PyTorch 2.10.0+cu128 / CUDA 12.8；成功运行 155.685 s；峰值 GPU 分配 446.349 MiB。

strict deterministic CUDA 开启，warn_only=False；flash/memory-efficient SDPA 关闭，math SDPA 开启；cuDNN benchmark 关闭、deterministic 开启。

结果 self-SHA：`fa9a5574d1239ee7b917b9c5c97154328dd106fbd3f089eca4555145bc83de22`。

Selection checkpoint SHA：`79c1b59bc2d55cbdbe1ebdbd661ee8ee6477406fc6d5e46b0f339b3100f230f2`。

完成结果与记录提交推送后停止；不继续 V18.3，不运行 official validation，不在本任务实现正式 V18.4 模型。
