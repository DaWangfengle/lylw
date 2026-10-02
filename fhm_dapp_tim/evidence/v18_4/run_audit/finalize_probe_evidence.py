"""Validate saved results without rerunning predictions; produce complete report."""
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
EVIDENCE = AUDIT.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result_path = EVIDENCE / "decoder_probe_v18_3.json"
markdown_path = EVIDENCE / "decoder_probe_v18_3.md"
assert result_path.is_file() and markdown_path.is_file()
result = read(result_path)
preflight = read(AUDIT / "preflight.json")
execution = read(AUDIT / "probe_04_execution.json")
runtime = read(AUDIT / "probe_04_runtime.json")
authorization = read(AUDIT / "radio_profile_authorization.json")
assert execution["exit_code"] == 0 and execution["scientific_files_unchanged"]
assert execution["completed_scientific_probe_count"] == 1
assert authorization["radio_nuisance_optimization_authorized"]
assert runtime["model_state_unchanged_bitwise"]
assert runtime["checkpoint_parameter_gradient_count"] == 0
assert runtime["backward_calls"] == runtime["optimizer_constructions"] == runtime["training_mode_requests"] == runtime["checkpoint_save_calls"] == 0
assert runtime["radio_profiler_calls"] == runtime["radio_nuisance_optimizer_constructions"] == 42
assert runtime["radio_nuisance_optimizer_steps"] == runtime["radio_nuisance_tensor_backward_calls"] == runtime["radio_nuisance_autograd_backward_calls"] == 252
assert runtime["prediction_scans"] == 5203
assert runtime["attempted_prediction_calls"] == runtime["prediction_calls"] == 42
assert not runtime["inference_target_coordinates_passed"]
assert not runtime["denied_accesses"]
assert not runtime["confirmation_lock_exists"] and not runtime["confirmation_result_exists"]
assert runtime["deterministic_algorithms"] and not runtime["deterministic_warn_only"]
assert not runtime["flash_sdpa"] and not runtime["memory_efficient_sdpa"] and runtime["math_sdpa"]
assert not runtime["cudnn_benchmark"] and runtime["cudnn_deterministic"]
assert set(runtime["allowed_scientific_reads"]) == {
    os.path.normcase(os.path.abspath(ROOT / path)) for path in preflight["required_inputs_present"]
}
for relative, expected in preflight["scientific_files_sha256_before"].items():
    assert sha(ROOT / relative) == expected, relative
canonical = {key: value for key, value in result.items() if key != "result_sha256"}
assert hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == result["result_sha256"]
assert result["decoders"] == ["posterior_mean", "map_component", "top8_mean", "geometric_median"]
assert result["topk_diagnostic"] == 8
assert result["frozen_posterior_temperature"] == 1.0 and result["frozen_scale_multiplier"] == 0.75
assert not any(result[key] for key in ["confirmation_accessed", "official_benchmark_accessed", "model_parameters_modified", "training_performed"])
decision = result["mechanism_decision"]
assert decision["rule"] == {"required_relative_mae_improvement_each_role": 0.10, "maximum_allowed_role_worsening_m": 0.25, "roles": ["early_stop", "calibration"]}


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


finite_tree(result)
counts = {"early_stop": (89, 2064), "calibration": (151, 3139)}
for role, (coordinates, rows) in counts.items():
    metrics = result["roles"][role]["decoder_metrics"]
    reference = metrics["posterior_mean"]["coordinate_macro_mae_m"]
    for name, values in metrics.items():
        assert values["coordinate_groups"] == coordinates and values["rows"] == rows
        assert values["coordinate_macro_mae_se_m"] >= 0
        assert values["row_median_error_m"] <= values["row_p75_error_m"] <= values["row_p90_error_m"]
        improvement = (reference - values["coordinate_macro_mae_m"]) / reference
        assert abs(improvement - values["relative_improvement_vs_mean"]) < 1e-12
        if name != "posterior_mean":
            paired = result["roles"][role]["paired_vs_posterior_mean"][name]
            assert abs(paired["mean_improvement_m"] - reference + values["coordinate_macro_mae_m"]) < 1e-10
            assert paired["improvement_se_m"] >= 0
            assert 0 <= paired["fraction_coordinates_improved"] <= 1


def passes(name):
    return all(
        result["roles"][role]["decoder_metrics"][name]["relative_improvement_vs_mean"] >= 0.10
        and result["roles"][role]["decoder_metrics"][name]["coordinate_macro_mae_m"]
        - result["roles"][role]["decoder_metrics"]["posterior_mean"]["coordinate_macro_mae_m"] <= 0.25
        for role in counts
    )


expected = ("loss_aligned_geometric_median_supported" if passes("geometric_median") else
            "posterior_tail_dilution_supported" if passes("top8_mean") else "posterior_learning_dominates")
assert expected == decision["conclusion"]
assert decision["geometric_median_supported"] == passes("geometric_median")
assert decision["top8_tail_contraction_supported"] == passes("top8_mean")
report = [
    "# V18.4 Decoder Diagnostic Gate 完整结果与执行记录", "",
    "本次是已开放开发角色上的诊断，不是新的盲测。原 probe JSON/Markdown 均保留原样。", "",
    f"固定机制结论：**{expected}**", "",
    f"分支 `{preflight['branch']}`；正式源码 `{preflight['formal_experiment_sha']}`。", "",
    "两项 py_compile 通过；指定测试 25 passed in 10.54s。CPU/CUDA 合成验证：审计前后 33 个预测张量逐位一致。", "",
    "用户已授权仅原 radio profiler 的两个临时 offset/gain 张量执行注册的 6 步 Adam。42 个预测 batch 共 252 次临时优化；网络参数梯度为 0，完整模型 state 逐位不变，磁盘输入与科学源码 SHA 不变。", "",
    "## 四个固定 decoder", "",
    "MAE、SE 和分位数的单位均为 m；相对改善以 posterior_mean 为基准。", "",
    "| 角色 | Decoder | 坐标宏 MAE | MAE SE | 行 MAE | 行 Median | 行 P75 | 行 P90 | 相对改善 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for role in counts:
    for name in result["decoders"]:
        m = result["roles"][role]["decoder_metrics"][name]
        values = [m[key] for key in ["coordinate_macro_mae_m", "coordinate_macro_mae_se_m", "row_mae_m", "row_median_error_m", "row_p75_error_m", "row_p90_error_m"]]
        report.append(f"| {role} | {name} | " + " | ".join(f"{v:.6f}" for v in values) + f" | {m['relative_improvement_vs_mean']:.2%} |")
report += ["", "## 坐标配对改善", "", "改善为 posterior_mean 误差减候选 decoder 误差；正值表示改善。", "",
           "| 角色 | Decoder | 平均改善 (m) | 改善 SE (m) | 改善坐标比例 |", "|---|---|---:|---:|---:|"]
for role in counts:
    for name, paired in result["roles"][role]["paired_vs_posterior_mean"].items():
        report.append(f"| {role} | {name} | {paired['mean_improvement_m']:.6f} | {paired['improvement_se_m']:.6f} | {paired['fraction_coordinates_improved']:.2%} |")
report += ["", "## Posterior 与 decoder 位移诊断", "",
           "| 角色 | 平均熵 (nats) | 平均有效 anchors | 平均 top1 概率 | GM 平均位移 (m) | GM P90 位移 (m) |",
           "|---|---:|---:|---:|---:|---:|"]
for role in counts:
    d = result["roles"][role]["posterior_diagnostics"]
    report.append(f"| {role} | " + " | ".join(f"{d[key]:.6f}" for key in ["mean_anchor_entropy_nats", "mean_effective_anchor_count", "mean_top1_probability", "mean_geometric_median_displacement_from_mean_m", "p90_geometric_median_displacement_from_mean_m"]) + " |")
report += ["", "## 判定与科学边界", "",
           "按两个角色均至少改善 10%、任一角色不恶化超过 0.25 m 的原规则独立核验通过；没有按数值排名新增 decoder 或修改阈值。", "",
           decision["v18_4_implication"], "",
           "5203 行：early_stop 89 坐标/2064 行，calibration 151 坐标/3139 行。epoch=17、temperature=1.0、scale=0.75。", "",
           "只读取允许的五个科学输入；没有打开 confirmation 或 official validation，没有创建/删除 confirmation 消费锁，没有训练或更新网络权重。完整 posterior 保持原计算，未把不同点 decoder 包装为不同 NLL/ES 模型。", "",
           "## 问题与修复记录", "",
           "1. 工具附件读取进程启动报 1056；顺序重试恢复，发生在科学运行前。", "",
           "2. 第一次 CLI 的辅助审计不兼容 SciPy bytes 临时路径；使用 os.fsdecode 修复，合成 bytes/Delaunay 验证通过。未加载角色 shard 或预测。", "",
           "3. 第二次 CLI 的辅助审计误拒绝 no_grad checkpoint 重建中的 train 标记恢复；守卫限定完整加载后的模型。CUDA 重建验证 state 逐位一致、eval=True、参数梯度为 0。未加载角色 shard 或预测。", "",
           "4. 第三次 CLI 在原 radio nuisance optimizer 创建前被拦截；early_stop 已加载，但没有返回完整预测或指标。查明原推理有 backward/Adam step，与附件字面禁止条款冲突。", "",
           "5. 用户明确授权仅临时 radio 参数优化后，加入 optimizer tensor 身份、独立叶子张量、注册函数范围、六步次数、网络零梯度与 state 不变守卫。CPU/CUDA 合成验证 33 个张量逐位一致，随后第四次 CLI 完成唯一一次完整科学 probe。", "",
           "实际记录：3 次失败启动 + 1 次成功运行；没有隐去失败或声称只有一次 CLI 调用，没有结果驱动重跑。所有原始失败记录、授权与验证保存于 run_audit。", "",
           "## 计算与来源", "",
           f"CUDA RTX 2060；PyTorch {runtime['torch_version']} / CUDA {runtime['cuda_version']}；成功运行 {execution['wall_seconds']:.3f} s；峰值 GPU 分配 {runtime['peak_gpu_memory_allocated_bytes']/1024**2:.3f} MiB。", "",
           "strict deterministic CUDA 开启，warn_only=False；flash/memory-efficient SDPA 关闭，math SDPA 开启；cuDNN benchmark 关闭、deterministic 开启。", "",
           f"结果 self-SHA：`{result['result_sha256']}`。", "",
           f"Selection checkpoint SHA：`{result['v18_3_selection_checkpoint_sha256']}`。", "",
           "完成结果与记录提交推送后停止；不继续 V18.3，不运行 official validation，不在本任务实现正式 V18.4 模型。", ""]
report_path = EVIDENCE / "DECODER_PROBE_REPORT_V18_4.md"
with report_path.open("x", encoding="utf-8") as handle:
    handle.write("\n".join(report))
qa = {
    "all_checks_passed": True,
    "result_self_sha_verified": result["result_sha256"],
    "original_output_file_sha256": {result_path.name: sha(result_path), markdown_path.name: sha(markdown_path)},
    "complete_report_sha256": sha(report_path),
    "mechanism_decision_independently_verified": expected,
    "runtime_and_protocol_checks_passed": True,
    "scientific_files_unchanged": True,
    "completed_scientific_probes": 1,
    "failed_cli_starts_preserved": 3,
    "successful_cli_starts": 1,
    "registered_thresholds_unchanged": True,
    "checkpoint_parameter_gradients": 0,
    "model_state_unchanged_bitwise": True,
    "confirmation_accessed": False,
    "official_validation_accessed": False,
    "predictions_rerun_for_report": False,
    "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
with (AUDIT / "result_verification.json").open("x", encoding="utf-8") as handle:
    json.dump(qa, handle, indent=2)
    handle.write("\n")
print(json.dumps({"verification": qa, "mechanism_decision": decision, "roles": result["roles"]}, indent=2))
