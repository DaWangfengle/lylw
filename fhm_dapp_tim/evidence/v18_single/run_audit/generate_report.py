"""Build the final report from existing V18 artifacts; never re-evaluate data."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
import torch
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from fhm_dapp_tim.fhm_dapp import uji_v18_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_single import V18_CONFIG
AUDIT = Path(__file__).resolve().parent
EVIDENCE = AUDIT.parent
def read(name):
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))
def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value: finite(child)

preflight = read("preflight.json")
protocol = read("registry_verification.json")
selection = read("selection_verification.json")
final = read("final_verification.json")
registry = exp.load_registry()
freeze = exp.load_freeze(registry=registry)
development = json.loads(exp.DEFAULT_DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
outer = json.loads(exp.DEFAULT_OUTER_RESULT.read_text(encoding="utf-8"))
for result in [development, outer]:
    assert result["result_sha256"] == exp._canonical_json_sha256(result, omit="result_sha256")
    assert result["registry_sha256"] == registry["registry_sha256"]
    finite(result)
assert outer["freeze_sha256"] == freeze["freeze_sha256"]
assert outer["final_checkpoint_sha256"] == exp._checkpoint_sha(exp.DEFAULT_FINAL_CHECKPOINT)
assert development["selection_checkpoint_sha256"] == exp._checkpoint_sha(exp.DEFAULT_SELECTION_CHECKPOINT)
assert outer["single_member"] is True
assert final["selected_epoch"] == selection["selected_epoch"] == freeze["frozen_selected_epoch"]
assert outer["outer_test_metrics"]["coordinate_groups"] == registry["roles"]["outer_test"]["coordinate_count"]
assert outer["outer_test_metrics"]["rows"] == registry["roles"]["outer_test"]["row_count"]
manifest = read("code_snapshot.json")
current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
assert current_sha == manifest["experiment_commit_sha"]
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest["files"].items())
phases = ["register", "train-selection", "freeze", "train-final", "outer"]
executions = {phase: read(phase + "_execution.json") for phase in phases}
assert all(record["exit_code"] == 0 and record["source_snapshot_matches_after_phase"]
           for record in executions.values())
for left, right in zip(phases[:-1], phases[1:]):
    assert dt.datetime.fromisoformat(executions[left]["finished_utc"]) < dt.datetime.fromisoformat(executions[right]["started_utc"])
assert (AUDIT / "outer_invocation.lock").is_file()
selection_payload = torch.load(exp.DEFAULT_SELECTION_CHECKPOINT, map_location="cpu", weights_only=True)
final_payload = torch.load(exp.DEFAULT_FINAL_CHECKPOINT, map_location="cpu", weights_only=True)
history = selection_payload["selection_history"]
trajectory = {"epoch_selection": history,
              "selection_refit": selection_payload["refit_history"],
              "final_refit": final_payload["refit_history"]}
finite(trajectory)
(AUDIT / "training_trajectory_v18.json").write_text(json.dumps(trajectory, indent=2, allow_nan=False), encoding="utf-8")
dm = development["development_metrics"]
om = outer["outer_test_metrics"]
audit = {
    "one_inference_checkpoint": True,
    "residual_experts_equals_one": True,
    "ensemble_or_consensus_called_in_formal_run": False,
    "member_weighting_used": False,
    "outer_role_scored_before_freeze": False,
    "outer_role_used_for_training": False,
    "outer_test_metrics_inspected_before_freeze": False,
    "outer_test_label_rows_materialized_by_shared_full_csv_loader_before_freeze": True,
    "literal_no_outer_label_loading_before_freeze_compliant": False,
    "role_isolation_and_outer_freeze_gate_compliant": True,
    "validationData_csv_accessed": False,
    "historical_V9_V17_results_inspected": False,
    "outcome_driven_tuning": False,
    "outer_cli_invocations": 1,
    "development_cli_invocations": 1,
    "source_hashes_unchanged_since_registration": True,
    "all_self_hashes_and_checkpoint_links_valid": True,
    "data_access_limitation": (
        "Registration and train-selection call experiment_v7._load_uji(), whose data.load_dataset() "
        "uses pandas.read_csv() on the entire trainingData.csv and constructs all raw coordinates. "
        "Outer rows are excluded from model fitting, anchor statistics, epoch selection and scoring "
        "until freeze, but their labels are physically loaded by the shared loader. "
        "Thus the literal pre-freeze label-read ban is not fully satisfied. "
        "The reported outer result remains one-shot; no corrective same-run evaluation is permitted."
    ),
    "ensemble_audit_basis": "V18 call-path inspection and fixed single-member checkpoint contracts; required V9 mechanism unit tests are outside the formal inference path."
}
peak = max(record["peak_sampled_total_gpu_memory_mib"] or 0 for record in executions.values())
summary = {"git_commit_sha": current_sha, "initial_commit_sha": preflight["initial_commit_sha"],
           "tests": preflight["verified_pytest"], "py_compile": preflight["py_compile"],
           "fixes": preflight["fixes"], "protocol": protocol, "selection_model": selection,
           "final_model": final, "development": dm, "outer_test": om,
           "registry_sha256": registry["registry_sha256"], "freeze_sha256": freeze["freeze_sha256"],
           "development_result_sha256": development["result_sha256"], "outer_result_sha256": outer["result_sha256"],
           "compute": {"gpu": preflight["gpu_name"], "selection_training_seconds": selection["training_seconds"],
                       "final_training_seconds": final["training_seconds"],
                       "peak_sampled_total_gpu_memory_mib": peak,
                       "memory_sampling_note": "Total device usage sampled every 2 seconds; allocator peak unavailable.",
                       "python_executable": preflight["python_executable"], "torch": preflight["torch_version"]},
           "audit": audit,
           "convergence": {"first_train_loss": history[0]["train"]["loss"],
                           "last_train_loss": history[-1]["train"]["loss"],
                           "first_early_stop_nll": history[0]["early_stop"]["coordinate_macro_mixture_nll"],
                           "last_early_stop_nll": history[-1]["early_stop"]["coordinate_macro_mixture_nll"],
                           "best_early_stop_nll": selection["best_early_stop_mixture_nll"]}}
finite(summary)
(EVIDENCE / "RUN_SUMMARY_V18.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

lines = [
"V18 单成员实验已完成 register → train-selection → freeze → train-final → outer，outer 只运行一次。"
"角色隔离、freeze gate 和数值检查通过，但共享 loader 在 freeze 前加载全表，"
"因此用户要求的“不得提前读取 outer-test 标签”按物理加载口径未完全满足。本报告明确保留这一审计缺口。",
"",
f"**Outer MAE：{om['coordinate_macro_mae_m']:.6f} m**",
f"**Outer Energy Score：{om['coordinate_macro_energy_score_m']:.6f} m**",
f"**Outer NLL：{om['coordinate_macro_mixture_nll']:.6f}**",
"",
f"Git 起始 SHA：{preflight['initial_commit_sha']}",
f"Git 正式实验 SHA：{current_sha}",
"分支：v18-single-member。全部正式阶段的源码哈希保持一致；结果和模型保存在本地，未推送。",
"",
"**TESTS**",
"",
"两个 py_compile 均 exit 0。指定 pytest：12 passed in 3.89s；没有删除、跳过、弱化或修改任何测试。"
"首次为 11 passed / 1 failed，原因是 V18 模块说明文字含协议测试禁止的官方验证文件名。",
"",
f"- ed2e5ab50cc1feedf218246b86357c505a554198：仅修改 uji_v18_experiment.py 的说明文字。",
f"- {current_sha}：修正 DEFAULT_TRAIN，使用仓库根目录 data/UJIndoorLoc/trainingData.csv。",
"- 两处修复均发生在正式 registration 之前。V18_CONFIG、模型定义、种子、目标函数、候选数均未改变。",
"- 额外合成数据核验：CUDA 前向/反向有限、权重归一化、checkpoint 重建后预测逐元素完全一致；不使用真实数据训练。",
"",
"**PROTOCOL**",
"",
f"总训练行：{protocol['training_rows']}；物理坐标：{protocol['training_coordinate_count']}；AP：{registry['n_aps']}。",
"",
"| Role | 坐标数 | 行数 |", "|---|---:|---:|"]
for role in ["member_fit", "early_stop", "selection_refit", "development", "outer_refit", "outer_test"]:
    value = protocol["roles"][role]
    lines.append(f"| {role} | {value['coordinates']} | {value['rows']} |")
lines += [
"",
"三项 disjoint intersection 全部为 0；三项并集恒等式成立；全部角色均包含 13 个 building-floor strata；"
"三项 hull_missing 检查全部为 0。registry self-hash、training 文件 SHA256 均有效；注册文件未修改。"
"official_validation_in_scope=false；checkpoint_count_at_inference=1；residual_experts=1。",
"",
"**MODEL**",
"",
f"selection：{selection['parameters']:,} 参数 / {selection['anchors']} anchors；"
f"final：{final['parameters']:,} 参数 / {final['anchors']} anchors。",
"residual_experts=1；scan_feature_mode=bounded_maximal_invariant；prototype_metric=none；"
"candidate_reranker=true；candidate_reranker_mode=summary；inference_top_k=64；"
"train_candidates=48；posterior_temperature=1.0；Student-t df=4。",
f"selected epoch={selection['selected_epoch']}；early-stop 实际运行 {selection['selection_epochs_run']} epochs；"
"selection_refit 和 final_refit 均只运行冻结的 1 epoch。",
f"early-stop 最优 coordinate-macro mixture NLL：{selection['best_early_stop_mixture_nll']:.9f}。",
"",
"**DEVELOPMENT / OUTER TEST**",
"",
"| 指标 | Development | Outer test |", "|---|---:|---:|"]
for label, key, percent in [
    ("coordinate-macro MAE (m)", "coordinate_macro_mae_m", False),
    ("coordinate-macro mixture NLL", "coordinate_macro_mixture_nll", False),
    ("coordinate-macro Energy Score (m)", "coordinate_macro_energy_score_m", False),
    ("coordinate-macro building error", "coordinate_macro_building_error", True),
    ("coordinate-macro floor error", "coordinate_macro_floor_error", True),
    ("mean anchor entropy (nats)", "mean_anchor_entropy_nats", False),
    ("mean radial predictive scale (m)", "mean_radial_scale_m", False)]:
    if percent:
        lines.append(f"| {label} | {dm[key]*100:.6f}% | {om[key]*100:.6f}% |")
    else:
        lines.append(f"| {label} | {dm[key]:.6f} | {om[key]:.6f} |")
lines += [
"",
"MAE / NLL / Energy Score / building error / floor error 按坐标分组宏平均；"
"entropy 与 radial scale 按现有协议为 scan mean。Energy Score 使用 64 次固定随机采样。",
"",
"**COMPUTE**",
"",
f"selection 训练（epoch selection + refit）：{selection['training_seconds']:.6f} 秒；"
f"final 训练：{final['training_seconds']:.6f} 秒。",
f"GPU：{preflight['gpu_name']}，6,144 MiB；各阶段合计 GPU 总占用采样峰值：{peak} MiB。",
"GPU 以 2 秒间隔用 nvidia-smi 采样，数值含其他进程占用；并非 PyTorch allocator 精确峰值。",
f"解释器：{preflight['python_executable']}；Python 3.12.12；PyTorch {preflight['torch_version']}。",
"现有 warn_only 确定性设置下，PyTorch attention backward 报告 nondeterministic kernel warning；"
"未因此修改冻结代码或配置，不能承诺 CUDA 训练逐位可复现。",
"",
"**AUDIT**",
"",
"- one inference checkpoint：yes（selection/final 是两个阶段产物，推理仅加载该阶段的一个 checkpoint）。",
"- residual_experts == 1：yes。",
"- 正式模型 ensemble / consensus / member weighting：no。",
"- outer 在 freeze 前参与训练或评分：no；outer 评分只运行一次且在 freeze 与 final 之后。",
"- outer 标签在 freeze 前是否被加载到内存：yes。共享 _load_uji / load_dataset 读取整张 trainingData；"
"随后用注册 row indices 限定训练和评分。因此不能将“无提前 outer 评分”写成“无提前 outer 标签加载”。",
"- validationData.csv accessed：no；未访问官方验证文件。",
"- V9/V17 历史结果 JSON / outer outcomes accessed：no。",
"- development 或 outer outcome-driven tuning：no；源码和配置自注册起保持一致。",
"- registry / freeze / development / outer self-hash、checkpoint SHA256 链：全部通过。",
"- 训练全部 loss trajectory、所有 checkpoint tensors / stats 均无 NaN/Inf。",
"",
"**运行判断与 V18.1 项目**",
"",
f"1. 数值稳定、训练目标下降：loss {history[0]['train']['loss']:.6f} → {history[-1]['train']['loss']:.6f}；"
f"但 early-stop NLL {history[0]['early_stop']['coordinate_macro_mixture_nll']:.6f} → "
f"{history[-1]['early_stop']['coordinate_macro_mixture_nll']:.6f}，最优在 epoch 1。"
"存在训练与未见坐标泛化分数的分离，不能称为已达到良好的泛化收敛。",
f"2. 外层均值 radial scale={om['mean_radial_scale_m']:.6f} m，entropy={om['mean_anchor_entropy_nats']:.6f} nats；"
f"entropy 最大参考值 log(64)={math.log(64):.6f}。均值未出现尺度接近零或熵接近 0 / 完全均匀的明显崩溃；"
"仅凭均值不能断言概率已校准。",
"3. radio reranker：训练各 epoch 的 reranker_loss、相关 checkpoint 参数与最终评估汇总均有限；未发现 NaN/Inf。",
f"4. Delaunay：selection 与 final 的全部 13 strata 使用 rank-2 Delaunay；transform 有限、对称、重建一致，"
f"最小 eigenvalue={final['delaunay_min_eigenvalue']:.9f}，邻域不跨 building-floor。",
"5. 下一次 V18.1 需要重新注册并优先处理按 role 的物理数据加载隔离；"
"泛化 NLL 恶化、epoch-1 选择、概率尺度校准、CUDA 严格确定性应作为新实验问题记录。"
"任何架构、loss、候选、温度或 epoch 选择规则修改均属于新实验；本次 outer 不再运行。",
"",
"**SHA256**",
"",
f"- registry self-hash：{registry['registry_sha256']}",
f"- selection checkpoint：{selection['checkpoint_sha256']}",
f"- development result self-hash：{development['result_sha256']}",
f"- freeze self-hash：{freeze['freeze_sha256']}",
f"- final checkpoint：{final['checkpoint_sha256']}",
f"- outer result self-hash：{outer['result_sha256']}",
"",
"完整机器审计：RUN_SUMMARY_V18.json；阶段命令、时间戳、GPU 采样、测试日志、源码哈希快照、"
"outer 单次调用锁、training trajectory 均位于 run_audit/。"
]
(EVIDENCE / "RUN_REPORT_V18.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

artifact_paths = [exp.DEFAULT_REGISTRY, exp.DEFAULT_SELECTION_CHECKPOINT, exp.DEFAULT_DEVELOPMENT_RESULT,
                  exp.DEFAULT_FREEZE, exp.DEFAULT_FINAL_CHECKPOINT, exp.DEFAULT_OUTER_RESULT,
                  EVIDENCE/"RUN_SUMMARY_V18.json", EVIDENCE/"RUN_REPORT_V18.md",
                  AUDIT/"training_trajectory_v18.json", AUDIT/"code_snapshot.json"]
artifact_manifest = {"experiment_commit_sha": current_sha,
                     "files": {str(path.relative_to(ROOT)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                             "bytes": path.stat().st_size}
                               for path in artifact_paths}}
(AUDIT/"artifact_manifest.json").write_text(json.dumps(artifact_manifest,indent=2),encoding="utf-8")
print(json.dumps({"report":str(EVIDENCE/"RUN_REPORT_V18.md"),"summary":str(EVIDENCE/"RUN_SUMMARY_V18.json"),
                  "hash_chain_valid":True,"outer_invocations":1,"audit":audit,
                  "convergence":summary["convergence"],"peak_sampled_total_gpu_memory_mib":peak},indent=2))
