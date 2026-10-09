from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import re
import sys
import numpy as np

repo=Path.cwd()
evidence=repo/"fhm_dapp_tim/evidence/v17_g5_bayes_action"
audit=evidence/"run_audit"
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1048576),b""):
            digest.update(chunk)
    return digest.hexdigest()
def read(name):
    return json.loads((evidence/name).read_text(encoding="utf-8"))
def self_verify(value,field):
    payload=dict(value);expected=payload.pop(field)
    actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()
    assert actual==expected,field
    return actual

manifest=read("manifest_v17_g5.json")
frozen=read("freeze_v17_g5.json")
lock=read("internal_confirmation.lock")
result=read("confirmation_v17_g5.json")
for value,field in ((manifest,"manifest_sha256"),(frozen,"result_sha256"),
                    (lock,"result_sha256"),(result,"result_sha256")):
    self_verify(value,field)
artifacts=["manifest_v17_g5.json","freeze_v17_g5.json","internal_confirmation.lock",
           "confirmation_v17_g5.json","RUN_REPORT_V17_G5.md"]
artifact_sha={name:sha(evidence/name) for name in artifacts}
assert frozen["manifest_sha256"]==manifest["manifest_sha256"]==result["manifest_sha256"]
assert frozen["manifest_file_sha256"]==artifact_sha["manifest_v17_g5.json"]
assert result["manifest_file_sha256"]==artifact_sha["manifest_v17_g5.json"]
assert lock["freeze_file_sha256"]==result["freeze_file_sha256"]==artifact_sha["freeze_v17_g5.json"]
assert result["lock_file_sha256"]==artifact_sha["internal_confirmation.lock"]
assert result["lock_result_sha256"]==lock["result_sha256"]

context=json.loads((audit/"EXECUTION_CONTEXT.json").read_text(encoding="utf-8"))
prep=json.loads((audit/"PREPARATION_REVIEW.json").read_text(encoding="utf-8"))
smoke=json.loads((audit/"FROZEN_FIT_GPU_SMOKE.json").read_text(encoding="utf-8"))
repairs=json.loads((audit/"PRE_MANIFEST_REPAIRS.json").read_text(encoding="utf-8"))
changed=[p for p,expected in context["historical_integrity_only_hashes"].items()
         if not (repo/p).is_file() or sha(repo/p)!=expected]
assert not changed,"protected scientific source/evidence changed"
for name,expected in manifest["source_sha256"].items():
    assert prep["source_sha256"][name]==expected
assert sha(repo/"fhm_dapp_tim/fhm_dapp/uji_v17_g5_bayes_action.py")==manifest["source_sha256"]["g5_code"]
assert sha(repo/"tests/test_uji_v17_g5_bayes_action.py")==repairs["after_sha256"]["tests"]
assert sha(evidence/"DESIGN_V17_G5.md")==manifest["source_sha256"]["g5_design"]

fit={tuple(k) for k in manifest["role_coordinate_keys"]["fit"]}
confirm={tuple(k) for k in manifest["role_coordinate_keys"]["confirmation"]}
keys=[tuple(k) for k in result["coordinate_keys"]]
assert fit.isdisjoint(confirm) and set(keys)==confirm and keys==sorted(confirm)
assert (len(fit),len(confirm),result["rows"],result["coordinates"])==(334,112,112,112)
assert len(set(keys))==len(keys)
vectors={name:np.asarray(value,dtype=np.float64) for name,value in result["coordinate_loss_vectors"].items()}
for name,vec in vectors.items():
    assert len(vec)==112 and np.isfinite(vec).all()
    assert math.isclose(float(vec.mean()),result["metrics"][name]["coordinate_macro_mae_m"],abs_tol=1e-12)
    assert math.isclose(result["metrics"][name]["row_mae_m"],float(vec.mean()),abs_tol=1e-12)
inference=result["paired_familywise_inference"]
assert (inference["family_size"],inference["bootstrap_repetitions"],inference["seed"],
        inference["confidence"],inference["coordinate_groups"])==(2,5000,19502,.95,112)
assert set(inference["comparisons"])=={"continuous_median_minus_mean","component_median_minus_mean"}
paired={}
for name in ("component_median","continuous_median"):
    delta=vectors[name]-vectors["mean"]
    saved=inference["comparisons"][name+"_minus_mean"]
    mean=float(delta.mean())
    se=float(delta.std(ddof=1)/math.sqrt(len(delta)))
    upper=mean+inference["critical_max_t"]*se
    assert math.isclose(mean,saved["mean_difference_m"],abs_tol=1e-12)
    assert math.isclose(se,saved["standard_error_m"],abs_tol=1e-12)
    assert math.isclose(upper,saved["simultaneous_upper_bound_m"],abs_tol=1e-12)
    paired[name]={"mean_difference_m":mean,"standard_error_m":se,"simultaneous_upper_bound_m":upper}
primary=paired["continuous_median"]
checks={"mae_gain_at_least_0p15m":primary["mean_difference_m"]<=-.15,
        "simultaneous_mae_upper_below_zero":primary["simultaneous_upper_bound_m"]<0}
assert checks==result["gate"]["checks"]
assert result["gate"]["passed"]==all(checks.values())
decision="PROMISING_ACTION_ALIGNMENT" if all(checks.values()) else "NO_INTERNAL_PROMOTION"
assert decision==result["decision"]
es=result["common_energy_score"]
assert es["all_actions_have_identical_predictive_law"] is True
assert es["energy_difference_between_actions_exactly_zero"] is True
assert math.isclose(float(np.mean(es["coordinate_energy_vector"])),es["coordinate_macro_energy_m"],abs_tol=1e-12)
gap=result["continuous_component_median_displacement"]
assert len(gap["coordinate_vector_m"])==112
assert math.isclose(float(np.mean(gap["coordinate_vector_m"])),gap["coordinate_macro_m"],abs_tol=1e-12)

commands=[json.loads(line) for line in (audit/"COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
assert len({x["stage"] for x in commands})==len(commands)
assert sum(x["stage"]=="confirm_once" for x in commands)==1
assert [x["stage"] for x in commands if x["stage"].startswith("train_member_")]==[
    "train_member_"+str(i).zfill(2) for i in range(1,6)]
assert json.loads((audit/"check_after_confirm.stdout.log").read_text(encoding="utf-8"))["status"]=="VERIFIED"
warning_stages=[]
for item in commands:
    name=item["stage"]
    assert sha(audit/(name+".stdout.log"))==item["stdout_sha256"]
    assert sha(audit/(name+".stderr.log"))==item["stderr_sha256"]
    assert sha(audit/(name+".access.json"))==item["access_trace_sha256"]
    trace=json.loads((audit/(name+".access.json")).read_text(encoding="utf-8"))
    assert not trace["denied_attempts"]
    assert all(event["path"] in trace["allowed_data_paths"] and not event["writing"]
               for event in trace["data_open_events"])
    stderr=(audit/(name+".stderr.log")).read_text(encoding="utf-8")
    if stderr.strip():
        warning_stages.append(name)
        assert "Memory Efficient attention" in stderr
    if name.startswith("train_member_"):
        number=int(name.rsplit("_",1)[1])
        epochs=manifest["epochs"][number-1]
        lines=(audit/(name+".stdout.log")).read_text(encoding="utf-8").splitlines()
        terminal=[json.loads(line) for line in lines if line.startswith('{"refit_epoch"')]
        assert terminal[-1]["refit_epoch"]==epochs==terminal[-1]["refit_total_epochs"]
        assert '"reused": false' in "\n".join(lines)
assert warning_stages==["train_member_"+str(i).zfill(2) for i in range(1,6)]
assert [(x["stage"],x["exit_code"]) for x in commands if x["exit_code"]]!=[ ]
assert [(x["stage"],x["exit_code"]) for x in commands if x["exit_code"]]==[("regression_before_fix",1)]
assert "40 passed" in (audit/"pytest_attempt02.stdout.log").read_text(encoding="utf-8")

members=smoke["members"]
for item,saved in zip(members,frozen["members"]):
    p=repo/"fhm_dapp_tim/models/v17_g5_bayes_action"/(
        "member_"+str(item["member"]).zfill(2)+"_seed"+str(item["seed"])+".pt")
    assert item["sha256"]==saved["sha256"]==sha(p)
assert lock["member_sha256"]==result["member_sha256"]==[x["sha256"] for x in members]
assert len(set(result["member_sha256"]))==5
started=dt.datetime.fromisoformat(context["started_utc"])
finished=dt.datetime.fromisoformat(commands[-1]["ended_utc"])
now=dt.datetime.now(dt.timezone.utc)
review={"reviewed_utc":now.isoformat(),"status":"FINAL_SAVED_ARTIFACTS_AND_INTEGRITY_VERIFIED",
        "no_new_prediction_or_bootstrap":True,"historical_files_verified":len(context["historical_integrity_only_hashes"]),
        "historical_changed_files":changed,"artifact_sha256":artifact_sha,"source_sha256":manifest["source_sha256"],
        "frozen_g5_test_sha256":repairs["after_sha256"]["tests"],
        "members":members,"role_groups":manifest["role_groups"],"role_rows":manifest["role_rows"],
        "physical_and_row_overlap":0,"coordinate_key_order_and_vector_means_verified":True,
        "paired_means_standard_errors_and_stored_critical_bounds_reconstructed":paired,
        "all_command_log_hashes_verified":True,"command_count":len(commands),
        "provided_test_access_attempts":0,"historical_outcomes_used_for_tuning":False,
        "confirmation_invocations":1,"model_restarts_or_reuse":0,
        "test_count":40,"decision":decision,"gate":result["gate"],
        "cuda_warning_stages":warning_stages,"start_utc":started.isoformat(),
        "final_check_utc":finished.isoformat(),"elapsed_to_final_check_seconds":(finished-started).total_seconds(),
        "five_training_command_seconds":sum(x["seconds"] for x in commands if x["stage"].startswith("train_member_"))}
with (audit/"FINAL_REVIEW.json").open("x",encoding="utf-8",newline="\n") as handle:
    json.dump(review,handle,ensure_ascii=False,indent=2,allow_nan=False)

env=prep["environment"]
out=["# V17-G5-A 执行摘要","",
     f"完整执行完成；最终 check=VERIFIED；科学决定 **{decision}**。仅 Tampere 训练文件内新分区的五成员探索，未进行提供测试集或外部泛化验证。",
     "",
     f"运行：{started.isoformat()} 至 {finished.isoformat()}（UTC），至最终检查 {(finished-started).total_seconds()/60:.3f} 分钟；五个训练命令合计 {review['five_training_command_seconds']:.3f} 秒，包含进程启动/加载时间。",
     f"环境：Python {env['python']}；解释器 {env['python_executable']}；Torch {env['torch']} / CUDA {env['cuda']}；{env['gpu']} 6GB；NumPy {env['numpy']}、pandas {env['pandas']}、SciPy {env['scipy']}。","",
     "数据只读 Training_rss.csv 与 Training_coordinates.csv：446 行、489 AP、446 物理坐标；FIT 334 行/334 坐标，CONFIRM 112 行/112 坐标。物理键及行交集均为 0。原始浮点坐标、适配器 float32、R0 居中后四位键在全部 446 个位置上一一对应，无压合。",
     f"楼层 z 映射：{prep['floor_height_mapping']}；只计算二维 x/y 欧氏误差。设备未知类别 -1 按原始 R0 映射为一个类别。FIT 坐标原点 {prep['fit_origin_float32']}。",
     "FIT/CONFIRM 分楼层坐标数："+json.dumps(prep["by_floor"],ensure_ascii=False,sort_keys=True)+"。",
     "键 SHA256（按排序列表、紧凑 JSON、UTF-8）："]
for role,h in prep["role_sorted_keys_sha256"].items():
    out.append(f"- {role}: {h}")
out+=["","训练固定原始 V17 R0 网络、损失、AdamW 和增广；锚点字典、原点、可见 AP、设备映射与训练 loss 均由 FIT 行建立。CONFIRM 无训练损失或提前误差评估；未用 G3 完整权重。五成员是同一 FIT 上不同种子的共享锚点训练，不构成五次独立实验。","",
      "| 成员 | seed | 完整 epoch | checkpoint SHA256 |","|---|---:|---:|---|"]
for item in members:
    out.append(f"| {item['member']} | {item['seed']} | {item['epochs']} | {item['sha256']} |")
out+=["","共享原型/RSS、锚点坐标/楼宇/楼层、坐标原点、可见 AP 掩码核验通过；五份哈希各异。freeze 后仅用前 16 个 FIT RSS 扫描进行 CUDA smoke；真实查询标签评分为 0，scorer 的目标为合成零向量。联合 law 16×960，动作 2×128 样本；峰值 allocated "+str(smoke["cuda_peak_allocated_bytes"])+" bytes、reserved "+str(smoke["cuda_peak_reserved_bytes"])+" bytes；原 R3 点均值逐位一致、grad 均为 None、checkpoint 文件未变。","",
      "确认只执行一次；锁在评分前创建并保留。三动作共用原 E5/L192/K192/alpha1、df4 Student-t law。动作 seed19503、风险 seed19504，均 2×128；共用 ES seed19505、2×64；Weiszfeld 64 步/1e-5m。","",
      "| 动作 | 坐标宏 MAE (m) | row MAE (m) | 独立抽样预测欧氏风险 (m) |","|---|---:|---:|---:|"]
for name in ("mean","component_median","continuous_median"):
    metric=result["metrics"][name]
    out.append(f"| {name} | {metric['coordinate_macro_mae_m']:.12f} | {metric['row_mae_m']:.12f} | {result['expected_posterior_euclidean_risk_m'][name]:.12f} |")
out+=["",
      f"唯一共用 ES：坐标宏 {es['coordinate_macro_energy_m']:.12f} m，row {es['row_energy_m']:.12f} m；概率法则未改变，动作间 ES 差严格为零。不宣称概率预测改善。",
      f"连续中位数相对均值的宏平均点位移 {result['coordinate_macro_median_mean_displacement_m']:.12f} m；连续/分量中位数之间 {gap['coordinate_macro_m']:.12f} m，112 个配对位置位移向量已保存。","",
      "| 预声明对照（候选减均值） | 宏平均差 (m) | SE (m) | 95% familywise 同时上界 (m) |","|---|---:|---:|---:|"]
for name in ("continuous_median","component_median"):
    p=paired[name]
    out.append(f"| {name} | {p['mean_difference_m']:+.12f} | {p['standard_error_m']:.12f} | {p['simultaneous_upper_bound_m']:+.12f} |")
out+=[f"","原始 studentized maxT：112 个物理坐标、5,000 次共同配对重采样、seed19502、两个 MAE 对照一个 family、单侧 confidence95%；critical="+str(inference["critical_max_t"])+"。最终核验只从保存向量和保存 critical 复原均值/SE/上界，没有重跑 bootstrap。","",
      f"主要改善 >=0.15m：{'PASS' if checks['mae_gain_at_least_0p15m'] else 'FAIL'}；同时上界 <0：{'PASS' if checks['simultaneous_mae_upper_below_zero'] else 'FAIL'}。固定最终决定 {decision}；没有豁免、不选次要动作晋级、不调整种子/阈值。","",
      f"描述性：连续中位数改善 {result['descriptive']['continuous_median_better_coordinate_groups']} 坐标、恶化 {result['descriptive']['continuous_median_worse_coordinate_groups']} 坐标、相同 {result['descriptive']['unchanged_coordinate_groups']}；配对差中位数 {result['descriptive']['median_paired_difference_m']:+.12f} m。",
      "分楼层主要对照（描述性，不能作为新的 gate）：",
      "| 编码楼层 | 坐标数 | 平均差 (m) | 中位差 (m) |","|---|---:|---:|---:|"]
for f,v in result["descriptive"]["by_encoded_floor"].items():
    out.append(f"| {f} | {v['coordinate_groups']} | {v['mean_delta_m']:+.12f} | {v['median_delta_m']:+.12f} |")
out+=["","row P90/P95 误差："+json.dumps({k:result["descriptive"][k] for k in ("row_error_p90_m","row_error_p95_m")},sort_keys=True)+"。","",
      "测试/修复：首次指定测试 39 passed；额外针对性回归在原实现复现两项 FAIL（regression_before_fix，退出1）：重复归一化破坏原 R3 float32 均值逐位一致，以及风险字典顺序导致报告 JSON 往返不一致。仅在 manifest 前修复 G5 新源码/测试，并补存指令要求的两种中位数位移诊断；最终完整测试 40 passed。原始失败、源码/测试快照、修复 diff/哈希完整保留在 PRE_MANIFEST_REPAIRS.* 与 G5_*_BEFORE_FIX.txt。未改科学参数或研究设计，冻结后未改源码。",
      "工具环境：首次默认 shell 进程因 helper setup refresh 错误未能启动；文件补丁工具两次未能写入新的审计 helper，随后用获准的 PowerShell 原生文件写入完成日志目录。均发生在 manifest 前，未执行科学命令、读取数据或覆盖结果；命令级实际输出见下表及原始日志。",
      "全部五名成员保留相同 CUDA Memory Efficient attention 后向非确定性 warning；沿用原训练实现，不为消除警告改变算法。固定 seed/CUBLAS 配置不保证重新训练逐位一致。无 OOM、训练中断、checkpoint 复用或确认评分失败；确认及最终 check stderr 为空。","",
      "| 阶段 | exit | 秒 |","|---|---:|---:|"]
for item in commands:
    out.append(f"| {item['stage']} | {item['exit_code']} | {item['seconds']:.3f} |")
out+=["",f"{len(commands)} 个记录阶段均保存独立原始 stdout/stderr、UTC 时间、退出码和日志 SHA256（COMMANDS.jsonl）。每阶段 Python open 记录只包含两份允许的训练文件，拒绝访问尝试 0；此证据是 Python 进程内 open hook，不是操作系统全局监控。历史证据仅作字节哈希核验，未解析标签结果或参与 G5 调参。",
      f"{review['historical_files_verified']} 个历史源代码/证据文件与初始哈希逐字匹配；G1/G3/G4 既有决定未修改。本轮未执行任何 Git 操作。五份模型保留本地 fhm_dapp_tim/models/v17_g5_bayes_action/。","",
      "输入/算法/设计 SHA256："]
for name,h in manifest["source_sha256"].items():
    out.append(f"- {name}: {h}")
out+=["- g5_tests: "+repairs["after_sha256"]["tests"],"",
      "冻结正式产物 SHA256："]
for name,h in artifact_sha.items():
    out.append(f"- {name}: {h}")
out+=["- manifest self: "+manifest["manifest_sha256"],
      "- freeze self: "+frozen["result_sha256"],
      "- lock self: "+lock["result_sha256"],
      "- result self: "+result["result_sha256"],"",
      "局限：这是训练文件内一次固定五成员内部探索，未完成独立提供测试集、全成员跨数据集或 Q1 证据链；绝不能与历史 E16/25 或 G3/G4 不同分区的绝对 MAE 直接排名。坐标空间相关使普通坐标 bootstrap 可能高估有效样本量。连续中位数是固定 256 点、最多64步的 Monte Carlo/Weiszfeld 近似；独立风险也只有256点，不能据小数差值认定连续中位数已达到精确 Bayes 最优。风险估计中分量中位数略低于连续中位数，按原样报告，不重抽种子或增大样本寻优。Student-t 抽样/Weiszfeld 与标准决策理论已在 V9 存在，不属于原创理论。",""]
with (audit/"EXECUTION_SUMMARY.md").open("x",encoding="utf-8",newline="\n") as handle:
    handle.write("\n".join(out))
assert {name:sha(evidence/name) for name in artifacts}==artifact_sha
print(json.dumps({"status":review["status"],"decision":decision,"command_count":len(commands),
                  "historical_files_verified":review["historical_files_verified"],
                  "summary_sha256":sha(audit/"EXECUTION_SUMMARY.md"),
                  "final_review_sha256":sha(audit/"FINAL_REVIEW.json"),
                  "formal_artifact_sha256":artifact_sha},ensure_ascii=False,indent=2))
