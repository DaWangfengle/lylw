"""Archive summary derived from frozen results; never reopen any provided test."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,re,sys
import numpy as np
from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
root=g3.REPO; e=g3.EVIDENCE; a=e/"run_audit"
ctx=json.loads((a/"CURRENT_EXECUTION_CONTEXT.json").read_text(encoding="utf-8"))
old=json.loads((a/"INITIAL_CONTEXT.json").read_text(encoding="utf-8"))
manifest=g3._load_json(g3.MANIFEST,"manifest_sha256")
frozen=g3._load_json(g3.FREEZE,"result_sha256")
locked=g3._load_json(g3.LOCK,"result_sha256")
r=g3._load_json(g3.RESULT,"result_sha256")
assert g3.REPORT.read_text(encoding="utf-8")==g3.make_report(r)
assert g3._frozen_source_hashes()==manifest["source_sha256"]
for rel,h in ctx["current_source_hashes"].items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
for rel,h in old["source_and_prior_evidence_sha256"].items():
 if rel.startswith(("fhm_dapp_tim/evidence/v17_g1_geometry_support/","fhm_dapp_tim/evidence/v17_g2_mechanism/")):
  assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
records=[]
for p in a.glob("*_attempt*.json"):
 if p.name.endswith(".access.json"): continue
 obj=json.loads(p.read_text(encoding="utf-8"))
 if "stage" in obj and "exit_code" in obj: records.append(obj)
records.sort(key=lambda d:d["started_utc"])
current_start=ctx["recorded_utc"]
current=[d for d in records if d["started_utc"]>=current_start]
assert all(d["exit_code"]==0 for d in current)
assert sum(d["stage"].startswith("train_member_") for d in current)==25
assert sum(d["stage"].startswith("evaluate_test_") for d in records)==1
for d in records:
 for stream in ["stdout","stderr"]:
  assert hashlib.sha256((root/d[stream]).read_bytes()).hexdigest()==d[stream+"_sha256"]
last=json.loads((a/"check_after_test_attempt01.stdout.log").read_text(encoding="utf-8"))
assert last["status"]=="VERIFIED" and last["result_sha256"]==r["result_sha256"]
traces=[json.loads(p.read_text(encoding="utf-8")) for p in a.glob("*.access.json")]
assert all(not t["denied_attempts"] for t in traces)
assert all(t["first_UTS_test_open"] is None for t in traces if t["stage"]!="evaluate_test_attempt01")
trace=json.loads((a/"evaluate_test_attempt01.access.json").read_text(encoding="utf-8"))
first=trace["first_UTS_test_open"]
assert first["lock_and_freeze_existed_before_first_open"] is True
assert first["lock_sha256_before_first_open"]==hashlib.sha256(g3.LOCK.read_bytes()).hexdigest()
assert first["freeze_sha256_before_first_open"]==hashlib.sha256(g3.FREEZE.read_bytes()).hexdigest()
assert locked["freeze_file_sha256"]==first["freeze_sha256_before_first_open"]
assert r["checkpoint_sha256"]==[m["sha256"] for m in frozen["members"]]
for m in frozen["members"]: assert hashlib.sha256(g3.member_path(m["member"]).read_bytes()).hexdigest()==m["sha256"]
keys=r["test_coordinate_keys"]
assert len(keys)==r["test_coordinate_groups"] and keys==sorted(keys)
assert len({tuple(k) for k in keys})==len(keys)
assert r["paired_familywise_inference"]["bootstrap_repetitions"]==5000
assert r["historical_G1_decision"]=="STOP_GEOMETRY_DENSITY_MECHANISM"
descriptions={}; deltas={}
for field,delta_field in [("exact_mixture_mean_mae_m","delta_coordinate_macro_mae_m"),("energy_score_2d_m","delta_coordinate_macro_energy_m")]:
 values={name:np.asarray(r["coordinate_group_vectors"][name][field],dtype=np.float64) for name in ["baseline","g1_gamma_0p7","permuted_density"]}
 assert all(len(v)==len(keys) and np.isfinite(v).all() for v in values.values())
 for name,v in values.items():
  metric="mae_2d_m" if field=="exact_mixture_mean_mae_m" else "energy_score_2d_m"
  assert np.isclose(v.mean(),r["metrics"][name][metric],rtol=0,atol=1e-12)
 delta=values["g1_gamma_0p7"]-values["baseline"]
 assert np.isclose(delta.mean(),r[delta_field],rtol=0,atol=1e-12)
 deltas[field]=delta
 descriptions[field]={"mean_candidate_minus_baseline_m":float(delta.mean()),"median_difference_m":float(np.median(delta)),"improved":int((delta<0).sum()),"worsened":int((delta>0).sum()),"unchanged":int((delta==0).sum())}
paired={"provenance":"Derived only from immutable external_result JSON; no data-file reopen, predictions, resampling or parameter selection.","result_self_sha256":r["result_sha256"],"result_file_sha256":hashlib.sha256(g3.RESULT.read_bytes()).hexdigest(),"difference_orientation":"fixed G1 minus V17 baseline; negative is improvement","coordinate_order":"exact saved sorted BUILDINGID/FLOOR/x/y keys under frozen four-decimal V17 grouping","summary":descriptions,"coordinates":[{"key":key,"delta_mae_m":float(deltas["exact_mixture_mean_mae_m"][i]),"delta_energy_score_m":float(deltas["energy_score_2d_m"][i])} for i,key in enumerate(keys)]}
with (a/"POST_RESULT_PAIRED_DIFFERENCES.json").open("x",encoding="utf-8",newline="\n") as f: json.dump(paired,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
local=lambda s:datetime.fromisoformat(s).astimezone(timezone(timedelta(hours=8))).isoformat()
end=next(d["ended_utc"] for d in records if d["stage"]=="check_after_test_attempt01")
wall=(datetime.fromisoformat(end)-datetime.fromisoformat(current_start)).total_seconds()
training=sum(d["runtime_seconds"] for d in current if d["stage"].startswith("train_member_"))
log_warnings={}
for d in records:
 lines=(root/d["stderr"]).read_text(encoding="utf-8",errors="replace").splitlines()
 for line in lines:
  if "Warning:" in line: log_warnings.setdefault(line,[]).append(d["stage"])
lines=["# V17-G3 UTS 执行摘要","","**执行完成；科学判定：TRANSFER_NOT_ESTABLISHED_ON_UTS（STOP）。最终 check=VERIFIED 仅表示产物一致性通过，不是科学 GO。**","",f"当前版本开始至最终 check（Asia/Shanghai）：{local(current_start)} → {local(end)}；历时 {wall/60:.3f} 分钟。25 名串行训练命令合计 {training/60:.3f} 分钟；唯一一次测试命令 {next(d['runtime_seconds'] for d in records if d['stage']=='evaluate_test_attempt01'):.3f} 秒。","",f"Python：{ctx['python_executable']}；{ctx['python_version']}。Torch {ctx['versions']['torch']} / CUDA {ctx['torch_cuda']}；显卡 {ctx['gpu']}（6GB）。NumPy {ctx['versions']['numpy']}，pandas {ctx['versions']['pandas']}，SciPy {ctx['versions']['scipy']}，pytest {ctx['versions']['pytest']}。","","## 冻结、测试访问与设计边界","","- 用户确认结果盲评，曾知晓文件格式；不声称所有测试文件字节从未被任何人打开。","- 训练仅使用 UTS_training.csv：9,108 行、1,466 个物理锚点、589 AP、1 个楼宇、16 个训练编码楼层；原始位置到 float32/四位锚点键无合并。Pos_x/Pos_y 按冻结设计与适配器的米单位解释；只核对了数值范围，没有另作外部单位验证。","- 训练 RSS 为原 V17 float32 [-1,1] 标度。5,045,233 个缺失单元均为 -1；可见 RSS 范围 [-96,-37] dBm，裁剪到 -1 的可见值数量为 0。","- gamma=0.70、10m、E16/L192/K192/alpha1、原 V17 R0 模型/损失/优化器/增强、seeds 17301..17325、原固定 epochs 全部保持。没有 UTS epoch 搜索。","- 25/25 完整训练，无失败/重启/恢复；25 份哈希唯一，锚点字典、原点和原型记忆逐位一致，均仅来自 UTS_training.csv。","- 训练和冻结前后的测试前共 40 份 Python 数据 open 轨迹仅见训练文件，无测试读取或拒绝尝试。该轨迹不是 OS 全局文件访问监控。",f"- 首次测试 open（Asia/Shanghai）：{local(first['utc'])}，当时锁与 freeze 已存在，哈希与最终文件一致。",f"- evaluate-test 仅执行 1 次，退出 0；同一次调用内部测试 open {sum(n for p,n in trace['python_data_open_counts'].items() if p.lower().endswith('uts_test.csv'))} 次（加载 CSV 与最终 SHA256），不是两次评估。之后仅读取结果 JSON 做本摘要。","- 测试：388 行 / 386 个坐标组。按冻结 V17 键定义，train/test 坐标组交集为 0。未访问 Tampere/TUJI 测试、UJI outer/official 文件，未执行 Git 操作。","- 25 个 E16 子集共享成员与测试坐标，不是 25 次独立实验。历史 G1 STOP、全部 G1/G2 输入哈希及所绑定科学源码保持不变。","- checkpoint val_mae=0 是旧 schema 的兼容占位字段；没有验证评分，不能把该值当成测量结果。","","## 唯一测试结果（m）","","| 方法 | Coordinate macro MAE | Coordinate macro ES | Row MAE | Row ES |","| --- | ---: | ---: | ---: | ---: |"]
for name,m in r["metrics"].items(): lines.append(f"| {name} | {m['mae_2d_m']:.9f} | {m['energy_score_2d_m']:.9f} | {m['mae_row_mean_m']:.9f} | {m['energy_row_mean_m']:.9f} |")
lines.extend(["","差值方向统一为 G1−baseline；正值表示恶化。","",f"MAE 差 {r['delta_coordinate_macro_mae_m']:+.9f}m；ES 差 {r['delta_coordinate_macro_energy_m']:+.9f}m。placebo−baseline MAE {r['placebo_minus_baseline_mae_m']:+.9f}m。","","原 G1 studentized-maxT：5000 次坐标级配对重采样，family=MAE+ES，confidence=0.95，seed=18113。","","| 固定条件 | 结果 | 是否通过 |","| --- | ---: | --- |",f"| MAE 差 ≤ −0.15m | {r['delta_coordinate_macro_mae_m']:+.9f} | FAIL |",f"| MAE 同时上界 < 0 | {r['paired_familywise_inference']['comparisons']['exact_mixture_mean_mae_m']['simultaneous_upper_bound_m']:+.9f} | FAIL |",f"| ES 同时上界 ≤ 0 | {r['paired_familywise_inference']['comparisons']['energy_score_2d_m']['simultaneous_upper_bound_m']:+.9f} | FAIL |",f"| placebo MAE > baseline | {r['placebo_minus_baseline_mae_m']:+.9f} | PASS |","","| 坐标级变化 | 改善 | 恶化 | 不变 | 差值中位数 (m) |","| --- | ---: | ---: | ---: | ---: |"])
for metric,d in descriptions.items(): lines.append(f"| {metric} | {d['improved']} | {d['worsened']} | {d['unchanged']} | {d['median_difference_m']:+.9f} |")
lines.extend(["","逐坐标三种方法的 MAE/ES 与 25 子集分布在不可变 external_result JSON；显式两种配对差在 POST_RESULT_PAIRED_DIFFERENCES.json。未再次推断、重采样或评分。点估计及门槛支持‘UTS 固定迁移未建立’；不把 MAE 的正点差说成已证明显著恶化。历史 UJI 5.726m 来自不同数据与训练参考，不能直接排名；本轮也不证明普遍外部泛化、因果机制或 Q1 完整证据链。","","## 错误、修复和警告","","- 早期预检发现 G3 错误继承 G1 的 candidate<placebo 判据，而本轮预注册的是 placebo>baseline；合成复现失败日志 gate_reproduction_attempt01 保留。PREPARE_GATE_IMPLEMENTATION_FIX.json/.diff、原始源码快照及新版 V17_G3_CODEX_RUN.txt 的修正说明均保留。","- 当前用户提供版本已使用独立 g3_transfer_gate，并含反例、严格零边界、原阈值和非有限输入回归。当前 compile、60 项 tests、preflight 全部通过后才 prepare；本次执行没有再改源码，prepare 后至结束源码哈希未变。","- 25 个训练 stderr 均记录同一 Memory Efficient attention backward 非确定性警告。固定 seeds / CUBLAS_WORKSPACE_CONFIG=:4096:8 不保证该 CUDA backward 逐位可复现；保存实际训练权重哈希，不为消除警告更换算法或重训。测试 stderr 为空。","- 当前版本所有命令退出码为 0；不利结果是科学门槛失败，不能当作工程故障重跑。","","## 单测与命令记录","","每条指定命令的原始 stdout/stderr、退出码、UTC 起止时间、运行秒数和日志 SHA256 均保存在同名 run_audit/*.json/*.stdout.log/*.stderr.log；不重复复制全部日志内容。","","| 阶段 | 退出码 | 秒数 |","| --- | ---: | ---: |"])
for d in records: lines.append(f"| {d['stage']} | {d['exit_code']} | {d['runtime_seconds']:.3f} |")
for d in records:
 if d["stage"].startswith("pytest_"):
  text=(root/d["stdout"]).read_text(encoding="utf-8")
  match=re.search(r"(\d+) passed[^\n]*",text)
  if match: lines.append(f"{d['stage']}：{match.group(0).strip()}。")
lines.extend(["","## 全 25 权重（均为 UTS train-only；保留本地）","","| 成员 | Seed | Epoch | Checkpoint SHA256 |","| --- | ---: | ---: | --- |"])
for m in frozen["members"]: lines.append(f"| {m['member']} | {m['seed']} | {m['epoch']} | {m['sha256']} |")
lines.extend(["","## 关键哈希","","| 文件/字段 | SHA256 |","| --- | --- |",f"| UTS_training.csv | {manifest['source_sha256']['training_csv']} |",f"| UTS_test.csv（仅引用评估内生成的值） | {r['provided_test_data_sha256']} |",f"| G3 source | {manifest['source_sha256']['g3_source']} |",f"| G3 design | {manifest['source_sha256']['design']} |",f"| Manifest self hash | {manifest['manifest_sha256']} |",f"| Freeze self hash | {frozen['result_sha256']} |",f"| Lock self hash | {locked['result_sha256']} |",f"| Result self hash | {r['result_sha256']} |"])
for p in [g3.MANIFEST,g3.FREEZE,g3.LOCK,g3.RESULT,g3.REPORT,a/"POST_RESULT_PAIRED_DIFFERENCES.json"]: lines.append(f"| {p.relative_to(e).as_posix()} | {hashlib.sha256(p.read_bytes()).hexdigest()} |")
lines.extend(["","Manifest/source_sha256 保存完整科学输入与代码哈希；CURRENT_EXECUTION_CONTEXT 保存当前测试和辅助代码版本；G1/G2 原始输入已逐一复核。未改写正式报告、结果、manifest、freeze 或 lock。","","结论保持 TRANSFER_NOT_ESTABLISHED_ON_UTS；原 G1 STOP 保持。没有后续调参、新架构、其他外部测试或 Git 操作。",""])
summary=a/"EXECUTION_SUMMARY.md"
with summary.open("x",encoding="utf-8",newline="\n") as f: f.write("\n".join(lines))
print(json.dumps({"status":"FINAL_SUMMARY_SAVED","official_check":last["status"],"decision":r["decision"],"current_command_count":len(current),"archived_command_count":len(records),"current_command_failures":0,"evaluate_test_count":1,"test_rows":r["test_rows"],"test_coordinates":r["test_coordinate_groups"],"paired_coordinate_counts":descriptions,"summary_sha256":hashlib.sha256(summary.read_bytes()).hexdigest(),"result_file_sha256":hashlib.sha256(g3.RESULT.read_bytes()).hexdigest(),"report_file_sha256":hashlib.sha256(g3.REPORT.read_bytes()).hexdigest()},ensure_ascii=True))
