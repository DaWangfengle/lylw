"""G6 summary from saved evidence only: no inference, selection or bootstrap rerun."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from fhm_dapp_tim.fhm_dapp import uji_v17_g6_calibration as g6
AUDIT=Path(__file__).resolve().parent
def read(name):
    return json.loads((AUDIT/name).read_text(encoding="utf-8-sig"))
manifest=g6._read(g6.MANIFEST,"manifest_sha256")
cal=g6._read(g6.CALIBRATION)
result=g6._read(g6.RESULT)
lock=g6._read(g6.LOCK)
panel=read("TRAINED_PANEL_REVIEW.json")
pre=read("PRE_MANIFEST_REVIEW.json")
historical=read("HISTORICAL_INTEGRITY_CHECK.json")
initial=read("INITIAL_INTEGRITY.json")
assert historical["all_unchanged"]
assert manifest["source_sha256"]==g6._source_hashes()
assert cal["selected_policy"]==g6.select_policy(cal["candidate_metrics"])
assert result["selected_policy"]==cal["selected_policy"]==lock["frozen_policy"]
hashes=[g6._sha256_file(g6.member_path(i)) for i in range(1,6)]
assert hashes==cal["member_sha256"]==result["member_sha256"]==lock["member_sha256"]
assert g6.REPORT.read_text(encoding="utf-8")==g6.make_report(result)
for role,scored in (("calibration",cal),("confirmation",result)):
    assert scored["coordinate_keys"]==manifest["role_coordinate_keys"][role]
    assert len(scored["coordinate_keys"])==manifest["role_site_counts"][role]
    for name,vec in scored["coordinate_vectors"].items():
        for metric,key in (("mae","mae_2d_m"),("es","energy_score_2d_m")):
            a=np.asarray(vec[metric],dtype=np.float64)
            assert len(a)==scored["coordinate_groups"] and np.isfinite(a).all()
            assert math.isclose(float(a.mean()),scored["candidate_metrics"][name][key],abs_tol=1e-10)
        for metric in ("row_mae_m","row_es_m"):
            devices=scored["row_metrics_by_device"][name]
            weighted=sum(x["rows"]*x[metric] for x in devices.values())/scored["rows"]
            assert math.isclose(weighted,scored["candidate_metrics"][name][metric],abs_tol=1e-10)
    for t in g6.T_VALUES:
        names=[g6.candidate_name(t,s) for s in g6.S_VALUES]
        if all(name in scored["coordinate_vectors"] for name in names):
            assert all(scored["coordinate_vectors"][name]["mae"]==
                scored["coordinate_vectors"][names[0]]["mae"] for name in names)
policy=result["selected_policy"]["policy"]
delta={m:np.asarray(result["coordinate_vectors"][policy][m])-
    np.asarray(result["coordinate_vectors"][g6.BASE][m]) for m in ("mae","es")}
gate=result["gate"]
for m,a in delta.items():
    assert math.isclose(float(a.mean()),gate[m+"_delta_m"],abs_tol=1e-12)
if policy!=g6.BASE:
    inference=result["paired_inference"]
    assert inference["seed"]==19603 and inference["bootstrap_repetitions"]==5000
    assert inference["family_size"]==2 and inference["coordinate_groups"]==139
    for m,a in delta.items():
        se=float(a.std(ddof=1)/math.sqrt(len(a)))
        upper=float(a.mean())+inference["critical_max_t"]*se
        assert math.isclose(se,inference["comparisons"][m]["standard_error_m"],abs_tol=1e-12)
        assert math.isclose(upper,gate[m+"_upper_m"],abs_tol=1e-12)
commands=[json.loads(line) for line in (AUDIT/"COMMANDS.jsonl").read_text(encoding="utf-8").splitlines()]
assert sum(c["stage"]=="calibrate" for c in commands)==1
assert sum(c["stage"]=="confirm" for c in commands)==1
assert next(c for c in commands if c["stage"]=="final_check")["exit_code"]==0
assert read("final_check.stdout.log")["status"]=="VERIFIED"
mat_calls=0
for c in commands:
    for stream in ("stdout","stderr"):
        assert g6._sha256_file(AUDIT/(c["stage"]+"."+stream+".log"))==c[stream+"_sha256"]
    path=AUDIT/(c["stage"]+".access.json")
    assert g6._sha256_file(path)==c["access_trace_sha256"]
    trace=json.loads(path.read_text(encoding="utf-8"))
    assert not trace["denied"]
    assert all(Path(x.replace("\\","/")).name in g6.ds.TUJI_DEVICE_FILES.values()
               for x in trace["data_open_counts"])
    for call in trace["restricted_mat_calls"]:
        assert call["variable_names"]==["results_train"] and not call["results_test_returned"]
        assert "results_test" not in call["returned_keys"]
    mat_calls+=len(trace["restricted_mat_calls"])
passed=int(re.search(r"(\d+) passed",(AUDIT/"tests_fixed.stdout.log").read_text()).group(1))
assert passed==47
start=dt.datetime.fromisoformat(initial["started_utc"])
end=dt.datetime.fromisoformat(next(c for c in commands if c["stage"]=="final_check")["ended_utc"])
tz=dt.timezone(dt.timedelta(hours=8))
env=pre["environment"]
out=["# V17-G6 执行摘要","",
 f"最终检查 VERIFIED；科学决定 **{result['decision']}**。CAL 冻结 {policy}（T={result['selected_policy']['T']}, s={result['selected_policy']['scale']}），CAL/CONFIRM 各一次。",
 "",f"北京时间 {start.astimezone(tz).isoformat(timespec='seconds')} 至 {end.astimezone(tz).isoformat(timespec='seconds')}，约 {(end-start).total_seconds()/60:.2f} 分钟；五成员训练命令合计 {sum(c['seconds'] for c in commands if c['stage'].startswith('train_member_')):.2f} 秒。",
 "",f"Python {env['python'].split()[0]}；Torch {env['torch']}；CUDA {env['cuda_runtime']}；{env['gpu']}（{env['gpu_total_memory_bytes']/1024**3:.2f} GiB）。NumPy {env['numpy']}，SciPy {env['scipy']}，pandas {env['pandas']}。解释器 {env['python_executable']}。",
 "", "仅用 TUJI1 五个 MAT 中的 results_train；每次 scipy.io.loadmat 均限制 variable_names=('results_train',)，返回字典无 results_test。物理容器被打开并整体哈希，不能声称测试字节隔离；测试变量未请求/解码/读取坐标或评分。未调用原 load_tuji1，未读取其他提供测试。历史证据只在命令外按原始字节哈希核验，没有解析或用于选择。",
 "",f"此前 {len(commands)} 条命令的 Python open/loadmat 轨迹核验通过，限制变量读取 {mat_calls} 次、禁止路径访问 0；不是操作系统全局监控。历史科学代码/证据 {historical['checked_files']} 文件 SHA256 不变。",
 "", "4595 行 / 919 物理位置 / 259 AP，building/floor=0/0；角色位置交集0。原始 double→适配 float32→FIT 原点居中四位键全角色一一对应，原始与适配四位键数值变化0。",
 "", "| 角色 | 位置 | 行 | 键 SHA256 |","|---|---:|---:|---|"]
for role in ("fit","calibration","confirmation"):
    out.append(f"| {role} | {manifest['role_site_counts'][role]} | {manifest['role_row_counts'][role]} | {pre['role_key_sha256'][role]} |")
out+=["","| 设备 | FIT 行 | CAL 行 | CONFIRM 行 |","|---|---:|---:|---:|"]
for i,name in enumerate(g6.EXPECTED_DEVICES):
    out.append("| "+name+" | "+" | ".join(str(manifest["role_device_rows"][r][str(i)]) for r in ("fit","calibration","confirmation"))+" |")
out+=["",f"共享 FIT 原点 {panel['coordinate_origin']}。5/5 完整训练、hash 各异、未复用历史权重，共享 FIT 锚点/原型/楼宇楼层/原点/encoder AP 掩码；只有 FIT 行参与 loss，没有 CAL/CONFIRM 早停。",
 "", "| 成员 | seed | epoch | checkpoint SHA256 |","|---:|---:|---:|---|"]
for m in panel["members"]:
    out.append(f"| {m['member']} | {m['seed']} | {m['epoch']} | {m['sha256']} |")
out+=["", "权重保留在本地 fhm_dapp_tim/models/v17_g6_distribution_calibration/。原 R0 网络/loss/AdamW/增强不变。E5/L192/K192/alpha1、df4、两流各64 draws、seed19602。FIT RSS/合成零目标 CUDA 检查证明基线 law/均值/MAE/ES/draws 与原 R3 逐位一致，同 T 所有 s 的 MAE 逐位一致，grad 为 None。",
 "", "| CAL 候选 | macro MAE m | macro ES m | MAE 差 m | ES 差 m | 合格 |","|---|---:|---:|---:|---:|---|"]
cb=cal["candidate_metrics"][g6.BASE]
for pair in g6.CANDIDATES:
    name=g6.candidate_name(*pair);m=cal["candidate_metrics"][name]
    out.append(f"| {name} | {m['mae_2d_m']:.9f} | {m['energy_score_2d_m']:.9f} | {m['mae_2d_m']-cb['mae_2d_m']:+.9f} | {m['energy_score_2d_m']-cb['energy_score_2d_m']:+.9f} | {name in cal['selected_policy']['eligible']} |")
out+=["", "仅用 CAL 的 MAE 改善≥0.10m 且 ES 改善≥0.02m 筛选，最小 MAE、ES/名称破平局；用保存指标复核选择一致，没有人工更选。",
 "", "| CONFIRM 方法 | macro MAE m | macro ES m | row MAE m | row ES m |","|---|---:|---:|---:|---:|"]
for name,m in result["candidate_metrics"].items():
    out.append(f"| {name} | {m['mae_2d_m']:.9f} | {m['energy_score_2d_m']:.9f} | {m['row_mae_m']:.9f} | {m['row_es_m']:.9f} |")
out+=["",f"正式 candidate−baseline：MAE **{gate['mae_delta_m']:+.9f}m**，ES **{gate['es_delta_m']:+.9f}m**；95% familywise 同时上界 MAE **{gate['mae_upper_m']:+.9f}m**、ES **{gate['es_upper_m']:+.9f}m**。",
 "",f"Gate：非 baseline={policy!=g6.BASE}；MAE 改善≥0.15m={gate['mae_delta_m']<=-.15}；MAE 上界<0={gate['mae_upper_m']<0}；ES 上界≤0={gate['es_upper_m']<=0}；总 gate={gate['passed']}。",
 "",f"描述性：改善 {result['descriptive']['mae_win_sites']} 位置，恶化 {result['descriptive']['mae_loss_sites']}，相等 {result['descriptive']['mae_ties']}；配对 MAE 差中位数 {result['descriptive']['median_mae_delta_m']:+.9f}m。单因素对照只能解释机制，不能替代冻结候选。"]
if policy!=g6.BASE:
    out.append(f"5000 次坐标配对 maxT，seed19603，MAE/ES 两指标 family，临界值 {result['paired_inference']['critical_max_t']:.9f}。用保存向量和原临界值复核均值/SE/上界，没有重新 bootstrap。")
else:
    out.append("Baseline 回退：确定性零差异，studentized 区间 N/A，未执行零方差 bootstrap；存储上界0表示无变化，不能晋级。")
out+=["","| CONFIRM 方法 | predicted radial spread m | 50%覆盖 | 80%覆盖 | 90%覆盖 |","|---|---:|---:|---:|---:|"]
for name,d in result["reliability"].items():
    out.append(f"| {name} | {d['macro_predicted_radius_m']:.9f} | {d['coverage']['0.5']:.6f} | {d['coverage']['0.8']:.6f} | {d['coverage']['0.9']:.6f} |")
out+=["", "所有被评分候选的逐坐标半径、50/80/90% credible-ball 半径/经验覆盖在 CAL/CONFIRM JSON；有限128样本估计，不是 conformal 保证，不参与选择。",
 "", "| CONFIRM 坐标尾部 | p90 MAE m | 最大 MAE m | >20m 位置 | >50m 位置 |","|---|---:|---:|---:|---:|"]
for name in dict.fromkeys((g6.BASE,policy)):
    a=np.asarray(result["coordinate_vectors"][name]["mae"])
    out.append(f"| {name} | {np.quantile(a,.9):.9f} | {a.max():.9f} | {int((a>20).sum())} | {int((a>50).sum())} |")
out+=["", "这些尾部数值是每物理坐标扫描平均后的误差，不是单扫描极值。设备 row MAE/ES、原始坐标键和对齐向量在结果 JSON。",
 "", "错误与修复：",
 "- 原始测试43 passed/1 failed：报告缺测试要求的 TRAIN 标识。三项新增回归在原 G6 源码上全部 FAIL，复现原始 double→float32 合并检查缺失、排序 JSON 后表格不一致、九候选可靠性输出缺失。",
 "- manifest 前仅修改新增 G6 源码/测试：补原始坐标映射断言和证明；固定 TRAIN 标识/表格排序；补原设计要求的半径/覆盖诊断。原始文件、失败日志和精确 diff 保留；分区/指标公式/随机流/训练/门槛不变。最终指定测试47 passed，编译/预检成功。",
 "- manifest 后，我编写的只读辅助核验错误引用 model.seen_ap_mask（正确位置 model.encoder.seen_ap_mask）。仅更正非科学性辅助检查，保留原稿/差异/失败 stderr，重跑 FIT/合成核验通过。冻结的科学源码/测试/manifest/模型未变，CAL/CONFIRM 未重跑。",
 "- 五训练 stderr 均保留 Memory Efficient attention 非确定性 backward warning；固定种子/CUBLAS 不保证再次训练逐位一致，未改变训练算法。无 OOM、训练失败、成员重启，CAL/CONFIRM 各一次成功。",
 "- 初始 apply_patch 创建日志辅助文件失败，改用 PowerShell 写入；一次摘要工具 JS 字符串解析失败，未启动命令，改为安全字面量。无科学产物受影响；未执行 Git。",
 "", "| 命令阶段 | exit code | 秒 |","|---|---:|---:|"]
for c in commands:
    out.append(f"| {c['stage']} | {c['exit_code']} | {c['seconds']:.3f} |")
out+=["", "本摘要生成命令记录另见 COMMANDS.jsonl。全部 stdout/stderr/退出码/时间/hash/访问轨迹保留，上表 FAIL 没有删除或豁免。","", "关键 SHA256："]
for name,path in (("manifest file",g6.MANIFEST),("calibration file",g6.CALIBRATION),("confirmation lock",g6.LOCK),("confirmation file",g6.RESULT),("report",g6.REPORT)):
    out.append("- "+name+": "+g6._sha256_file(path))
out.append("- manifest self: "+manifest["manifest_sha256"]+"；confirmation self: "+result["result_sha256"])
out.append("- G6 science source: "+manifest["source_sha256"]["g6_code"]+"；tests: "+manifest["source_sha256"]["g6_tests"])
for key,value in manifest["source_sha256"].items():
    if key.startswith("train_container_"):
        out.append("- "+key+": "+value)
out+=["", "仅为 TUJI1 训练变量内的探索性试验；五成员共享 FIT 数据，不是五次独立实验。空间相关可能使普通坐标 bootstrap 偏乐观；ES/覆盖有 Monte Carlo 数值误差，全局 T/s 可能依赖数据集。没有新外部泛化验证或 Q1 完整证据链，G1–G5 历史决定不变。",""]
with (AUDIT/"EXECUTION_SUMMARY.md").open("x",encoding="utf-8",newline="\n") as stream:
    stream.write("\n".join(out))
print(json.dumps({"status":"SAVED_EVIDENCE_REVIEW_VERIFIED","decision":result["decision"],
 "policy":policy,"gate":gate,"tests_passed":passed,"prior_commands":len(commands),
 "summary_sha256":g6._sha256_file(AUDIT/"EXECUTION_SUMMARY.md")},ensure_ascii=False))
