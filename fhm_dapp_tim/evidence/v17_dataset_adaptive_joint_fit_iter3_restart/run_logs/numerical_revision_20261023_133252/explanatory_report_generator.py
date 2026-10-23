"""Explain completed native results; this does not aggregate or run experiments."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(r"D:\liuyanlunwen")
OUT = ROOT / "fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart"
PREFIX = "numerical_revision_20261023_133252"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")


audit_log = OUT / f"run_logs/{PREFIX}_18_audit_ALL.stdout.log"
entries = [line.removeprefix("READ_ONLY_RESULT=") for line in audit_log.read_text(encoding="utf-8").splitlines()
           if line.startswith("READ_ONLY_RESULT=")]
assert len(entries) == 1
audit = json.loads(entries[0])
assert audit["verified_runs"] == audit["native_completed_runs"] == 27
assert audit["status"] == "READ_ONLY_SCOPE_PASS" and audit["native_status"] == "PASS"
assert audit["datasets"] == ["tuji", "uts", "uji"]
revision = read(OUT / f"run_logs/{PREFIX}/REVISION_FREEZE.json")
context = read(OUT / "run_logs/continuation_27of27_20261023_125047_CONTEXT.json")
metadata = {name: read(OUT / f"run_logs/{label}.json") for name, label in (
    ("UTS failed before repair", "continuation_27of27_20261023_125047_UTS"),
    ("UTS authorized restart", PREFIX + "_15_run_UTS"),
    ("UTS 18-run audit", PREFIX + "_16_audit_UTS"),
    ("UJI", PREFIX + "_17_run_UJI"),
    ("Final 27-run audit", PREFIX + "_18_audit_ALL"))}
for name, meta in metadata.items():
    assert meta["exit_code"] == (1 if name == "UTS failed before repair" else 0)
    assert sha(ROOT / meta["stdout"]) == meta["stdout_sha256"]
    assert sha(ROOT / meta["stderr"]) == meta["stderr_sha256"]

native = read(OUT / "INTEGRITY_REVIEW.json")
native_selection = read(OUT / "JOINT_FIT_SELECTION.json")
assert native["status"] == native_selection["status"] == "PASS"
assert native["completed_runs"] == native_selection["cal_eval_report_count"] == len(native_selection["runs"]) == 27
with (OUT / "NEAR_OPTIMAL_SENSITIVITY.csv").open(encoding="utf-8-sig", newline="") as handle:
    sensitivity_count = sum(1 for _ in csv.DictReader(handle))
assert sensitivity_count == 54
with (OUT / "CANDIDATE_METRICS.csv").open("rb") as handle:
    candidate_rows = sum(1 for _ in handle) - 1
assert candidate_rows == 27 * 52489
native_md = (OUT / "CAL_EVAL_REPORT.md").read_text(encoding="utf-8")
for dataset in ("tuji", "uts", "uji"):
    assert f"| {dataset} |" in native_md
assert "Completed run reports: 27" in native_md

aggregates = ("CANDIDATE_METRICS.csv", "PARETO_FRONTIER.csv", "NEAR_OPTIMAL_SENSITIVITY.csv",
              "CAL_EVAL_REPORT.md", "JOINT_FIT_SELECTION.json", "INTEGRITY_REVIEW.json",
              "RUN_REPORT_V17_DATASET_ADAPTIVE_JOINT_FIT.md")
final = {**audit, "native_aggregate_candidate_rows": candidate_rows,
         "native_sensitivity_rows": sensitivity_count,
         "command_metadata": metadata,
         "native_aggregate_sha256": {name: sha(OUT / name) for name in aggregates},
         "read_only_auditor_sha256": sha(OUT / f"run_logs/{PREFIX}/readonly_auditor.py"),
         "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
target = OUT / "CONTINUATION_READ_ONLY_VERIFICATION.json"
with target.open("x", encoding="utf-8") as handle:
    json.dump(final, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")

summaries = [row for row in audit["summary"] if "method" in row]
deltas = [row for row in audit["summary"] if "comparison" in row]
evaluation = audit["evaluation_metrics"]
pairs = audit["paired_checkpoint_deltas"]
methods = ("joint_calibrated_base", "joint_lambda_0.02", "joint_lambda_0.05", "old_combo_1D_2A")
names = {"joint_calibrated_base": "clean", "joint_lambda_0.02": "joint λ=.02",
         "joint_lambda_0.05": "joint λ=.05", "old_combo_1D_2A": "旧 1D+2A"}
fields = ("macro_mae_full_m", "macro_es_m", "macro_transport_risk_m", "macro_joint_nll",
          "bf_cross_entropy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m", "macro_bf_accuracy")
lines = ["# ITER3 三域 27/27 实测与续跑报告", "",
         "本报告从原程序完成的 CAL-fit 选择、CAL-eval 站点向量和独立只读核算生成。"
         "TUJI 9/9、UTS 9/9、UJI 9/9，原生汇总 PASS；不是只包含最后一个数据域的报告。", "",
         "本轮为已观察训练语料内的校准/机制比较。CAL-fit 选参，CAL-eval 只报告；"
         "internal TEST 与 official TEST 评分数均为 0，没有训练或更新 checkpoint。"
         "CAL-fit 不退化约束不能保证 CAL-eval 不退化；两者必须区分。", "",
         "## 执行、环境与版本", "",
         f"Python `{context['python']}`；{context['python_version']}；Torch {context['torch']}；"
         f"CUDA {context['cuda']}；{context['gpu']}，6GB。", "",
         "UTS 完成并通过只读核算后才启动 UJI，两域没有并行。下表为真实日志元数据；"
         "首次 UTS 失败不计为成功实验，也没有删除。"]
table(lines, ["步骤", "退出码", "耗时秒", "开始", "结束"],
      [(name, value["exit_code"], f"{value['elapsed_seconds']:.3f}", value["started_at"], value["finished_at"])
       for name, value in metadata.items()])
for name, meta in metadata.items():
    lines.extend([f"**{name}**：`{' '.join(meta['command'])}`", "",
                  f"stdout：`{meta['stdout']}`；stderr：`{meta['stderr']}`。", ""])
lines.extend([f"原 TUJI 核心源码 SHA256：`{revision['old_tuji_source_sha256']}`。", "",
              f"授权修订后 UTS/UJI 核心源码 SHA256：`{revision['new_uts_uji_source_sha256']}`。", "",
              "重要版本限制：TUJI 保留原完成的 9 组结果，UTS/UJI 使用授权数值修订。"
              "27/27 是产物完整性，不能称为 27 次逐字节相同源码执行。所有科学常量、"
              "参数池、模型、优化目标、选择规则、数据划分和门禁保持原定义。", "",
              "## 数值故障及修复", "",
              "首次 UTS 17301/C 的 clean MAE 比参考多 1.430511474609375e-6m，触发原 1e-6m 门禁。"
              "独立回放证明根因是候选批量求和、附加零概率列以及宏平均的浮点归约顺序；"
              "与此前 TUJI 距离截断的 float32/float64 问题不同。", "",
              "授权修复统一 beta=0 的物理 softmax、点乘求和及按站点宏平均路径，并将参考 BF accuracy "
              "恢复为导出站点计数比例的双精度汇总。没有抄写基准 MAE、删断言或放宽门禁。"
              "修复过程中的夹具错误、残余数值差、被原测试拦截的准确率汇总尝试和基准舍入问题均保留。", "",
              "最终编译通过、38 项测试通过；UTS/UJI 18 个 checkpoint × 12 几何配置，"
              "216 个 clean 对照 MAE 差全为 0，BF accuracy/BF-CE 门禁均通过。"
              "完整失败与修复版本见 `run_logs/numerical_revision_20261023_133252/FIX_RECORD_CN.md`、"
              "`APPLIED_NUMERICAL_FIX_FINAL.patch`、`REVISION_FREEZE.json`；"
              "独立根因证据见 `run_logs/diagnosis_clean_anchor_20261023_130525/CAUSE_REPORT_CN.md`。", "",
              "## 分域完整指标", "",
              "每行是同域 9 个冻结 checkpoint（3 seeds × C/D/E）的指标等权描述均值。"
              "同一域反复使用同一 CAL-eval 站点，这些均值及 checkpoint 计数不构成独立样本显著性检验。"
              "Transport 列为原程序导出的 transport risk（W1 项），保留其原定义。"
              "P95/P99/worst 是每 checkpoint 的站点尾部指标再求均值，不能解释为合并站点分布的分位数或全局最大。"])
table(lines, ["域", "方法", "MAE m", "ES m", "Transport m", "Joint NLL", "BF-CE", "P95 m", "P99 m", "Worst m", "BF acc"],
      [(row["dataset"], names[row["method"]], *(f"{row[k]:.6f}" for k in fields)) for row in summaries])
lines.extend(["## 主要对照相对 clean 的变化", "", "负 Δ MAE/ES/Transport/NLL/尾部表示下降。BF accuracy 越高越好；BF-CE 越低越好。"])
table(lines, ["域", "λ", "ΔMAE", "ΔES", "ΔTransport", "ΔNLL", "ΔBF-CE", "ΔP95", "ΔP99", "ΔWorst", "ΔBF acc", "MAE 改/坏/同"],
      [(row["dataset"], row["comparison"].split()[0].removeprefix("joint_lambda_"),
        *(f"{row['mean_delta_'+k]:+.6f}" for k in fields),
        f"{row['mae_improved_checkpoints_strict']}/{row['mae_worsened_checkpoints_strict']}/{row['mae_unchanged_checkpoints']}")
       for row in deltas])
lines.extend(["旧组合相对 clean 的 checkpoint MAE 计数（严格差值符号，不加事后阈值）：", ""])
old_rows = []
for dataset in ("tuji", "uts", "uji"):
    group = []
    for seed in (17301, 17302, 17303):
        for arm in "CDE":
            clean = next(x for x in evaluation if x["dataset"] == dataset and x["seed"] == seed and x["backbone"] == arm and x["method"] == methods[0])
            old = next((x for x in evaluation if x["dataset"] == dataset and x["seed"] == seed and x["backbone"] == arm and x["method"] == methods[3]), None)
            if old is not None:
                group.append(old["macro_mae_full_m"]-clean["macro_mae_full_m"])
    old_rows.append((dataset, len(group), f"{statistics.mean(group):+.6f}" if group else "N/A", sum(x < 0 for x in group), sum(x > 0 for x in group), sum(x == 0 for x in group)))
table(lines, ["域", "旧组合可用数 /9", "旧组合 ΔMAE 均值", "改善", "恶化", "不变"], old_rows)
lines.extend(["## 分骨干（3 seeds 均值）", ""])
backbone_rows = []
for dataset in ("tuji", "uts", "uji"):
    for arm in "CDE":
        base = [row for row in evaluation if row["dataset"] == dataset and row["backbone"] == arm and row["method"] == methods[0]]
        for lam in (.02, .05):
            group = [row for row in pairs if row["dataset"] == dataset and row["backbone"] == arm and row["lambda"] == lam]
            backbone_rows.append((dataset, arm, lam, f"{statistics.mean(x['macro_mae_full_m'] for x in base):.6f}",
                                  *(f"{statistics.mean(x['delta_'+k] for x in group):+.6f}" for k in fields[:3])))
table(lines, ["域", "骨干", "λ", "clean MAE", "ΔMAE", "ΔES", "ΔTransport"], backbone_rows)
lines.extend(["## 实测判读", ""])
for row in deltas:
    label = row["comparison"].split()[0].removeprefix("joint_lambda_")
    mae = row["mean_delta_macro_mae_full_m"]
    es = row["mean_delta_macro_es_m"]
    p99 = row["mean_delta_site_p99_full_m"]
    worst = row["mean_delta_worst_site_mae_m"]
    lines.extend([f"{row['dataset'].upper()} λ={label}：平均 ΔMAE={mae:+.6f}m，ΔES={es:+.6f}m，"
                  f"ΔP99={p99:+.6f}m，Δworst={worst:+.6f}m；"
                  f"MAE 严格改善/恶化/不变 checkpoint 为 {row['mae_improved_checkpoints_strict']}/"
                  f"{row['mae_worsened_checkpoints_strict']}/{row['mae_unchanged_checkpoints']}。"
                  + ("MAE 与 ES 均值均下降。" if mae < 0 and es < 0 else "这组结果不支持同时改善 MAE 与 ES 均值。")
                  + ("尾部存在代价，不能称为可靠性已解决。" if p99 > 0 or worst > 0 else "该尾部均值未上升；仍须查阅逐 checkpoint/逐位置伤害。"), ""])
lines.extend(["## 尾部伤害与退化选择", "", "以下位置计数在每 checkpoint 内配对，跨 checkpoint 汇总时重复位置会重复计数；不声称它们相互独立。"])
tail_rows = []
for row in deltas:
    group = [x for x in pairs if x["dataset"] == row["dataset"] and f"joint_lambda_{x['lambda']:.2f}" in row["comparison"]]
    tail_rows.append((row["dataset"], row["comparison"].split()[0].removeprefix("joint_lambda_"),
                      row["alpha_obs_zero_count"], row["beta_zero_count"],
                      sum(all(x["selected_parameters"][k] == 0 for k in ("alpha_prior", "alpha_obs", "beta")) for x in group),
                      sum(x["improved_sites_gt_1e6m"] for x in group),
                      sum(x["harmed_sites_gt_1e6m"] for x in group),
                      f"{max(x['largest_site_harm_m'] for x in group):.6f}",
                      sum(x["delta_worst_site_mae_m"] > 0 for x in group)))
table(lines, ["域", "λ", "alpha_obs=0 /9", "beta=0 /9", "三空间项均0 /9", "位置改善计数和", "位置恶化计数和", "最大单位置恶化 m", "Worst 退化 checkpoint /9"],
      tail_rows)
lines.extend(["## 几何、门禁及身份核查", ""])
geometry_rows = []
for dataset in ("tuji", "uts", "uji"):
    group = [row for row in audit["runs"] if row["dataset"] == dataset]
    geometry_rows.append((dataset, len(group),
                          f"{min(x['s1_edge_range'][0] for x in group)}–{max(x['s1_edge_range'][1] for x in group)}",
                          f"{min(x['s2_edge_range'][0] for x in group)}–{max(x['s2_edge_range'][1] for x in group)}",
                          sum(x["empty_s1_geometries"] for x in group), sum(x["empty_s2_geometries"] for x in group),
                          sum(x["exact_clean_anchors"] for x in group),
                          "/".join(str(v) for v in sorted({x["cal_eval_sites"] for x in group})),
                          "/".join(str(v) for v in sorted({x["cal_eval_scans"] for x in group}))))
table(lines, ["域", "完成", "S1 边数范围", "S2 边数范围", "空 S1", "空 S2", "可行 clean 行", "CAL-eval 位置", "扫描"], geometry_rows)
lines.extend([f"原生候选汇总 {candidate_rows:,} 行；54 行近优敏感性摘要；27 份报告、27 份选择与27份候选表。", "",
              "只读核算逐候选重算 MAE ≤ clean+1e-6m、BF accuracy ≥ clean−1e-12、"
              "BF-CE ≤ clean×1.02+0.005、仅惩罚 BF-CE 上升的归一化项、λ=.02/.05目标及原破平局顺序，"
              "并核对全部几何与各自域的预检、45% 支持门禁。没有把 TUJI 边数套到 UTS/UJI。", "",
              f"站点向量到宏指标最大核算误差 {audit['maximum_site_reduction_error']:.12g}；"
              f"最终 324 个 clean 行最大 MAE 漂移 {audit['maximum_exact_clean_anchor_mae_drift']:.12g}m；"
              f"历史 clean/old-combo 参考最大指标差 {audit['maximum_historical_reference_metric_drift']:.12g}。", "",
              "138 个源码/测试/协议文件与授权修订清单匹配，其中136个与原冻结逐字节相同，"
              "另2个是记录在案的 core/test 数值修订。27 个 checkpoint、7 个训练输入、3 个预检文件、"
              "36 个 TUJI 原始结果文件及5个先前审计文件 SHA 不变。完整清单见独立复核 JSON 和修订 freeze。", "",
              "所有选择和报告中的 internal_test_rows_scored=0、official_test_rows_scored=0。"
              "该结论结合源码角色守卫与导出计数核验；本轮未安装操作系统级文件访问跟踪器。"
              "CAL-eval BF-CE 的逐扫描真类概率未导出，因此该值只能核对保存的汇总，不能从站点 JSON 完整独立重建。", "",
              "## 已知限制与材料位置", "",
              "CAL-fit 多候选筛选后的 CAL-eval 改善或退化只作描述；没有新 TEST/独立外部验证，不能宣称模型晋级。"
              "λ=.02 和 .05 都完整保留，不按 CAL-eval 换赢家。均值改善不能掩盖 ES、楼层或尾部损害。", "",
              "标准原生汇总：`CAL_EVAL_REPORT.md`、`JOINT_FIT_SELECTION.json`、`INTEGRITY_REVIEW.json`；"
              "每 checkpoint 科学原始结果在 `runs/<dataset>_seed<seed>/<C|D|E>/`；"
              "独立核算为 `CONTINUATION_READ_ONLY_VERIFICATION.json`。", "",
              "旧 TUJI-only 报告、publication archive 与审计保持其原阶段快照，不是本次27/27的完整发布包。"
              "原生根汇总由原程序自动更新，旧7份汇总已有备份。大候选 CSV 按既有忽略规则留本地；本轮无 Git 操作。", "",
              "两次成功正式命令的 stderr 原文随日志保留。下列长度为字节；非空内容必须结合原日志解释。", ""])
table(lines, ["步骤", "stderr 字节", "日志 SHA256"],
      [(name, (ROOT / metadata[name]["stderr"]).stat().st_size, metadata[name]["stderr_sha256"])
       for name in ("UTS authorized restart", "UJI")])
lines.extend(["## 训练输入 SHA256", ""])
table(lines, ["文件", "SHA256"], [(name, value) for name, value in revision["training_input_sha256"].items()])
lines.extend(["## 27 个冻结 checkpoint SHA256", ""])
table(lines, ["文件", "SHA256"], [(name, value) for name, value in revision["checkpoint_sha256"].items()])
report_path = OUT / "RUN_REPORT_ITER3_27OF27_CN.md"
with report_path.open("x", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(json.dumps({"report": str(report_path), "verification": str(target), "counts": {"tuji": 9, "uts": 9, "uji": 9},
                  "summary_deltas": deltas}, ensure_ascii=False, allow_nan=False))
