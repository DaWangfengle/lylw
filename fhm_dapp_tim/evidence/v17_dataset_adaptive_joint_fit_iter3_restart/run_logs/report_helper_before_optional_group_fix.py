"""Render completed TUJI results from saved files; no predictions or selection."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics

root = Path(r"D:\liuyanlunwen")
out = root / "fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3_restart"
old = root / "fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(name):
    with (out / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value):
    return f"{float(value):.8g}"


audit = load(out / "READ_ONLY_VERIFICATION.json")
assert audit["status"] == "TUJI_SCOPE_PASS"
context = load(out / "run_logs/EXECUTION_CONTEXT.json")
frozen = load(out / "EXECUTION_FROZEN.json")
all_metrics = rows("EVALUATION_SUMMARY.csv")
pairs = rows("PAIRED_CAL_EVAL_DELTAS.csv")
parameters = rows("SELECTED_PARAMETER_SUMMARY.csv")
coverage = rows("CANDIDATE_COVERAGE.csv")
fields = ("macro_mae_full_m", "macro_es_m", "macro_transport_risk_m", "macro_joint_nll", "macro_bf_accuracy",
          "bf_cross_entropy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m")
commands = [load(out / "run_logs" / (label + ".json")) for label in
            ("01_compile", "02_pytest", "03_preflight", "04_inspect_preflight", "05_run_tuji", "06_readonly_audit")]
means = {}
for arm in ("C", "D", "E"):
    methods = sorted({row["method"] for row in all_metrics if row["backbone"] == arm})
    for method in methods:
        group = [row for row in all_metrics if row["backbone"] == arm and row["method"] == method]
        assert len(group) == 3
        means[(arm, method)] = {field: statistics.mean(float(row[field]) for row in group) for field in fields}
md = ["# ITER3 TUJI 完整重启与独立审计报告", "",
      "状态：TUJI_SCOPE_PASS，授权范围内 9/9 组完成。只完成 TUJI；UTS/UJI 不进行定位评分。", "",
      f"- 完成报告时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}。",
      f"- 完整输出：`{out}`；失败尝试原样保留：`{old}`。",
      f"- 分支 `{context['branch']}`，基准提交 `{context['head_commit']}`。",
      "- 正式编译退出 0；单测 32 passed，退出 0；九组 FIT 预检退出 0；TUJI 完整重启命令退出 0。",
      "- 原生聚合器固定期待三域 27 组，因此其 INTEGRITY_REVIEW/JOINT_FIT_SELECTION 保留 INCOMPLETE 9/27。此为指令要求的 TUJI 阶段结束状态，不修改原聚合器为 PASS；独立范围审计为 TUJI_SCOPE_PASS 9/9。", "",
      "## 环境、命令与身份", "", f"- Python：`{context['sys_executable']}`；`{context['python']}`。",
      f"- 包版本：`{json.dumps(context['packages'], ensure_ascii=False)}`。",
      f"- Torch CUDA：{context['torch_cuda']}；GPU：{context['cuda_device']['name']}，显存 {context['cuda_device']['total_memory_bytes']} bytes。",
      "- 只读 Git 状态查询曾提示用户全局 .config/git/ignore 访问 Permission denied；查询仍返回成功，未修改权限或 Git 配置。科学命令原始 stderr 独立保存在 run_logs，不把该查询警告当成模型运行失败。",
      f"- 冻结 {audit['source_hashes_verified']} 份源码/测试/协议、9 个 TUJI checkpoint 与 7 份原训练文件 SHA256，运行后全部一致。身份清单见 EXECUTION_FROZEN.json。",
      "- 三份重启前后预检产物 SHA256 完全一致；没有更换 FIT 角色、半径或模型。", "",
      "| 命令 | 退出码 | 秒 |", "|---|---:|---:|"]
for command in commands:
    md.append(f"| {' '.join(command['command'][1:])} | {command['exit_code']} | {command['elapsed_seconds']:.6f} |")
md += ["", "以上是命令实际墙钟时长，不是单独 CUDA kernel/GPU 计算时间。完整命令、stdout/stderr、退出码、时间与日志 SHA256 保存在 run_logs。", "",
       "## 错误、修复与运行次数", "",
       "- 首次 ITER3：30 项单测及预检通过；TUJI seed17301/C 在 geometry_01 失败，退出 1，12.436673 秒。预检按 float64 保留 2209/2946 条 S2 边，运行按 float32 保留 2216 条；两者均超过最低 1326 条，失败来自计数不一致。尚未产生完整选择或 CAL-eval 结果。",
       "- 用户明确批准最小修复与新目录完整重启。实际只在 run、_make_probability 两处对距离计算用坐标添加 float64 转换，并加入两项合成回归。32 项正式测试全部通过。详见 FIX_RECORD.md，以及旧目录授权原文、修复前源文件、APPLIED_PRECISION_FIX.patch 和失败日志。",
       "- 新旧预检输出逐字节相同。45% 支持门禁、1e-6m MAE 门禁、半径、目标、系数、随机数、连续分量及物理角色未改；未删除任何门禁。",
       "- 原源码上的两项修订合成回归均失败，临时预览补丁共 32 passed。首版合成运行夹具及辅助替换助手的准备错误也已在旧 STOPPED 报告披露，未混作额外正式评分。",
       "- 真实 TUJI 命令累计 2 次启动：第一次失败，第二次经授权完整成功。未进行第三次寻优；重启后的每个 checkpoint 只完成一次评分。", "",
       "## 候选和支持覆盖", "",
       f"- 每组 12 × 4,374 = 52,488 个候选，加 1 行 clean reference，共 52,489 行。9 组共 {audit['nominal_evaluated_candidates_excluding_references']:,} 候选；汇总 CSV 含参考共 {audit['root_csv_rows']['CANDIDATE_METRICS.csv']:,} 行。",
       f"- 108 个几何各有且仅有一个 exact clean anchor，全部可行；无因空 S2 排除的正 beta 组合。Pareto 行数 {audit['root_csv_rows']['PARETO_FRONTIER.csv']:,}，近优摘要 {audit['root_csv_rows']['NEAR_OPTIMAL_SENSITIVITY.csv']} 行。",
       "- 每个配置的实际 S1/S2 边数与预检一致；半径均来自实际 union-kNN 边分位数，核温度来自规定的 FIT 邻居/边尺度。几何与覆盖明细见 GEOMETRY_SUPPORT_SUMMARY.csv 和 CANDIDATE_COVERAGE.csv。", "",
       "| FIT 几何组 | 锚点 | 基础边 | S1边范围 | S2边范围 | Infinity对照数 |", "|---|---:|---:|---|---|---:|"]
for key, spec in frozen["geometry_preflight_summary"].items():
    md.append(f"| {key} | {spec['fit_anchors']} | {spec['base_edges']} | {spec['s1_edge_count_range']} | {spec['s2_edge_count_range']} | {spec['infinite_tau_controls']} |")
md += ["", "| seed/模型 | 可行搜索候选（不含参考） | CAL-eval站点 | CAL-eval扫描 |", "|---|---:|---:|---:|"]
for item in audit["runs"]:
    md.append(f"| {item['seed']}/{item['backbone']} | {item['feasible_search_candidates']} | {item['cal_eval_sites']} | {item['cal_eval_scans']} |")
md += ["", "## 所有方案的三种子均值", "",
       "C/D/E 是固定历史 checkpoint 组，不能把训练方法差异或三个种子说成独立采集。全部逐 seed、逐方案原值见 EVALUATION_SUMMARY.csv。ES 为原连续 Student-t 固定随机流 MC32×2 估计；W1/transport 是程序定义的分量中心到真值的加权距离风险，不等同于点均值 MAE。", "",
       "| 模型 | 方案 | MAE(m) | ES(m) | W1(m) | 联合NLL | BF准确率 | BF-CE | P95(m) | P99(m) | 最坏站点(m) |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for (arm, method), values in means.items():
    md.append(f"| {arm} | {method} | " + " | ".join(f(values[field]) for field in fields) + " |")
md += ["", "## 主要候选相对 clean 的配对差", "",
       "下表先在每个 seed 内与相同 checkpoint、相同物理站点的 clean baseline 配对，再取三种子均值。负 MAE/ES/W1/NLL/BF-CE/尾部差值表示下降；BF accuracy 的正差表示提高。最坏站点差是每个方案 worst-site 指标之差，两个方案的最坏位置可能不同。", "",
       "| 模型 | lambda | ΔMAE | ΔES | ΔW1 | ΔNLL | ΔBF准确率 | ΔBF-CE | ΔP95 | ΔP99 | Δ最坏站点 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
signal = []
for arm in ("C", "D", "E"):
    for lam in (.02, .05):
        group = [row for row in pairs if row["backbone"] == arm and float(row["lambda"]) == lam]
        assert len(group) == 3
        values = {field: statistics.mean(float(row["delta_"+field]) for row in group) for field in fields}
        signal.append(values)
        md.append(f"| {arm} | {lam:.2f} | " + " | ".join(f(values[field]) for field in fields) + " |")
md += ["", "### 每个 seed 的配对差与位置伤害", "",
       "改善/恶化站点计数使用配对站点 MAE 差低于 -1e-6m / 高于 +1e-6m，仅作数值报告，不用于选参。完整表见 PAIRED_CAL_EVAL_DELTAS.csv。", "",
       "| seed/模型 | lambda | ΔMAE | ΔES | ΔW1 | ΔNLL | ΔBF准确率 | ΔBF-CE | ΔP95 | ΔP99 | Δ最坏 | 改善/恶化站点 |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in pairs:
    md.append(f"| {row['seed']}/{row['backbone']} | {row['lambda']} | " + " | ".join(f(row['delta_'+field]) for field in fields)
              + f" | {row['improved_sites_1e6m']}/{row['harmed_sites_1e6m']} |")
md += ["", "## 冻结参数", "",
       "CAL-eval 只做报告；两种预声明 lambda 均保留。beta=0 表示没有采用虚拟质量；alpha_prior=alpha_obs=0 表示没有采用对应推断扩散。相应空间参数在这些边界上不能视为已被唯一辨识。", "",
       "| seed/模型 | lambda | 几何ID | R1(m) | tau_graph | R2(m) | tau_virt | T | theta[v,s] | alpha_prior | alpha_obs | beta |",
       "|---|---:|---|---:|---:|---:|---|---:|---|---:|---:|---:|"]
for row in parameters:
    md.append(f"| {row['seed']}/{row['backbone']} | {row['lambda']} | {row['geometry_id']} | {f(row['R_max_S1_m'])} | {f(row['tau_graph_m'])} | {f(row['R_max_S2_m'])} | {row['tau_virt_m']} | {f(row['temperature'])} | {row['theta']} | {row['alpha_prior']} | {row['alpha_obs']} | {row['beta']} |")
mae_better = sum(item["macro_mae_full_m"] < 0 for item in signal)
es_better = sum(item["macro_es_m"] < 0 for item in signal)
joint_better = sum(item["macro_mae_full_m"] < 0 and item["macro_es_m"] < 0 for item in signal)
md += ["", "## 独立审计与解释范围", "",
       "- 从每组原始 CSV 独立重算全部候选的 MAE/BF accuracy/BF-CE 可行性、BF-CE 一侧惩罚、两个 lambda 的目标函数、排序破平局和近优候选数。所选候选全部满足 CAL-fit MAE 不退化（原 1e-6m 门禁）、BF accuracy 不退化及 BFCE<=1.02×clean+0.005。",
       "- 目标严格为 MAE/MAE_clean +0.10×归一化 W1 差 +0.25×max(0,BFCE-BFCE_clean)/max(BFCE_clean,0.05) +lambda×归一化 NLL 差。BF-CE 改善不产生负奖励。",
       f"- 108 个 exact anchor 的最大 MAE 数值漂移 {audit['maximum_exact_anchor_mae_drift']:.12g}m；所有 anchor 通过原可行性断言，没有扩大容差。",
       f"- 逐站点宏平均/尾部和扫描数独立复算；最大站点→宏平均差 {audit['maximum_site_macro_reduction_error']:.12g}。clean 相对历史同 checkpoint 复现最大差 {audit['maximum_clean_baseline_drift']:.12g}；旧 1D+2A 对照最大差 {audit['maximum_old_combo_drift']:.12g}。",
       "- CAL-eval BF-CE 保存为原逐扫描、站点等权的加权汇总；产物未导出逐扫描真类概率，无法仅凭 site JSON 对该值做第二次完整独立重构。CAL-fit 的 BF-CE 门禁和目标可由原始 CSV 独立核算。",
       f"- 六个 C/D/E×lambda 三种子均值主比较中，MAE 下降 {mae_better}/6，ES 下降 {es_better}/6，同时下降 {joint_better}/6；尾部及位置伤害完整保留，不仅报告最好方案。",
       "- CAL-fit 门禁不保证 CAL-eval 或外部 TEST 改善。历史 CAL-eval 已被观察，当前仅为回顾性报告，三个 seeds 共享物理样本资源；不生成新的独立显著性结论或正式晋级。ES 的有限抽样、站点间空间和设备/会话相关是实际限制。",
       "- 不根据 CAL-eval 重新挑 lambda、几何或参数；没有执行训练、UTS/UJI 定位评分、内部或官方 TEST 预测/评分，也没有 Git 管理或上传操作。",
       "- 按本轮要求在 TUJI 审计后停止。UTS/UJI 后续定位阶段保持 NOT RUN，等待用户复核与另行授权。", ""]
with (out / "RUN_REPORT_ITER3_TUJI_CN.md").open("x", encoding="utf-8") as handle:
    handle.write("\n".join(md))
print(json.dumps({"status": "TUJI_SCOPE_PASS", "report": str(out / "RUN_REPORT_ITER3_TUJI_CN.md"),
                  "mae_better_mean_comparisons": mae_better, "es_better_mean_comparisons": es_better,
                  "both_better_mean_comparisons": joint_better}, ensure_ascii=True), flush=True)
