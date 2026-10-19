"""Append execution facts and paired summaries after independent closeout."""
from pathlib import Path
import csv
import json
import re
import statistics

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    independent = load("INDEPENDENT_CLOSEOUT.json")
    assert independent["mandatory_protocol_audits_status"] == "PASS"
    execution, manifest, revision = (load(n) for n in
        ("EXECUTION_RECORD.json", "EXECUTION_MANIFEST.json", "ENGINEERING_REVISION.json"))
    report = ROOT / "RUN_REPORT_V17_SPATIAL_STRUCTURE.md"
    assert "## 执行收尾与授权例外" not in report.read_text(encoding="utf-8")
    lines = ["", "## 执行收尾与授权例外", "",
             "正式 Phase F 共18组骨干成功（含保留的TUJI六组和用户授权恢复的UTS一组）。原UTS失败尝试及锁原样保留，不能宣称所有原始命令都一次成功。",
             "用户于2026-10-19明确允许最小ES数值修复并继续。USER_AUTHORIZATION_20261019.json、ENGINEERING_REVISION.json和run_logs/authorized_es_and_resume.diff保留完整修订链；原EXECUTION_MANIFEST.json与33份既有F产物逐字不变。",
             "ES修复只在内部采样时忽略全零质量列，输入完整N+E混合仍用于概率指标；不改变任何正权重、中心/尺度、抽样种子、网格、容差或分区。历史ES科学代码未改。TUJI已完成六组保留原结果，不重复评分。",
             "失败UTS恢复严格使用原CAL_FIT_SELECTION.json，SHA256为2b4db10a47dfe653be8fe62939768f485b638311d58f7cdde206ac3955aacaaf；没有重新拟合或重选。原失败前部分CAL-eval指标未持久化，未编造或补写。",
             "首次CUDA父G13布局修复、原beta=0 ES故障、失败回归及pytest临时目录问题均见FIX_RECORD.md，原始stdout/stderr从未覆盖。修复后完整50项回归通过；包含默认路径损失/梯度/预测逐位一致和一次编码器前向复用检查。",
             "Phase T完成45条80轮历史/新训练记录：18条O/A严格核验后复用，27条C/D/E为本轮真实新训练，共2160新轮次。不得将复用的1440历史轮次包装成本轮新训练。各臂/种子仅训练一次，未早停、未延长预算。",
             "批量64、AdamW、原增强与配对种子固定。C/D/E全锚点连续头是预注册损失所需计算；不声称与旧Top32训练的FLOPs或真实GPU时间相同。CUDA确定性算法未启用。", "",
             "|命令标签|状态|退出码|墙钟秒|", "|---|---|---:|---:|"]
    for command in execution["commands"]:
        if "exit_code" in command:
            lines.append(f"|{command['label']}|{command['status']}|{command['exit_code']}|{command['elapsed_seconds']:.3f}|")
    scientific = [c for c in execution["commands"] if c["label"] in ("frozen_phase", "frozen_authorized_resume", "phase_T", "aggregate")]
    seconds = sum(c.get("elapsed_seconds", 0.) for c in scientific)
    lines += ["", f"上述科学阶段命令合计墙钟 {seconds / 60:.2f} 分钟（包括失败F尝试，排除等待用户授权的暂停时间）。", "",
              "## 数据和站点身份", "",
              "|域|FIT站点/扫描|DEV站点/扫描|CAL-fit站点/扫描|CAL-eval站点/扫描|TEST站点/扫描（仅原角色元数据）|",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("tuji", "uts", "uji"):
        r = manifest["runs"][f"{dataset}_seed17301"]
        roles = r["roles"]["site_counts"]
        fmt = lambda role: f"{roles[role]['sites']}/{roles[role]['rows']}"
        lines.append(f"|{dataset}|{fmt('fit')}|{fmt('dev')}|{r['roles']['cal_fit_sites']}/{r['cal_fit_rows']}|{r['roles']['cal_eval_sites']}/{r['cal_eval_rows']}|{fmt('test')}|")
    lines += ["", "原训练容器坐标仅用于复原不可变角色键；没有建立TEST推断包、加载官方测试用于实验或计算其性能。TUJI物理MAT容器包含测试变量，但只解码results_train。", "",
              "7份输入完整SHA256、全部角色键SHA见EXECUTION_MANIFEST.json；45份训练链检查点完整SHA和选中epoch见INDEPENDENT_CLOSEOUT.json的checkpoint_identity。原冻结权重没有被改写。", "",
              "## 预声明训练链的配对差", "",
              "以下保留全部三个种子的均值±样本SD；负MAE/ES为改善。用于回顾性机制判断，不按CAL-eval选择新的训练方案。", "",
              "|域|对照→候选|ΔMAE m|ΔES m|ΔP95 m|ΔP99 m|单站点最大伤害 m|",
              "|---|---|---:|---:|---:|---:|---:|"]
    with (ROOT / "PAIRED_DELTAS.csv").open(encoding="utf-8-sig", newline="") as handle:
        paired = list(csv.DictReader(handle))
    for dataset in ("tuji", "uts", "uji"):
        for reference, candidate in (("O", "A"), ("A", "C"), ("C", "D"), ("D", "E")):
            rows = [r for r in paired if r["phase"] == "T" and r["dataset"] == dataset
                    and r["reference"] == reference and r["candidate"] == candidate]
            assert len(rows) == 3
            def fmt(key):
                v = [float(r[key]) for r in rows]
                return f"{statistics.mean(v):+.6f} ± {statistics.stdev(v):.6f}"
            harm = max(float(r["max_single_site_harm_m"]) for r in rows)
            lines.append(f"|{dataset}|{reference}→{candidate}|{fmt('mae_delta_m')}|{fmt('es_delta_m')}|{fmt('p95_delta_m')}|{fmt('p99_delta_m')}|{harm:.6f}|")
    lines += ["", "## 独立审计的实际范围", "",
              f"独立只读核算全部63组结果、405条运行指标、135条三种子汇总、{independent['paired_rows']}条种子配对、{independent['paired_site_rows']}条站点配对；全部JSON可解析且数值有限。",
              f"站点到宏指标最大浮点差为{independent['maximum_site_reduction_difference']:.12g}；从C/D/E轮次日志重建损失公式的最大差为{independent['maximum_training_composition_rounding_difference']:.12g}。后者为序列化浮点均值审计，未放宽训练的数学不变量。",
              "每条训练历史都是1–80轮，独立回放21–80最优DEV epoch；逐一核验检查点SHA、有限值、E的alpha、全网格确定性选择、beta=0门禁、Jensen界及原始BF质量保持。",
              "194个历史源码/测试、7份原数据及全部原冻结产物仍匹配SHA；单独保存规范图/边ID的SHA256。原始逐扫描指标没有由该运行器输出，因此只声明从已保存站点指标重算宏指标，不能伪称独立从扫描重建站点。",
              "内部TEST评分行数=0；官方测试评分行数=0；未进行任何Git操作。", "",
              "## 额外跨阶段数值诊断：不完全恒等", "",
              "独立审计状态为PASS_WITH_DISCLOSED_CROSS_PHASE_DIFFERENCES：协议强制门禁通过，但额外的跨阶段逐站点恒等检查没有成立，不能笼统称所有附加诊断通过。",
              "18组相同O/A检查点的F与T结果：最大逐站点MAE差0.0006201863m（小于1mm），最大逐站点ES差0.09826636m；BF准确率逐站点完全相同。各组MAE宏差绝对值均小于2e-5m，ES宏差绝对值最大0.001526876m。完整向量见CROSS_PHASE_OA_NUMERICAL_REVIEW.json。",
              "F以8条查询块构建包含虚拟场的pack，T以32条查询块构建原后验pack；CUDA非确定性/形状相关浮点前向与CDF舍入会影响离散MC抽样。已有JSON没有保存逐组件q/中心张量，因此只报告观察到的差异与实现事实，不声称已由新前向唯一证明物理原因。",
              "该额外跨阶段断言并非协议要求的同pack beta=0数学不变量；失败原文保留，没有修改科学源码、容差、权重、结果或重跑评分来凑一致。F内beta=0的严格父权重/中心恒等及NLL/ES allclose仍全部通过。分别解释F/T内部配对结果，避免把跨阶段细小ES差当成模型收益。", ""]
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({"report_append": "complete", "scientific_wall_minutes": seconds / 60,
                      "new_training_epochs": 2160, "reused_training_epochs": 1440}))


if __name__ == "__main__":
    main()
