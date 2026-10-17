"""Render the complete saved development evidence without further model inference."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def main():
    summary, data, record, freeze = map(read, ("SUMMARY.json", "DATA_AUDIT.json", "EXECUTION_RECORD.json", "METHOD_FREEZE.json"))
    assert summary["status"] == "PASS" and len(summary["runs"]) == 9
    runs = summary["runs"]
    commands = {v["label"]: v for v in record["commands"]}
    full_commands = [v for v in record["commands"] if v["label"].startswith("full_")]
    assert len(full_commands) == 9 and all(v["exit_code"] == 0 for v in full_commands)
    env = data["environment"]
    lines = ["# MSEP-Loc Joint-location v1：三域、三种子开发实验报告", "",
             "本轮仅为已暴露训练语料上的 split-internal exploratory development。所有正式指标来自 CAL-eval；TEST 未推断、未评分、未参与梯度或选择。三个训练种子共享同一组评价站点，不是三个独立数据集或新盲测。", "",
             f"分支：`{freeze['initial_branch']}`；只读起始提交：`{freeze['initial_commit']}`。冻结时间：{freeze['frozen_at']}。", "",
             f"方法冻结 SHA256：`{hashlib.sha256((HERE/'METHOD_FREEZE.json').read_bytes()).hexdigest()}`。完整源文件、7 个输入、角色行索引/键 SHA 见 METHOD_FREEZE.json 与 DATA_AUDIT.json。", "",
             "## 执行与环境", "",
             f"Python `{env['python_executable']}`，{env['python'].splitlines()[0]}；Torch {env['torch']}，CUDA {env['cuda_runtime']}；NumPy {env['numpy']}、SciPy {env['scipy']}、pandas {env['pandas']}。GPU：{env['gpu']}。", "",
             "初始数学单元测试 8/8 通过；附加合成独立核查通过。三域 smoke 均退出 0，只用于运行正确性，不解释其分数。9 次正式运行均退出 0；每个输出目录首次创建，未覆盖或重跑正式结果。科学源码、数学定义、数据适配器及常数未修改，没有 bug 修复或被隐藏的失败。", "",
             f"九次正式命令墙钟合计 {sum(c['duration_seconds'] for c in full_commands)/60:.2f} 分钟。该计时包括载入、训练、校准、评分和保存，不是纯 GPU 内核时间。OMP/MKL 线程上限 4；batch 保持 64，未减小。未强制所有 CUDA 算子确定性，不作跨机器逐位复现承诺。", "",
             "逐命令原始 stdout、stderr、退出码、时长、日志 SHA 位于 run_logs/ 与 EXECUTION_RECORD.json；本轮只有只读 Git 身份查询，没有提交、上传或切分支。", "",
             "## 数据隔离与角色", "",
             "按 (building,floor,原始编码坐标四位小数) 物理站点分组；split_seed=20261017，CAL 再以 20261116 分为 CAL-fit/CAL-eval。所有角色物理位置互斥、扫描不跨角色；只有 FIT 构建参考字典、AP mask、原型、原点和 G13 统计。DEV 仅供基础模型早停，CAL-fit 仅供冻结后验校准，CAL-eval 仅描述性比较。", "",
             "|域|FIT 站点/扫描|DEV|CAL-fit|CAL-eval|TEST 仅分区计数|模型 AP|FIT 重复扫描站点|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in ("tuji", "uts", "uji"):
        a = data["datasets"][d]
        cols = [f"{a['roles'][r]['sites']}/{a['roles'][r]['rows']}" for r in ("fit", "dev", "cal_fit", "cal_eval", "test")]
        lines.append(f"|{d.upper()}|"+"|".join(cols)+f"|{a['fit_seen_aps']}|{a['fit_sites_with_at_least_two_scans']}/{a['fit_anchor_count']}|")
    lines += ["", "TUJI 的 5 个 MAT 是含训练/测试变量的同一物理容器；文件被打开和哈希，但受限适配器只请求并返回 results_train，未解码 results_test。UTS/UJI 仅读取各 training CSV；没有打开官方 test/validation、Tampere 或旧 TEST 结果用于方法选择。此为调用路径和限制加载器证据，并非 OS 级全系统 I/O 追踪。", "",
              "原始训练文件含未来分区元数据，载入后仅按许可角色切片使用；不能声称这些容器或字节从未被打开。RSS>-0.999 作为 G13 检测规则保持历史定义，-1 可能混合缺失与弱信号裁剪；重复扫描不假设相互独立。UJI 原适配器使用 float32 坐标，本轮核实居中前后物理键一一映射，无分组碰撞；空间误差以米计算。", "",
              "## 冻结方法与训练预算", "",
              "两种基础训练从同一初始化、同 seed、同批次顺序开始，只替换邻居标签。原 V17 网络/Student-t df4/30m 残差/原训练增强保持不变；沿用 MSEP 单成员训练接口，domain_adversarial_weight=0。基础最多 80 epoch，DEV 改善须超过 0.01m 保存，20 epoch 后距上次改进 12 epoch 停止；这与历史 E5 训练资源不同。", "",
              "两个基础模型的共同上限/早停规则相同，但实际训练轮数和选中 epoch 可以不同；Raw→Floor 对比包含早停路径差异，不能称为固定 optimizer.update 次数对照。同基座 XY-only→BF-XY 才是相同起点与微调预算的主目标消融。", "",
              "每个基座独立执行 XY-only、BF-XY、BF-XY+G13 三组微调，均 10 epoch、网络 lr=3e-5、AdamW、相同 FIT 批次序列；证据 theta lr=1e-3，初值 [.2,.2] 并限制 [0,1.5]。每组独立深拷贝，不共享训练参数；no-evidence 是独立同预算训练，不是同一模型推断开关。训练只在同 BF 有其他参考时排除自身锚点，单锚点 BF 保留自身。", "",
              "BF-XY 使用全参考字典归一化的精确 -log p(B,F,x,y|RSS)，另加训练期真 BF 条件化 0.3 点风险和 0.02 KL。XY-only 保留二维边缘 NLL、0.3 全混合点风险和 0.02 KL。KL 参考是当前网络分数 detach 后的 T=.6 概率，不是固定原基座的函数，因而不能解释成冻结原模型漂移约束。推断绝不输入查询真 BF；predicted-BF mean 由模型概率最大 BF 再求均值，full mean 则跨所有 BF。TUJI 单 BF，类别监督没有额外信息。", "",
              "冻结 Raw 后验的 3 个校准优化均仅 CAL-fit、站点等权，110 次 Adam(lr=.035)，连续 XY Student-t NLL+0.3 点风险；T∈[.25,3]、theta∈[0,1.5]。不使用 CAL-eval 选择优化迭代、模型或阈值。所有 Raw 校准共享同一连续中心/尺度；学习型 retrieval law 的反事实也仅替换概率评分，未对齐重训。", "",
              "连续 ES 为全 XY Student-t 混合的两流各 32 样本 MC 近似，种子 seed+900，各方法共享随机数；并非 BF 条件 ES。joint NLL 同时衡量 BF 及位置，两者概率对象不同。真实目标仅用于合法训练监督/事后评分。", "",
              "|域/种子|Raw 选中/实际 epoch|Floor 选中/实际 epoch|命令墙钟秒|结果目录|",
              "|---|---:|---:|---:|---|"]
    for r in runs:
        train = r["base_training"]
        cmd = commands[f"full_{r['dataset']}_{r['seed']}"]
        lines.append(f"|{r['dataset']}/{r['seed']}|{train['raw']['best_epoch']}/{len(train['raw']['history'])}|{train['floor_neighbors']['best_epoch']}/{len(train['floor_neighbors']['history'])}|{cmd['duration_seconds']:.2f}|`{r['output']}`|")
    unstable = [(r["dataset"], r["seed"], p, v["best_epoch"], len(v["history"]))
                for r in runs for p, v in r["base_training"].items()
                if v["best_epoch"] <= 2]
    if unstable:
        lines += ["", "较早选中 checkpoint 的基座：" + "；".join(
            f"{d}/{s}/{p} 选中 {best}、实际训练 {total} epoch" for d, s, p, best, total in unstable) +
            "。这是预设早停规则下的真实结果，提示种子/训练路径不稳定；未为这些组单独延长训练。它限制了将 Raw→Floor 差异归因于邻居污染修复的力度。", ""]
    lines += ["", "## CAL-eval 正式指标：全部 8 个训练对照", "",
              "以下均为每站点先平均扫描、再站点等权宏平均。BF acc 为站点宏准确率；P95 对站点平均 full-mean 误差取分位数。所有负结果完整保留。", "",
              "|域/seed|策略|Full MAE m|Predicted-BF MAE m|BF acc %|连续 ES m|Joint NLL|P95 m|",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    policies = ("raw", "floor_neighbors", "raw_xy", "raw_joint", "raw_joint_with_evidence",
                "floor_neighbors_xy", "floor_neighbors_joint", "floor_neighbors_joint_with_evidence")
    for r in runs:
        for p in policies:
            m = r["metrics"][p]
            lines.append(f"|{r['dataset']}/{r['seed']}|{p}|{m['macro_mae_full_m']:.6f}|{m['macro_mae_predicted_bf_m']:.6f}|{100*m['macro_bf_accuracy']:.3f}|{m['macro_es_m']:.6f}|{m['macro_joint_nll']:.6f}|{m['site_p95_full_m']:.6f}|")
    lines += ["", "## 三种子描述性均值与离散程度", "",
              "均值±样本标准差来自同一组站点上的三次训练；ES 还包含随 seed+900 变化的 Monte Carlo 随机流误差。它不是独立站点置信区间，不据此生成 p 值、PASS 或部署选择。每个 seed 内的方法使用配对共同随机数。", "",
              "|域|策略|Full MAE mean±SD m|ES mean±SD m|BF acc mean±SD %|",
              "|---|---|---:|---:|---:|"]
    for d, group in summary["across_seed_descriptive"].items():
        for p in policies:
            m, e, b = (group[p][k] for k in ("macro_mae_full_m", "macro_es_m", "macro_bf_accuracy"))
            lines.append(f"|{d}|{p}|{m['mean']:.6f}±{m['sample_sd']:.6f}|{e['mean']:.6f}±{e['sample_sd']:.6f}|{100*b['mean']:.3f}±{100*b['sample_sd']:.3f}|")
    lines += ["", "## 同站点配对：目标与观测证据", "",
              "差值为后者减前者，误差负值有利；胜/负/平使用 ±1e-6m。SUMMARY.json 保存每个位置的配对向量和最严重伤害/收益，非只保留均值。", "",
              "|域/seed|对照→候选|ΔFull MAE m|ΔPredicted-BF MAE m|ΔES m|ΔBF acc pp|胜/负/平|最大伤害 m|",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        for c in r["comparisons"]:
            if c["before"].startswith("cal_"):
                continue
            lines.append(f"|{r['dataset']}/{r['seed']}|{c['before']}→{c['after']}|{c['mean_delta_mae_m']:+.6f}|{c['mean_delta_mae_predicted_bf_m']:+.6f}|{c['mean_delta_es_m']:+.6f}|{100*c['mean_delta_bf_accuracy']:+.3f}|{c['improved_sites']}/{c['harmed_sites']}/{c['ties_within_1e-6m']}|{c['worst_harm']['delta_mae_m']:+.6f}|")
    lines += ["", "### 各域主对照中跨三种子的最严重站点伤害", "",
              "这里只按预定配对对照描述最大伤害，不把这些真值位置用于模型或阈值选择。其余全部站点在 SUMMARY.json。", "",
              "|域|对照→候选|seed|物理位置 (B,F,X,Y)|原 MAE m|新 MAE m|ΔMAE m|",
              "|---|---|---:|---|---:|---:|---:|"]
    for d in ("tuji", "uts", "uji"):
        for a, b in (("raw_xy", "raw_joint"), ("floor_neighbors_xy", "floor_neighbors_joint"),
                     ("raw_joint", "raw_joint_with_evidence"), ("floor_neighbors_joint", "floor_neighbors_joint_with_evidence")):
            candidates = [(r["seed"], c["worst_harm"]) for r in runs if r["dataset"] == d
                          for c in r["comparisons"] if c["before"] == a and c["after"] == b]
            seed, harm = max(candidates, key=lambda value: value[1]["delta_mae_m"])
            lines.append(f"|{d}|{a}→{b}|{seed}|`{json.dumps(harm['site_key'])}`|{harm['mae_before_m']:.6f}|{harm['mae_after_m']:.6f}|{harm['delta_mae_m']:+.6f}|")
    lines += ["", "## 冻结 Raw：T 与 G13 的同条件 CAL-fit / CAL-eval 对照", "",
              "|域/seed|策略|T|theta vis/strength|Full MAE m|Predicted-BF MAE m|BF acc %|ES m|Joint NLL|P95 m|",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        for p, par in r["calibration_parameters"].items():
            m = r["metrics"]["cal_"+p]
            lines.append(f"|{r['dataset']}/{r['seed']}|{p}|{par['temperature']:.6f}|{par['theta'][0]:.6f}/{par['theta'][1]:.6f}|{m['macro_mae_full_m']:.6f}|{m['macro_mae_predicted_bf_m']:.6f}|{100*m['macro_bf_accuracy']:.3f}|{m['macro_es_m']:.6f}|{m['macro_joint_nll']:.6f}|{m['site_p95_full_m']:.6f}|")
    lines += ["", "### 校准与评分律逐种子配对", "",
              "|域/seed|对照→候选|ΔMAE m|ΔES m|胜/负/平|最大伤害 m|",
              "|---|---|---:|---:|---:|---:|"]
    for r in runs:
        for c in r["comparisons"]:
            if not c["before"].startswith("cal_"):
                continue
            lines.append(f"|{r['dataset']}/{r['seed']}|{c['before']}→{c['after']}|{c['mean_delta_mae_m']:+.6f}|{c['mean_delta_es_m']:+.6f}|{c['improved_sites']}/{c['harmed_sites']}/{c['ties_within_1e-6m']}|{c['worst_harm']['delta_mae_m']:+.6f}|")
    lines += ["", "### 同一 Raw 连续分量：训练评分律与 R3 固定评分律", "",
              "|域/seed|评分律|Full MAE m|Predicted-BF MAE m|BF acc %|ES m|Joint NLL|P95 m|",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        for p, label in (("cal_raw", "固定 R3"), ("learned_training_score_same_centers", "原学习型 retrieval")):
            m = r["metrics"][p]
            lines.append(f"|{r['dataset']}/{r['seed']}|{label}|{m['macro_mae_full_m']:.6f}|{m['macro_mae_predicted_bf_m']:.6f}|{100*m['macro_bf_accuracy']:.3f}|{m['macro_es_m']:.6f}|{m['macro_joint_nll']:.6f}|{m['site_p95_full_m']:.6f}|")
    lines += ["", "## 机制检查与解释边界", "",
              "旧邻居污染为邻居子分布的错误 BF 正质量；基础 loss 另有 0.5 自身标签，因此完整软检索目标中的污染质量为其一半。新邻居的零权重填充均留在自身 BF，稀疏层和单锚点层数见 DATA_AUDIT。旧/新邻居矩阵不同并不单独证明 MAE 变化由污染造成，必须读取同初始化配对结果。", "",
              "|域|旧邻居错误 BF 质量|完整软目标错误 BF 质量|新邻居错误 BF 质量|",
              "|---|---:|---:|---:|"]
    for d, a in data["datasets"].items():
        lines.append(f"|{d}|{a['neighbor_mass_wrong_bf_original']:.6f}|{a['effective_total_soft_target_wrong_bf_mass_original']:.6f}|{a['neighbor_mass_wrong_bf_new']:.6f}|")
    lines += ["", "### 数据实际支持的目标/证据效应", ""]
    for d in ("tuji", "uts", "uji"):
        lines.append(f"**{d.upper()}**（同一 CAL-eval，三个 seed）：")
        lines.append("")
        for before, after in (("raw_xy", "raw_joint"), ("floor_neighbors_xy", "floor_neighbors_joint"),
                              ("raw_joint", "raw_joint_with_evidence"),
                              ("floor_neighbors_joint", "floor_neighbors_joint_with_evidence"),
                              ("cal_temp_fit", "cal_temp_plus_g13")):
            cs = [c for r in runs if r["dataset"] == d for c in r["comparisons"] if c["before"] == before and c["after"] == after]
            ds = [c["mean_delta_mae_m"] for c in cs]
            es = [c["mean_delta_es_m"] for c in cs]
            lines.append(f"- {before}→{after}：三个种子 ΔMAE=[{', '.join(f'{v:+.6f}' for v in ds)}]m，平均 {np.mean(ds):+.6f}m；ΔES=[{', '.join(f'{v:+.6f}' for v in es)}]m。MAE 改善 {sum(v<0 for v in ds)}/3，ES 不恶化 {sum(v<=0 for v in es)}/3。")
        lines.append("")
    lines += ["以上只能支持所列干预在本轮站点和训练预算下的描述性结果。BF-XY 同时改变联合类别 NLL 和训练期点风险的条件化，不能把效果拆成唯一类别监督因果贡献；TUJI 单 BF 的 XY/Joint 理论目标退化一致，数值小差异可能由浮点路径/非确定性产生。CAL 参数触边、零系数和域间不一致均须如实保留，不可为恢复增量再调参。", "",
              "冻结评分律反事实保持中心/尺度相同，因此显示的差异属于评分权重与决策；不证明按 R3 规律重新训练会产生同样收益。概率质量不是真实隐含锚点的可识别校准真值；ES 与 joint NLL、MAE 的权衡不能被单一平均分数替代。", "",
              "最严重配对伤害已保留完整坐标键和基线/候选误差，不得依其真值构建部署门控。共享站点、重复扫描、相邻位置相关、CAL-eval 较小和 ES Monte Carlo 误差限制结论；不将三个训练种子当三次独立站点验证。旧 2.316/4.599/6.760m 来自另一划分，不能直接比较。", "",
              "## 可复核产物与命令", "",
              "每个正式目录：RESULT.json、SITE_DIAGNOSTICS.json；seed17301 另保留 8 个本地 checkpoint，其余两个 seed 按协议不保存 checkpoint。Checkpoint SHA 位于 SUMMARY.json。三个 smoke 目录完整保留，不与正式分数混合。", "",
              "仅用已保存的逐站点 JSON 独立复算宏 MAE、ES、预测 BF MAE、BF accuracy、P95，最大绝对差为 "
              f"{max(r['max_saved_site_reaggregation_difference'] for r in runs):.9g}。原脚本未保存逐扫描误差与逐站点 joint NLL，因而不声称完成这两项的原始扫描级独立复算；宏 joint NLL 原值保留于 RESULT。", "",
              "完整参数与运行历史在 RESULT 与原日志；完整命令如下（解释器均为指定 Python，原始 argv/时间见 EXECUTION_RECORD）：", "", "```text"]
    for cmd in full_commands:
        lines.append(" ".join(cmd["command"]))
    lines += ["```", "", "本轮已完成授权范围内的三域三种子开发消融及只读核查。没有开启 TEST、外部数据、多成员、残差截断或额外参数网格；下一步由研究者基于完整正负证据另行决定，未实施新模型。", ""]
    def paired(domain, before, after):
        return [c for r in runs if r["dataset"] == domain for c in r["comparisons"]
                if c["before"] == before and c["after"] == after]

    def mean_delta(domain, before, after, metric="mean_delta_mae_m"):
        return float(np.mean([c[metric] for c in paired(domain, before, after)]))

    uji_group = [r for r in runs if r["dataset"] == "uji"]
    before_bf = np.mean([r["metrics"]["raw_xy"]["macro_bf_accuracy"] for r in uji_group])*100
    after_bf = np.mean([r["metrics"]["raw_joint"]["macro_bf_accuracy"] for r in uji_group])*100
    frozen_uji = paired("uji", "cal_temp_fit", "cal_temp_plus_g13")
    findings = ["## 本轮最重要的实测现象", "",
        f"1. **BF 识别修复与连续 MAE 改善不同步。** UJI 原邻居基座的 XY-only→BF-XY，三种子平均 BF 准确率从 {before_bf:.3f}% 升至 {after_bf:.3f}%，但 MAE 平均增加 {mean_delta('uji','raw_xy','raw_joint'):.6f}m，三个种子均恶化。UTS 同楼层基座该对照的 MAE/ES 平均分别增加 {mean_delta('uts','floor_neighbors_xy','floor_neighbors_joint'):.6f}/{mean_delta('uts','floor_neighbors_xy','floor_neighbors_joint','mean_delta_es_m'):.6f}m，三种子均恶化；不能把类别改善当成全面定位成功。",
        f"2. **观测证据存在明确域差异。** 同楼层 BF-XY 基座加入证据后，TUJI/UTS/UJI 的平均 MAE 变化分别为 {mean_delta('tuji','floor_neighbors_joint','floor_neighbors_joint_with_evidence'):+.6f}/{mean_delta('uts','floor_neighbors_joint','floor_neighbors_joint_with_evidence'):+.6f}/{mean_delta('uji','floor_neighbors_joint','floor_neighbors_joint_with_evidence'):+.6f}m。TUJI/UTS 三个种子的 MAE 和 ES 均改善，UJI 只有两个种子改善、一个恶化；原邻居 UJI 的证据组三个种子 MAE 均变差，平均 {mean_delta('uji','raw_joint','raw_joint_with_evidence'):+.6f}m。整体均值不能覆盖站点伤害或跨域失败。",
        "3. **固定 T=.6 与联合校准表现不同，但增量不普遍。** UJI 固定 .6 学 G13 的两个 theta 在三个种子均为零，逐指标等同固定 .6；联合学 T/G13 后 T≈1.38–1.51、visibility≈1.48–1.5、strength=0，相对单独学 T 的三次 ΔMAE 为 [" +
        ", ".join(f"{c['mean_delta_mae_m']:+.6f}" for c in frozen_uji) +
        "]m，最后一次仅约 0.3mm。UTS 联合校准仅两个种子提供 MAE/ES 增益，17302 两个 theta 为零。TUJI 增量较一致，但若干 theta 达到预设上界；不能据此扩边界、重挑 T，或声称已证实真实隐含锚点概率校准。",
        f"4. **训练评分律与固定 R3 的差异真实存在，不能外推为对齐重训收益。** 在完全相同 Raw 连续分量上换回学习型 retrieval law，三种子平均 MAE 相对 R3 在 TUJI/UTS/UJI 分别变化 {mean_delta('tuji','cal_raw','learned_training_score_same_centers'):+.6f}/{mean_delta('uts','cal_raw','learned_training_score_same_centers'):+.6f}/{mean_delta('uji','cal_raw','learned_training_score_same_centers'):+.6f}m。TUJI/UTS 三次均变差，UJI 有正有负。",
        "5. **零邻居污染已经核实，跨基座收益归因仍受早停路径影响。** UTS/UJI 旧邻居错误 BF 质量约 93.61%/79.01%，新邻居为零；但 UTS 有基座选中第 1/2 epoch 并在 20 epoch 停止。Raw→Floor 包含训练路径差异，且 TUJI 单 BF 的类别目标退化为无新增类别信息的负对照。本轮没有形成可直接宣布独立泛化或正式晋级的证据。", ""]
    index = lines.index("## 执行与环境")
    lines[index:index] = findings
    with (HERE/"RUN_REPORT.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"report": str(HERE/"RUN_REPORT.md"), "lines": len(lines),
                      "formal_runs": len(runs), "rendered_at": datetime.datetime.now().astimezone().isoformat()}))


if __name__ == "__main__":
    main()
