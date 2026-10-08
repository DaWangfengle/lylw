"""Finalize G1 evidence from saved outcomes without rerunning any prediction."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_experiment as run
from fhm_dapp_tim.fhm_dapp import uji_v17_g1_protocol as g1

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
assert ROOT == Path(r"D:\liuyanlunwen")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()
def load(path):
    return json.loads(path.read_text(encoding="utf-8"))
def write_new(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                     "| " + " | ".join("---" for _ in headers) + " |"]
                    + ["| " + " | ".join(map(str, row)) + " |" for row in rows])
def fmt(value):
    return "N/A" if value is None else f"{float(value):.9f}"

def main():
    for path in (run.REGISTRY, run.CALIBRATION, run.FREEZE, run.CONFIRMATION):
        if not path.is_file():
            raise RuntimeError(f"Cannot finalize: stage result missing: {path}")
    initial = load(AUDIT / "INITIAL_EXECUTION_AUDIT.json")
    gate = load(AUDIT / "PREPARE_GATE.json")
    registered = load(AUDIT / "REGISTRY_VERIFICATION.json")
    registry = run._read_hashed(run.REGISTRY, "registry_sha256")
    calibration = run._read_hashed(run.CALIBRATION, "result_sha256")
    freeze = run._read_hashed(run.FREEZE, "result_sha256")
    confirmation = run._read_hashed(run.CONFIRMATION, "result_sha256")
    assert sha(run.REGISTRY) == registered["registry_file_sha256"]
    assert sha(run.REFIT_CSV) == registered["selection_refit_csv_sha256"]
    assert sha(ROOT/"tests/test_uji_v17_g1.py") == registered["G1_test_source_sha256"]
    assert freeze["calibration_result_sha256"] == calibration["result_sha256"]
    assert freeze["calibration_file_sha256"] == sha(run.CALIBRATION)
    assert confirmation["calibration_file_sha256"] == sha(run.CALIBRATION)
    assert confirmation["freeze_file_sha256"] == sha(run.FREEZE)
    assert float(freeze["chosen_gamma"]) == float(confirmation["chosen_gamma"])
    assert float(freeze["chosen_gamma"]) == g1.choose_gamma(calibration["metrics"])
    assert freeze["confirmation_metrics_accessed"] is False
    for result in (calibration, freeze, confirmation):
        assert result["registry_sha256"] == registry["registry_sha256"]
    for name, entry in initial["protected_historical_assets"].items():
        path = ROOT / name
        if "sha256" in entry:
            assert sha(path) == entry["sha256"], name
        else:
            assert path.stat().st_size == entry["bytes"], name
            assert path.stat().st_mtime_ns == entry["mtime_ns"], name
    for name, module in [("g1_protocol",run.g1),("g1_execution",run),
                         ("r0_training",run.r0),("r3_scoring",run.r3)]:
        assert sha(Path(module.__file__)) == registry["source_sha256"][name], name
    for path in (run.DESIGN, ROOT/"V17_G1_CODEX_RUN.txt"):
        assert sha(path) == initial["initial_G1_sources"][path.relative_to(ROOT).as_posix()]["sha256"]
    assert sha(run.TRAIN_DATA) == registry["input_sha256"] == g1.EXPECTED_TRAIN_SHA256
    assert registry["seeds"] == list(g1.SEEDS) and registry["epochs"] == list(g1.EPOCHS)
    for role, result in (("calibration",calibration),("confirmation",confirmation)):
        assert result["groups"] == gate[role]["coordinates"]
        assert result["rows"] == gate[role]["rows"]
        assert result["group_key_sha256"] == gate[role]["coordinate_key_sha256"]
        assert result["estimand"] == "average_loss_over_25_fixed_balanced_E16_subsets"
        assert result["official_validation_predictions"] == result["outer_test_predictions"] == result["external_predictions"] == 0
    check = run.check_members()
    assert check["all_complete"] and check["complete_members"] == 25
    members = []
    for index in range(1,26):
        path = run.member_path(index)
        member = torch.load(path, map_location="cpu", weights_only=True)
        stage = load(AUDIT / f"train-member-{index:02d}_attempt01.json")
        stdout = (AUDIT / f"train-member-{index:02d}_attempt01.stdout.log").read_text(encoding="utf-8")
        saved_result = json.JSONDecoder().raw_decode(stdout[stdout.rfind("\n{")+1:])[0]
        digest = sha(path)
        assert stage["exit_code"] == 0 and saved_result["resumed"] is False
        assert saved_result["sha256"] == digest
        assert member["epoch"] == len(member["refit_history"]) == g1.EPOCHS[index-1]
        assert member["member_seed"] == g1.SEEDS[index-1]
        assert member["registry_sha256"] == registry["registry_sha256"]
        assert member["selection_refit_csv_sha256"] == sha(run.REFIT_CSV)
        assert member["split_contract"]["training_roles"] == ["selection_refit"]
        assert all(value is False for key,value in member["split_contract"].items()
                   if key != "training_roles")
        for result in (calibration,confirmation):
            assert result["input_member_sha256"][index-1]["sha256"] == digest
        members.append({"member":index,"seed":g1.SEEDS[index-1],
                        "epoch":member["epoch"],"checkpoint_sha256":digest,
                        "training_seconds":member["training_seconds"]})
        del member
    inference = confirmation["paired_familywise_inference"]
    gamma = float(freeze["chosen_gamma"])
    if gamma == 0:
        assert inference["status"] == "not_applicable_identity_candidate"
        assert not confirmation["gate"]["passed"]
        for endpoint in ("exact_mixture_mean_mae_m","energy_score_2d_m"):
            assert confirmation["coordinate_group_vectors"]["baseline"][endpoint] == confirmation["coordinate_group_vectors"]["selected"][endpoint]
            assert confirmation["coordinate_group_vectors"]["baseline"][endpoint] == confirmation["coordinate_group_vectors"]["placebo"][endpoint]
    for result in (calibration,confirmation):
        for name, metric in result["metrics"].items():
            realizations = result["realization_metrics"][name]
            assert len(realizations) == 25
            mean_mae = statistics.mean(r["exact_mixture_mean_mae_coordinate_macro_m"] for r in realizations)
            mean_es = statistics.mean(r["energy_score_coordinate_macro_m"] for r in realizations)
            assert abs(mean_mae - metric["mae_2d_m"]) < 1e-8
            assert abs(mean_es - metric["energy_score_2d_m"]) < 1e-8
    stages = []
    for path in sorted(AUDIT.glob("*_attempt*.json")):
        item = load(path)
        if "command" not in item or "exit_code" not in item:
            continue
        assert sha(Path(item["stdout_path"])) == item["stdout_sha256"], path
        assert sha(Path(item["stderr_path"])) == item["stderr_sha256"], path
        stages.append(item)
    confirmation_stages = [s for s in stages if s["stage"] == "confirm"]
    assert len(confirmation_stages) == 1 and confirmation_stages[0]["exit_code"] == 0
    assert (AUDIT / "CONFIRMATION_COMMAND_STARTED.json").exists()
    confirm_started = confirmation_stages[0]["started_utc"]
    calibrate_stage = [s for s in stages if s["stage"] == "calibrate"][0]
    assert calibrate_stage["exit_code"] == 0 and calibrate_stage["ended_utc"] < confirm_started
    warning_records = []
    for path in sorted(AUDIT.glob("*.stderr.log")):
        lines = path.read_text(encoding="utf-8",errors="replace").splitlines()
        for line in lines:
            if "Warning" in line or "non-deterministic" in line:
                warning_records.append({"file":path.name,"warning":line})
    failures = [s for s in stages if s["exit_code"] != 0]
    verification = {
        "status":"FINAL_EXECUTION_VERIFICATION_PASS",
        "recorded_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "registry_file_and_self_hashes_valid":True,
        "calibration_freeze_confirmation_chain_valid":True,
        "protected_historical_assets_unchanged":len(initial["protected_historical_assets"]),
        "training_input_and_frozen_source_hashes_unchanged":True,
        "all_25_fresh_selection_refit_members_verified":True,
        "all_25_checkpoint_hashes_identical_before_and_after_evaluation":True,
        "all_stage_log_hashes_verified":True,
        "unique_confirmation_command_count":1,
        "calibration_before_confirmation":True,
        "calibration_confirmation_coordinate_overlap":0,
        "calibration_confirmation_row_overlap":0,
        "official_validation_predictions":0,"outer_test_predictions":0,"external_predictions":0,
        "historical_next_stage_authorized":False,
        "frozen_gamma":gamma,"decision":confirmation["decision"],
        "gate":confirmation["gate"],
        "test_result":gate["pytest"], "member_records":members,
        "retained_failed_stage_records":failures,"warning_records":warning_records,
        "scientific_source_repair_after_freeze":False,
        "runtime_seconds_sum":sum(s["runtime_seconds"] for s in stages),
        "member_training_seconds_sum":sum(m["training_seconds"] for m in members),
        "Git_operations_performed":False,
        "new_model_or_next_stage_implemented":False,
        "artifact_file_sha256":{p.name:sha(p) for p in
            (run.REGISTRY,run.REFIT_CSV,run.CALIBRATION,run.FREEZE,run.CONFIRMATION)}
    }

    chunks = [
        "# V17-G1 本地执行报告",
        f"生成时间（UTC）：{verification['recorded_utc']}",
        f"**最终决定：{confirmation['decision']}；冻结 gamma={gamma:g}。**",
        "本轮是用户独立授权的 development 几何支持机制探索。历史 V17 selected 的 outer coordinate-macro MAE 为 5.725985 m，V9 为 6.355559 m，历史 next_stage_authorized=false 原样保留。本轮不构成历史实验 PASS、waiver 或独立 outer-test 验证。",
        "## 环境、耗时和执行顺序",
        table(["项目","记录"],[
            ["Python 解释器",initial["python_executable"]],
            ["Python",initial["python_version"].replace("\n"," ")],
            ["操作系统",initial["platform"]],
            ["Torch / CUDA runtime",f"{initial['torch_version']} / {initial['cuda_runtime']}"],
            ["GPU",f"{initial['gpu']['name']}，{initial['gpu']['memory_bytes']} bytes"],
            ["25 成员训练秒数合计",fmt(verification["member_training_seconds_sum"])],
            ["全部日志阶段运行秒数合计",fmt(verification["runtime_seconds_sum"])],
            ["最终测试","37 passed，0 failed / skipped / xfailed"],
            ["确认命令执行次数","1"],
        ]),
        "执行顺序为编译、完整测试、CUDA 预检、prepare、成员 1..25 串行训练、check-members、calibrate 与 freeze、唯一一次 confirm。每条阶段命令的 stdout/stderr、退出码、起止 UTC、运行秒数与日志 SHA256 均独立保存在 run_audit。",
        "## 注册数据、源码和不可变记录",
        table(["文件或绑定","SHA256"],[
            ["trainingData.csv",registry["input_sha256"]],
            ["历史 V17 registry（self）",registry["historical_v17_registry_sha256"]],
            ["历史 V17 registry（原始文件）",registry["historical_v17_registry_file_sha256"]],
            ["G1 DESIGN",registry["design_sha256"]],
            ["G1 registry（self）",registry["registry_sha256"]],
            ["G1 registry（原始文件）",sha(run.REGISTRY)],
            ["selection_refit_v17_g1.csv",sha(run.REFIT_CSV)],
            ["G1 测试文件",registered["G1_test_source_sha256"]],
        ] + list(registry["source_sha256"].items())),
        f"{len(initial['protected_historical_assets'])} 项受保护历史资产核验未变。历史文本源码和证据按 SHA256 检查；历史二进制仅检查文件大小及 mtime，未读取或复用历史模型权重。新 G1 生产模块及设计文件保留初始字节，registry 冻结后未修改这些源码、registry 或模型。保存文件的 checkpoint 哈希在训练完成、calibration 和 confirmation 之间一致。",
        "## 坐标切分与无重叠证明",
        table(["角色","RSS 行数","物理坐标数","坐标键 SHA256","原始行索引 SHA256"],
              [[role,gate[role]["rows"],gate[role]["coordinates"],
                gate[role]["coordinate_key_sha256"],gate[role]["row_index_sha256"]]
               for role in ("selection_refit","calibration","confirmation")]),
        "development 共 151 个物理坐标 / 3,154 行，按 building/floor 与固定 hash seed 18111 分为 calibration 72 组和 confirmation 79 组。同一物理坐标的全部 RSS 行属于同一侧。两侧坐标交集和行索引交集均为 0，并且并集等于 development；两侧与 selection_refit、outer_test 坐标无重叠。切分在任何 G1 预测前固定，证明见 run_audit/PREPARE_GATE.json。",
        "## 25 名成员训练来源及 checkpoint",
        "所有成员使用原始 V17 R0 网络、损失、AdamW、增强、数据变换与固定 epoch，仅重新训练 selection_refit 12,717 行 / 595 anchors。25 个 seeds 为 17301..17325；epoch 源自历史 member_fit/early_stop 决策。本轮未重新选择 epoch，未复用历史 outer_refit checkpoint，也未用 development / confirmation / outer / official 标签拟合或提前停止。",
        table(["成员","seed","epoch","训练秒数","checkpoint SHA256"],
              [[m["member"],m["seed"],m["epoch"],fmt(m["training_seconds"]),
                m["checkpoint_sha256"]] for m in members]),
        "## Calibration、ES 约束与冻结选择",
        f"候选 gamma 固定为 0 / 0.35 / 0.70，邻域半径固定 10m。选择规则为 calibration coordinate-macro MAE 最小，且 ES ≤ gamma=0 ES+0.02m；并列时选择较小 gamma。本轮冻结 gamma={gamma:g}。",
    ]
    base_es = calibration["metrics"]["gamma_0"]["energy_score_2d_m"]
    chunks += [
        table(["gamma","coordinate-macro MAE (m)","coordinate-macro ES (m)",
               "row MAE (m)","ES guard"],
              [[g,fmt(calibration["metrics"][g1.gamma_name(g)]["mae_2d_m"]),
                fmt(calibration["metrics"][g1.gamma_name(g)]["energy_score_2d_m"]),
                fmt(calibration["metrics"][g1.gamma_name(g)]["mae_row_mean_m"]),
                calibration["metrics"][g1.gamma_name(g)]["energy_score_2d_m"] <= base_es+0.02]
               for g in g1.GAMMAS]),
        f"ES guard 上限：{fmt(base_es+0.02)} m。",
        table(["冻结绑定","SHA256"],[
            ["calibration（self）",calibration["result_sha256"]],
            ["calibration（原始文件）",sha(run.CALIBRATION)],
            ["freeze（self）",freeze["result_sha256"]],
            ["freeze（原始文件）",sha(run.FREEZE)],
        ]),
        "calibration 和 freeze 完成后才执行 confirmation。冻结记录的 confirmation_metrics_accessed=false，冻结链和命令时间顺序均已核验。confirmation 未触发重新选择 gamma。",
        "## 唯一一次 Confirmation",
        "baseline 为 gamma=0，selected 为冻结 gamma，placebo 使用相同 gamma 并在训练 building/floor 内以 seed 18112 置乱 log-density。使用原始 R3 E16/L192/K192/alpha1，t4 分布，64 个共同 draws，固定 scoring seed 18114。",
        table(["方案","coordinate-macro MAE (m)","coordinate-macro ES (m)",
               "row MAE (m)","row ES (m)"],
              [[name,fmt(confirmation["metrics"][name]["mae_2d_m"]),
                fmt(confirmation["metrics"][name]["energy_score_2d_m"]),
                fmt(confirmation["metrics"][name]["mae_row_mean_m"]),
                fmt(confirmation["metrics"][name]["energy_row_mean_m"])]
               for name in ("baseline","selected","placebo")]),
        f"confirmation 文件 self hash：{confirmation['result_sha256']}；原始文件 SHA256：{sha(run.CONFIRMATION)}。",
        "统计量为固定 25 个平衡 E16 子集的平均损失：每成员恰好出现 16 次，子集互相重叠。先对每个子集计算损失，再平均；不把平均预测分布当作该估计量，也不把 25 个子集当作 25 次独立重复实验或一个固定 16 成员部署模型。",
        "### 25 子集分布",
    ]
    distribution_rows = []
    for name in ("baseline","selected","placebo"):
        for label,field in [("MAE","exact_mixture_mean_mae_coordinate_macro_m"),
                            ("ES","energy_score_coordinate_macro_m")]:
            values = np.asarray([r[field] for r in confirmation["realization_metrics"][name]])
            distribution_rows.append([name,label,fmt(values.mean()),fmt(values.std(ddof=1)),
                                      *[fmt(v) for v in np.quantile(values,[0,.1,.5,.9,1])]])
    chunks += [
        table(["方案","指标","mean","sample SD","min","q10","median","q90","max"],
              distribution_rows),
        "上表 SD 是重叠子集结果的描述统计，不能解释为 25 个独立重复的标准误。",
        table(["子集","baseline MAE","selected MAE","placebo MAE",
               "baseline ES","selected ES","placebo ES"],
              [[i+1] +
               [fmt(confirmation["realization_metrics"][name][i]["exact_mixture_mean_mae_coordinate_macro_m"])
                for name in ("baseline","selected","placebo")] +
               [fmt(confirmation["realization_metrics"][name][i]["energy_score_coordinate_macro_m"])
                for name in ("baseline","selected","placebo")]
               for i in range(25)]),
        "## 配对推断、familywise bounds 与 gate",
    ]
    applicable = gamma != 0
    if applicable:
        chunks.append("按 confirmation 物理坐标组进行 5,000 次 paired studentized max-T，seed 18113；同时覆盖 MAE 和 Energy Score 两个端点，95% familywise simultaneous upper bounds。推断条件于本轮固定模型、子集和确认坐标样本。")
    else:
        chunks.append("gamma=0 是原始后验的精确 identity，selected、baseline、placebo 的逐坐标向量完全一致。差值为零且零方差，studentized max-T 不适用（N/A），未进行或伪造 bootstrap 区间，决定为 STOP。原始 JSON 中 upper_bound=0 是不适用分支的 gate 哨兵值，并非统计置信上界；本报告统一呈现 N/A。")
    paired_rows = []
    for endpoint,delta in [
        ("exact_mixture_mean_mae_m",confirmation["delta_coordinate_macro_mae_m"]),
        ("energy_score_2d_m",confirmation["delta_coordinate_macro_energy_m"])
    ]:
        comparison = inference["comparisons"][endpoint]
        upper = comparison["simultaneous_upper_bound_m"] if applicable else None
        paired_rows.append([endpoint,fmt(delta),fmt(upper)])
    chunks += [
        table(["端点","selected − baseline (m)","95% familywise simultaneous upper (m)"],paired_rows),
        f"placebo − baseline 的 coordinate-macro MAE：{fmt(confirmation['placebo_delta_mae_m'])} m。",
        table(["GO 条件","结果"],list(confirmation["gate"]["checks"].items())),
        f"全部条件同时通过：{confirmation['gate']['passed']}；原始决定：{confirmation['decision']}。",
        "逐坐标差值、max-T 完整数值及所有子集成员列表见 confirmation_v17_g1.json；没有对不利结果豁免、回头重选、增加确认次数或实现后续模型。",
        "## 失败、修复和 CUDA 确定性",
        "首轮测试：34 passed / 1 failed。旧 R3 测试引用的 DESIGN_V17_R3_OUTER_CONFIRMATION.md 在本地不存在。仅在新 tests/test_uji_v17_g1.py 中增加 session 隔离 fixture，文档明确标为 G1 测试依赖，并引用未修改的原始 R3 评分源码；未宣称其为恢复的原始预注册设计。旧 R3 测试的数值断言及历史文件保持原样。另加独立相反预测的期望损失检验，以及 selection_refit 序列化与已验证 resume 的接口回归。",
        "第二轮测试：36 passed / 1 failed。新增回归使用了错误的嵌套 mae 字段名，产生 KeyError；修正为原始 R3 返回的 exact_mixture_mean_mae_coordinate_macro_m。第三轮完整测试：37 passed / 0 failed，无 skipped / xfailed。G1 生产模块、参数、切分、损失、优化器和 gate 均未改动，修复发生在 prepare 前。",
        "原始 stdout/stderr 和失败退出码均保留。差异与修复证明见 PRECHECK_FIXTURE_REPAIR.json/.diff、PRECHECK_REGRESSION_FIELD_REPAIR.json/.diff；两次失败的 pytest 临时目录也留存于新 G1 模型目录。初始进程启动器失败通过同范围审查执行恢复；第一次修复工具发生的字符串解析错误在执行前中止，无文件变动，已记入修复审计。",
        table(["阶段","attempt","退出码","秒数","stdout SHA256","stderr SHA256"],
              [[s["stage"] + (f"-{s['member_index']:02d}" if s.get("member_index") else ""),
                s["attempt"],s["exit_code"],fmt(s["runtime_seconds"]),
                s["stdout_sha256"],s["stderr_sha256"]] for s in stages]),
        f"stderr 中保存 {len(warning_records)} 条警告记录。原始 CUDA Memory Efficient attention backward 警告说明该内核默认非确定；本轮保留原训练配置和精度，固定 seed 不能承诺 GPU 训练逐位相同。完整警告见 run_audit/FINAL_EXECUTION_VERIFICATION.json 与各成员 stderr 日志。",
        "## 证据范围",
        "本轮只在 V17 development 内探索和确认几何密度机制。595-anchor selection_refit 模型及本轮 confirmation cohort 与历史 746-anchor outer_refit / outer-test 不同，不能把这里的结果与旧 5.725985 m 当作同一测试集比较。",
        "official validation、outer-test 和外部数据预测次数均为 0；未进行外部泛化验证，未完成 Q1 完整证据链，没有部署或最终模型结论。历史 next_stage_authorized=false 未修改。本轮所有操作在本地执行，Git 操作由用户自行负责。",
    ]
    report = run.EVIDENCE / "RUN_REPORT_V17_G1.md"
    with report.open("x",encoding="utf-8",newline="\n") as f:
        f.write("\n\n".join(chunks) + "\n")
    verification["report_file_sha256"] = sha(report)
    write_new(AUDIT / "FINAL_EXECUTION_VERIFICATION.json",verification)
    print(json.dumps({"status":verification["status"],"decision":confirmation["decision"],
                      "chosen_gamma":gamma,"report":str(report),
                      "report_sha256":verification["report_file_sha256"]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
