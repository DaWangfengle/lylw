"""Independently audit saved files and describe all results; never run a model."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATASETS = ("tuji", "uts", "uji")
SEEDS = (17301, 17302, 17303)
VARIANTS = ("raw_neighbors", "same_floor_neighbors")
PRIMARY = "best_dev_epoch_21_80"
CHECKPOINTS = ("legacy_early_stop", "best_dev_epoch_1_80", PRIMARY)
POLICIES = ("raw", "temperature_only", "g13_fixed_t06", "temperature_plus_g13")
MAX_FLOAT32_REDUCTION_DIFFERENCE = 2e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    result = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
    def visit(value):
        if isinstance(value, dict):
            for x in value.values():
                visit(x)
        elif isinstance(value, list):
            for x in value:
                visit(x)
        elif isinstance(value, float):
            assert math.isfinite(value)
    visit(result)
    return result


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def replay(history):
    assert [x["epoch"] for x in history] == list(range(1, 81))
    full = min(history, key=lambda x: x["dev_mae_m"])
    post = min(history[20:], key=lambda x: x["dev_mae_m"])
    best, best_epoch, stop = math.inf, None, None
    for row in history:
        if row["dev_mae_m"] < best - .01:
            best, best_epoch = row["dev_mae_m"], row["epoch"]
        if row["epoch"] >= 20 and row["epoch"] - best_epoch >= 12:
            stop = row["epoch"]
            break
    return {
        "legacy_early_stop": (best_epoch, best, stop),
        "best_dev_epoch_1_80": (full["epoch"], full["dev_mae_m"], None),
        PRIMARY: (post["epoch"], post["dev_mae_m"], None),
    }


def stats(values):
    return float(np.mean(values)), statistics.stdev(values) if len(values) > 1 else 0.0


def mean_sd(values, digits=4):
    mean, sd = stats(values)
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def get_sites(diag, stage, variant, policy):
    if stage == "stage1":
        return diag["stage1_cal_eval"][variant][policy]
    return diag["stage2_frozen_calibration"][variant][policy]["sites"]


def get_metrics(result, stage, variant, policy):
    if stage == "stage1":
        return result["stage1"]["cal_eval_checkpoint_metrics"][variant][policy]["metrics"]
    return result["stage2"]["variants"][variant]["calibration"][policy]["metrics"]


def compare_sites(reference, candidate):
    assert set(reference) == set(candidate)
    deltas = {k: candidate[k]["mae_full_m"] - reference[k]["mae_full_m"] for k in reference}
    worst = max(deltas, key=deltas.get)
    best = min(deltas, key=deltas.get)
    return {
        "improved": sum(x < 0 for x in deltas.values()),
        "worsened": sum(x > 0 for x in deltas.values()),
        "unchanged": sum(x == 0 for x in deltas.values()),
        "worst_site": reference[worst]["site"], "worst_delta_m": deltas[worst],
        "best_site": reference[best]["site"], "best_delta_m": deltas[best],
    }


def main():
    target = ROOT / "INTEGRITY_REVIEW.json"
    report_path = ROOT / "RUN_REPORT_V17_TWO_STAGE.md"
    assert not target.exists() and not report_path.exists(), "do not overwrite previous audit/report"
    before = load(ROOT / "BEFORE_RUN_AUDIT.json")
    execution = load(ROOT / "EXECUTION_RECORD.json")
    expected_labels = {"compile", "pytest", "audit", "aggregate"} | {f"{d}_seed{s}" for d in DATASETS for s in SEEDS}
    commands = {x["label"]: x for x in execution["commands"]}
    assert len(commands) == len(execution["commands"]) and expected_labels <= set(commands)
    for label in expected_labels:
        command = commands[label]
        assert command["exit_code"] == 0 and command["status"] == "COMPLETE", label
        for stream in ("stdout", "stderr"):
            assert sha(REPO / command[stream]) == command[f"{stream}_sha256"], (label, stream)
    assert "13 passed" in (REPO / commands["pytest"]["stdout"]).read_text(encoding="utf-8")
    for group in ("source_sha256", "input_sha256"):
        for path, expected in before[group].items():
            assert sha(REPO / path) == expected, (group, path)

    results, diagnostics, artifact_hashes, checkpoint_hashes = {}, {}, {}, {}
    paired_site_reviews = {}
    maximum_reduction_difference = 0.0
    expected_metrics = []
    for dataset in DATASETS:
        for seed in SEEDS:
            name = f"{dataset}_seed{seed}"
            directory = ROOT / name
            result = load(directory / "RESULT.json")
            diag = load(directory / "SITE_DIAGNOSTICS.json")
            manifest = load(directory / "RUN_MANIFEST.json")
            results[(dataset, seed)], diagnostics[(dataset, seed)] = result, diag
            for filename in ("RESULT.json", "SITE_DIAGNOSTICS.json", "RUN_MANIFEST.json", "EPOCH_HISTORY.csv", "CHECKPOINT_SELECTION.csv", "REFERENCE_STATE.pt"):
                artifact_hashes[f"{name}/{filename}"] = sha(directory / filename)
            assert manifest["result_sha256"] == artifact_hashes[f"{name}/RESULT.json"]
            assert manifest["site_diagnostics_sha256"] == artifact_hashes[f"{name}/SITE_DIAGNOSTICS.json"]
            assert manifest["reference_state_sha256"] == artifact_hashes[f"{name}/REFERENCE_STATE.pt"]
            assert manifest["epochs_per_variant"] == 80 and manifest["variant_count"] == 2
            assert not manifest["joint_refit_implemented"] and not manifest["joint_refit_run"]
            assert manifest["evaluation_role"] == "CAL-eval" and manifest["official_test_rows_scored"] == 0
            protocol = result["data_protocol"]
            assert protocol["official_test_rows_scored"] == protocol["internal_test_role_rows_scored"] == 0
            assert protocol["cal_fit_eval_site_overlap"] == 0
            assert result["config"]["calibration_fit_split_seed"] == 20261116
            assert result["config"]["epochs_per_variant"] == 80 and result["config"]["batch"] == 64
            assert result["config"]["es_draws"] == 32 and result["config"]["threads"] == 4
            assert not result["config"]["learning_rate_warmup_schedule_added"]
            for path, expected in result["source_sha256"].items():
                assert sha(REPO / "fhm_dapp_tim/fhm_dapp" / path) == expected

            reference = torch.load(directory / "REFERENCE_STATE.pt", map_location="cpu", weights_only=False)
            role_keys = {role: set(map(tuple, keys)) for role, keys in reference["role_site_keys"].items()}
            assert set(role_keys) == {"fit", "dev", "cal"}
            assert all(not role_keys[a] & role_keys[b] for a in role_keys for b in role_keys if a != b)
            cal_sites = sorted(role_keys["cal"])
            positions = set(np.random.default_rng(20261116).permutation(len(cal_sites))[:len(cal_sites)//2].tolist())
            cal_fit = {key for i, key in enumerate(cal_sites) if i in positions}
            cal_eval = set(cal_sites) - cal_fit
            assert len(cal_fit) == protocol["cal_fit_sites"] and len(cal_eval) == protocol["cal_eval_sites"]
            assert not cal_fit & cal_eval
            for role, keys in role_keys.items():
                got = hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest()
                assert got == protocol["role_site_key_sha256"][role]
                if role != "cal":
                    assert got == before["datasets"][dataset]["roles"][role]["site_keys_sha256"]
            for role, keys in (("cal_fit", cal_fit), ("cal_eval", cal_eval)):
                assert hashlib.sha256(json.dumps(sorted(keys), separators=(",", ":")).encode()).hexdigest() == before["datasets"][dataset]["roles"][role]["site_keys_sha256"]

            history_csv = read_csv(directory / "EPOCH_HISTORY.csv")
            selection_csv = read_csv(directory / "CHECKPOINT_SELECTION.csv")
            assert len(history_csv) == 160 and len(selection_csv) == 6
            for variant in VARIANTS:
                training = result["stage1"]["training"][variant]
                history = training["all_epoch_history"]
                assert training["epochs_completed"] == len(history) == 80
                assert all(math.isfinite(float(v)) for row in history for k, v in row.items() if k not in ("dataset", "variant"))
                rows = [x for x in history_csv if x["variant"] == variant]
                for a, b in zip(history, rows, strict=True):
                    assert set(a) == set(b)
                    assert all(str(a[k]) == b[k] if isinstance(a[k], str) else math.isclose(float(a[k]), float(b[k]), rel_tol=1e-12, abs_tol=1e-12) for k in a)
                independent_selection = replay(history)
                states = {}
                for policy in CHECKPOINTS:
                    selection = training["checkpoint_selection"][policy]
                    epoch, score, stop = independent_selection[policy]
                    assert (selection["selected_epoch"], selection["simulated_stop_epoch"]) == (epoch, stop)
                    assert selection["dev_mae_m"] == score
                    path = directory / f"{variant}_{policy}.pt"
                    checkpoint_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    assert ckpt["selected_epoch"] == epoch and ckpt["dev_mae_m"] == score
                    assert ckpt["epochs_trained_in_this_run"] == 80 and ckpt["no_joint_refit"]
                    assert all(not torch.is_floating_point(v) or torch.isfinite(v).all() for v in ckpt["state_dict"].values())
                    states[policy] = ckpt["state_dict"]
                for a in CHECKPOINTS:
                    for b in CHECKPOINTS:
                        if independent_selection[a][0] == independent_selection[b][0]:
                            assert all(torch.equal(states[a][k], states[b][k]) for k in states[a])
                del states

                for stage, policies in (("stage1", CHECKPOINTS), ("stage2", POLICIES)):
                    for policy in policies:
                        sites = get_sites(diag, stage, variant, policy)
                        metrics = get_metrics(result, stage, variant, policy)
                        assert {tuple(x["site"]) for x in sites.values()} == cal_eval
                        assert len(sites) == metrics["n_sites"] == len(cal_eval)
                        assert sum(x["n_scans"] for x in sites.values()) == metrics["n_scans"] == before["datasets"][dataset]["roles"]["cal_eval"]["scans"]
                        pairs = (("mae_full_m", "macro_mae_full_m"), ("mae_predicted_bf_m", "macro_mae_predicted_bf_m"), ("energy_score_m", "macro_es_m"), ("bf_accuracy", "macro_bf_accuracy"), ("joint_nll", "macro_joint_nll"))
                        for small, big in pairs:
                            difference = abs(float(np.mean([s[small] for s in sites.values()])) - metrics[big])
                            maximum_reduction_difference = max(maximum_reduction_difference, difference)
                            assert difference <= MAX_FLOAT32_REDUCTION_DIFFERENCE, (name, variant, stage, policy, big, difference)
                        p95 = float(np.quantile([s["mae_full_m"] for s in sites.values()], .95))
                        assert abs(p95 - metrics["site_p95_full_m"]) <= MAX_FLOAT32_REDUCTION_DIFFERENCE
                        weighted_bf = sum(s["n_scans"] * s["bf_accuracy"] for s in sites.values()) / metrics["n_scans"]
                        assert abs(weighted_bf - metrics["row_bf_accuracy"]) <= 1e-6
                        expected_metrics.append((dataset, seed, variant, stage, policy, metrics))

                stage1_raw = get_sites(diag, "stage1", variant, PRIMARY)
                stage2_raw = get_sites(diag, "stage2", variant, "raw")
                assert stage1_raw == stage2_raw, (name, variant, "raw identity")
                for policy in POLICIES:
                    entry = result["stage2"]["variants"][variant]["calibration"][policy]
                    t, theta = entry["params"]["temperature"], entry["params"]["theta"]
                    assert .25 - 1e-6 <= t <= 3 + 1e-6 and all(0 <= x <= 1.5 for x in theta)
                    if policy == "raw":
                        assert t == 1 and theta == [0, 0]
                    elif policy == "temperature_only":
                        assert theta == [0, 0]
                    elif policy == "g13_fixed_t06":
                        assert t == .6
                    item = diag["stage2_frozen_calibration"][variant][policy]
                    calculated = compare_sites(stage2_raw, item["sites"])
                    paired_site_reviews[f"{name}/{variant}/{policy}"] = calculated
                    for key, delta in item["delta_vs_raw"].items():
                        for saved, metric in (("mae_delta_m", "mae_full_m"), ("energy_score_delta_m", "energy_score_m"), ("bf_accuracy_delta", "bf_accuracy")):
                            assert delta[saved] == item["sites"][key][metric] - stage2_raw[key][metric]
            for policy in CHECKPOINTS:
                raw_sites = get_sites(diag, "stage1", "raw_neighbors", policy)
                floor_sites = get_sites(diag, "stage1", "same_floor_neighbors", policy)
                for key, delta in diag["stage1_neighbor_deltas"][policy].items():
                    for saved, metric in (("mae_delta_m", "mae_full_m"), ("energy_score_delta_m", "energy_score_m"), ("bf_accuracy_delta", "bf_accuracy")):
                        assert delta[saved] == floor_sites[key][metric] - raw_sites[key][metric]
            del reference

    # Remaining aggregation checks and the report are appended below.
    finish(before, execution, results, diagnostics, expected_metrics, artifact_hashes,
           checkpoint_hashes, paired_site_reviews, maximum_reduction_difference)


def finish(before, execution, results, diagnostics, expected_metrics, artifact_hashes,
           checkpoint_hashes, paired_site_reviews, maximum_reduction_difference):
    run_rows = read_csv(ROOT / "RUN_LEVEL_METRICS.csv")
    summary_rows = read_csv(ROOT / "AGGREGATE_SUMMARY.csv")
    paired_rows = read_csv(ROOT / "PAIRED_DELTAS.csv")
    site_rows = read_csv(ROOT / "SITE_DELTAS_ALL_RUNS.csv")
    assert len(run_rows) == 126 and len(summary_rows) == 42 and len(paired_rows) == 81
    assert len(site_rows) == 3 * 11 * sum(before["datasets"][d]["roles"]["cal_eval"]["sites"] for d in DATASETS)
    expected = {}
    for dataset, seed, variant, stage, policy, metrics in expected_metrics:
        key = (dataset, seed, variant, "stage1_raw_checkpoint" if stage == "stage1" else "stage2_frozen_calibration", policy if stage == "stage1" else PRIMARY, "raw" if stage == "stage1" else policy)
        expected[key] = metrics
    assert len(expected) == len(run_rows)
    groups = {}
    seen = set()
    metric_names = ("macro_mae_full_m", "macro_mae_predicted_bf_m", "macro_es_m", "macro_bf_accuracy", "row_bf_accuracy", "macro_joint_nll")
    for row in run_rows:
        key = (row["dataset"], int(row["seed"]), row["variant"], row["stage"], row["checkpoint_policy"], row["calibration_policy"])
        assert key not in seen and key in expected
        seen.add(key)
        for metric in (*metric_names, "n_scans", "n_sites"):
            assert math.isclose(float(row[metric]), expected[key][metric], rel_tol=1e-12, abs_tol=1e-12)
        group = (row["stage"], row["dataset"], row["variant"], row["checkpoint_policy"], row["calibration_policy"])
        groups.setdefault(group, []).append(row)
    for row in summary_rows:
        key = tuple(row[k] for k in ("stage", "dataset", "variant", "checkpoint_policy", "calibration_policy"))
        members = groups[key]
        assert len(members) == int(row["n_seeds"]) == 3
        for metric in (*metric_names, "selected_epoch"):
            mean, sd = stats([float(x[metric]) for x in members])
            assert math.isclose(mean, float(row[f"{metric}_mean"]), rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(sd, float(row[f"{metric}_sd"]), rel_tol=1e-12, abs_tol=1e-12)
    for row in paired_rows:
        result = results[(row["dataset"], int(row["seed"]))]
        if row["stage"] == "stage1_same_floor_vs_raw_neighbors":
            ref = get_metrics(result, "stage1", "raw_neighbors", row["policy"])
            cand = get_metrics(result, "stage1", "same_floor_neighbors", row["policy"])
        else:
            ref = get_metrics(result, "stage2", row["variant"], "raw")
            cand = get_metrics(result, "stage2", row["variant"], row["policy"])
        for col, metric in (("mae_delta_m", "macro_mae_full_m"), ("es_delta_m", "macro_es_m"), ("bf_accuracy_delta", "macro_bf_accuracy")):
            assert math.isclose(float(row[col]), cand[metric] - ref[metric], rel_tol=1e-12, abs_tol=1e-12)
    for row in site_rows:
        diag = diagnostics[(row["dataset"], int(row["seed"]))]
        if row["stage"] == "stage1_same_floor_vs_raw_neighbors":
            ref = get_sites(diag, "stage1", "raw_neighbors", row["policy"])[row["site_key"]]
            cand = get_sites(diag, "stage1", "same_floor_neighbors", row["policy"])[row["site_key"]]
        else:
            ref = get_sites(diag, "stage2", row["variant"], "raw")[row["site_key"]]
            cand = get_sites(diag, "stage2", row["variant"], row["policy"])[row["site_key"]]
        assert int(row["n_scans"]) == ref["n_scans"] == cand["n_scans"]
        for col, metric in (("mae_delta_m", "mae_full_m"), ("es_delta_m", "energy_score_m"), ("bf_accuracy_delta", "bf_accuracy")):
            assert math.isclose(float(row[col]), cand[metric] - ref[metric], rel_tol=1e-12, abs_tol=1e-12)
    aggregate = load(ROOT / "AGGREGATE_SUMMARY.json")
    assert aggregate["completed_runs"] == aggregate["expected_runs"] == 9
    for filename in ("RUN_LEVEL_METRICS.csv", "AGGREGATE_SUMMARY.csv", "PAIRED_DELTAS.csv", "SITE_DELTAS_ALL_RUNS.csv", "AGGREGATE_SUMMARY.json"):
        artifact_hashes[filename] = sha(ROOT / filename)
    warnings = []
    for command in execution["commands"]:
        if command.get("status") == "COMPLETE":
            err = REPO / command["stderr"]
            if err.stat().st_size:
                warnings.append({"command": command["label"], "stderr": str(err.relative_to(REPO)), "bytes": err.stat().st_size})
    review = {
        "status": "PASS", "completed_paired_jobs": 9, "trained_backbones": 18,
        "full_training_epochs": 1440, "checkpoint_files": len(checkpoint_hashes),
        "reference_bundles": 9, "run_metric_rows": len(run_rows),
        "summary_rows": len(summary_rows), "paired_seed_rows": len(paired_rows),
        "paired_site_rows": len(site_rows), "pytest_passed": 13,
        "source_files_unchanged": len(before["source_sha256"]),
        "training_input_files_unchanged": len(before["input_sha256"]),
        "maximum_float32_site_reduction_difference": maximum_reduction_difference,
        "float32_reduction_audit_tolerance": MAX_FLOAT32_REDUCTION_DIFFERENCE,
        "checkpoint_selection_independently_replayed": True,
        "stage1_primary_raw_equals_stage2_raw_per_site": True,
        "cal_fit_eval_sites_disjoint": True, "internal_and_official_test_rows_scored": 0,
        "audit_limit": "Saved site metrics independently reaggregated. Raw per-scan errors are not emitted by this runner, so a scan-to-site rerun is not claimed.",
        "artifact_sha256": artifact_hashes, "checkpoint_sha256": checkpoint_hashes,
        "stage2_site_harms_vs_raw": paired_site_reviews,
        "nonempty_stderr_streams": warnings,
    }
    sys.path.insert(0, str(REPO))
    from fhm_dapp_tim.fhm_dapp.train_anchor import _neighbor_targets
    from fhm_dapp_tim.fhm_dapp.msep_joint_science_v1 import same_floor_neighbors
    tuji_ref = torch.load(ROOT / "tuji_seed17301/REFERENCE_STATE.pt", map_location="cpu", weights_only=False)
    coords, buildings, floors = (tuji_ref[k] for k in ("fit_anchor_coords", "fit_anchor_building", "fit_anchor_floor"))
    old_ids, old_weights = _neighbor_targets(coords, buildings, neighbor_count=8, temperature_m=8.)
    new_ids, new_weights = same_floor_neighbors(coords, buildings, floors, count=8, temperature_m=8.)
    review["tuji_neighbor_numerics"] = {
        "ids_bitwise_equal": torch.equal(old_ids, new_ids),
        "weights_bitwise_equal": torch.equal(old_weights, new_weights),
        "weights_allclose_atol_rtol_1e_7": torch.allclose(old_weights, new_weights, atol=1e-7, rtol=1e-7),
        "weight_max_absolute_difference": float((old_weights-new_weights).abs().max()),
        "unequal_weight_entries": int((old_weights!=new_weights).sum()),
        "weight_entries": old_weights.numel(),
        "interpretation": "Same geometric neighbor targets within declared numeric tolerance, but not bitwise identical weights. CUDA deterministic algorithms were not enabled. Do not attribute TUJI training differences to floor exclusion.",
    }
    assert review["tuji_neighbor_numerics"]["ids_bitwise_equal"]
    assert review["tuji_neighbor_numerics"]["weights_allclose_atol_rtol_1e_7"]
    assert len(checkpoint_hashes) == 54
    (ROOT / "INTEGRITY_REVIEW.json").write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(before, execution, results, diagnostics, review, summary_rows, paired_rows)
    print(json.dumps({"integrity": "PASS", "paired_jobs": 9, "epochs": 1440, "checkpoint_files": 54, "reference_bundles": 9, "site_delta_rows": len(site_rows), "maximum_reduction_difference": maximum_reduction_difference}, ensure_ascii=False))


def write_report(before, execution, results, diagnostics, review, summary_rows, paired_rows):
    lines = ["# V17 双阶段冻结校准实验实测报告", "", "本轮为内部 CAL-eval 收尾比较：3 数据域 × 3 配对种子，18 条单成员训练轨迹，均完成 80 轮，共 1440 轮。没有 Joint 微调，未评估内部 TEST 或官方测试。", "", "三种检查点是同一训练轨迹的选择敏感性比较；唯一预声明的 Stage 2 主检查点为 `best_dev_epoch_21_80`。以下均值 ± 样本标准差描述相同评价站点上三个种子的波动，不是三个独立数据集的显著性检验。", "", "## 执行与数据边界", "", f"Python `{before['python_executable']}`；{before['python']}。Torch {before['torch']}，CUDA {before['torch_cuda']}，{before['gpu']}（{before['gpu_total_bytes']/1024**3:.2f} GiB）；NumPy {before['numpy']}，SciPy {before['scipy']}，pandas {before['pandas']}，pytest {before['pytest']}。", "", "指定编译与 13 项单测通过。全部命令及 stdout/stderr、退出码、墙钟时长、日志 SHA256 保存在 EXECUTION_RECORD.json 和 run_logs/。命令为指定 Python：compileall；两个指定 pytest 文件；按 tuji/uts/uji、17301/17302/17303 串行执行 run（batch64、threads4、ES draws32）；最后 aggregate。", "", "|域|FIT 站点/扫描|DEV|CAL-fit|CAL-eval|TEST（仅角色元数据）|FIT 锚点/AP|每骨干更新次数|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        audit = before["datasets"][dataset]
        counts = [f"{audit['roles'][role]['sites']}/{audit['roles'][role]['scans']}" for role in ("fit", "dev", "cal_fit", "cal_eval", "test")]
        lines.append(f"|{dataset}|" + "|".join(counts) + f"|{audit['fit_anchor_count']}/{audit['fit_seen_aps']}|{audit['optimizer_updates_per_variant']}|")
    total_wall = sum(x["elapsed_seconds"] for x in execution["commands"] if x.get("status") == "COMPLETE")
    training_wall = sum(x["elapsed_seconds"] for x in execution["commands"] if x.get("label", "").startswith(DATASETS) and x.get("status") == "COMPLETE")
    lines += ["", f"正式 9 次作业累计墙钟 {training_wall/60:.2f} 分钟；审计前已完成命令累计 {total_wall/60:.2f} 分钟。每个 paired job 包含两次完整训练及校准/评分，未分别计时单个骨干 GPU 核函数。", "", "物理站点所有扫描角色不交叉；原坐标与居中 float32 四位分组一一对应。G13 原型/检出率/条件均值方差、AP seen mask 只由 FIT 得到。TUJI 五个 MAT 容器被打开和哈希，但仅请求 results_train，results_test 未反序列化；UTS/UJI 只读 training CSV。", "", "## Stage 1：完整训练与检查点敏感性", "", "两方案在每个 dataset/seed 从相同初始化复制，重置相同训练与 DataLoader 种子，AdamW/损失/增强/扫描/批量/80 轮预算相同，唯一训练改动为邻居标签限制建筑和楼层。排除 1–20 轮仅影响主检查点选择，没有增加学习率 warmup。旧规则实现为 score < best−0.01、epoch≥20 且 epoch−best_epoch≥12，模拟停止不影响实际 80 轮执行。", "", "|域|方案|旧选中轮/模拟停止轮（三种子）|全80最优轮|21–80最优轮|", "|---|---|---|---|---|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            selections = [results[(dataset, s)]["stage1"]["training"][variant]["checkpoint_selection"] for s in SEEDS]
            legacy = "; ".join(f"{x['legacy_early_stop']['selected_epoch']}/{x['legacy_early_stop']['simulated_stop_epoch']}" for x in selections)
            full = ",".join(str(x["best_dev_epoch_1_80"]["selected_epoch"]) for x in selections)
            primary = ",".join(str(x[PRIMARY]["selected_epoch"]) for x in selections)
            lines.append(f"|{dataset}|{variant}|{legacy}|{full}|{primary}|")
    lines += ["", "各轮 FIT 总损失、全部 training_loss 分量和 DEV 五项指标完整保存在每个 EPOCH_HISTORY.csv/RESULT.json；没有按 CAL-eval 选 checkpoint。", "", "|域|方案|检查点|CAL-eval MAE(m)|连续 ES(m)|BF准确率|预测BF条件MAE(m)|联合NLL|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row["stage"] != "stage1_raw_checkpoint":
            continue
        values = [f"{float(row[m+'_mean']):.4f} ± {float(row[m+'_sd']):.4f}" for m in ("macro_mae_full_m", "macro_es_m", "macro_bf_accuracy", "macro_mae_predicted_bf_m", "macro_joint_nll")]
        lines.append(f"|{row['dataset']}|{row['variant']}|{row['checkpoint_policy']}|" + "|".join(values) + "|")
    lines += ["", "同楼层减原邻居的配对差（负 MAE/ES 为改善，正 BF 为改善）：", "", "|域|检查点|ΔMAE(m)|ΔES(m)|ΔBF准确率|MAE改善种子数|", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for policy in CHECKPOINTS:
            rows = [r for r in paired_rows if r["stage"] == "stage1_same_floor_vs_raw_neighbors" and r["dataset"] == dataset and r["policy"] == policy]
            values = [[float(r[m]) for r in rows] for m in ("mae_delta_m", "es_delta_m", "bf_accuracy_delta")]
            lines.append(f"|{dataset}|{policy}|" + "|".join(mean_sd(v) for v in values) + f"|{sum(v<0 for v in values[0])}/3|")
    lines += ["", "## Stage 2：冻结主检查点后的四方案", "", "每个骨干共用同一 CAL-fit/CAL-eval pack，候选 ID、连续分量中心/尺度均不变；只有 T 和两项 G13 系数改变。校准只用 CAL-fit，连续 Student-t NLL + 0.30 二维点风险，站点等权，Adam lr0.035、110 次更新；T∈[.25,3]、theta∈[0,1.5]。所有 ES 使用相同 seed+900、两流各32次抽样；比较跨种子波动包含模型和 MC 随机流差异。", "", "|域|方案|校准|MAE(m)|ES(m)|BF准确率|预测BF条件MAE(m)|联合NLL|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        if row["stage"] != "stage2_frozen_calibration":
            continue
        values = [f"{float(row[m+'_mean']):.4f} ± {float(row[m+'_sd']):.4f}" for m in ("macro_mae_full_m", "macro_es_m", "macro_bf_accuracy", "macro_mae_predicted_bf_m", "macro_joint_nll")]
        lines.append(f"|{row['dataset']}|{row['variant']}|{row['calibration_policy']}|" + "|".join(values) + "|")
    lines += ["", "校准参数均来自 CAL-fit，不能从下表或 CAL-eval 再选一个方案作为部署结论：", "", "|域/种子|骨干|校准|T|theta visibility|theta strength|", "|---|---|---|---:|---:|---:|"]
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                for policy in POLICIES[1:]:
                    p = results[(dataset,seed)]["stage2"]["variants"][variant]["calibration"][policy]["params"]
                    lines.append(f"|{dataset}/{seed}|{variant}|{policy}|{p['temperature']:.6f}|{p['theta'][0]:.6f}|{p['theta'][1]:.6f}|")
    append_interpretation(lines, before, results, diagnostics, review)
    (ROOT / "RUN_REPORT_V17_TWO_STAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_interpretation(lines, before, results, diagnostics, review):
    lines += ["", "## 温度之外的额外证据收益", "", "下面固定比较 temperature_plus_g13 − temperature_only，避免把温度本身的收益归给 G13。没有据此替换预声明方案或重训。", "", "|域|骨干|ΔMAE(m)|ΔES(m)|ΔBF准确率|MAE/ES同时严格降低的种子数|", "|---|---|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            differences = []
            for seed in SEEDS:
                result = results[(dataset,seed)]
                ref = get_metrics(result,"stage2",variant,"temperature_only")
                cand = get_metrics(result,"stage2",variant,"temperature_plus_g13")
                differences.append([cand[m]-ref[m] for m in ("macro_mae_full_m","macro_es_m","macro_bf_accuracy")])
            cols = list(zip(*differences))
            lines.append(f"|{dataset}|{variant}|" + "|".join(mean_sd(list(v)) for v in cols) + f"|{sum(d[0]<0 and d[1]<0 for d in differences)}/3|")
    lines += ["", "## 尾部与逐站点伤害", "", "以下是站点内扫描均值的 P95/P99/最坏站点误差；不是扫描误差分位数。害处计数以同骨干同主检查点的 raw 为参照，按 seed17301/17302/17303 顺序列出。原始全站点向量和配对差在 SITE_DIAGNOSTICS.json 与 SITE_DELTAS_ALL_RUNS.csv；不能作为推断门控。", "", "|域|骨干|校准|P95(m)|P99(m)|最坏站点MAE(m)|较raw恶化站点数（三种子）|", "|---|---|---|---:|---:|---:|---|"]
    for dataset in DATASETS:
        for variant in VARIANTS:
            for policy in POLICIES:
                p95, p99, maximum, harms = [], [], [], []
                for seed in SEEDS:
                    sites = get_sites(diagnostics[(dataset,seed)],"stage2",variant,policy)
                    errors = [x["mae_full_m"] for x in sites.values()]
                    p95.append(float(np.quantile(errors,.95)))
                    p99.append(float(np.quantile(errors,.99)))
                    maximum.append(max(errors))
                    harms.append(review["stage2_site_harms_vs_raw"][f"{dataset}_seed{seed}/{variant}/{policy}"]["worsened"])
                lines.append(f"|{dataset}|{variant}|{policy}|{mean_sd(p95)}|{mean_sd(p99)}|{mean_sd(maximum)}|{','.join(map(str,harms))}|")
    lines += ["", "每个域和校准方法最严重的负迁移例子（跨两骨干/三种子，逐站点相对其 raw）：", "", "|域|校准|骨干/种子|位置(B,F,X,Y)|站点ΔMAE(m)|", "|---|---|---|---|---:|"]
    for dataset in DATASETS:
        for policy in POLICIES[1:]:
            options = [(review["stage2_site_harms_vs_raw"][f"{dataset}_seed{seed}/{variant}/{policy}"],variant,seed) for variant in VARIANTS for seed in SEEDS]
            worst, variant, seed = max(options,key=lambda x:x[0]["worst_delta_m"])
            lines.append(f"|{dataset}|{policy}|{variant}/{seed}|{json.dumps(worst['worst_site'],ensure_ascii=False)}|{worst['worst_delta_m']:+.6f}|")
    lines += ["", "## 实测判断与局限", ""]
    for dataset in DATASETS:
        improvements, es_improvements, bf_improvements = [], [], []
        for seed in SEEDS:
            raw = get_metrics(results[(dataset,seed)],"stage1","raw_neighbors",PRIMARY)
            floor = get_metrics(results[(dataset,seed)],"stage1","same_floor_neighbors",PRIMARY)
            improvements.append(floor["macro_mae_full_m"]-raw["macro_mae_full_m"])
            es_improvements.append(floor["macro_es_m"]-raw["macro_es_m"])
            bf_improvements.append(floor["macro_bf_accuracy"]-raw["macro_bf_accuracy"])
        lines.append(f"- {dataset}：主检查点同楼层修复 ΔMAE {mean_sd(improvements)} m、ΔES {mean_sd(es_improvements)} m、ΔBF {mean_sd(bf_improvements)}；MAE 严格改善 {sum(x<0 for x in improvements)}/3 个种子。不同指标的正负结果均保留，不能仅以分类准确率或平均 MAE 宣称全面成功。")
    tuji_differences = [max(abs(a["dev_mae_m"]-b["dev_mae_m"]) for a,b in zip(results[("tuji",seed)]["stage1"]["training"]["raw_neighbors"]["all_epoch_history"],results[("tuji",seed)]["stage1"]["training"]["same_floor_neighbors"]["all_epoch_history"],strict=True)) for seed in SEEDS]
    lines += ["", f"TUJI 单建筑楼层的邻居 ID 完全相同，权重通过原程序 atol=rtol=1e-7 等价检查，但不是逐位相同：最大权重差 {review['tuji_neighbor_numerics']['weight_max_absolute_difference']:.10g}，{review['tuji_neighbor_numerics']['unequal_weight_entries']}/{review['tuji_neighbor_numerics']['weight_entries']} 项不同。配对轨迹逐轮 DEV MAE 最大绝对差（三种子）为 {tuji_differences} m。没有强制全 CUDA 确定性，不能唯一分离浮点归一化与GPU随机执行的影响，不能把 TUJI 的差异归因于排除跨楼层邻居。", "", "T=1 与 T=.6 是本轮四方案定义的一部分。g13_fixed_t06 与 raw 的差包含温度变化，不能全部归因于观测证据；temperature_plus_g13 与 temperature_only 才是本轮可报告的额外证据对照，但二者来自各自 CAL-fit 优化，不是识别物理因果。参数触边和种子差异须连同校准表解释，不能扩网格追结果。", "", "所有骨干为单成员且同一划分下公平重训；不能直接与旧 E5/G13 的不同预算和划分绝对 MAE 作改进或显著性比较。三个种子共享 CAL-eval 站点，站点也可能有空间/设备/会话相关性。CAL-eval 来自此前已研究的训练语料内部位置，非独立外部盲测。连续 ES 是有限 MC 近似；BF-XY NLL 的概率对象与仅 XY NLL 不同，不能跨定义排优劣。", "", "本轮未加入新网络、门控、Joint 微调或 TEST 选择。是否推进后续机制应另行预注册，不能用本轮 CAL-eval 再挑部署策略；当前收尾结果本身不构成正式外部泛化证明。", "", "## 完整性、异常及文件", "", f"独立核验 PASS：9 次作业，18×80 轮，54 个检查点与9个共享参考 bundle，126 条 run-level、42 条聚合、81 条 seed 配对、{review['paired_site_rows']} 条逐站点配对记录。{review['source_files_unchanged']} 个源码/测试/指令文件及7个训练输入 SHA256 全部保持不变。检查点选择由独立程序回放；Stage1 主检查点 raw 与 Stage2 raw 逐站点 JSON 完全一致。", "", f"从保存站点向量复算总体指标的最大绝对差为 {review['maximum_float32_site_reduction_difference']:.10g}，来自原程序 float32 reduction 与序列化后 double 归并的舍入差；审计允许2e-5。这里没有更改任何科学门禁。源码未输出原始逐扫描误差，故不声称独立重建 scan→site；本次独立复核的是 saved site→macro/CSV/paired deltas。", "", "随机种子固定，未额外开启原脚本未设定的全 CUDA 确定性内核。未新增学习率调度、改变 batch/epochs、重新选划分或改模型源代码。", ""]
    if review["nonempty_stderr_streams"]:
        lines.append("stderr 非空记录（保留原始日志；内容须按日志解释）：")
        for warning in review["nonempty_stderr_streams"]:
            lines.append(f"- {warning['command']}：{warning['stderr']}，{warning['bytes']} 字节。")
    else:
        lines.append("已完成科学命令 stderr 均为空；未发生训练/评分失败、重试、OOM 或科学代码修复。")
    lines += ["", "源文件和输入逐项 SHA 在 BEFORE_RUN_AUDIT.json；各输出/54权重 SHA 和独立数值复核在 INTEGRITY_REVIEW.json；每作业 RUN_MANIFEST.json 绑定 RESULT/SITE_DIAGNOSTICS/REFERENCE_STATE 和核心源文件。模型 .pt 仅在本地保存。本轮未执行任何 Git 命令或上传。", "", "完整命令记录：", "", "```text"]
    for command in execution["commands"]:
        if command.get("status") == "COMPLETE":
            lines.append(f"exit={command['exit_code']} wall_s={command['elapsed_seconds']:.3f}: " + " ".join(command["command"]))
    lines += ["```", ""]


if __name__ == "__main__":
    main()
