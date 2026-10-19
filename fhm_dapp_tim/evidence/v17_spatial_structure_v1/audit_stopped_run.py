"""Read-only saved-file audit after the Phase F beta-zero failure. No inference."""
from pathlib import Path
import hashlib
import json
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as study

ROOT = Path(__file__).resolve().parent
manifest = study.freeze_guard(ROOT)
complete, failed, untouched = [], [], []
maximum = 0.
for dataset in study.DATASETS:
    for seed in study.SEEDS:
        for variant in study.VARIANTS.values():
            folder = ROOT / "frozen" / f"{dataset}_seed{seed}" / variant
            item = {"dataset": dataset, "seed": seed, "variant": variant}
            if not (folder / "RESULT.json").exists():
                (failed if (folder / "FROZEN_RUN.lock").exists() else untouched).append(item)
                continue
            result = study.read_json(folder / "RESULT.json")
            sites = study.read_json(folder / "SITE_DIAGNOSTICS.json")
            for name, entry in result["methods"].items():
                rows = list(sites[name].values())
                assert len(rows) == entry["metrics"]["n_sites"]
                assert sum(r["n_scans"] for r in rows) == entry["metrics"]["n_scans"]
                for metric, field in (("macro_mae_full_m", "mae_full_m"), ("macro_es_m", "energy_score_m"),
                                      ("macro_bf_accuracy", "bf_accuracy"), ("macro_joint_nll", "joint_nll")):
                    error = abs(statistics.mean(r[field] for r in rows) - entry["metrics"][metric])
                    assert error < 2e-5
                    maximum = max(maximum, error)
            complete.append({**item, "metrics_by_method": {k: v["metrics"] for k, v in result["methods"].items()}})

hashes = {}
for p in ROOT.rglob("*"):
    if p.is_file() and not any(part in ("pytest_tmp", "__pycache__") for part in p.parts):
        hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = study.sha(p)
record = {
    "status": "STOPPED_AWAITING_AUTHORIZATION", "phase_F_completed_backbones": len(complete),
    "phase_F_expected_backbones": 18, "completed": complete, "failed": failed, "not_run": untouched,
    "phase_T_status": "NOT RUN", "aggregation_status": "NOT RUN",
    "maximum_site_reduction_difference": maximum,
    "original_manifest_and_frozen_sources_unchanged": True,
    "historical_source_data_and_artifact_hashes_verified": True,
    "failure": "UTS/17301/raw_neighbors: beta=0 expanded-vs-parent NLL/ES combined assertion failed",
    "formal_failure_separate_NLL_ES_values": "not persisted by the failed process; no invented values",
    "confirmed_synthetic_bug": study.read_json(ROOT / "BETA_ZERO_SYNTHETIC_DIAGNOSIS.json"),
    "proposed_patch_applied": False, "failed_CAL_eval_rerun": False,
    "preserved_artifact_sha256": hashes, **study.TEST_COUNTS,
}
study.write_json(ROOT / "STOPPED_INTEGRITY_REVIEW.json", record)
lines = ["# V17 空间可辨识性：中止与待授权记录", "",
         "状态：STOPPED_AWAITING_AUTHORIZATION。正式实验尚未完成，不能称全域通过。", "",
         "42项主回归测试通过；初始数学/历史回归30项通过。冻结前CUDA父G13特征逐位恒等问题已修复并保留日志、原源码副本和diff。",
         "预检已核验9组来源、18份主检查点、7个训练输入、所有原始角色及194个历史源码/测试文件，均匹配。FIT-only CUDA烟雾检查没有optimizer更新。", "",
         "## 正式运行与失败", "",
         f"Phase F 完成 {len(complete)}/18 个骨干，仅TUJI三个种子×两骨干；UTS17301/raw_neighbors在beta=0 NLL/ES联合恒等断言失败。所有锁、选择文件、原始stdout/stderr保留。",
         "UTS失败骨干已冻结CAL-fit参数并开始固定CAL-eval对照评分；失败前的部分指标只存在于已结束进程内存，未伪造输出或补写。",
         "Phase T 全部 NOT RUN；全域聚合 NOT RUN。没有重训任何模型，也没有读取或评分内部TEST/官方测试。", "",
         "## 已确认的数值原因与未确认范围", "",
         "纯合成CUDA数据B=414、N=1052、追加5000个零权重分量，CDF父前缀最大差4.172325134277344e-7；同一随机流的ES最大差0.14511680603027344，其中1行不同，allclose失败。此数值是合成诊断，不是UTS真实误差。",
         "原始失败断言合并NLL与ES，未保存二者各自差值，不能捏造真实NLL差值。合成证据已独立证明原ES抽样路径不保证追加零分量恒等。",
         "候选修复仅在ES抽样内部排除全零概率列，保留输入的N+E表示、所有正权重/中心/尺度、随机种子和方法；4项纯合成测试通过。补丁尚未应用到冻结源码。", "",
         "## 已完成TUJI结果（仅描述性部分数据）", "",
         "|种子|骨干|旧raw MAE|新联合基线MAE|2A MAE|2B MAE|", "|---:|---|---:|---:|---:|---:|"]
for r in complete:
    m = r["metrics_by_method"]
    lines.append(f"|{r['seed']}|{r['variant']}|{m['historical_raw']['macro_mae_full_m']:.6f}|{m['joint_calibrated_base']['macro_mae_full_m']:.6f}|{m['2A']['macro_mae_full_m']:.6f}|{m['2B']['macro_mae_full_m']:.6f}|")
lines += ["", "所有20组方案与逐站点向量保存在对应RESULT/SITE_DIAGNOSTICS中；本表不能包装成三域结论。", "",
          "## 继续所需的例外", "",
          "设计明确CAL-eval只进行一次最终评价，因此已经触及UTS CAL-eval后的评分恢复需要用户明确授权。原manifest和锁不可静默重写或删除；应另建修订审计，复用已冻结参数并完整保留失败记录、已完成TUJI结果。",
          "不改变网格、阈值、种子、角色、原始权重或历史源码；未经授权不重复CAL-eval评分，不启动Phase T。", ""]
(ROOT / "STOPPED_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"status": record["status"], "completed": len(complete), "failed": failed,
                  "not_run": len(untouched), "maximum_reduction_difference": maximum}, ensure_ascii=False))
