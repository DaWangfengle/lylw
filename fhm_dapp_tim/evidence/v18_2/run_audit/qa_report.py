"""Validate the completed records/report without importing or running a model."""
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
EVIDENCE = AUDIT.parent
PHASES = ["prepare", "train-selection", "calibrate-freeze", "stability", "train-confirmation", "confirm", "train-final", "benchmark"]
HEADINGS = ["GIT", "TESTS", "PROTOCOL", "MODEL", "TRAINING", "CALIBRATION", "STABILITY", "CONFIRMATION", "PRE-EXPOSED BENCHMARK", "COMPUTE", "AUDIT", "INTERPRETATION"]
SOURCES = ["invariant", "prototype", "learned", "marked_radio", "floor", "geometry"]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def digest(value, omitted):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != omitted}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)

report = read(EVIDENCE / "RUN_REPORT_V18_2.json")
markdown = (EVIDENCE / "RUN_REPORT_V18_2.md").read_text(encoding="utf-8")
assert report["report_sha256"] == digest(report, "report_sha256")
finite(report)
assert re.findall(r"^## (.+)$", markdown, re.M) == HEADINGS
assert not re.search(r"\b(?:TODO|TBD|FIXME|NaN|Infinity)\b", markdown)
assert len(report["INTERPRETATION"]) == 8
assert report["TESTS"]["py_compile_passed"] == 3
assert report["TESTS"]["pytest_passed"] == 28 and report["TESTS"]["pytest_failed"] == 0
assert "28 passed" in (AUDIT / "pytest.log").read_text(encoding="utf-8")
assert report["GIT"]["branch"] == "v18-2-adaptive-evidence"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == report["GIT"]["formal_experiment_sha"]
assert not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *report["GIT"]["source_snapshot"]["files"]], cwd=ROOT, text=True).strip()
assert report["AUDIT"]["confirm_command_count"] == 1
assert report["AUDIT"]["denied_accesses"] == 0
assert report["AUDIT"]["lock_written_before_confirmation_load"]
assert not report["AUDIT"]["teacher_used"] and not report["AUDIT"]["ensemble_used"]
assert not report["AUDIT"]["best_seed_selected"] and not report["AUDIT"]["prediction_averaging"]
assert all(report["PRE_CONFIRMATION_GATE"].values()) and len(report["PRE_CONFIRMATION_GATE"]) == 12

for section, path, label, metric_field in [
    ("CONFIRMATION", EVIDENCE / "confirmation_result_v18_2.json", "CONFIRMATION", "confirmation_metrics"),
    ("PRE_EXPOSED_BENCHMARK", EVIDENCE / "benchmark_result_v18_2.json", "BENCHMARK", "benchmark_metrics"),
]:
    original = read(path)
    assert original["result_sha256"] == digest(original, "result_sha256")
    assert report[section] == original
    m = original[metric_field]
    for label_suffix, field, unit in [("MAE", "coordinate_macro_mae_m", " m"), ("ENERGY SCORE", "coordinate_macro_energy_score_m", " m"), ("NLL", "coordinate_macro_mixture_nll", "")]:
        assert f"**{label} {label_suffix} = {m[field]:.6f}{unit}**" in markdown
    assert abs(sum(m["mean_gate_weight_" + source] for source in SOURCES) - 1.0) < 1e-5

bench = report["PRE_EXPOSED_BENCHMARK"]
assert bench["benchmark_was_pre_exposed_before_v18_2"]
assert not bench["blind_confirmation_claim"] and not bench["used_for_model_selection"]
assert "blind confirmation = NO；pre-exposed benchmark = YES；used_for_model_selection = NO" in markdown
stab = report["STABILITY"]
assert stab["seeds"] == [18321, 18322, 18323, 18324, 18325]
assert [r["seed"] for r in stab["records"]] == stab["seeds"]
for metric, summary in stab["summary"].items():
    values = [record["metrics"][metric] for record in stab["records"]]
    assert math.isclose(summary["mean"], statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(summary["sample_std"], statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-12)
for source, summary in stab["six_gate_seed_summary"].items():
    values = [record["metrics"]["mean_gate_weight_" + source] for record in stab["records"]]
    assert math.isclose(summary["mean"], statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(summary["sample_std"], statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-12)
assert not stab["ensemble_constructed"] and not stab["best_seed_selected"] and not stab["used_for_model_selection"]

previous_end = None
for phase in PHASES:
    execution = report["COMPUTE"]["executions"][phase]
    assert execution["exit_code"] == 0 and execution["sources_unchanged"]
    assert execution["command"] == [report["TESTS"]["python_executable"], "-m", "fhm_dapp_tim.fhm_dapp.uji_v18_2_experiment", phase] + ([] if phase == "prepare" else ["--device", "cuda"])
    start, end = [dt.datetime.fromisoformat(execution[k]) for k in ("started_utc", "finished_utc")]
    assert end > start and (previous_end is None or start > previous_end)
    previous_end = end
    assert report["COMPUTE"]["runtimes"][phase]["io_counts"]["denied"] == 0
for phase in ["train-selection", "stability", "train-confirmation", "train-final"]:
    runtime = report["COMPUTE"]["runtimes"][phase]
    assert runtime["deterministic_algorithms"] and not runtime["deterministic_warn_only"]
    assert not runtime["flash_sdpa"] and not runtime["memory_efficient_sdpa"] and runtime["math_sdpa"]
    assert runtime["cudnn_deterministic"] and not runtime["cudnn_benchmark"] and runtime["cublas_workspace_config"] == ":4096:8"

result = {"all_checks_passed": True, "model_inference_performed_by_qa": False, "report_self_sha256": report["report_sha256"], "required_sections": HEADINGS, "commands_in_registered_order": PHASES, "five_seed_sample_sd_valid": True, "strict_cuda_training_valid": True, "one_shot_confirmation_valid": True, "pre_exposed_benchmark_label_valid": True, "formal_scientific_source_unchanged": True, "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
destination = AUDIT / "report_qa.json"
assert not destination.exists()
with destination.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(result, ensure_ascii=False, indent=2))
