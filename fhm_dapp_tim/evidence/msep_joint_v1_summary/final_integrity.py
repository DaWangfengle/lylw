"""Final saved-file audit; no model forward pass, training, or TEST access."""
from pathlib import Path
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    frozen = read(HERE/"METHOD_FREEZE.json")
    for item in frozen["source_files"]:
        assert sha(ROOT/item["path"]) == item["sha256"], item["path"]
    for path, expected in frozen["input_sha256"].items():
        assert sha(ROOT/path) == expected, path
    for key, name in (("data_audit_sha256", "DATA_AUDIT.json"), ("smoke_audit_sha256", "SMOKE_AUDIT.json"),
                      ("numerical_audit_sha256", "NUMERICAL_AUDIT.json")):
        assert sha(HERE/name) == frozen[key]
    assert sha(HERE/"original/MSEP_JOINT_V1_CODEX_RUN.txt") == frozen["original_instructions_sha256"]
    summary, record = read(HERE/"SUMMARY.json"), read(HERE/"EXECUTION_RECORD.json")
    assert summary["status"] == "PASS" and len(summary["runs"]) == 9
    completed = {v["label"]: v for v in record["commands"]}
    for cmd in frozen["formal_commands"]:
        actual = completed[cmd["label"]]
        assert actual["command"][1:] == cmd["arguments"] and actual["exit_code"] == 0
    assert sum(v["label"].startswith("full_") for v in record["commands"]) == 9
    logs = 0
    for command in record["commands"]:
        for log in command["logs"].values():
            assert sha(ROOT/log["path"]) == log["sha256"]
            logs += 1
    checkpoint_count, checkpoint_bytes = 0, 0
    identical_calibration_cases = []
    checkpoint_tensor_reviews = []
    import torch
    for run in summary["runs"]:
        for name, expected in run["output_sha256"].items():
            assert sha(ROOT/run["output"]/name) == expected
        sites = read(ROOT/run["output"]/"SITE_DIAGNOSTICS.json")
        for policy, reference in (("g13_t06", "temp_t06"), ("temp_plus_g13", "temp_fit")):
            if run["calibration_parameters"][policy] == run["calibration_parameters"][reference]:
                assert run["metrics"]["cal_"+policy] == run["metrics"]["cal_"+reference]
                assert sites["cal_"+policy] == sites["cal_"+reference]
                identical_calibration_cases.append([run["dataset"], run["seed"], reference, policy])
        for name, item in run["checkpoints"].items():
            p = ROOT/run["output"]/name
            assert sha(p) == item["sha256"] and p.stat().st_size == item["bytes"]
            saved = torch.load(p, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(value).all() for value in saved["model"].values())
            if "theta" in saved:
                assert all(0 <= v <= 1.5 for v in saved["theta"])
            checkpoint_tensor_reviews.append({"path": p.relative_to(ROOT).as_posix(),
                                               "model_tensors": len(saved["model"]),
                                               "all_finite": True})
            del saved
            checkpoint_count += 1
            checkpoint_bytes += item["bytes"]
    assert checkpoint_count == 24
    report = HERE/"RUN_REPORT.md"
    assert report.is_file() and report.stat().st_size > 1000
    warnings = [{"label": c["label"], "stderr_bytes": c["logs"]["stderr"]["bytes"]}
                for c in record["commands"] if c["logs"]["stderr"]["bytes"]]
    review = {"status": "PASS", "reviewed_at": datetime.datetime.now().astimezone().isoformat(),
              "unchanged_frozen_source_files": len(frozen["source_files"]),
              "unchanged_training_inputs": len(frozen["input_sha256"]),
              "formal_run_count": 9, "formal_retries": 0, "test_evaluations": 0,
              "verified_completed_command_logs": logs,
              "local_checkpoint_count": checkpoint_count, "checkpoint_bytes": checkpoint_bytes,
              "equal_parameter_calibration_metric_and_site_identities": identical_calibration_cases,
              "checkpoint_tensor_reviews": checkpoint_tensor_reviews,
              "commands_with_stderr": warnings,
              "nonzero_command_exits": [c["label"] for c in record["commands"] if c["exit_code"] != 0],
              "scientific_source_or_protocol_repairs": [],
              "report_sha256": sha(report), "summary_sha256": sha(HERE/"SUMMARY.json"),
              "method_freeze_sha256": sha(HERE/"METHOD_FREEZE.json"),
              "review_scope": "all completed logs before this audit; own stdout/stderr recorded by command runner after exit; no new inference"}
    with (HERE/"INTEGRITY_REVIEW.json").open("x", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(review))


if __name__ == "__main__":
    main()
