"""Run the exact preregistered CLI once and preserve execution evidence."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
preflight = json.loads((AUDIT / "preflight.json").read_text(encoding="utf-8"))
assert preflight["all_checks_passed"]
assert sys.executable.lower() == preflight["python_executable"].lower()
for relative, expected in preflight["scientific_files_sha256_before"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
for relative in preflight["required_absences_verified"]:
    assert not (ROOT / relative).exists(), relative
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == preflight["formal_experiment_sha"]
recovery = sys.argv[1:] in (["--recover-before-prediction"], ["--recover-before-prediction=3"], ["--radio-profile-authorized"])
assert not sys.argv[1:] or recovery, "Unknown launch argument"
attempt = 4 if sys.argv[1:] == ["--radio-profile-authorized"] else (3 if sys.argv[1:] == ["--recover-before-prediction=3"] else (2 if recovery else 1))
prefix = "probe" if attempt == 1 else "probe_0" + str(attempt)
if recovery:
    previous_prefixes = ["probe"] + ["probe_0" + str(i) for i in range(2, attempt)]
    for previous_prefix in previous_prefixes:
        previous = json.loads((AUDIT / (previous_prefix + "_execution.json")).read_text(encoding="utf-8"))
        runtime = json.loads((AUDIT / (previous_prefix + "_runtime.json")).read_text(encoding="utf-8"))
        assert previous["exit_code"] != 0 and previous["scientific_files_unchanged"]
        assert runtime["prediction_calls"] == runtime["prediction_scans"] == 0
        if previous_prefix == "probe_03":
            assert runtime["model_state_unchanged_bitwise"]
        else:
            assert runtime["model_load_count"] == 0
        assert not runtime["denied_accesses"]
        if previous_prefix != "probe_03":
            assert not any("/role_shards/" in name.replace("\\", "/")
                           for name in runtime["allowed_scientific_reads"])
    assert (AUDIT / "audit_path_fix_verification.json").is_file()
    assert json.loads((AUDIT / "audit_path_fix_verification.json").read_text(encoding="utf-8"))["all_checks_passed"]
    if attempt >= 3:
        assert json.loads((AUDIT / "checkpoint_reconstruction_verification.json").read_text(encoding="utf-8"))["all_checks_passed"]
    if attempt == 4:
        assert json.loads((AUDIT / "radio_profile_authorization.json").read_text(encoding="utf-8"))["radio_nuisance_optimization_authorized"]
        assert json.loads((AUDIT / "radio_prediction_equivalence.json").read_text(encoding="utf-8"))["all_checks_passed"]
lock = AUDIT / (prefix + "_invocation.lock")
with lock.open("x", encoding="utf-8") as handle:
    handle.write(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")

device = "cuda" if preflight["cuda_environment"]["cuda_available"] else "cpu"
command = [sys.executable, "-m", "fhm_dapp_tim.fhm_dapp.v18_4_decoder_probe", "--device", device]
environment = os.environ.copy()
environment["V184_DECODER_AUDIT_DIR"] = str(AUDIT)
environment["V184_DECODER_RUNTIME_PATH"] = str(AUDIT / (prefix + "_runtime.json"))
environment["PYTHONPATH"] = os.pathsep.join([str(AUDIT), str(ROOT), environment.get("PYTHONPATH", "")])
environment["PYTHONUNBUFFERED"] = "1"
record = {
    "command": command,
    "formal_experiment_sha": preflight["formal_experiment_sha"],
    "device": device,
    "invocation_count": 1,
    "total_cli_invocations_including_pre_prediction_failure": attempt,
    "recovery_scope": ("Explicitly authorized radio nuisance inference after three failed starts with no complete predictions"
                       if attempt == 4 else ("Audit compatibility corrections after model-build failures with zero role reads and predictions" if recovery else None)),
    "radio_nuisance_exception_authorized": attempt == 4,
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "instrumentation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [AUDIT / "sitecustomize.py", Path(__file__)]},
}
execution = AUDIT / (prefix + "_execution.json")
execution.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("[audit] invoking " + " ".join(command), flush=True)
start = time.perf_counter()
with (AUDIT / (prefix + ".log")).open("x", encoding="utf-8") as log:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    record["python_process_id"] = process.pid
    execution.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    returncode = process.wait()
record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
record["wall_seconds"] = time.perf_counter() - start
record["exit_code"] = returncode
record["completed_scientific_probe_count"] = 1 if returncode == 0 else 0
record["scientific_files_unchanged"] = all(
    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for relative, expected in preflight["scientific_files_sha256_before"].items())
execution.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
assert record["scientific_files_unchanged"], "Scientific input or source changed"
print("[audit] probe exit code: " + str(returncode), flush=True)
raise SystemExit(returncode)
