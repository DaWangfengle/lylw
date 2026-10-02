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
lock = AUDIT / "probe_invocation.lock"
with lock.open("x", encoding="utf-8") as handle:
    handle.write(dt.datetime.now(dt.timezone.utc).isoformat() + "\n")

device = "cuda" if preflight["cuda_environment"]["cuda_available"] else "cpu"
command = [sys.executable, "-m", "fhm_dapp_tim.fhm_dapp.v18_4_decoder_probe", "--device", device]
environment = os.environ.copy()
environment["V184_DECODER_AUDIT_DIR"] = str(AUDIT)
environment["PYTHONPATH"] = os.pathsep.join([str(AUDIT), str(ROOT), environment.get("PYTHONPATH", "")])
environment["PYTHONUNBUFFERED"] = "1"
record = {
    "command": command,
    "formal_experiment_sha": preflight["formal_experiment_sha"],
    "device": device,
    "invocation_count": 1,
    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "instrumentation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [AUDIT / "sitecustomize.py", Path(__file__)]},
}
execution = AUDIT / "probe_execution.json"
execution.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("[audit] invoking " + " ".join(command), flush=True)
start = time.perf_counter()
with (AUDIT / "probe.log").open("x", encoding="utf-8") as log:
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
record["scientific_files_unchanged"] = all(
    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for relative, expected in preflight["scientific_files_sha256_before"].items())
execution.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
assert record["scientific_files_unchanged"], "Scientific input or source changed"
print("[audit] probe exit code: " + str(returncode), flush=True)
raise SystemExit(returncode)
