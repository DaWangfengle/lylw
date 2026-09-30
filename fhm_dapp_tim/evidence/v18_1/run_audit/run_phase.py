"""Execute the exact registered module CLI with I/O and runtime auditing."""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
PHASES = ["prepare", "train-selection", "calibrate-freeze", "stability", "train-final", "external"]
phase = sys.argv[1]
assert phase in PHASES
files = ["fhm_dapp_tim/fhm_dapp/v18_1_single.py",
         "fhm_dapp_tim/fhm_dapp/uji_v18_1_experiment.py",
         "fhm_dapp_tim/evidence/v18_1/DESIGN_V18_1.md",
         "tests/test_uji_v18_1_experiment.py", "V18_1_CODEX_RUN.txt"]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def head(): return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
current = {"formal_experiment_sha": head(), "files": {p: sha(ROOT/p) for p in files}}
snapshot = AUDIT/"source_snapshot.json"
if snapshot.exists():
    assert json.loads(snapshot.read_text(encoding="utf-8")) == current, "Source changed"
else:
    assert phase == "prepare"
    snapshot.write_text(json.dumps(current, indent=2), encoding="utf-8")
log_path = AUDIT/(phase+".log")
execution_path = AUDIT/(phase+"_execution.json")
assert not log_path.exists() and not execution_path.exists(), "Phase already invoked"
if phase == "external":
    assert (AUDIT/"pre_external_gate.json").is_file()
    with (AUDIT/"external_invocation.lock").open("x", encoding="utf-8") as handle:
        handle.write(now()+"\n")
command = [sys.executable, "-m", "fhm_dapp_tim.fhm_dapp.uji_v18_1_experiment", phase]
if phase != "prepare": command += ["--device", "cuda"]
env = dict(os.environ)
env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "V181_AUDIT_PHASE": phase,
            "V181_AUDIT_DIR": str(AUDIT), "V181_REPO_ROOT": str(ROOT)})
env["PYTHONPATH"] = str(AUDIT) + os.pathsep + env.get("PYTHONPATH", "")
record = {"phase": phase, "command": command, "started_utc": now(),
          "formal_experiment_sha": current["formal_experiment_sha"]}
print(json.dumps({"event": "started", **record}), flush=True)
started = time.perf_counter()
flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
with log_path.open("x", encoding="utf-8") as log:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                creationflags=flags)
    record["python_process_id"] = proc.pid
    for line in proc.stdout:
        log.write(line); log.flush()
        if phase != "prepare": print(line, end="", flush=True)
    code = proc.wait()
record.update({"finished_utc": now(), "wall_seconds": time.perf_counter()-started,
               "exit_code": code, "sources_unchanged": current ==
                   {"formal_experiment_sha": head(), "files": {p: sha(ROOT/p) for p in files}}})
execution_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps({"event": "finished", **record}), flush=True)
sys.exit(code)
