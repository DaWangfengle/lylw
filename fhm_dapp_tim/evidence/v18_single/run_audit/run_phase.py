"""Launch the exact V18 module CLI and record logs and sampled GPU memory."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
MODULE = "fhm_dapp_tim.fhm_dapp.uji_v18_experiment"
def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["register", "train-selection", "freeze", "train-final", "outer"])
args = parser.parse_args()
manifest_path = AUDIT / "code_snapshot.json"
sources = sorted((ROOT / "fhm_dapp_tim" / "fhm_dapp").glob("*.py"))
sources += [ROOT / "tests" / "test_coherent_marked_v9.py",
            ROOT / "tests" / "test_uji_v18_experiment.py",
            ROOT / "V18_CODEX_RUN.md",
            ROOT / "fhm_dapp_tim" / "evidence" / "v18_single" / "DESIGN_V18.md"]
current = {"experiment_commit_sha": git_sha(),
           "files": {str(path.relative_to(ROOT)): file_sha(path) for path in sources}}
if manifest_path.exists():
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError("Experiment source/commit changed after registration")
else:
    if args.phase != "register":
        raise RuntimeError("Registration must create the immutable source snapshot")
    manifest_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
evidence = ROOT / "fhm_dapp_tim" / "evidence" / "v18_single"
if args.phase == "outer":
    lock = AUDIT / "outer_invocation.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(now() + "\n")
command = [sys.executable, "-m", MODULE, args.phase]
if args.phase != "register":
    command += ["--device", "cuda"]
environment = dict(os.environ)
environment["PYTHONUNBUFFERED"] = "1"
environment["PYTHONDONTWRITEBYTECODE"] = "1"
environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
record = {"phase": args.phase, "command": command, "started_utc": now(),
          "experiment_commit_sha": current["experiment_commit_sha"],
          "source_snapshot_verified": True, "gpu_memory_sampling_interval_seconds": 2,
          "environment": {key: environment[key] for key in
                          ["CUBLAS_WORKSPACE_CONFIG", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"]}}
stopped = threading.Event()
samples = []
creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

def monitor():
    with (AUDIT / (args.phase + "_gpu_samples.jsonl")).open("x", encoding="utf-8") as handle:
        while not stopped.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, creationflags=creation_flags, timeout=10).strip()
                name, total, used, utilization = [value.strip() for value in output.splitlines()[0].split(",")]
                sample = {"utc": now(), "name": name, "memory_total_mib": int(total),
                          "memory_used_mib": int(used), "utilization_percent": int(utilization)}
                samples.append(sample)
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
            except Exception as error:
                handle.write(json.dumps({"utc": now(), "error": str(error)}) + "\n")
                handle.flush()
            stopped.wait(2)

log_path = AUDIT / (args.phase + ".log")
record_path = AUDIT / (args.phase + "_execution.json")
if log_path.exists() or record_path.exists():
    raise FileExistsError("A phase execution log already exists")
print(json.dumps({"event": "started", **record}), flush=True)
started = time.perf_counter()
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
with log_path.open("x", encoding="utf-8") as handle:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", creationflags=creation_flags)
    record["python_process_id"] = process.pid
    for line in process.stdout:
        handle.write(line)
        handle.flush()
        if args.phase != "register":
            print(line, end="", flush=True)
    result = process.wait()
stopped.set()
thread.join(timeout=12)
record.update({"finished_utc": now(), "exit_code": result,
               "wall_seconds": time.perf_counter() - started,
               "gpu_memory_samples": len(samples),
               "peak_sampled_total_gpu_memory_mib": max((sample["memory_used_mib"] for sample in samples), default=None),
               "gpu_memory_measurement": "nvidia-smi total device usage, 2-second samples; not allocator peak",
               "source_snapshot_matches_after_phase": current ==
                   {"experiment_commit_sha": git_sha(),
                    "files": {str(path.relative_to(ROOT)): file_sha(path) for path in sources}}})
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps({"event": "finished", **record}), flush=True)
sys.exit(result)
