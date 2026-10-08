"""Operational CLI logger; does not alter reproduction scientific computation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "fhm_dapp_tim/evidence/historical_reproduction/run_audit"
MODEL_ROOT = ROOT / "fhm_dapp_tim/models/historical_reproduction"
PYTHON = Path(r"D:\miniminiconda\envs\liuyan\python.exe")
MODULE = "fhm_dapp_tim.fhm_dapp.historical_reproduction"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "compile":
        command = [str(PYTHON), "-m", "py_compile",
                   "fhm_dapp_tim/fhm_dapp/historical_reproduction.py"]
    elif args.stage == "pytest":
        command = [str(PYTHON), "-m", "pytest", "-q",
                   "tests/test_historical_reproduction.py",
                   "tests/test_uji_v9_experiment.py",
                   "tests/test_uji_v17_mother.py"]
    else:
        commands = {"preflight", "prepare", "run-v9", "prepare-v17",
                    "run-v17", "check-v17-members", "evaluate-v17", "summarize"}
        if args.stage not in commands:
            parser.error("unsupported stage")
        command = [str(PYTHON), "-m", MODULE, args.stage]
        if args.stage in {"preflight", "prepare", "run-v9", "run-v17", "evaluate-v17"}:
            command.extend(["--device", "cuda"])
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if args.stage == "compile":
        env["PYTHONPYCACHEPREFIX"] = str(MODEL_ROOT / "runtime_cache/pycache")
    if args.stage == "pytest":
        fixture_root = MODEL_ROOT / f"runtime_cache/pytest_attempt{args.attempt:02d}"
        fixture_root = fixture_root.resolve()
        assert fixture_root.is_relative_to(MODEL_ROOT.resolve())
        if fixture_root.exists():
            raise FileExistsError("refusing to reuse/delete existing pytest fixture directory")
        env["PYTEST_ADDOPTS"] = f"-p no:cacheprovider --basetemp={fixture_root.as_posix()}"
    log = AUDIT / f"{args.stage}_attempt{args.attempt:02d}.log"
    record = log.with_suffix(".json")
    if log.exists() or record.exists():
        raise FileExistsError("operational stage log already exists")
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.perf_counter()
    with log.open("x", encoding="utf-8", newline="\n") as sink:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   bufsize=1)
        for line in process.stdout:
            sink.write(line)
            sink.flush()
            print(line, end="", flush=True)
        code = process.wait()
    payload = {"stage": args.stage, "attempt": args.attempt, "command": command,
               "execution_started_utc": started_utc,
               "runtime_seconds": time.perf_counter() - started, "exit_code": code,
               "log_path": str(log),
               "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
               "operational_environment_changes": {
                   "unbuffered_stdout": True, "disable_implicit_bytecode_writes": True,
                   "compile_bytecode_output_under_allowed_model_root": args.stage == "compile",
                   "disable_pytest_cacheprovider": args.stage == "pytest"},
               "scientific_parameters_modified": False}
    with record.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print("HISTORICAL_STAGE_EXECUTION=" + json.dumps(payload, sort_keys=True), flush=True)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
