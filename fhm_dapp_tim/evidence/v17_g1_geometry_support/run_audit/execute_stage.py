"""Capture every V17-G1 stage without changing its scientific arguments."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
MODELS = ROOT / "fhm_dapp_tim/models/v17_g1_geometry_support"
MODULE = "fhm_dapp_tim.fhm_dapp.uji_v17_g1_experiment"
PYTHON = Path(r"D:\miniminiconda\envs\liuyan\python.exe")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--member-index", type=int)
    parser.add_argument("--test-filter")
    args = parser.parse_args()
    assert ROOT == Path(r"D:\liuyanlunwen")
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    if args.stage == "compile":
        child = [str(PYTHON), "-m", "compileall", "-q",
                 "fhm_dapp_tim/fhm_dapp/uji_v17_g1_protocol.py",
                 "fhm_dapp_tim/fhm_dapp/uji_v17_g1_experiment.py",
                 "tests/test_uji_v17_g1.py"]
    elif args.stage == "pytest":
        child = [str(PYTHON), "-m", "pytest", "-q",
                 "tests/test_uji_v17_g1.py",
                 "tests/test_historical_reproduction.py",
                 "tests/test_uji_v17_mother_r3.py",
                 "--basetemp", "fhm_dapp_tim/models/v17_g1_geometry_support/pytest_tmp"]
        if args.test_filter:
            child += ["-k", args.test_filter]
        temporary = MODELS / "pytest_tmp"
        if temporary.exists():
            preserved = MODELS / f"pytest_tmp_preserved_before_attempt{args.attempt:02d}"
            assert temporary.resolve().is_relative_to(MODELS.resolve())
            assert preserved.resolve().is_relative_to(MODELS.resolve())
            if preserved.exists():
                raise FileExistsError(preserved)
            temporary.rename(preserved)
    elif args.stage in {"preflight","prepare","train-member","check-members","calibrate","confirm"}:
        child = [str(PYTHON), "-m", MODULE, args.stage]
        if args.stage == "train-member":
            if args.member_index is None or not 1 <= args.member_index <= 25:
                raise ValueError("member 1..25 required")
            child += ["--member-index", str(args.member_index)]
        if args.stage != "check-members":
            child += ["--device", "cuda"]
    else:
        raise ValueError(args.stage)
    if args.stage == "confirm":
        sentinel = AUDIT / "CONFIRMATION_COMMAND_STARTED.json"
        if sentinel.exists():
            raise RuntimeError("Confirmation command already started; no automatic retry allowed")
        with sentinel.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "command": child, "attempt": args.attempt}, handle, indent=2)
            handle.write("\n")
    label = args.stage
    if args.stage == "train-member":
        label += f"-{args.member_index:02d}"
    label += f"_attempt{args.attempt:02d}"
    record_path = AUDIT / (label + ".json")
    stdout_path = AUDIT / (label + ".stdout.log")
    stderr_path = AUDIT / (label + ".stderr.log")
    if any(p.exists() for p in (record_path, stdout_path, stderr_path)):
        raise FileExistsError(label)
    cache = MODELS / "runtime_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_ADDOPTS": "-p no:cacheprovider",
                "MPLCONFIGDIR": str(cache / "matplotlib"),
                "TORCH_HOME": str(cache / "torch"),
                "XDG_CACHE_HOME": str(cache)})
    record = {"stage": args.stage, "member_index": args.member_index,
              "attempt": args.attempt, "command": child, "working_directory": str(ROOT),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
              "scientific_parameters_changed_by_logger": False,
              "output_encoding": "utf-8", "pytest_cacheprovider": "disabled"}
    started = time.perf_counter()
    with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
        process = subprocess.Popen(child, cwd=ROOT, env=env, stdout=out, stderr=err)
        print(json.dumps({"stage": label, "child_pid": process.pid, "command": child,
                          "stdout": str(stdout_path), "stderr": str(stderr_path)}), flush=True)
        record["child_pid"] = process.pid
        code = process.wait()
    record.update({"ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "runtime_seconds": time.perf_counter() - started,
                   "exit_code": code, "stdout_sha256": sha(stdout_path),
                   "stderr_sha256": sha(stderr_path)})
    with record_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"stage": label, "exit_code": code,
                      "runtime_seconds": record["runtime_seconds"],
                      "audit": str(record_path)}), flush=True)
    print(stdout_path.read_text(encoding="utf-8", errors="replace")[-1800:], flush=True)
    if code:
        print(stderr_path.read_text(encoding="utf-8", errors="replace")[-2400:], flush=True)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
