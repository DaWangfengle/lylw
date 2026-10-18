"""Save unmodified command streams and execution metadata; no scientific logic."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PYTHON = Path(r"D:\miniminiconda\envs\liuyan\python.exe")
ZONE = dt.timezone(dt.timedelta(hours=8))


def timestamp():
    return dt.datetime.now(ZONE).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(label, arguments):
    logs = ROOT / "run_logs"
    logs.mkdir(exist_ok=True)
    stdout, stderr = logs / f"{label}.stdout.log", logs / f"{label}.stderr.log"
    if stdout.exists() or stderr.exists():
        raise FileExistsError(f"refusing to overwrite command logs: {label}")
    record_path = ROOT / "EXECUTION_RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {
        "started_at": timestamp(), "python": str(PYTHON), "commands": [],
        "scientific_git_operations": False,
    }
    command = [str(PYTHON), *arguments]
    item = {"label": label, "command": command, "cwd": str(REPO),
            "started_at": timestamp(), "stdout": str(stdout.relative_to(REPO)),
            "stderr": str(stderr.relative_to(REPO)), "status": "RUNNING"}
    record["commands"].append(item)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "command_start", "label": label, "time": item["started_at"]}), flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    started = time.perf_counter()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        result = subprocess.run(command, cwd=REPO, env=environment, stdout=out, stderr=err)
    item.update(exit_code=result.returncode, elapsed_seconds=time.perf_counter()-started,
                ended_at=timestamp(), stdout_sha256=digest(stdout), stderr_sha256=digest(stderr),
                status="COMPLETE" if result.returncode == 0 else "FAILED")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "command_end", "label": label, "exit_code": result.returncode,
                      "elapsed_seconds": item["elapsed_seconds"]}), flush=True)
    if result.returncode:
        print(stderr.read_text(encoding="utf-8", errors="replace")[-16000:], file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    if sys.argv[1:] == ["--formal-sequence"]:
        for dataset in ("tuji", "uts", "uji"):
            for seed in (17301, 17302, 17303):
                run(f"{dataset}_seed{seed}", ["-m", "fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1",
                    "run", "--dataset", dataset, "--seed", str(seed), "--output",
                    str(ROOT / f"{dataset}_seed{seed}"), "--device", "auto", "--batch", "64",
                    "--threads", "4", "--es-draws", "32"])
        run("aggregate", ["-m", "fhm_dapp_tim.fhm_dapp.msep_v17_two_stage_freeze_v1",
                          "aggregate", "--root", str(ROOT)])
    else:
        run(sys.argv[1], sys.argv[2:])
