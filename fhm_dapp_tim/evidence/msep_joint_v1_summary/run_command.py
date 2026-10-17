"""Record the authorized Python command verbatim, preserving separate raw logs."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    label, *args = sys.argv[1:]
    log_paths = {k: HERE / "run_logs" / f"{label}.{k}.log"
                 for k in ("stdout", "stderr")}
    if any(p.exists() for p in log_paths.values()):
        raise FileExistsError(label)
    command = [sys.executable, *args]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    started = datetime.datetime.now().astimezone().isoformat()
    tic = time.perf_counter()
    print(json.dumps({"event": "start", "label": label, "command": command}), flush=True)
    with log_paths["stdout"].open("xb") as out, log_paths["stderr"].open("xb") as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    item = {"label": label, "command": command, "cwd": str(ROOT),
            "started_at": started,
            "ended_at": datetime.datetime.now().astimezone().isoformat(),
            "duration_seconds": time.perf_counter() - tic,
            "exit_code": result.returncode,
            "environment_overrides": {k: env[k] for k in
                ("PYTHONUTF8", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "logs": {k: {"path": p.relative_to(ROOT).as_posix(),
                         "bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                     for k, p in log_paths.items()}}
    path = HERE / "EXECUTION_RECORD.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["commands"].append(item)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", **item}), flush=True)
    print(log_paths["stdout"].read_text(encoding="utf-8", errors="replace")[-6500:])
    if result.returncode:
        print(log_paths["stderr"].read_text(encoding="utf-8", errors="replace")[-12000:], file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
