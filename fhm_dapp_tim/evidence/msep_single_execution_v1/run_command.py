"""Local command evidence recorder; does not alter experiment arguments."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent


def main():
    label, *arguments = sys.argv[1:]
    stdout = AUDIT / "run_logs" / f"{label}.stdout.log"
    stderr = AUDIT / "run_logs" / f"{label}.stderr.log"
    if stdout.exists() or stderr.exists():
        raise FileExistsError(label)
    command = [sys.executable, *arguments]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    start = datetime.datetime.now().astimezone().isoformat()
    tic = time.perf_counter()
    print(json.dumps({"event": "start", "label": label, "command": command}, ensure_ascii=False), flush=True)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    elapsed = time.perf_counter() - tic
    item = {"label": label, "command": command, "cwd": str(ROOT), "started_at": start,
            "ended_at": datetime.datetime.now().astimezone().isoformat(),
            "duration_seconds": elapsed, "exit_code": completed.returncode,
            "logs": {kind: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for kind, path in (("stdout", stdout), ("stderr", stderr))}}
    record_path = AUDIT / "EXECUTION_RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["commands"].append(item)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "finished", **item}, ensure_ascii=False), flush=True)
    print(stdout.read_text(encoding="utf-8", errors="replace")[-9000:])
    if completed.returncode:
        print(stderr.read_text(encoding="utf-8", errors="replace")[-12000:], file=sys.stderr)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
