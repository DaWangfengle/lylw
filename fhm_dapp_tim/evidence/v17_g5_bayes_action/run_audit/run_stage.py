"""Small G5 command logger: raw outputs, timing, exit status and hashes."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[3]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def historical_hashes():
    files = set((REPO / "fhm_dapp_tim/fhm_dapp").rglob("*.py"))
    files.update((REPO / "tests").glob("test_uji_v17_g[1-4]*.py"))
    for name in ("v17_g1_geometry_support", "v17_g2_mechanism",
                 "v17_g3_uts_frozen_transfer", "v17_g4_observable_support"):
        root = REPO / "fhm_dapp_tim/evidence" / name
        files.update(p for p in root.rglob("*") if p.is_file()
                     and "runtime_cache" not in p.parts and "__pycache__" not in p.parts)
    files = {p for p in files if p.name != "uji_v17_g5_bayes_action.py"}
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(files)}

def main():
    name, *command = sys.argv[1:]
    if name == "snapshot":
        target = AUDIT / "EXECUTION_CONTEXT.json"
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "python": sys.executable, "repository": str(REPO),
                       "no_git_operations": True,
                       "historical_integrity_only_hashes": historical_hashes(),
                       "initial_g5_hashes": {
                           p.relative_to(REPO).as_posix(): digest(p) for p in (
                               REPO / "fhm_dapp_tim/fhm_dapp/uji_v17_g5_bayes_action.py",
                               REPO / "tests/test_uji_v17_g5_bayes_action.py",
                               AUDIT.parent / "DESIGN_V17_G5.md")}},
                      handle, ensure_ascii=False, indent=2)
        print(json.dumps({"context": str(target), "sha256": digest(target)}))
        return 0
    if not command:
        raise ValueError("stage needs a command")
    cache = AUDIT / "runtime_cache"
    (cache / "pytest").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
                "PYTHONPATH": os.pathsep.join((str(AUDIT / "access_guard"), str(REPO))),
                "PYTEST_ADDOPTS": '-p no:cacheprovider --basetemp="' +
                    (cache / "pytest" / name).as_posix() + '"',
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "G5_REPO": str(REPO), "G5_TRACE": str(AUDIT / (name + ".access.json"))})
    stdout = AUDIT / (name + ".stdout.log")
    stderr = AUDIT / (name + ".stderr.log")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    clock = time.perf_counter()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        code = subprocess.run(command, cwd=REPO, env=env, stdout=out, stderr=err).returncode
    record = {"stage": name, "command": command, "started_utc": started,
              "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "seconds": time.perf_counter() - clock, "exit_code": code,
              "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
              "access_trace_sha256": digest(AUDIT / (name + ".access.json"))}
    with (AUDIT / "COMMANDS.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    print(stdout.read_text(encoding="utf-8", errors="replace")[-5000:], flush=True)
    if code:
        print(stderr.read_text(encoding="utf-8", errors="replace")[-7000:], flush=True)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
