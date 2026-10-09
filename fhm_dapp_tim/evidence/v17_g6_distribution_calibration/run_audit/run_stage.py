"""Small G6 command logger and TRAIN-variable access guard; no scoring logic."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
G6_EVIDENCE = AUDIT.parent
G6_MODELS = REPO / "fhm_dapp_tim/models/v17_g6_distribution_calibration"
MAT_NAMES = ("Formatted_data_A12.mat", "Formatted_data_POCO.mat",
             "Formatted_data_S20.mat", "Formatted_data_S7.mat", "Formatted_data_tabS7.mat")
ALLOWED = {(REPO / "data/TUJI1" / n).resolve() for n in MAT_NAMES}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def within(path, root):
    return path == root or root in path.parents

def child(stage, args):
    trace = {"scope": "Python open events and scipy.io.loadmat calls, not OS-wide monitoring",
             "data_open_counts": {}, "restricted_mat_calls": [], "denied": []}
    def audit(event, values):
        if event != "open" or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode = values[1] if len(values) > 1 else None
        flags = values[2] if len(values) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        reason = None
        if within(path, REPO / "data"):
            if path not in ALLOWED or writing:
                reason = "only five TUJI containers may be read; data writes forbidden"
            else:
                key = str(path.relative_to(REPO))
                trace["data_open_counts"][key] = trace["data_open_counts"].get(key, 0) + 1
        elif within(path, REPO / ".git"):
            reason = "Git access forbidden in experiment"
        elif within(path, REPO / "fhm_dapp_tim/evidence") and not within(path, G6_EVIDENCE):
            reason = "historical outcome files forbidden in G6 commands"
        elif within(path, REPO / "fhm_dapp_tim/models") and not within(path, G6_MODELS):
            reason = "historical model weights forbidden"
        elif writing and (within(path, REPO / "fhm_dapp_tim/fhm_dapp") or within(path, REPO / "tests")):
            reason = "scientific source/test writes forbidden in run commands"
        if reason:
            trace["denied"].append({"path": str(path), "reason": reason})
            raise PermissionError(reason + ": " + str(path))
    sys.addaudithook(audit)
    try:
        import scipy.io
        original = scipy.io.loadmat
        def restricted_loadmat(path, *positional, **kwargs):
            if Path(path).resolve() in ALLOWED:
                if positional or kwargs.get("variable_names") != ("results_train",):
                    raise PermissionError("TUJI must request exactly results_train")
                payload = original(path, **kwargs)
                record = {"file": Path(path).name, "variable_names": ["results_train"],
                          "returned_keys": sorted(payload), "results_test_returned": "results_test" in payload}
                trace["restricted_mat_calls"].append(record)
                if record["results_test_returned"]:
                    raise PermissionError("unexpected test variable returned")
                return payload
            return original(path, *positional, **kwargs)
        scipy.io.loadmat = restricted_loadmat
        if len(args) < 2 or args[0] != "-m":
            raise ValueError("logger accepts only Python module commands")
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    finally:
        with open(AUDIT / (stage + ".access.json"), "x", encoding="utf-8") as stream:
            json.dump(trace, stream, ensure_ascii=False, indent=2)

def run(stage, args):
    for suffix in (".stdout.log", ".stderr.log", ".access.json"):
        if (AUDIT / (stage + suffix)).exists():
            raise FileExistsError(stage + suffix)
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX=str(AUDIT / "runtime_cache/pycache"),
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    temp = AUDIT / "runtime_cache/pytest" / stage
    temp.parent.mkdir(parents=True, exist_ok=True)
    env["PYTEST_ADDOPTS"] = '-p no:cacheprovider --basetemp="' + temp.as_posix() + '"'
    env["PYTHONPATH"] = str(REPO)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    clock = time.perf_counter()
    stdout = AUDIT / (stage + ".stdout.log")
    stderr = AUDIT / (stage + ".stderr.log")
    with stdout.open("xb") as out, stderr.open("xb") as err:
        proc = subprocess.run([sys.executable, str(Path(__file__)), "--child", stage, *args],
                              cwd=REPO, env=env, stdout=out, stderr=err)
    record = {"stage": stage, "command": [sys.executable, *args],
              "instrumentation": "runpy module execution with TRAIN-variable access guard",
              "started_utc": started, "ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "seconds": time.perf_counter()-clock, "exit_code": proc.returncode,
              "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr)}
    trace_path = AUDIT / (stage + ".access.json")
    if trace_path.exists():
        record["access_trace_sha256"] = sha(trace_path)
    with (AUDIT / "COMMANDS.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if proc.returncode:
        print(stdout.read_text(encoding="utf-8"), flush=True)
        print(stderr.read_text(encoding="utf-8"), file=sys.stderr, flush=True)
    return proc.returncode

if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(run(sys.argv[1], sys.argv[2:]))
