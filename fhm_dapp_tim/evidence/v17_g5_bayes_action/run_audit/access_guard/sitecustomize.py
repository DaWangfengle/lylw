"""Process-scoped Python open audit for the G5 train-only run."""
import atexit
import json
import os
from pathlib import Path
import sys

repo = Path(os.environ["G5_REPO"]).resolve()
data_root = repo / "data"
allowed = {(data_root / "Tampere" / name).resolve()
           for name in ("Training_rss.csv", "Training_coordinates.csv")}
protected = [repo / "fhm_dapp_tim" / kind / name
             for kind in ("evidence", "models")
             for name in ("v17_g1_geometry_support", "v17_g2_mechanism",
                          "v17_g3_uts_frozen_transfer", "v17_g4_observable_support")]
events = []
denied = []

def audit(event, args):
    if event != "open" or isinstance(args[0], int):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1], args[2]
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
        isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
    reason = None
    if path.is_relative_to(data_root):
        events.append({"path": str(path), "mode": mode, "writing": writing})
        if path not in allowed or writing:
            reason = "Only two Tampere training files may be read; data writes forbidden"
    if any(path.is_relative_to(root) for root in protected):
        reason = "Historical labelled evidence and model files are not G5 inputs"
    if path.is_relative_to(repo / ".git"):
        reason = "No Git operations in G5"
    if writing and path.suffix == ".py" and (
            path.is_relative_to(repo / "fhm_dapp_tim/fhm_dapp") or
            path.is_relative_to(repo / "tests")):
        reason = "Scientific source writes forbidden in experiment processes"
    if reason:
        denied.append({"path": str(path), "reason": reason})
        raise PermissionError(reason + ": " + str(path))

sys.addaudithook(audit)

@atexit.register
def save_trace():
    with open(os.environ["G5_TRACE"], "x", encoding="utf-8", newline="\n") as handle:
        json.dump({"scope": "Python open events; not an OS-wide monitor",
                   "allowed_data_paths": sorted(map(str, allowed)),
                   "data_open_events": events, "denied_attempts": denied},
                  handle, ensure_ascii=False, indent=2)
