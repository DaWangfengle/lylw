"""Read-only initial source/environment snapshot for the G1 pilot."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
assert ROOT == Path(r"D:\liuyanlunwen")
def sha(path):
    with path.open("rb") as handle:
        h = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()
def entry(path, digest=True):
    stat = path.stat()
    result = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if digest:
        result["sha256"] = sha(path)
    return result

protected = {}
for parent in (ROOT / "fhm_dapp_tim/fhm_dapp", ROOT / "tests"):
    for path in sorted(parent.rglob("*.py")):
        if path.name in {"uji_v17_g1_protocol.py","uji_v17_g1_experiment.py","test_uji_v17_g1.py"}:
            continue
        protected[path.relative_to(ROOT).as_posix()] = entry(path)
for path in sorted((ROOT / "fhm_dapp_tim/evidence").rglob("*")):
    if not path.is_file() or "v17_g1_geometry_support" in path.parts or "__pycache__" in path.parts:
        continue
    protected[path.relative_to(ROOT).as_posix()] = entry(path, path.suffix.lower() in {".json",".md",".txt",".csv",".py"})
training = ROOT / "data/UJIndoorLoc/trainingData.csv"
protected[training.relative_to(ROOT).as_posix()] = entry(training)
expected = "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168"
assert protected["data/UJIndoorLoc/trainingData.csv"]["sha256"] == expected
new_sources = ["fhm_dapp_tim/fhm_dapp/uji_v17_g1_protocol.py",
               "fhm_dapp_tim/fhm_dapp/uji_v17_g1_experiment.py",
               "tests/test_uji_v17_g1.py",
               "fhm_dapp_tim/evidence/v17_g1_geometry_support/DESIGN_V17_G1.md",
               "V17_G1_CODEX_RUN.txt"]
snapshots = AUDIT / "initial_sources"
snapshots.mkdir(exist_ok=True)
new_records = {}
for name in new_sources:
    path = ROOT / name
    snapshot = snapshots / path.name
    with snapshot.open("xb") as handle:
        handle.write(path.read_bytes())
    new_records[name] = entry(path)
gpu = torch.cuda.get_device_properties(0)
record = {
    "status": "INITIAL_SOURCE_AND_ENVIRONMENT_AUDIT_PASS",
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "workspace": str(ROOT), "local_ref": (ROOT / ".git/HEAD").read_text().strip(),
    "python_executable": sys.executable, "python_version": sys.version,
    "platform": platform.platform(), "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    "gpu": {"name": gpu.name, "memory_bytes": gpu.total_memory,
            "free_memory_bytes": torch.cuda.mem_get_info()[0]},
    "workspace_free_bytes": shutil.disk_usage(ROOT).free,
    "training_data_sha256": expected, "initial_G1_sources": new_records,
    "protected_historical_assets": protected,
    "historical_binary_content_read": False,
    "old_checkpoint_weights_used": False,
    "official_or_external_data_opened": False,
    "outer_test_predictions": 0,
    "initial_process_launcher_failures": [
        {"operation": "read-only local file/environment inspection",
         "reason": "helper_unknown_error: setup refresh had errors",
         "recovery": "same scoped read under reviewed execution"}
    ],
    "initial_inspection_notes": [
        "An rg inspection initially named two nonexistent optional module paths; actual imports and existing source paths were then read."
    ],
    "G1_result_files_present_at_start": []
}
for name in ["registry_v17_g1.json","calibration_v17_g1.json","freeze_v17_g1.json","confirmation_v17_g1.json"]:
    assert not (AUDIT.parent / name).exists(), name
output = AUDIT / "INITIAL_EXECUTION_AUDIT.json"
with output.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
    handle.write("\n")
print(json.dumps({"status":record["status"],"protected_assets":len(protected),
                  "gpu":record["gpu"],"workspace_free_bytes":record["workspace_free_bytes"]}))
