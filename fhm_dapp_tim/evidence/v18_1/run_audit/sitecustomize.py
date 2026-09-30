"""Observe V18.1 I/O and CUDA runtime without changing model computation."""
import atexit
import json
import os
import sys
import time
from pathlib import Path
phase = os.environ.get("V181_AUDIT_PHASE")
audit = os.environ.get("V181_AUDIT_DIR")
if phase and audit:
    audit = Path(audit)
    root = Path(os.environ["V181_REPO_ROOT"])
    lock = root / "fhm_dapp_tim/evidence/v18_1/external_validation_consumed.lock"
    log = (audit / (phase + "_io.jsonl")).open("a", encoding="utf-8")
    counts = {"validation_reads": 0, "training_csv_reads": 0, "denied": 0}
    allowed_roles = {
        "prepare": {"member_fit", "early_stop", "selection_refit", "calibration"},
        "verify-prepare": {"member_fit", "early_stop", "selection_refit", "calibration"},
        "train-selection": {"member_fit", "early_stop", "selection_refit"},
        "calibrate-freeze": {"calibration"},
        "stability": {"member_fit", "early_stop"},
        "train-final": {"selection_refit", "calibration"},
    }
    def observe(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0]))).replace("\\", "/")
        mode, flags = args[1], args[2]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
        kind = None
        permitted = True
        present = None
        if path.endswith("/validationdata.csv"):
            kind = "validation"
            if reading:
                present = lock.is_file()
                permitted = phase == "external" and present
                counts["validation_reads"] += 1
        elif path.endswith("/trainingdata.csv"):
            kind = "training_csv"
            if reading:
                counts["training_csv_reads"] += 1
                permitted = phase in {"prepare", "verify-prepare"}
        elif "/evidence/" in path and not "/evidence/v18_1/" in path and path.endswith(".json") and reading:
            kind = "historical_evidence"
            permitted = False
        elif "/role_shards/" in path and path.endswith(".pt"):
            kind = "role_shard"
            if reading:
                role = Path(path).stem
                permitted = role in allowed_roles.get(phase, set())
        elif path.endswith("/external_validation_consumed.lock"):
            kind = "external_lock"
        elif "/models/v18_1/" in path and path.endswith(".pt"):
            kind = "checkpoint"
        if kind is None:
            return
        entry = {"time_ns": time.time_ns(), "phase": phase, "kind": kind,
                 "path": path, "mode": mode, "reading": reading,
                 "permitted": permitted, "validation_lock_present": present}
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if not permitted:
            counts["denied"] += 1
            raise PermissionError("V18.1 access audit denied " + kind + " in " + phase)
    sys.addaudithook(observe)
    def finish():
        record = {"phase": phase, "io_counts": counts}
        t = sys.modules.get("torch")
        if t is not None:
            try:
                record.update({
                    "torch_version": t.__version__, "cuda_version": t.version.cuda,
                    "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
                    "deterministic_warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
                    "flash_sdpa": t.backends.cuda.flash_sdp_enabled(),
                    "memory_efficient_sdpa": t.backends.cuda.mem_efficient_sdp_enabled(),
                    "math_sdpa": t.backends.cuda.math_sdp_enabled(),
                    "cudnn_benchmark": t.backends.cudnn.benchmark,
                    "cudnn_deterministic": t.backends.cudnn.deterministic,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                })
                if t.cuda.is_initialized():
                    record.update({"gpu": t.cuda.get_device_name(0),
                        "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": t.cuda.max_memory_reserved()})
            except Exception as error:
                record["runtime_audit_error"] = str(error)
        (audit / (phase + "_runtime.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
        log.close()
    atexit.register(finish)
