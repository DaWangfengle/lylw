"""Read-only boundary audit for the single registered V18.4 decoder probe."""
import os


def _install():
    import atexit
    import datetime as dt
    import json
    from pathlib import Path
    import sys
    import torch
    from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as experiment
    from fhm_dapp_tim.fhm_dapp import v18_3_single as single

    audit = Path(os.environ["V184_DECODER_AUDIT_DIR"])
    root = audit.parents[3]
    evidence = root / "fhm_dapp_tim/evidence/v18_3"
    allowed = [
        evidence / "registry_v18_3.json",
        evidence / "selection_freeze_v18_3.json",
        evidence / "role_shards/early_stop.pt",
        evidence / "role_shards/calibration.pt",
        root / "fhm_dapp_tim/models/v18_3/selection.pt",
    ]

    def normalized(path):
        return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(path))))

    allowed_names = {normalized(path) for path in allowed}
    record = {
        "audit_armed": True,
        "allowed_scientific_reads": {},
        "denied_accesses": [],
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "training_mode_requests": 0,
        "checkpoint_save_calls": 0,
        "prediction_calls": 0,
        "prediction_scans": 0,
        "inference_target_coordinates_passed": False,
        "instrumentation_changes_rng_or_predictions": False,
    }

    def io_hook(event, args):
        if event != "open" or not args or isinstance(args[0], int):
            return
        name = normalized(args[0])
        mode = args[1] or ""
        flags = args[2] or 0
        writing = any(c in mode for c in "wax+") or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        )
        scientific = (
            name.startswith(normalized(root / "data") + os.sep)
            or (
                name.startswith(normalized(root / "fhm_dapp_tim/evidence") + os.sep)
                and name.endswith((".json", ".pt"))
                and not name.startswith(normalized(audit.parent) + os.sep)
            )
            or name.startswith(normalized(root / "fhm_dapp_tim/models") + os.sep)
        )
        if name in allowed_names:
            if writing:
                record["denied_accesses"].append(name)
                raise PermissionError("Probe input is read-only: " + name)
            record["allowed_scientific_reads"][name] = (
                record["allowed_scientific_reads"].get(name, 0) + 1
            )
        elif scientific:
            record["denied_accesses"].append(name)
            raise PermissionError("Unregistered scientific input: " + name)

    sys.addaudithook(io_hook)

    def forbidden(counter):
        def reject(*args, **kwargs):
            record[counter] += 1
            raise RuntimeError("Decoder diagnostic forbids " + counter)
        return reject

    torch.Tensor.backward = forbidden("backward_calls")
    torch.autograd.backward = forbidden("backward_calls")
    torch.optim.Optimizer.__init__ = forbidden("optimizer_constructions")
    torch.save = forbidden("checkpoint_save_calls")
    original_train = torch.nn.Module.train

    def audited_train(self, mode=True):
        if mode:
            record["training_mode_requests"] += 1
            raise RuntimeError("Decoder diagnostic forbids training mode")
        return original_train(self, mode)

    torch.nn.Module.train = audited_train
    models = []
    original_loader = experiment.load_v183_checkpoint

    def audited_loader(*args, **kwargs):
        model, checkpoint = original_loader(*args, **kwargs)
        before = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
        models.append((model, before))
        record["selected_epoch_verified"] = int(checkpoint["selected_epoch"])
        return model, checkpoint

    experiment.load_v183_checkpoint = audited_loader
    original_predict = single.v183_predict

    def audited_predict(model, rss, **kwargs):
        if kwargs.get("target_coords") is not None:
            record["inference_target_coordinates_passed"] = True
            raise RuntimeError("Probe inference cannot receive target coordinates")
        result = original_predict(model, rss, **kwargs)
        record["prediction_calls"] += 1
        record["prediction_scans"] += len(rss)
        if record["prediction_calls"] % 4 == 0 or record["prediction_scans"] == 5203:
            print("[audit] frozen posterior scans: "
                  + str(record["prediction_scans"]) + "/5203", flush=True)
        return result

    single.v183_predict = audited_predict

    def finish():
        record["model_state_unchanged_bitwise"] = bool(models) and all(
            all(torch.equal(value.detach().cpu(), before[key])
                for key, value in model.state_dict().items())
            for model, before in models
        )
        record["model_load_count"] = len(models)
        record["confirmation_lock_exists"] = (evidence / "confirmation_consumed.lock").exists()
        record["confirmation_result_exists"] = (evidence / "confirmation_result_v18_3.json").exists()
        record["torch_version"] = torch.__version__
        record["cuda_version"] = torch.version.cuda
        record["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        record["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        record["flash_sdpa"] = torch.backends.cuda.flash_sdp_enabled()
        record["memory_efficient_sdpa"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        record["math_sdpa"] = torch.backends.cuda.math_sdp_enabled()
        record["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        record["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        record["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        Path(os.environ.get("V184_DECODER_RUNTIME_PATH", str(audit / "probe_runtime.json"))).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")

    atexit.register(finish)


if os.environ.get("V184_DECODER_AUDIT_DIR"):
    _install()
