"""Read-only boundary audit for the single registered V18.4 decoder probe."""
import os


def _install():
    import atexit
    import contextvars
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
    authorization_path = audit / "radio_profile_authorization.json"
    radio_authorized = authorization_path.is_file() and json.loads(
        authorization_path.read_text(encoding="utf-8")
    )["radio_nuisance_optimization_authorized"]
    record = {
        "audit_armed": True,
        "allowed_scientific_reads": {},
        "denied_accesses": [],
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "training_mode_requests": 0,
        "model_reconstruction_mode_restore_calls": 0,
        "checkpoint_save_calls": 0,
        "prediction_calls": 0,
        "prediction_scans": 0,
        "attempted_prediction_calls": 0,
        "radio_nuisance_exception_authorized": bool(radio_authorized),
        "radio_profiler_calls": 0,
        "radio_nuisance_optimizer_constructions": 0,
        "radio_nuisance_tensor_backward_calls": 0,
        "radio_nuisance_autograd_backward_calls": 0,
        "radio_nuisance_optimizer_steps": 0,
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

    active_profile = contextvars.ContextVar("registered_radio_profile", default=None)
    original_tensor_backward = torch.Tensor.backward
    original_autograd_backward = torch.autograd.backward
    original_optimizer_init = torch.optim.Optimizer.__init__
    original_adam_step = torch.optim.Adam.step
    original_profiler = single.profile_shared_affine_marked_energy

    def audited_tensor_backward(*args, **kwargs):
        if active_profile.get() is None:
            return forbidden("backward_calls")(*args, **kwargs)
        record["radio_nuisance_tensor_backward_calls"] += 1
        return original_tensor_backward(*args, **kwargs)

    def audited_autograd_backward(*args, **kwargs):
        if active_profile.get() is None:
            return forbidden("backward_calls")(*args, **kwargs)
        record["radio_nuisance_autograd_backward_calls"] += 1
        return original_autograd_backward(*args, **kwargs)

    def audited_optimizer_init(self, params, defaults):
        context = active_profile.get()
        if context is None:
            return forbidden("optimizer_constructions")(self, params, defaults)
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not original_profiler.__code__:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Optimizer must belong to the registered radio profiler")
        expected = [frame.f_locals["raw_offset"], frame.f_locals["raw_log_scale"]]
        actual = list(params)
        checkpoint_ids = {id(value) for model, _ in models
                          for value in list(model.parameters()) + list(model.buffers())}
        if len(actual) != 2 or any(a is not b for a, b in zip(actual, expected)):
            raise RuntimeError("Only raw_offset and raw_log_scale may be optimized")
        if any(id(value) in checkpoint_ids or not value.is_leaf or not value.requires_grad
               or value.shape != (len(frame.f_locals["query"]),) for value in actual):
            raise RuntimeError("Radio nuisance tensors must be fresh scan-length leaves")
        if float(defaults["lr"]) != 0.2:
            raise RuntimeError("Registered radio profiling learning rate drifted")
        original_optimizer_init(self, actual, defaults)
        context["optimizers"][id(self)] = {id(value) for value in actual}
        record["radio_nuisance_optimizer_constructions"] += 1

    def audited_adam_step(self, *args, **kwargs):
        context = active_profile.get()
        if context is None or id(self) not in context["optimizers"]:
            raise RuntimeError("Optimizer step outside registered radio nuisance inference")
        if {id(p) for group in self.param_groups for p in group["params"]} != context["optimizers"][id(self)]:
            raise RuntimeError("Radio optimizer parameter identities changed")
        result = original_adam_step(self, *args, **kwargs)
        record["radio_nuisance_optimizer_steps"] += 1
        context["steps"] += 1
        return result

    def audited_profiler(*args, **kwargs):
        if not radio_authorized:
            return original_profiler(*args, **kwargs)
        if int(kwargs["steps"]) != 6 or float(kwargs["learning_rate"]) != 0.2:
            raise RuntimeError("Registered radio profiling configuration drifted")
        if any(value.requires_grad for value in [args[0], args[3], *args[2].values(), kwargs["reliability"]]):
            raise RuntimeError("Radio profiler inputs must be detached")
        context = {"optimizers": {}, "steps": 0}
        token = active_profile.set(context)
        try:
            result = original_profiler(*args, **kwargs)
            if context["steps"] != 6 or len(context["optimizers"]) != 1:
                raise RuntimeError("Registered radio profiler did not perform exactly six nuisance steps")
            if any(p.grad is not None for model, _ in models for p in model.parameters()):
                raise RuntimeError("Checkpoint parameter received gradients")
            record["radio_profiler_calls"] += 1
            return result
        finally:
            active_profile.reset(token)

    torch.Tensor.backward = audited_tensor_backward
    torch.autograd.backward = audited_autograd_backward
    torch.optim.Optimizer.__init__ = audited_optimizer_init
    torch.optim.Adam.step = audited_adam_step
    single.profile_shared_affine_marked_energy = audited_profiler
    torch.save = forbidden("checkpoint_save_calls")
    original_train = torch.nn.Module.train
    loaded_module_ids = set()

    def audited_train(self, mode=True):
        if mode and id(self) in loaded_module_ids:
            record["training_mode_requests"] += 1
            raise RuntimeError("Decoder diagnostic forbids training mode")
        if mode:
            record["model_reconstruction_mode_restore_calls"] += 1
        return original_train(self, mode)

    torch.nn.Module.train = audited_train
    models = []
    original_loader = experiment.load_v183_checkpoint

    def audited_loader(*args, **kwargs):
        model, checkpoint = original_loader(*args, **kwargs)
        before = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
        models.append((model, before))
        loaded_module_ids.update(id(module) for module in model.modules())
        record["selected_epoch_verified"] = int(checkpoint["selected_epoch"])
        return model, checkpoint

    experiment.load_v183_checkpoint = audited_loader
    original_predict = single.v183_predict

    def audited_predict(model, rss, **kwargs):
        record["attempted_prediction_calls"] += 1
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
        record["checkpoint_parameter_gradient_count"] = sum(
            p.grad is not None for model, _ in models for p in model.parameters())
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
