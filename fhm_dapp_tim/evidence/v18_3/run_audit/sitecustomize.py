"""Read-only V18.3 runtime instrumentation; never changes tensors or RNG."""
import atexit
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

_armed = False


def activate():
    global _armed
    phase = os.environ.get("V183_AUDIT_PHASE")
    folder = os.environ.get("V183_AUDIT_DIR")
    if _armed or not phase or not folder:
        return
    _armed = True
    folder = Path(folder)
    root = Path(os.environ["V183_REPO_ROOT"])
    evidence = root / "fhm_dapp_tim/evidence/v18_3"
    lock = evidence / "confirmation_consumed.lock"
    counts = dict(training_csv_reads=0, confirmation_reads=0, benchmark_reads=0,
                  historical_checkpoint_reads=0, denied=0, v18_2_import_attempts=0)
    checked = dict(structured_calls=0, training_scans=0, inference_scans=0,
                   dropped_training_scans=0, training_loss_calls=0,
                   prediction_calls=0, builds=0, violations=0)
    ranges = {}
    stability_histories = []
    fit_started = {}
    log = (folder / (phase + "_io.jsonl")).open("x", encoding="utf-8")
    roles_allowed = {
        "prepare": {"member_fit", "early_stop", "selection_refit", "calibration", "outer_refit", "confirmation"},
        "verify-prepare": {"member_fit", "early_stop", "selection_refit", "calibration", "outer_refit"},
        "train-selection": {"member_fit", "early_stop", "selection_refit"},
        "calibrate-freeze": {"calibration"}, "stability": {"member_fit", "early_stop"},
        "train-confirmation": {"outer_refit"}, "confirm": {"confirmation"},
        "train-final": {"outer_refit", "confirmation"},
    }

    def observe(event, args):
        if event == "import" and str(args[0]).endswith((".v18_2_single", ".uji_v18_2_experiment")):
            counts["v18_2_import_attempts"] += 1
            counts["denied"] += 1
            raise ImportError("V18.3 runtime independence audit denied " + str(args[0]))
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0]))).replace("\\", "/")
        mode, flags = args[1], args[2]
        reading = ("r" in mode or "+" in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
        kind = None
        permitted = True
        present = None
        purpose = None
        if path.endswith("/trainingdata.csv"):
            kind = "training_csv"
            if reading:
                counts["training_csv_reads"] += 1
                permitted = phase in {"prepare", "verify-prepare"}
        elif path.endswith("/validationdata.csv"):
            kind = "pre_exposed_benchmark"
            if reading:
                counts["benchmark_reads"] += 1
                permitted = phase == "benchmark" and (folder / "train-final_execution.json").is_file() and (evidence / "confirmation_result_v18_3.json").is_file()
        elif "/role_shards/" in path and path.endswith(".pt"):
            kind = "role_shard"
            if reading:
                role = Path(path).stem
                current_shard_dir = os.path.normcase(str(evidence / 'role_shards')).replace('\\', '/')
                permitted = path.startswith(current_shard_dir + '/') and role in roles_allowed.get(phase, set())
                if role == "confirmation":
                    counts["confirmation_reads"] += 1
                    present = lock.is_file()
                    if phase == "prepare":
                        purpose = "registered_prepare_hash_only"
                    elif phase == "confirm":
                        permitted = permitted and present
                        purpose = "one_shot_sequential_internal_confirmation"
                    elif phase == "train-final":
                        permitted = permitted and present and (evidence / "confirmation_result_v18_3.json").is_file()
                        purpose = "frozen_final_refit_after_valid_confirmation"
                    else:
                        permitted = False
        elif "/models/" in path and path.endswith((".pt", ".pth", ".ckpt")):
            kind = "checkpoint"
            if reading and "/models/v18_3/" not in path:
                counts["historical_checkpoint_reads"] += 1
                permitted = False
        elif "/evidence/" in path and "/evidence/v18_3/" not in path and path.endswith(".json") and reading:
            kind = "historical_evidence"
            permitted = False
        elif "/role_shards/" not in path and path.endswith("/confirmation_consumed.lock"):
            kind = "confirmation_lock"
        if kind is None:
            return
        log.write(json.dumps(dict(time_ns=time.time_ns(), phase=phase, kind=kind, path=path,
                                  mode=mode, reading=reading, permitted=permitted,
                                  confirmation_lock_present=present, purpose=purpose)) + "\n")
        log.flush()
        if not permitted:
            counts["denied"] += 1
            raise PermissionError("V18.3 audit denied " + kind + " in " + phase)

    sys.addaudithook(observe)

    if phase == 'stability':
        def capture_fit(frame, event, value):
            if frame.f_code.co_name != 'fit_fixed' or not frame.f_code.co_filename.endswith('uji_v18_3_experiment.py'):
                return
            if event == 'call':
                fit_started[id(frame)] = time.perf_counter()
            elif event == 'return' and isinstance(value, tuple) and len(value) == 5:
                stability_histories.append(dict(seed=int(frame.f_locals['seed']), history=value[-1],
                    fit_wall_seconds=time.perf_counter() - fit_started.pop(id(frame))))
        sys.setprofile(capture_fit)

    def instrument(module):
        torch = sys.modules["torch"]
        tol = 2e-5

        def require(condition, label):
            if not bool(condition):
                checked["violations"] += 1
                raise RuntimeError("V18.3 fail-closed runtime audit: " + label)

        def finite(values):
            tensors = [v.detach() for v in values.values() if torch.is_tensor(v)]
            require(torch.stack([torch.isfinite(v).all() for v in tensors]).all().item(), "non-finite output")

        def remember(names, values):
            for name, low, high in zip(names, values[::2], values[1::2]):
                old = ranges.setdefault(name, [low, high])
                old[0] = min(old[0], low)
                old[1] = max(old[1], high)

        original = module.structured_evidence_opinions

        @functools.wraps(original)
        def structured(*args, **kwargs):
            result = original(*args, **kwargs)
            finite(result)
            weights = result["inference_family_weights"].detach()
            within = result["within_family_weights"].detach()
            keep = result["family_keep_mask"].detach()
            active = result["family_weights"].detach()
            temperature = result["disagreement_temperature"].detach()
            d = result["global_disagreement"].detach()
            predicates = [
                (weights >= .15 - tol).all(), (weights <= .70 + tol).all(),
                (weights.sum(1) - 1).abs().max() <= tol,
                (within >= .30 - tol).all(), (within <= .70 + tol).all(),
                (within.sum(2) - 1).abs().max() <= tol,
                ((keep == 0) | (keep == 1)).all(), ((keep == 0).sum(1) <= 1).all(),
                (active.sum(1) - 1).abs().max() <= tol,
                (active - weights * keep / (weights * keep).sum(1, keepdim=True)).abs().max() <= tol,
                (temperature >= 1 - tol).all(), (temperature <= 2 + tol).all(),
                (temperature - (1 + d).clamp(1, 2)).abs().max() <= tol,
            ]
            for name in ["source_probability", "family_probability", "base_consensus_probability", "consensus_probability"]:
                probability = result[name].detach()
                predicates += [(probability >= 0).all(), (probability.sum(-1) - 1).abs().max() <= tol]
            for name in ["source_entropy", "family_entropy", "family_disagreement", "global_disagreement", "radio_mismatch"]:
                value = result[name].detach()
                predicates += [(value >= -tol).all(), (value <= 1 + tol).all()]
            if not kwargs["training"]:
                predicates += [(keep == 1).all()]
            require(torch.stack(predicates).all().item(), "family/source weights, probabilities, temperature or dropout")
            columns = [weights[:, i] for i in range(3)] + [within[:, f, s] for f in range(3) for s in range(2)] + [d, temperature, result["radio_mismatch"].detach()]
            names = ["family_" + n for n in module.EVIDENCE_FAMILIES] + ["within_" + str(f) + "_" + str(s) for f in range(3) for s in range(2)] + ["global_disagreement", "disagreement_temperature", "radio_mismatch"]
            values = torch.stack([v for column in columns for v in (column.min(), column.max())] + [(keep == 0).any(1).sum()]).cpu().tolist()
            remember(names, values[:-1])
            batch = len(weights)
            checked["structured_calls"] += 1
            checked["training_scans" if kwargs["training"] else "inference_scans"] += batch
            if kwargs["training"]:
                checked["dropped_training_scans"] += int(values[-1])
            return result

        module.structured_evidence_opinions = structured
        for name, counter in [("v183_training_loss", "training_loss_calls"), ("v183_predict", "prediction_calls")]:
            def wrap(function, key):
                @functools.wraps(function)
                def observed(*args, **kwargs):
                    if key == "prediction_calls":
                        require(float(kwargs.get("posterior_temperature", 1.0)) >= 1.0, "sharpening calibration temperature")
                    result = function(*args, **kwargs)
                    finite(result)
                    checked[key] += 1
                    return result
                return observed
            setattr(module, name, wrap(getattr(module, name), counter))

        build = module.build_v183_model

        @functools.wraps(build)
        def observed_build(*args, **kwargs):
            result = build(*args, **kwargs)
            model, _geometry, neighbours, weights = result
            require(not any("gate" in name.lower() for name, _p in model.named_parameters()), "trainable gate parameter")
            require(not hasattr(model, "evidence_gate"), "evidence_gate attribute")
            require(not (((neighbours == torch.arange(model.n_anchors, device=neighbours.device)[:, None]) & (weights > 0)).any().item()), "positive self-anchor target")
            require(80 + neighbours.shape[1] <= 96, "candidate-union target capacity")
            checked["builds"] += 1
            return result

        module.build_v183_model = observed_build

    target = "fhm_dapp_tim.fhm_dapp.v18_3_single"

    class Loader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original
        def create_module(self, spec):
            return self.original.create_module(spec)
        def exec_module(self, module):
            self.original.exec_module(module)
            instrument(module)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.endswith((".v18_2_single", ".uji_v18_2_experiment")):
                counts["v18_2_import_attempts"] += 1
                counts["denied"] += 1
                raise ImportError("V18.3 runtime independence audit denied " + fullname)
            if fullname == "fhm_dapp_tim.fhm_dapp.v18_3_single":
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                spec.loader = Loader(spec.loader)
                return spec
            return None

    if target in sys.modules:
        instrument(sys.modules[target])
    else:
        sys.meta_path.insert(0, Finder())

    def finish():
        historical = [name for name in sys.modules if name.endswith((".v18_2_single", ".uji_v18_2_experiment"))]
        record = dict(phase=phase, audit_armed=True, io_counts=counts,
                      tensor_checks=checked, observed_ranges=ranges,
                      v18_2_runtime_modules=historical,
                      instrumentation_changes_rng_or_model_outputs=False)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                record.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    flash_sdpa=torch.backends.cuda.flash_sdp_enabled(),
                    memory_efficient_sdpa=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    math_sdpa=torch.backends.cuda.math_sdp_enabled(),
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
                if torch.cuda.is_initialized():
                    record.update(gpu=torch.cuda.get_device_name(0),
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception as error:
                record["runtime_audit_error"] = str(error)
        (folder / (phase + "_runtime.json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
        if phase == 'stability':
            sys.setprofile(None)
            (folder / 'stability_full_training_histories.json').write_text(json.dumps(stability_histories, indent=2), encoding='utf-8')
        log.close()

    atexit.register(finish)


if os.environ.get("V183_AUDIT_PHASE"):
    try:
        activate()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(75)
