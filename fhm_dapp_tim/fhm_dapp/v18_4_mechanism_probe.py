"""Write-once, read-only decomposition of the frozen V18.4 UJI method.

No fitting, checkpoint writes, official validation, or external evaluation.
The frozen radio profiler's temporary Adam requires a recorded human exception;
the runtime guard permits only its two nuisance tensors and six steps per batch.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch

from .coherent_marked_v9 import common_student_t_randomness, continuous_energy_score_mc
from .uji_v18_4_development import (
    MODEL_SEED, PREDICTIVE_DRAWS, SCORING_SEED, _coordinate_group_ids,
    _load_model_checkpoint, _load_registered_data, _prediction_entropy,
    _reference_from_payload, _row_nll, _set_seed,
)
from .v18_4_core import (
    kl_p_to_q, reliability_bounded_posterior_correction,
    tail_mass_diagnostics, weighted_component_geometric_median,
)
from .v18_4_model import V184_CONFIG, apply_correction, raw_correction_energies, v184_distribution


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "fhm_dapp_tim/evidence/v18_4"
OUTPUT_JSON = EVIDENCE / "MECHANISM_PROBE_V18_4.json"
OUTPUT_MD = EVIDENCE / "MECHANISM_PROBE_V18_4.md"
EXECUTION_JSON = EVIDENCE / "run_audit/MECHANISM_PROBE_EXECUTION_V18_4.json"
EXECUTION_MD = EVIDENCE / "run_audit/MECHANISM_PROBE_EXECUTION_V18_4.md"
OUTPUTS = (OUTPUT_JSON, OUTPUT_MD, EXECUTION_JSON, EXECUTION_MD)
PROBE_VERSION = "v18_4_read_only_mechanism_probe_v1"
FROZEN_SOURCE_SHA = "5b3a1888707e193c649007ebeb208c2a9d02ff45"
EPSILON_KL = 0.025
# Already accepted float32 casting tolerance (test_v18_4_core and revision audit).
KL_TOLERANCE = 1e-7
PRIMARY = ("development_native", "calibration_frozen")
MODES = ("none", "radio_only", "geometry_only", "full")
DECODERS = ("posterior_mean", "map_component", "top8_mean", "geometric_median")
DELTA_METRICS = ("energy_score_m", "nll", "posterior_mean_mae_m")
SCENARIOS = (
    {"name": PRIMARY[0], "primary": True, "role": "development", "temperature": 1.0,
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_lambda_0p10.pt", "reference_role": "fit"},
    {"name": PRIMARY[1], "primary": True, "role": "calibration", "temperature": 1.5,
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_selection_refit.pt", "reference_role": "selection_refit"},
    {"name": "calibration_T1_control", "primary": False, "role": "calibration", "temperature": 1.0,
     "checkpoint": "fhm_dapp_tim/models/v18_4/uji_selection_refit.pt", "reference_role": "selection_refit"},
)
FORMAL_EXPECTED = {
    "fhm_dapp_tim/evidence/v18_4/uji_registry_v18_4.json": "3470b9ff247117fbe9cefcd2f8135a55a8efe440b497a4ac96e8346339c26fb3",
    "fhm_dapp_tim/evidence/v18_4/uji_development_v18_4.json": "8b7a30e2de836c26631d8fcda965dc6d7bbeb522c6d01ed53f3a4c25b1dc2019",
    "fhm_dapp_tim/evidence/v18_4/uji_calibration_v18_4.json": "77500418abd927629b5687e2639c6d77a56c7464f144fab81089899dc4076082",
    "fhm_dapp_tim/evidence/v18_4/uji_method_freeze_v18_4.json": "9901da1203e05af9a73ff8b7a3d6c6f75d77364b2f6d3a52d093e7d72b6c874f",
    "fhm_dapp_tim/models/v18_4/uji_lambda_0p10.pt": "460ce4c0a3fa22976bca164deec6ce0cbf60c4de796811acf7df22a880bf6419",
    "fhm_dapp_tim/models/v18_4/uji_selection_refit.pt": "e460d90876deb893466c8e8af9ac167b0f2deb145ed47e0698362ee8fc08b72b",
    "fhm_dapp_tim/evidence/v18_4/run_audit/FORMAL_UJI_EXECUTION_V18_4.json": "b3c63ab7d650f1c7f013acd7ecf29ed599b6ffbe08399eac880fc00f3f971610",
}
TEST_FILES = (
    "tests/test_v18_4_core.py", "tests/test_v18_4_model.py",
    "tests/test_uji_v18_4_development.py", "tests/test_v18_4_external.py",
    "tests/test_v18_4_protocol.py", "tests/test_v18_4_datasets.py",
    "tests/test_v18_4_capacity_audit.py", "tests/test_v18_4_mechanism_probe.py",
)
BOUNDARIES = {
    "training_performed": False, "model_parameters_modified": False,
    "model_gradients_created": False, "official_validation_accessed": False,
    "external_data_accessed": False, "external_evaluation_run": False,
    "v18_5_changes": False, "correction_references_refitted": False,
    "method_freeze_modified": False, "network_optimizer_invoked": False,
    "network_loss_backward_invoked": False,
}


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Mapping, self_field: str) -> str:
    clean = dict(payload)
    clean.pop(self_field, None)
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def canonical_config(config: Mapping) -> dict:
    """JSON stores the frozen tuple grids as arrays; compare their same law."""
    return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))


def seal(payload: dict, field: str) -> dict:
    payload[field] = canonical_hash(payload, field)
    return payload


def json_bytes(payload: Mapping) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def assert_unused(paths: Sequence[Path]) -> None:
    for path in paths:
        require(not path.exists(), f"write-once path already exists: {path}")


def assert_finite(value, name: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            assert_finite(item, f"{name}[{i}]")
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(float(value)), f"non-finite metric: {name}")
    elif isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"non-finite tensor: {name}")
    elif isinstance(value, np.ndarray):
        require(bool(np.isfinite(value).all()), f"non-finite array: {name}")


def normalized_path(path) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(path)))))


class FileOpenAudit:
    """Deny non-training dataset opens and writes to all pre-existing evidence.

    Python audit hooks cannot be unregistered. The context disables its hook
    on exit; the hook does not change any file operation or suppress a failure.
    """

    def __init__(self, root: Path = REPO_ROOT):
        self.root = root.resolve()
        self.data = normalized_path(self.root / "data")
        self.train = normalized_path(self.root / "data/UJIndoorLoc/trainingData.csv")
        self.evidence = normalized_path(self.root / "fhm_dapp_tim/evidence")
        self.models = normalized_path(self.root / "fhm_dapp_tim/models")
        self.sources = normalized_path(self.root / "fhm_dapp_tim/fhm_dapp")
        self.allowed_outputs = {normalized_path(path) for path in OUTPUTS}
        self.active = False
        self.opens: list[dict] = []
        self.denied: list[dict] = []

    def check(self, path, mode=None, flags: int = 0) -> None:
        if isinstance(path, int):
            return
        full = normalized_path(path)
        suffix = Path(full).suffix.lower()
        parts = {part.lower() for part in Path(full).parts}
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        in_data = full == self.data or full.startswith(self.data + os.sep)
        external_component = bool(parts & {"uts", "tampere", "tuji", "tuji1", "tuji1_new"})
        official = Path(full).name.lower() == "validationdata.csv"
        external_result = ("external" in parts and suffix in {".json", ".md", ".csv", ".tsv", ".npz", ".pt"}) or (
            suffix in {".json", ".md", ".csv"} and "external" in Path(full).name.lower()
            and any(token in Path(full).stem.lower() for token in ("result", "evaluation", "metrics")))
        dataset = in_data or official or (external_component and suffix in {".csv", ".tsv", ".json", ".mat", ".h5", ".zip", ".npy", ".npz"})
        reason = None
        if dataset:
            self.opens.append({"path": full, "mode": mode, "flags": int(flags),
                               "allowed": full == self.train and not writing})
            if full != self.train or writing:
                reason = "dataset boundary permits only read-only UJI trainingData.csv"
        if external_result:
            reason = "external evaluation result access forbidden"
        protected_write = writing and (
            full.startswith(self.models + os.sep)
            or (full.startswith(self.evidence + os.sep) and full not in self.allowed_outputs
                and suffix not in {".log", ".out", ".err"})
            or (full.startswith(self.sources + os.sep) and suffix == ".py"
                and Path(full).name != "v18_4_mechanism_probe.py"))
        if protected_write:
            reason = "frozen model/source/evidence write forbidden"
        if reason:
            self.denied.append({"path": full, "reason": reason})
            raise RuntimeError(f"file-open audit denied {full}: {reason}")

    def __enter__(self):
        self.active = True

        def hook(event, args):
            if self.active and event == "open":
                self.check(args[0], args[1], args[2] if isinstance(args[2], int) else 0)

        sys.addaudithook(hook)
        return self

    def __exit__(self, *_):
        self.active = False


def tensor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def randomness_hash(randomness: Mapping) -> str:
    tensors = {k: v for k, v in randomness.items() if isinstance(v, torch.Tensor)}
    metadata = {k: v for k, v in randomness.items() if not isinstance(v, torch.Tensor)}
    return hashlib.sha256((tensor_state_hash(tensors) + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()


class ReadOnlyModelGuard:
    """Allow temporary radio Adam only; reject all other optimizer/backward calls."""

    def __init__(self, model: torch.nn.Module, authorization: Mapping):
        self.model = model
        self.authorization = dict(authorization)
        self.in_radio = False
        self.rows = 0
        self.allowed_leaf_ids: set[int] = set()
        self.optimizer = None
        self.counts = {"temporary_adam_initializations": 0, "temporary_adam_steps": 0,
                       "temporary_nuisance_backward_calls": 0, "radio_batches": 0}
        self.before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.before_hash = tensor_state_hash(self.before)
        self.model_tensors = list(model.parameters()) + list(model.buffers())
        self.model_storage = {(str(t.device), t.untyped_storage().data_ptr()) for t in self.model_tensors if t.numel()}
        self.requires_grad_before = [p.requires_grad for p in model.parameters()]
        self.verification: dict = {}

    def assert_model_clean(self) -> None:
        require(all(not module.training for module in self.model.modules()), "model entered training mode")
        require(all(p.grad is None for p in self.model.parameters()), "model parameter gradient created")

    def _check_optimizer(self, optimizer) -> None:
        require(self.in_radio and optimizer is self.optimizer, "optimizer outside authorized radio profiler")
        params = [p for group in optimizer.param_groups for p in group["params"]]
        require(len(params) == 2 and {id(p) for p in params} == self.allowed_leaf_ids,
                "optimizer parameter set changed")
        self.assert_model_clean()

    @staticmethod
    def _backward_leaves(tensors) -> set[int]:
        tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)
        leaves: set[int] = set()
        pending = [t.grad_fn for t in tensors if t.grad_fn is not None]
        leaves.update(id(t) for t in tensors if t.is_leaf and t.requires_grad)
        seen = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            variable = getattr(node, "variable", None)
            if variable is not None and variable.requires_grad:
                leaves.add(id(variable))
            pending.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
        return leaves

    def __enter__(self):
        self.assert_model_clean()
        require(self.authorization.get("radio_temporary_adam_allowed") is True,
                "human radio nuisance optimizer exception required")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.originals = (torch.nn.Module.train, torch.optim.Optimizer.__init__,
                          torch.optim.Adam.step, torch.autograd.backward, torch.autograd.grad)
        original_train, original_init, original_step, original_backward, _ = self.originals
        guard = self

        def training(module, mode=True):
            require(not mode, "training mode forbidden during probe")
            return original_train(module, mode)

        def optimizer_init(optimizer, params, defaults):
            require(guard.in_radio and type(optimizer) is torch.optim.Adam,
                    "optimizer forbidden outside authorized temporary radio Adam")
            params = list(params)
            require(len(params) == 2 and guard.optimizer is None, "radio optimizer must contain exactly two fresh tensors")
            for param in params:
                require(isinstance(param, torch.Tensor) and param.is_leaf and param.requires_grad
                        and param.shape == (guard.rows,) and param.grad is None,
                        "radio optimizer received non-nuisance tensor")
                require((str(param.device), param.untyped_storage().data_ptr()) not in guard.model_storage,
                        "radio optimizer received model storage")
            guard.allowed_leaf_ids = {id(p) for p in params}
            require(len(guard.allowed_leaf_ids) == 2, "radio nuisance tensors must be distinct")
            guard.optimizer = optimizer
            guard.counts["temporary_adam_initializations"] += 1
            return original_init(optimizer, params, defaults)

        def step(optimizer, *args, **kwargs):
            guard._check_optimizer(optimizer)
            guard.counts["temporary_adam_steps"] += 1
            return original_step(optimizer, *args, **kwargs)

        def backward(tensors, *args, **kwargs):
            require(guard.in_radio and guard.optimizer is not None, "backward forbidden outside radio nuisance profiling")
            require(guard._backward_leaves(tensors) == guard.allowed_leaf_ids,
                    "backward graph contains leaves other than the two temporary nuisance tensors")
            guard._check_optimizer(guard.optimizer)
            guard.counts["temporary_nuisance_backward_calls"] += 1
            result = original_backward(tensors, *args, **kwargs)
            guard.assert_model_clean()
            return result

        def autograd_grad(*_args, **_kwargs):
            raise RuntimeError("autograd.grad forbidden during read-only probe")

        torch.nn.Module.train = training
        torch.optim.Optimizer.__init__ = optimizer_init
        torch.optim.Adam.step = step
        torch.autograd.backward = backward
        torch.autograd.grad = autograd_grad
        return self

    @contextmanager
    def radio_batch(self, rows: int):
        require(not self.in_radio, "nested radio profiling forbidden")
        previous = dict(self.counts)
        self.in_radio, self.rows, self.optimizer = True, int(rows), None
        self.allowed_leaf_ids = set()
        try:
            yield
            require(self.counts["temporary_adam_initializations"] - previous["temporary_adam_initializations"] == 1,
                    "radio batch must initialize Adam exactly once")
            for key in ("temporary_adam_steps", "temporary_nuisance_backward_calls"):
                require(self.counts[key] - previous[key] == 6, "radio batch must use exactly six registered steps/backwards")
            self.assert_model_clean()
            self.counts["radio_batches"] += 1
        finally:
            self.in_radio, self.optimizer = False, None
            self.allowed_leaf_ids = set()

    def __exit__(self, *_):
        (torch.nn.Module.train, torch.optim.Optimizer.__init__, torch.optim.Adam.step,
         torch.autograd.backward, torch.autograd.grad) = self.originals
        for parameter, needed in zip(self.model.parameters(), self.requires_grad_before):
            parameter.requires_grad_(needed)
        self.assert_model_clean()
        after = self.model.state_dict()
        require(set(after) == set(self.before), "model state_dict keys changed")
        require(all(torch.equal(value.detach().cpu(), self.before[key]) for key, value in after.items()),
                "model parameter or buffer changed bitwise")
        after_hash = tensor_state_hash(after)
        require(after_hash == self.before_hash, "model state hash changed")
        self.verification = {"before_sha256": self.before_hash, "after_sha256": after_hash,
                             "all_parameters_and_buffers_bitwise_unchanged": True,
                             "all_parameter_gradients_none": True, "training_mode_invoked": False,
                             **self.counts}


def decompose_correction(q: torch.Tensor, mask: torch.Tensor, production: Mapping,
                         epsilon_kl: float = EPSILON_KL) -> dict[str, dict]:
    """Use one production correction and its already-normalized source fields."""
    zero = torch.zeros(len(q), dtype=torch.float64, device=q.device)
    reliability_zero = torch.zeros(len(q), dtype=q.dtype, device=q.device)
    energy_zero = torch.zeros_like(q)
    none = {"posterior": q, "rho": zero.clone(), "realized_kl": zero.clone(),
            "epsilon_kl_effective": zero.clone(), "radio_reliability": reliability_zero.clone(),
            "geometry_reliability": reliability_zero.clone(),
            "radio_energy": energy_zero.clone(), "geometry_energy": energy_zero.clone()}
    modes = {"none": none}
    for name, active in (("radio_only", "radio"), ("geometry_only", "geometry")):
        radio_energy = production["radio_energy"] if active == "radio" else energy_zero
        geometry_energy = production["geometry_energy"] if active == "geometry" else energy_zero
        rr = production["radio_reliability"] if active == "radio" else reliability_zero
        rg = production["geometry_reliability"] if active == "geometry" else reliability_zero
        result = reliability_bounded_posterior_correction(
            q, radio_energy, geometry_energy, radio_reliability=rr,
            geometry_reliability=rg, epsilon_kl=float(epsilon_kl), mask=mask)
        result.update(radio_energy=radio_energy, geometry_energy=geometry_energy)
        expected = production[f"{active}_reliability"].to(q).double() * float(epsilon_kl)
        require(torch.equal(result["epsilon_kl_effective"], expected), "source-only effective budget mismatch")
        inactive = "geometry" if active == "radio" else "radio"
        require(bool((result[f"{inactive}_energy"] == 0).all()) and
                bool((result[f"{inactive}_reliability"] == 0).all()), "inactive source was not exactly zero")
        modes[name] = result
    modes["full"] = dict(production)
    require(torch.equal(modes["none"]["posterior"], q), "none posterior is not bitwise q_T")
    require(torch.equal(modes["full"]["posterior"], production["posterior"]), "full differs from production")
    for name, result in modes.items():
        p = result["posterior"]
        assert_finite(result, name)
        require(bool((p >= 0).all()), "negative posterior probability")
        require(bool((p[~mask] == 0).all()), "padding received posterior mass")
        require(bool(torch.allclose(p.double().sum(1), torch.ones(len(p), device=p.device, dtype=torch.float64),
                                   atol=4 * torch.finfo(p.dtype).eps, rtol=0)), "posterior is not normalized")
        actual_kl = kl_p_to_q(p.double(), q.double(), mask).clamp(min=0)
        require(bool((actual_kl <= result["epsilon_kl_effective"] + KL_TOLERANCE).all()), "KL exceeds row budget")
        require(bool(torch.allclose(actual_kl, result["realized_kl"], atol=1e-12, rtol=1e-9)), "reported KL differs from actual posterior")
    return modes


def decode_points(probability: torch.Tensor, locations: torch.Tensor,
                  mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Label-free point actions; stable component order resolves Top-8 ties."""
    require(probability.ndim == 2 and locations.shape == (*probability.shape, 2), "decoder input shape mismatch")
    require(bool(mask.any(1).all()), "decoder received empty valid support")
    require(bool((probability[~mask] == 0).all()), "decoder padding probability is not zero")
    mean = (probability.unsqueeze(2) * locations).sum(1)
    rows = torch.arange(len(probability), device=probability.device)
    masked = torch.where(mask, probability, torch.full_like(probability, float("-inf")))
    map_location = locations[rows, masked.argmax(1)]
    indices = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :min(8, probability.shape[1])]
    selected = probability.gather(1, indices) * mask.gather(1, indices).to(probability.dtype)
    selected = selected / selected.sum(1, keepdim=True).clamp(min=1e-30)
    top_locations = locations.gather(1, indices.unsqueeze(2).expand(-1, -1, 2))
    top8 = (selected.unsqueeze(2) * top_locations).sum(1)
    gm = weighted_component_geometric_median(probability, locations, mask=mask)
    result = dict(zip(DECODERS, (mean, map_location, top8, gm)))
    assert_finite(result, "decoders")
    return result


def as_array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().double().numpy()
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and len(array) > 0, "statistics require a non-empty row vector")
    assert_finite(array)
    return array


def group_means(values, groups: Sequence) -> np.ndarray:
    values = as_array(values)
    require(len(values) == len(groups), "metric/group row count mismatch")
    members: dict[object, list[int]] = {}
    for row, key in enumerate(groups):
        members.setdefault(key, []).append(row)
    return np.asarray([values[indices].mean() for indices in members.values()])


def mean_se(values) -> tuple[float, float]:
    values = as_array(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def paired_statistics(first, second, groups: Sequence, *, positive_is_better: bool = False) -> dict:
    first, second = as_array(first), as_array(second)
    require(first.shape == second.shape, "paired metric shapes differ")
    grouped = group_means(first - second, groups)
    mean, se = mean_se(grouped)
    improved = grouped > 0 if positive_is_better else grouped < 0
    return {"mean_delta": mean, "paired_se": se, "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
            "fraction_groups_improved": float(improved.mean()), "coordinate_groups": len(grouped)}


def decoder_metrics(errors, groups: Sequence) -> dict:
    errors = as_array(errors)
    macro, se = mean_se(group_means(errors, groups))
    return {"coordinate_macro_mae_m": macro, "coordinate_macro_mae_se_m": se,
            "row_mae_m": float(errors.mean()), "row_median_error_m": float(np.median(errors)),
            "row_p75_error_m": float(np.quantile(errors, .75)), "row_p90_error_m": float(np.quantile(errors, .9))}


def average_ranks(values) -> np.ndarray:
    values = as_array(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = .5 * (start + stop - 1)
        start = stop
    return ranks


def pearson(first, second) -> float:
    first, second = as_array(first), as_array(second)
    require(first.shape == second.shape, "correlation vector shapes differ")
    x, y = first - first.mean(), second - second.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    # A constant vector has undefined correlation; preregister deterministic 0.
    return 0.0 if denominator == 0 else float(np.clip(np.dot(x, y) / denominator, -1, 1))


def spearman(first, second) -> float:
    return pearson(average_ranks(first), average_ranks(second))


def reliability_quartiles(reliability) -> list[np.ndarray]:
    reliability = as_array(reliability)
    require(len(reliability) >= 4, "four non-empty reliability quartiles need at least four rows")
    quartiles = list(np.array_split(np.argsort(reliability, kind="stable"), 4))
    merged = np.concatenate(quartiles)
    require(np.array_equal(np.sort(merged), np.arange(len(reliability))), "reliability quartiles do not partition all rows")
    return quartiles


def source_energy_alignment(energy, distance, q, mask) -> dict:
    energy, distance, q = map(np.asarray, (energy, distance, q))
    mask = np.asarray(mask, dtype=bool)
    rho, selected_error, map_error = [], [], []
    for row in range(len(q)):
        valid = np.flatnonzero(mask[row])
        rho.append(spearman(energy[row, valid], distance[row, valid]))
        selected = valid[np.argmin(energy[row, valid])]
        map_index = valid[np.argmax(q[row, valid])]
        selected_error.append(distance[row, selected])
        map_error.append(distance[row, map_index])
    rho, selected_error, map_error = map(as_array, (rho, selected_error, map_error))
    return {"mean_row_spearman": float(rho.mean()), "median_row_spearman": float(np.median(rho)),
            "p10_row_spearman": float(np.quantile(rho, .1)), "p90_row_spearman": float(np.quantile(rho, .9)),
            "fraction_spearman_positive": float((rho > 0).mean()), "fraction_spearman_negative": float((rho < 0).mean()),
            "source_argmin_candidate_mean_error_m": float(selected_error.mean()),
            "q_T_map_candidate_mean_error_m": float(map_error.mean()),
            "fraction_source_argmin_beats_q_T_map": float((selected_error < map_error).mean())}


def reliability_utility(reliability, oracle_gain) -> dict:
    reliability, gain = as_array(reliability), as_array(oracle_gain)
    require(reliability.shape == gain.shape, "reliability/gain row count mismatch")
    quartiles = reliability_quartiles(reliability)
    assignment = np.empty(len(gain), dtype=np.int64)
    reports = []
    for number, rows in enumerate(quartiles, 1):
        assignment[rows] = number
        reports.append({"quartile": f"Q{number}", "row_count": len(rows),
                        "mean_reliability": float(reliability[rows].mean()),
                        "mean_oracle_gain_m": float(gain[rows].mean()),
                        "median_oracle_gain_m": float(np.median(gain[rows])),
                        "fraction_oracle_gain_positive": float((gain[rows] > 0).mean())})
    return {"mean_oracle_gain_m": float(gain.mean()), "median_oracle_gain_m": float(np.median(gain)),
            "fraction_oracle_gain_positive": float((gain > 0).mean()), "fraction_oracle_gain_negative": float((gain < 0).mean()),
            "spearman_reliability_oracle_gain": spearman(reliability, gain),
            "pearson_reliability_oracle_gain": pearson(reliability, gain),
            "quartiles": reports, "quartile_assignment_role_row_order": assignment.tolist(),
            "every_row_in_exactly_one_quartile": True}


def source_agreement(radio, geometry, locations, mask, rr, rg) -> dict:
    rhos, same, separation = [], [], []
    for row in range(len(radio)):
        valid = np.flatnonzero(mask[row])
        rhos.append(spearman(radio[row, valid], geometry[row, valid]))
        r = valid[np.argmin(radio[row, valid])]
        g = valid[np.argmin(geometry[row, valid])]
        same.append(r == g)
        separation.append(np.linalg.norm(locations[row, r] - locations[row, g]))
    rhos = as_array(rhos)
    return {"mean_row_energy_spearman": float(rhos.mean()), "median_row_energy_spearman": float(np.median(rhos)),
            "fraction_row_energy_spearman_positive": float((rhos > 0).mean()),
            "fraction_same_energy_argmin_candidate": float(np.mean(same)),
            "mean_argmin_spatial_distance_m": float(np.mean(separation)),
            "pearson_source_reliabilities": pearson(rr, rg), "spearman_source_reliabilities": spearman(rr, rg)}


def concat_support_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate role batches with exact zero padding for variable widths."""
    require(len(arrays) > 0, "cannot concatenate empty support batches")
    if arrays[0].ndim == 1:
        return np.concatenate(arrays)
    width = max(array.shape[1] for array in arrays)
    padded = [np.pad(array, ((0, 0), (0, width - array.shape[1])) + ((0, 0),) * (array.ndim - 2))
              for array in arrays]
    return np.concatenate(padded, axis=0)


def primary_correction_decision(scenarios: Mapping) -> dict:
    classifications, conditions = {}, {}
    for mode in MODES[1:]:
        deltas = {name: {metric: scenarios[name]["paired_correction_effects"][mode][metric]["mean_delta"]
                         for metric in DELTA_METRICS} for name in PRIMARY}
        benefit = all(d["energy_score_m"] < 0 and d["nll"] < 0 and d["posterior_mean_mae_m"] <= .05 for d in deltas.values())
        harm = all(d["energy_score_m"] > 0 and d["nll"] > 0 and d["posterior_mean_mae_m"] >= 0 for d in deltas.values())
        classifications[mode] = "CONSISTENTLY_BENEFICIAL" if benefit else "CONSISTENTLY_HARMFUL" if harm else "MIXED_OR_NEUTRAL"
        conditions[mode] = deltas
    good_sources = [m for m in ("radio_only", "geometry_only") if classifications[m] == "CONSISTENTLY_BENEFICIAL"]
    if classifications["full"] == "CONSISTENTLY_BENEFICIAL":
        conclusion = "full_correction_supported"
    elif len(good_sources) == 1:
        conclusion = f"single_source_correction_supported:{good_sources[0].removesuffix('_only')}"
    elif not good_sources and classifications["full"] == "CONSISTENTLY_HARMFUL":
        conclusion = "structured_correction_not_supported"
    else:
        conclusion = "correction_evidence_mixed"
    return {"primary_scenarios": list(PRIMARY), "classifications": classifications,
            "primary_deltas": conditions, "conclusion": conclusion}


def decoder_promotion_gate(base_mae: float, candidate_mae: float) -> dict:
    base, candidate = Decimal(str(base_mae)), Decimal(str(candidate_mae))
    require(base > 0 and candidate >= 0, "decoder promotion requires positive baseline and nonnegative candidate MAE")
    improvement = candidate <= Decimal("0.90") * base
    no_excess_worsening = candidate - base <= Decimal("0.25")
    return {"improvement_at_least_10_percent": improvement, "worsening_at_most_0p25_m": no_excess_worsening,
            "passes": improvement and no_excess_worsening}


def primary_decoder_decision(scenarios: Mapping) -> dict:
    gates, supported = {}, []
    directly_best = True
    for name in PRIMARY:
        metrics = scenarios[name]["modes"]["none"]["decoders"]
        base = metrics["posterior_mean"]["coordinate_macro_mae_m"]
        directly_best &= all(base <= metrics[d]["coordinate_macro_mae_m"] for d in DECODERS[1:])
    for decoder in DECODERS[1:]:
        gates[decoder] = {name: decoder_promotion_gate(
            scenarios[name]["modes"]["none"]["decoders"]["posterior_mean"]["coordinate_macro_mae_m"],
            scenarios[name]["modes"]["none"]["decoders"][decoder]["coordinate_macro_mae_m"]) for name in PRIMARY}
        if all(record["passes"] for record in gates[decoder].values()):
            supported.append(decoder)
    return {"primary_scenarios": list(PRIMARY), "correction_mode": "none", "promotion_gates": gates,
            "supported_alternatives": supported, "posterior_mean_directly_best_both_roles": bool(directly_best),
            "ties_count_as_directly_best": True,
            "conclusion": "decoder_alternatives_supported_over_mean" if supported else "posterior_mean_supported_as_default"}


def temperature_interaction(scenarios: Mapping) -> dict:
    result = {}
    for mode in MODES[1:]:
        comparisons = {}
        for metric in DELTA_METRICS:
            frozen = scenarios["calibration_frozen"]["paired_correction_effects"][mode][metric]["mean_delta"]
            control = scenarios["calibration_T1_control"]["paired_correction_effects"][mode][metric]["mean_delta"]
            comparisons[metric] = {"delta_T1p0": control, "delta_T1p5": frozen,
                                   "sign_T1p0": int(np.sign(control)), "sign_T1p5": int(np.sign(frozen)),
                                   "same_sign": int(np.sign(control)) == int(np.sign(frozen))}
        result[mode] = {"classification": "sign_stable" if all(c["same_sign"] for c in comparisons.values()) else "sign_changed",
                        "metrics": comparisons}
    return result


def verify_integrity() -> tuple[dict, dict]:
    files, payloads, self_hashes = {}, {}, {}
    for relative, expected in FORMAL_EXPECTED.items():
        actual = sha256_file(REPO_ROOT / relative)
        require(actual == expected, f"formal file-byte hash mismatch: {relative}")
        files[relative] = actual
        if relative.endswith(".json"):
            payloads[Path(relative).name] = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
    fields = {"uji_registry_v18_4.json": "registry_sha256", "uji_development_v18_4.json": "development_result_sha256",
              "uji_calibration_v18_4.json": "calibration_result_sha256", "uji_method_freeze_v18_4.json": "freeze_sha256",
              "FORMAL_UJI_EXECUTION_V18_4.json": "execution_record_sha256"}
    for name, field in fields.items():
        require(canonical_hash(payloads[name], field) == payloads[name][field], f"canonical self-hash mismatch: {name}")
        self_hashes[name] = {"field": field, "sha256": payloads[name][field], "verified": True}
    registry, development, calibration, freeze = (payloads[n] for n in fields if n != "FORMAL_UJI_EXECUTION_V18_4.json")
    for name in ("uji_registry_v18_4.json", "uji_development_v18_4.json", "uji_calibration_v18_4.json", "uji_method_freeze_v18_4.json"):
        require(payloads[name]["source_commit_sha"] == FROZEN_SOURCE_SHA, f"frozen source provenance mismatch: {name}")
        require(payloads[name]["protocol_version"] == registry["protocol_version"], f"protocol link mismatch: {name}")
        require(payloads[name]["external_test_performance_used_for_v184_selection"] is False and
                payloads[name]["uji_official_validation_used_for_v184_selection"] is False, "formal boundary flags mismatch")
    links = []
    def link(label, actual, expected):
        require(actual == expected, f"formal link mismatch: {label}")
        links.append({"link": label, "value": actual, "verified": True})
    for name, payload in (("development", development), ("calibration", calibration), ("freeze", freeze)):
        link(f"{name}->registry", payload["registry_sha256"], registry["registry_sha256"])
    link("calibration->development", calibration["development_result_sha256"], development["development_result_sha256"])
    link("freeze->development", freeze["development_result_sha256"], development["development_result_sha256"])
    link("freeze->calibration", freeze["calibration_result_sha256"], calibration["calibration_result_sha256"])
    development_checkpoint = FORMAL_EXPECTED[SCENARIOS[0]["checkpoint"]]
    refit_checkpoint = FORMAL_EXPECTED[SCENARIOS[1]["checkpoint"]]
    link("development->lambda0.10 checkpoint", development["selected"]["checkpoint_sha256"], development_checkpoint)
    reports = [r for r in development["lambda_reports"] if r["lambda_spatial"] == .1]
    require(len(reports) == 1, "lambda0.10 report is not unique")
    link("lambda0.10 report->checkpoint", reports[0]["checkpoint_sha256"], development_checkpoint)
    for name, payload in (("calibration", calibration), ("freeze", freeze)):
        link(f"{name}->selection_refit checkpoint", payload["selection_refit_checkpoint_sha256"], refit_checkpoint)
    expected_freeze = {"lambda_spatial": .1, "epsilon_kl": .025, "selected_epoch": 16,
                       "posterior_temperature": 1.5, "scale_multiplier": 1.,
                       "decoder": "posterior_weighted_component_center_geometric_median"}
    for key, expected in expected_freeze.items():
        link(f"freeze.{key}", freeze[key], expected)
    for key in ("lambda_spatial", "epsilon_kl", "selected_epoch"):
        link(f"development.selected.{key}", development["selected"][key], freeze[key])
        link(f"calibration.{key}", calibration[key], freeze[key])
    link("calibration.selected.temperature", calibration["selected"]["temperature"], freeze["posterior_temperature"])
    link("calibration.selected.scale", calibration["selected"]["scale_multiplier"], freeze["scale_multiplier"])
    link("registry.model_config", canonical_config(registry["model_config"]), canonical_config(V184_CONFIG))
    link("freeze.model_config", canonical_config(freeze["model_config"]), canonical_config(V184_CONFIG))
    execution = payloads["FORMAL_UJI_EXECUTION_V18_4.json"]
    link("formal_execution.freeze_artifact.source", execution["freeze_artifact"]["source_commit_sha"], FROZEN_SOURCE_SHA)
    link("formal_execution.freeze_artifact.self_hash", execution["freeze_artifact"]["freeze_sha256"], freeze["freeze_sha256"])
    sources = {}
    for relative, expected in execution["unchanged_source_file_sha256"].items():
        actual = sha256_file(REPO_ROOT / relative)
        require(actual == expected, f"frozen source hash mismatch: {relative}")
        sources[relative] = actual
    required = {"fhm_dapp_tim/fhm_dapp/" + n for n in (
        "v18_4_core.py", "v18_4_model.py", "uji_v18_4_development.py", "v18_4_protocol.py", "model.py", "coherent_marked_v9.py")}
    require(required <= set(sources), "formal record missing required frozen sources")
    preserved = dict(files)
    for item in execution["formal_artifacts"]:
        path = Path(item["path"])
        relative = path.relative_to(REPO_ROOT).as_posix()
        actual = sha256_file(path)
        require(actual == item["file_sha256"], f"formal execution artifact mismatch: {relative}")
        preserved[relative] = actual
    # Preserve all existing evidence and checkpoint bytes, without opening external results.
    snapshots = {}
    for directory in (EVIDENCE, REPO_ROOT / "fhm_dapp_tim/models/v18_4"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".pt", ".pth", ".ckpt"}:
                if "external" in {part.lower() for part in path.parts}:
                    continue
                relative = path.relative_to(REPO_ROOT).as_posix()
                snapshots[relative] = sha256_file(path)
    return {"status": "PASS", "formal_file_byte_sha256": files, "canonical_self_hash_validation": self_hashes,
            "link_validation": links, "verified_source_file_sha256": sources,
            "verified_source_file_count": len(sources), "preserved_formal_file_sha256": preserved,
            "pre_existing_artifact_sha256": snapshots}, registry


def validate_checkpoint(payload: Mapping, scenario: Mapping, registry: Mapping) -> dict:
    require(payload["lambda_spatial"] == .1 and payload["selected_epoch"] == 16, "checkpoint method/epoch mismatch")
    require(canonical_config(payload["model_config"]) == canonical_config(V184_CONFIG)
            and payload["protocol_version"] == registry["protocol_version"], "checkpoint config/protocol mismatch")
    reference_role = scenario["reference_role"]
    count = registry["roles"][reference_role]["row_count"]
    ref = payload["correction_reference"]
    for source in ("radio", "geometry"):
        require(ref[f"{source}_mismatch_reference"].shape == (count,), "checkpoint correction reference role size mismatch")
        require(bool(torch.isfinite(ref[f"{source}_mismatch_reference"]).all()), "checkpoint reference contains non-finite values")
        require(bool(torch.isfinite(ref[f"{source}_energy_scale"]).all()) and float(ref[f"{source}_energy_scale"]) > 0,
                "checkpoint energy scale invalid")
    anchors = registry["roles"][reference_role]["coordinate_count"]
    require(len(payload["memory"]["coords"]) == anchors, "checkpoint memory role size mismatch")
    expected_phase = "fit_candidate" if reference_role == "fit" else "selection_refit"
    require(payload["phase"] == expected_phase, "checkpoint role phase mismatch")
    if reference_role == "selection_refit":
        require(payload["epsilon_kl"] == EPSILON_KL, "refit checkpoint epsilon mismatch")
    return {"reference_role": reference_role, "reference_rows": count, "memory_anchors": anchors,
            "reference_tensor_sha256": tensor_state_hash(ref), "checkpoint_phase": expected_phase}


def run_scenario(scenario: Mapping, registry: Mapping, data, roles: Mapping,
                 rss_all: torch.Tensor, device: torch.device, authorization: Mapping) -> dict:
    started = time.perf_counter()
    rows = roles[scenario["role"]]
    groups = _coordinate_group_ids(data, rows)
    require(len(rows) == registry["roles"][scenario["role"]]["row_count"] and
            len(set(groups)) == registry["roles"][scenario["role"]]["coordinate_count"], "role row/group count mismatch")
    checkpoint = torch.load(REPO_ROOT / scenario["checkpoint"], map_location="cpu", weights_only=False)
    reference_validation = validate_checkpoint(checkpoint, scenario, registry)
    model = _load_model_checkpoint(checkpoint, device=device)
    reference = _reference_from_payload(checkpoint["correction_reference"]).to(device)
    reference_before = tensor_state_hash(checkpoint["correction_reference"])
    centered = data.coords_m - checkpoint["origin"].cpu().numpy()
    randomness = common_student_t_randomness(len(rows), draws=PREDICTIVE_DRAWS,
                                             degrees_freedom=float(V184_CONFIG["student_df"]), seed=SCORING_SEED)
    initial_randomness_hash = randomness_hash(randomness)
    collected = {mode: {} for mode in MODES}
    oracle_batches = {"radio": [], "geometry": []}
    agreement_batches = []
    batch_provenance = []
    calls = {"raw_correction_energies": 0, "apply_correction": 0,
             "energy_score_calls": {mode: 0 for mode in MODES}}
    guard = ReadOnlyModelGuard(model, authorization)

    def collect(mode, key, value):
        assert_finite(value, f"{scenario['name']}.{mode}.{key}")
        collected[mode].setdefault(key, []).append(value.detach().cpu().double())

    print(f"START {scenario['name']}: {len(rows)} rows / {len(set(groups))} groups / T={scenario['temperature']}", flush=True)
    with guard, torch.no_grad():
        for start in range(0, len(rows), int(V184_CONFIG["batch_size"])):
            stop = min(start + int(V184_CONFIG["batch_size"]), len(rows))
            role_rows = rows[start:stop]
            rss = rss_all[role_rows].to(device)
            distribution = v184_distribution(model, rss, final_temperature=scenario["temperature"], scale_multiplier=1.)
            q = distribution["calibrated_base_probability"]
            mask = distribution["candidate_mask"]
            locations, scales = distribution["component_locations"], distribution["component_scales"]
            assert_finite(q, "q_T")
            assert_finite(locations, "component locations")
            assert_finite(scales, "component scales")
            require(bool((scales > 0).all()), "non-positive component scale")
            candidate_snapshot = tensor_state_hash({k: distribution[k] for k in (
                "candidate_indices", "candidate_mask", "calibrated_base_probability", "component_locations", "component_scales")})
            with guard.radio_batch(len(role_rows)):
                raw = raw_correction_energies(model, rss, q, distribution)
                calls["raw_correction_energies"] += 1
            production = apply_correction(base_probability=q, candidate_mask=mask, raw=raw,
                                          reference=reference, epsilon_kl=EPSILON_KL)
            calls["apply_correction"] += 1
            source_snapshot = tensor_state_hash({**raw, "radio_energy": production["radio_energy"],
                                                "geometry_energy": production["geometry_energy"]})
            modes = decompose_correction(q, mask, production)
            # Labels enter only scoring/oracle diagnostics after all weights have been computed.
            target = torch.from_numpy(centered[role_rows].astype(np.float32)).to(device)
            shared_draws = {key: (value[start:stop].to(device) if isinstance(value, torch.Tensor) else value)
                            for key, value in randomness.items()}
            draw_hash = randomness_hash(shared_draws)
            row_distance = torch.linalg.vector_norm(locations - target[:, None, :], dim=2)
            base_expected_distance = (q * row_distance).sum(1)
            for mode in MODES:
                correction = modes[mode]
                p = correction["posterior"]
                require(randomness_hash(shared_draws) == draw_hash, "Student-t draws changed between correction modes")
                es, _, _ = continuous_energy_score_mc(p, locations, scales, target, shared_draws)
                calls["energy_score_calls"][mode] += 1
                require(randomness_hash(shared_draws) == draw_hash, "Energy Score mutated shared Student-t draws")
                collect(mode, "energy_score_m", es)
                collect(mode, "nll", _row_nll(p, locations, scales, target, model))
                entropy = _prediction_entropy(p)
                collect(mode, "entropy_nats", entropy)
                collect(mode, "effective_anchor_count", entropy.exp())
                collect(mode, "top1_probability", p.amax(1))
                for key in ("realized_kl", "epsilon_kl_effective", "rho", "radio_reliability", "geometry_reliability"):
                    collect(mode, key, correction[key])
                points = decode_points(p, locations, mask)
                for decoder in DECODERS:
                    collect(mode, f"error_{decoder}", torch.linalg.vector_norm(points[decoder] - target, dim=1))
                collect(mode, "gm_displacement_m", torch.linalg.vector_norm(points["geometric_median"] - points["posterior_mean"], dim=1))
                tails = tail_mass_diagnostics(p, locations, target_coords=target, mask=mask)
                for key, value in tails.items():
                    if key != "geometric_median":
                        collect(mode, key, value)
                if scenario["primary"] and mode in ("radio_only", "geometry_only"):
                    collect(mode, "oracle_gain_m", base_expected_distance - (p * row_distance).sum(1))
            if scenario["primary"]:
                distance_cpu = row_distance.cpu().double().numpy()
                q_cpu, mask_cpu = q.cpu().double().numpy(), mask.cpu().numpy()
                for source in ("radio", "geometry"):
                    oracle_batches[source].append((production[f"{source}_energy"].cpu().double().numpy(), distance_cpu, q_cpu, mask_cpu))
                agreement_batches.append((production["radio_energy"].cpu().double().numpy(),
                                          production["geometry_energy"].cpu().double().numpy(),
                                          locations.cpu().double().numpy(), mask_cpu,
                                          production["radio_reliability"].cpu().double().numpy(),
                                          production["geometry_reliability"].cpu().double().numpy()))
            require(candidate_snapshot == tensor_state_hash({k: distribution[k] for k in (
                "candidate_indices", "candidate_mask", "calibrated_base_probability", "component_locations", "component_scales")}),
                "candidate support/components/q_T mutated across modes")
            require(source_snapshot == tensor_state_hash({**raw, "radio_energy": production["radio_energy"],
                                                          "geometry_energy": production["geometry_energy"]}), "source energies mutated across modes")
            batch_provenance.append({"role_start": start, "role_stop": stop, "shared_draw_sha256": draw_hash,
                                     "candidate_and_component_sha256": candidate_snapshot,
                                     "shared_source_energy_sha256": source_snapshot,
                                     "raw_calls": 1, "production_apply_calls": 1, "ES_draw_reuse_count": 4})
            if len(batch_provenance) % 5 == 0 or stop == len(rows):
                print(f"PROGRESS {scenario['name']}: {stop}/{len(rows)} rows", flush=True)
    require(reference_before == tensor_state_hash(checkpoint["correction_reference"]), "checkpoint reference mutated")
    reference_after = tensor_state_hash({name: getattr(reference, name) for name in checkpoint["correction_reference"]})
    require(reference_after == reference_before, "device correction reference mutated")
    require(randomness_hash(randomness) == initial_randomness_hash, "scenario Student-t draws mutated")
    batches = len(batch_provenance)
    require(calls["raw_correction_energies"] == calls["apply_correction"] == batches, "raw/full production call counts not exactly once per batch")
    require(all(count == batches for count in calls["energy_score_calls"].values()), "missing shared-draw ES comparison")
    arrays = {mode: {key: torch.cat(values).numpy() for key, values in record.items()} for mode, record in collected.items()}
    report_modes, paired_correction = {}, {}
    for mode in MODES:
        values = arrays[mode]
        require(all(len(value) == len(rows) for value in values.values()), "collected metric row count mismatch")
        probabilistic = {}
        for metric in ("energy_score_m", "nll"):
            macro, se = mean_se(group_means(values[metric], groups))
            probabilistic[f"coordinate_macro_{metric}"] = macro
            probabilistic[f"coordinate_macro_{metric}_se"] = se
        for metric in ("entropy_nats", "effective_anchor_count", "top1_probability", "realized_kl",
                       "epsilon_kl_effective", "rho", "radio_reliability", "geometry_reliability"):
            probabilistic[f"mean_{metric}"] = float(values[metric].mean())
        decoders = {decoder: decoder_metrics(values[f"error_{decoder}"], groups) for decoder in DECODERS}
        paired_decoders = {}
        for decoder in DECODERS[1:]:
            paired = paired_statistics(values["error_posterior_mean"], values[f"error_{decoder}"], groups, positive_is_better=True)
            paired["improvement_m"] = paired["mean_delta"]
            paired["relative_coordinate_macro_improvement"] = (
                decoders["posterior_mean"]["coordinate_macro_mae_m"] - decoders[decoder]["coordinate_macro_mae_m"]
            ) / decoders["posterior_mean"]["coordinate_macro_mae_m"]
            paired_decoders[decoder] = paired
        tail = {key: float(values[key].mean()) for key in values if key.startswith(("truth_far_mass_", "mode_far_mass_"))}
        tail.update(mean_gm_displacement_from_mean_m=float(values["gm_displacement_m"].mean()),
                    p90_gm_displacement_from_mean_m=float(np.quantile(values["gm_displacement_m"], .9)),
                    mean_gm_spatial_leverage_m=float(values["gm_spatial_leverage_m"].mean()),
                    pearson_truth_far_mass_20m_posterior_mean_error=pearson(values["truth_far_mass_20m"], values["error_posterior_mean"]),
                    pearson_mode_far_mass_20m_mean_gm_displacement=pearson(values["mode_far_mass_20m"], values["gm_displacement_m"]))
        report_modes[mode] = {"probabilistic_metrics": probabilistic, "decoders": decoders,
                              "paired_decoder_effects": paired_decoders, "tail_diagnostics": tail,
                              "decoder_ranking": sorted(DECODERS, key=lambda d: (decoders[d]["coordinate_macro_mae_m"], DECODERS.index(d)))}
        if mode != "none":
            paired_correction[mode] = {metric: paired_statistics(values["error_posterior_mean" if metric == "posterior_mean_mae_m" else metric],
                                                                arrays["none"]["error_posterior_mean" if metric == "posterior_mean_mae_m" else metric], groups)
                                       for metric in DELTA_METRICS}
    alignments, reliability_reports = {}, {}
    agreement = None
    if scenario["primary"]:
        for source in ("radio", "geometry"):
            # Concatenate first: role medians/quantiles must not average batch summaries.
            joined = [concat_support_arrays([batch[i] for batch in oracle_batches[source]]) for i in range(4)]
            alignments[source] = source_energy_alignment(*joined)
            source_values = arrays[f"{source}_only"]
            reliability_reports[source] = reliability_utility(source_values[f"{source}_reliability"], source_values["oracle_gain_m"])
        joined_agreement = [concat_support_arrays([batch[i] for batch in agreement_batches]) for i in range(6)]
        agreement = source_agreement(*joined_agreement)
    result = {"definition": {**scenario, "scale_multiplier": 1., "epsilon_kl": EPSILON_KL,
                              "lambda_spatial": .1, "selected_epoch": 16, "rows": len(rows), "coordinate_groups": len(set(groups)),
                              "row_index_sha256": registry["roles"][scenario["role"]]["row_index_sha256"],
                              "checkpoint_sha256": FORMAL_EXPECTED[scenario["checkpoint"]], **reference_validation},
              "modes": report_modes, "paired_correction_effects": paired_correction,
              "source_energy_oracle_alignment": alignments, "reliability_utility_alignment": reliability_reports,
              "radio_geometry_agreement": agreement, "model_integrity": guard.verification,
              "correction_reference_before_sha256": reference_before, "correction_reference_after_sha256": reference_after,
              "shared_ES_randomness": {"seed": SCORING_SEED, "draws": PREDICTIVE_DRAWS, "degrees_freedom": float(V184_CONFIG["student_df"]),
                                       "full_role_sha256": initial_randomness_hash, "generation_calls": 1,
                                       "unchanged": True, "batch_provenance": batch_provenance},
              "production_call_counts": calls, "runtime_seconds": time.perf_counter() - started}
    assert_finite(result)
    print(f"DONE {scenario['name']}: {result['runtime_seconds']:.1f} seconds; model state unchanged", flush=True)
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    def render(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(render(value) for value in row) + " |" for row in rows])


def scientific_markdown(result: Mapping) -> str:
    scenarios = result["scenarios"]
    sections = ["# V18.4 只读 correction × decoder 机制探针\n",
                "仅在已开放的 UJI trainingData 开发/校准角色作机制诊断；不构成盲测确认，也不改变方法冻结。",
                "已获人工授权：冻结 radio profiler 仅对两个临时 offset/gain 张量做每批 6 步 Adam。网络优化、训练和网络损失反向传播均被禁止。",
                "ES/NLL/MAE 的主报告为坐标组等权均值；其余后验、尾部及 oracle 汇总为行等权。paired correction Δ = correction − none（负值改善）；decoder improvement = mean − alternative（正值改善）。",
                "同场景共用 Student-t draws（seed 184412，64 对独立流抽样）。常量/单候选相关系数按预注册约定记为 0；Spearman 用平均秩。可靠性四分位按稳定排序等数切分，相同可靠性依角色行序分配。Top-8/argmin/MAP 并列依候选顺序；decoder 排名依 mean、MAP、Top-8、GM 顺序破并列。mean 并列最优计为 directly best。KL 的 float32 转换容差为既有门禁 1e-7。",
                "## 1. 场景\n"]
    sections.append(markdown_table(("Scenario", "Role", "Primary", "Rows", "Groups", "T", "Scale", "epsilon", "Reference"),
        [[name, r["definition"]["role"], r["definition"]["primary"], r["definition"]["rows"], r["definition"]["coordinate_groups"],
          r["definition"]["temperature"], 1., .025, r["definition"]["reference_role"]] for name, r in scenarios.items()]))
    sections.append("## 2. Correction 概率指标\n")
    sections.append(markdown_table(("Scenario", "Mode", "ES m", "NLL", "Entropy", "Eff anchors", "Top-1", "KL", "Eff epsilon", "rho", "r radio", "r geometry"),
        [[name, mode] + [r["modes"][mode]["probabilistic_metrics"][key] for key in (
            "coordinate_macro_energy_score_m", "coordinate_macro_nll", "mean_entropy_nats", "mean_effective_anchor_count", "mean_top1_probability",
            "mean_realized_kl", "mean_epsilon_kl_effective", "mean_rho", "mean_radio_reliability", "mean_geometry_reliability")]
         for name, r in scenarios.items() for mode in MODES]))
    sections.append("## 3. Decoder 坐标宏 MAE（m）\n")
    sections.append(markdown_table(("Scenario", "Mode", "Mean", "MAP", "Top-8", "GM", "Ranking"),
        [[name, mode] + [r["modes"][mode]["decoders"][d]["coordinate_macro_mae_m"] for d in DECODERS] +
         [" > ".join(r["modes"][mode]["decoder_ranking"])] for name, r in scenarios.items() for mode in MODES]))
    sections.append("## 4. Paired correction Δ 对 none\n")
    sections.append(markdown_table(("Scenario", "Mode", "Metric", "Mean delta", "SE", "95% lower", "95% upper", "Groups improved"),
        [[name, mode, metric, p["mean_delta"], p["paired_se"], *p["normal_95_ci"], p["fraction_groups_improved"]]
         for name, r in scenarios.items() for mode, metrics in r["paired_correction_effects"].items() for metric, p in metrics.items()]))
    sections.append("## 5. None 下 decoder 对 mean 的改善\n")
    sections.append(markdown_table(("Scenario", "Decoder", "Improvement m", "SE", "95% lower", "95% upper", "Groups improved", "Relative improvement"),
        [[name, d, p["improvement_m"], p["paired_se"], *p["normal_95_ci"], p["fraction_groups_improved"], p["relative_coordinate_macro_improvement"]]
         for name, r in scenarios.items() for d, p in r["modes"]["none"]["paired_decoder_effects"].items()]))
    sections.append("## 6. Source energy → truth\n")
    sections.append(markdown_table(("Primary scenario", "Source", "Mean rho", "Median rho", "P10", "P90", "rho > 0", "rho < 0", "Argmin error m", "q_T MAP error m", "Argmin beats MAP"),
        [[name, source] + [p[key] for key in ("mean_row_spearman", "median_row_spearman", "p10_row_spearman", "p90_row_spearman",
            "fraction_spearman_positive", "fraction_spearman_negative", "source_argmin_candidate_mean_error_m", "q_T_map_candidate_mean_error_m", "fraction_source_argmin_beats_q_T_map")]
         for name in PRIMARY for source, p in scenarios[name]["source_energy_oracle_alignment"].items()]))
    for number, source in ((7, "radio"), (8, "geometry")):
        sections.append(f"## {number}. {source} reliability 四分位\n")
        sections.append(markdown_table(("Primary scenario", "Quartile", "Rows", "Mean reliability", "Mean gain m", "Median gain m", "Gain > 0"),
            [[name, p["quartile"], p["row_count"], p["mean_reliability"], p["mean_oracle_gain_m"], p["median_oracle_gain_m"], p["fraction_oracle_gain_positive"]]
             for name in PRIMARY for p in scenarios[name]["reliability_utility_alignment"][source]["quartiles"]]))
        for name in PRIMARY:
            p = scenarios[name]["reliability_utility_alignment"][source]
            sections.append(f"{name}: oracle gain 均值 {p['mean_oracle_gain_m']:.6f} m；中位数 {p['median_oracle_gain_m']:.6f} m；"
                            f"正/负比例 {p['fraction_oracle_gain_positive']:.6f}/{p['fraction_oracle_gain_negative']:.6f}；"
                            f"reliability→gain Spearman {p['spearman_reliability_oracle_gain']:.6f}，Pearson {p['pearson_reliability_oracle_gain']:.6f}。")
    sections.append("## 9. Radio-vs-geometry agreement\n")
    sections.append(markdown_table(("Primary scenario", "Mean energy rho", "Median energy rho", "rho > 0", "Same argmin", "Argmin distance m", "r Pearson", "r Spearman"),
        [[name] + [scenarios[name]["radio_geometry_agreement"][key] for key in ("mean_row_energy_spearman", "median_row_energy_spearman",
            "fraction_row_energy_spearman_positive", "fraction_same_energy_argmin_candidate", "mean_argmin_spatial_distance_m", "pearson_source_reliabilities", "spearman_source_reliabilities")]
         for name in PRIMARY]))
    sections.append("## 10. Calibration T=1 vs T=1.5 符号\n")
    sections.append(markdown_table(("Mode", "Metric", "T1 delta", "T1.5 delta", "T1 sign", "T1.5 sign", "Mode conclusion"),
        [[mode, metric, p["delta_T1p0"], p["delta_T1p5"], p["sign_T1p0"], p["sign_T1p5"], report["classification"]]
         for mode, report in result["temperature_interaction"].items() for metric, p in report["metrics"].items()]))
    sections.append("## 预注册结论\n")
    correction, decoder = result["primary_correction_decision"], result["primary_decoder_decision"]
    for mode, classification in correction["classifications"].items():
        sections.append(f"- {mode}: **{classification}**")
    sections.append(f"\nOverall correction: **{correction['conclusion']}**。")
    sections.append(f"Decoder: **{decoder['conclusion']}**；supported alternatives={decoder['supported_alternatives']}；"
                    f"posterior_mean_directly_best_both_roles={decoder['posterior_mean_directly_best_both_roles']}。")
    sections.append("主判定仅使用 A+B；C 仅检查温度交互。decoder ≥10% / ≤0.25m 门槛和 correction 规则按指令执行，无事后调整。"
                    "全部 SE/CI、行分位数、四种 mode 的 paired decoder effects 及尾部诊断见 JSON。冻结 decoder 保持原状，下一步仅科学审阅。")
    sections.append(f"Result canonical SHA256: `{result['result_sha256']}`。训练、模型修改/梯度、官方验证数据、外部评价、V18.5 变更均为 false。")
    return "\n\n".join(sections) + "\n"


def environment(device: torch.device) -> dict:
    return {"python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(),
            "torch_version": torch.__version__, "numpy_version": np.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def gate_receipt() -> dict:
    raw = os.environ.get("V184_MECHANISM_GATE_RECEIPT")
    require(raw is not None, "successful compile/full-regression and human-exception receipt required before execution")
    receipt = json.loads(raw)
    require(receipt["compile"]["exit_code"] == 0 and receipt["pytest"]["exit_code"] == 0, "compile/regression gate failed")
    require(receipt["pytest"]["test_files"] == list(TEST_FILES), "full regression test list mismatch")
    require(receipt["pytest"]["passed"] == 69 + receipt["pytest"]["new_tests"] and receipt["pytest"]["skipped"] == 0
            and receipt["pytest"]["xfailed"] == 0 and receipt["pytest"]["failed"] == 0, "regression test count/status mismatch")
    require(receipt["human_authorization"]["radio_temporary_adam_allowed"] is True, "human radio exception missing")
    for relative, expected in receipt["tested_file_sha256"].items():
        require(sha256_file(REPO_ROOT / relative) == expected, f"source changed after regression gate: {relative}")
    require({"fhm_dapp_tim/fhm_dapp/v18_4_mechanism_probe.py", "tests/test_v18_4_mechanism_probe.py"} <= set(receipt["tested_file_sha256"]),
            "new module/test gate hashes missing")
    return receipt


def execution_markdown(record: Mapping) -> str:
    texts = ["# V18.4 Mechanism probe execution record\n", f"Status: **{record['status']}**。",
             f"Evidence HEAD: `{record.get('current_evidence_commit_sha')}`；frozen source: `{FROZEN_SOURCE_SHA}`。",
             "人工例外只允许生产 radio profiler 的两个临时张量 Adam；每批精确 6 步。禁止网络训练、网络优化和网络损失 backward。",
             "## 门禁及执行\n"]
    commands = []
    for key in ("compile", "pytest"):
        item = record.get("gate", {}).get(key, {})
        commands.append([key, item.get("exit_code"), item.get("runtime_seconds"), item.get("passed", ""), item.get("command", "")])
    commands.append(["probe", record.get("probe_exit_code"), record.get("probe_runtime_seconds"), "", record["probe_command"]])
    texts.append(markdown_table(("Step", "Exit", "Seconds", "Passed tests", "Command"), commands))
    texts.append("## 输出 SHA256\n")
    texts.append(markdown_table(("Path", "File-byte SHA256"), [[p, h] for p, h in record.get("output_sha256", {}).items()]))
    texts.append("## 问题与处理\n")
    for problem in record.get("problems_and_resolutions", []):
        texts.append(f"- {problem['problem']}\n  {problem['resolution']}")
    if record.get("failure"):
        texts.append("```text\n" + record["failure"] + "\n```")
    texts.append("## 完整性及边界\n")
    texts.append("```json\n" + json.dumps({"boundary_confirmations": record.get("boundary_confirmations"),
                                         "dataset_opens": record.get("dataset_opens"),
                                         "model_integrity": record.get("model_integrity"),
                                         "source_artifact_postcheck": record.get("postcheck")}, ensure_ascii=False, indent=2) + "\n```")
    texts.append(f"Execution canonical SHA256: `{record['execution_record_sha256']}`。硬停止；下一步为科学审阅。")
    return "\n\n".join(texts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    # This check precedes all dataset/model work. Existing outputs are never touched.
    assert_unused(OUTPUTS)
    started = time.perf_counter()
    device = torch.device(args.device)
    receipt = gate_receipt()
    _set_seed(MODEL_SEED)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "requested CUDA is unavailable")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    audit = FileOpenAudit()
    record = {"record_type": PROBE_VERSION, "status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
              "current_evidence_commit_sha": current_head, "frozen_source_commit_sha": FROZEN_SOURCE_SHA,
              "environment": environment(device), "gate": receipt,
              "probe_command": f"{sys.executable} -m fhm_dapp_tim.fhm_dapp.v18_4_mechanism_probe --device {args.device}",
              "human_authorization": receipt["human_authorization"],
              "problems_and_resolutions": receipt.get("problems_and_resolutions", [])}
    scenario_results = {}
    try:
        with audit:
            integrity, registry = verify_integrity()
            record["integrity"] = integrity
            train_path = REPO_ROOT / "data/UJIndoorLoc/trainingData.csv"
            require(sha256_file(train_path) == registry["train_sha256"], "UJI training source hash mismatch")
            data, roles = _load_registered_data(registry)
            require(len(data.coords_m) == registry["training_rows"], "UJI total training rows mismatch")
            rss_all = torch.from_numpy(data.model_rss())
            for scenario in SCENARIOS:
                scenario_results[scenario["name"]] = run_scenario(scenario, registry, data, roles, rss_all, device,
                                                                 receipt["human_authorization"])
            require(len(scenario_results) == 3, "exactly three scenarios required")
            # Full second integrity pass also verifies all pre-existing evidence bytes.
            after, _ = verify_integrity()
            require(after == integrity, "source/formal artifact snapshot changed during probe")
            require(sha256_file(train_path) == registry["train_sha256"], "UJI training source changed during probe")
            require(not audit.denied and all(item["allowed"] for item in audit.opens), "dataset boundary violation recorded")
            result = {"probe_version": PROBE_VERSION,
                      "scientific_status": "READ_ONLY_MECHANISM_DIAGNOSTIC_ON_ALREADY_OPEN_UJI_DEVELOPMENT_CALIBRATION_ROLES",
                      "frozen_source_commit_sha": FROZEN_SOURCE_SHA, "current_evidence_commit_sha": current_head,
                      "verified_source_hashes": integrity["verified_source_file_sha256"],
                      "formal_file_byte_hashes": integrity["formal_file_byte_sha256"],
                      "canonical_self_hash_validation": integrity["canonical_self_hash_validation"],
                      "link_validation": integrity["link_validation"], "source_artifact_integrity": "PASS",
                      "dataset_opens": list(audit.opens), "train_sha256": registry["train_sha256"],
                      "human_authorization": receipt["human_authorization"], **BOUNDARIES,
                      "temporary_radio_optimizer_used_under_human_exception": True,
                      "conventions": {"correction_delta": "correction - none; negative improves",
                                      "decoder_improvement": "posterior mean error - candidate error; positive improves",
                                      "macro": "equal coordinate groups; sample group SE; normal 95% CI using 1.96",
                                      "other_metrics": "equal rows", "undefined_correlation": "constant or single-candidate vectors -> 0",
                                      "spearman_ties": "average ranks", "quartiles": "stable ascending reliability; balanced array_split; role row order breaks ties",
                                      "decoder_ranking_ties": list(DECODERS), "top8_ties": "stable candidate order; valid-only",
                                      "argmin_map_ties": "first valid candidate order", "KL_casting_tolerance": KL_TOLERANCE},
                      "scenarios": scenario_results, "primary_correction_decision": primary_correction_decision(scenario_results),
                      "primary_decoder_decision": primary_decoder_decision(scenario_results),
                      "temperature_interaction": temperature_interaction(scenario_results)}
            assert_finite(result)
            seal(result, "result_sha256")
            science_json = json_bytes(result)
            science_md = scientific_markdown(result).encode("utf-8")
            record.update(status="PASS", probe_exit_code=0, probe_runtime_seconds=time.perf_counter() - started,
                          dataset_opens=list(audit.opens), denied_file_opens=list(audit.denied),
                          boundary_confirmations=dict(BOUNDARIES), postcheck="PASS: all frozen sources and pre-existing formal/evidence/checkpoint bytes unchanged",
                          model_integrity={name: report["model_integrity"] for name, report in scenario_results.items()},
                          output_sha256={OUTPUT_JSON.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(science_json).hexdigest(),
                                         OUTPUT_MD.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(science_md).hexdigest()},
                          result_self_hash=result["result_sha256"],
                          finished_utc=datetime.now(timezone.utc).isoformat(), hard_stop_after_outputs=True)
            assert_finite(record)
            seal(record, "execution_record_sha256")
            execution_json, execution_md = json_bytes(record), execution_markdown(record).encode("utf-8")
            # Prepare all four payloads before the first write; each write is exclusive.
            assert_unused(OUTPUTS)
            for path, content in zip(OUTPUTS, (science_json, science_md, execution_json, execution_md)):
                write_once(path, content)
            for path in OUTPUTS:
                print(f"OUTPUT {path.relative_to(REPO_ROOT).as_posix()} SHA256 {sha256_file(path)}", flush=True)
            print("PASS; hard stop for scientific review", flush=True)
    except Exception:
        record.update(status="FAIL_CLOSED", probe_exit_code=1, probe_runtime_seconds=time.perf_counter() - started,
                      failure=traceback.format_exc(), dataset_opens=list(audit.opens), denied_file_opens=list(audit.denied),
                      completed_scenarios=list(scenario_results), finished_utc=datetime.now(timezone.utc).isoformat())
        seal(record, "execution_record_sha256")
        if not EXECUTION_JSON.exists() and not EXECUTION_MD.exists():
            write_once(EXECUTION_JSON, json_bytes(record))
            write_once(EXECUTION_MD, execution_markdown(record).encode("utf-8"))
        raise


if __name__ == "__main__":
    main()
