"""Locked V17 spatial-identifiability study; no TEST scoring or Git operations.

New opt-in subclassing keeps every historical source byte unchanged. Frozen
calibration uses canonical anchor IDs and cached sufficient quantities, never
CAL-eval labels. All writes are exclusive and failures retain their locks.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import msep_v17_spatial_operators_v1 as op
from . import msep_v17_two_stage_freeze_v1 as old
from .msep_single_study import _load, _prepare_fit, _seed, _site_weights, _json_safe
from .msep_single_posterior import (
    anchor_site_keys, split_physical_sites, energy_score_mc,
    HISTORICAL_NEURAL_SCALE, HISTORICAL_PROTOTYPE_WEIGHT,
)
from .msep_joint_science_v1 import (
    same_floor_neighbors, class_structure, student_t_logpdf, spatial_metrics,
)
from .model import AnchorStudentTModel
from .train_anchor_v2 import FROZEN_TRAINING

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "fhm_dapp_tim/evidence/v17_two_stage_freeze_v1"
DEFAULT_OUTPUT = REPO / "fhm_dapp_tim/evidence/v17_spatial_structure_v1"
DATASETS, SEEDS = old.DATASETS, old.SEEDS
ALPHAS = (0., .05, .1, .2, .35, .5, .65, .8, 1.)
BETAS = (0., .1, .2, .35, .5, .65, .8)
GAMMAS = (0., .005, .01, .025, .05, .1, .2)
COARSE_T = tuple(float(x) for x in np.geomspace(.25, 3., 25))
COARSE_THETA = tuple(float(x) for x in np.arange(0., 1.50001, .25))
ARMS = ("O", "A", "C", "D", "E")
VARIANTS = {"O": "raw_neighbors", "A": "same_floor_neighbors"}
TEST_COUNTS = {"internal_test_rows_scored": 0, "official_test_rows_scored": 0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def emit(stage, **values):
    print(json.dumps({"stage": stage, **_json_safe(values)}, allow_nan=False), flush=True)


def key_sha(keys):
    return hashlib.sha256(json.dumps(sorted(set(keys)), separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    files = list((REPO / "fhm_dapp_tim/fhm_dapp").glob("*.py"))
    files += list((REPO / "tests").glob("*.py"))
    files += [DEFAULT_OUTPUT / "DESIGN_V17_SPATIAL_STRUCTURE.md",
              REPO / "V17_SPATIAL_STRUCTURE_CODEX_RUN.txt"]
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in sorted(files)}


def verify_hashes(mapping, root=REPO):
    for name, expected in mapping.items():
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact missing: {path}")
        actual = sha(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {path}: {expected} != {actual}")


def freeze_guard(output):
    output = Path(output)
    manifest_path = output / "EXECUTION_MANIFEST.json"
    manifest = read_json(manifest_path)
    expected = dict(manifest["source_sha256"])
    revision_path = output / "ENGINEERING_REVISION.json"
    if revision_path.is_file():
        revision = read_json(revision_path)
        assert revision["original_manifest_sha256"] == sha(manifest_path)
        authorization_path = output / "USER_AUTHORIZATION_20261019.json"
        assert revision["authorization_sha256"] == sha(authorization_path)
        authorization = read_json(authorization_path)
        verify_hashes(authorization["preserved_frozen_artifact_sha256"], output)
        for name, change in revision["source_changes"].items():
            assert expected[name] == change["before_sha256"]
            expected[name] = change["after_sha256"]
        expected.update(revision["added_source_sha256"])
    verify_hashes(expected)
    verify_hashes(manifest["input_sha256"])
    verify_hashes(manifest["historical_sha256"], Path(manifest["input_root"]))
    return manifest


def reconstruct(dataset, seed, root, device):
    """Recreate FIT statistics/splits; never index TEST targets for model scoring."""
    run = Path(root) / f"{dataset}_seed{seed}"
    historical = read_json(run / "RESULT.json")
    _seed(seed)
    data = _load(dataset, None)
    keys = anchor_site_keys(data.coords_m, data.building, data.floor)
    roles = split_physical_sites(keys, seed=historical["split_seed"])
    cal_fit, cal_eval = old.split_cal(roles["cal"], keys, historical["split_seed"] + 99)
    prepared = _prepare_fit(data, roles, device)
    rss, xy, origin, memory, sample_anchor, obs, device_ids, model, ni, nw = prepared
    reference = torch.load(run / "REFERENCE_STATE.pt", map_location="cpu", weights_only=False)
    assert reference["ap_ids"] == list(data.ap_ids), "AP schema changed"
    assert np.array_equal(reference["reference_origin"], origin), "coordinate origin changed"
    for name, actual in (("fit_anchor_coords", memory["coords"]),
                         ("fit_anchor_building", memory["building"]),
                         ("fit_anchor_floor", memory["floor"])):
        assert torch.equal(reference[name], actual.cpu()), f"FIT dictionary changed: {name}"
    for name, value in reference["observation_memory"].items():
        assert torch.equal(value, obs[name].cpu()), f"FIT observation memory changed: {name}"
    for role, indices in roles.items():
        rkeys = [keys[int(i)] for i in indices]
        assert key_sha(rkeys) == historical["data_protocol"]["role_site_key_sha256"][role]
        if role != "test":
            assert rkeys == [tuple(k) for k in reference["role_site_keys"][role]]
    sets = [set(keys[int(i)] for i in rows) for rows in roles.values()]
    assert all(not (a & b) for a, b in itertools.combinations(sets, 2)), "site leakage"
    assert not set(keys[int(i)] for i in cal_fit) & set(keys[int(i)] for i in cal_eval)
    graph, edges = op.build_doubly_stochastic_floor_graph(
        memory["coords"], memory["building"], memory["floor"])
    class_pairs, classes = class_structure(memory["building"], memory["floor"])
    return dict(data=data, keys=keys, roles=roles, cal_fit=cal_fit, cal_eval=cal_eval,
                rss=rss, xy=xy, memory=memory, origin=origin, sample_anchor=sample_anchor,
                obs=obs, device_ids=device_ids, initial=model, old_neighbors=ni,
                old_weights=nw, graph=graph.to(device), edges=edges.to(device),
                classes=classes.to(device), class_pairs=class_pairs.to(device),
                historical=historical, historical_run=run)


def checkpoint(root, dataset, seed, variant):
    return Path(root) / f"{dataset}_seed{seed}" / f"{variant}_{old.PRIMARY_CHECKPOINT}.pt"


def preflight(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    before, review = read_json(root / "BEFORE_RUN_AUDIT.json"), read_json(root / "INTEGRITY_REVIEW.json")
    historical_sha = {**review["artifact_sha256"], **review["checkpoint_sha256"]}
    for name in ("BEFORE_RUN_AUDIT.json", "INTEGRITY_REVIEW.json", "RUN_REPORT_V17_TWO_STAGE.md"):
        historical_sha[name] = sha(root / name)
    verify_hashes(historical_sha, root)
    verify_hashes(before["input_sha256"])
    records = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            ctx = reconstruct(dataset, seed, root, torch.device("cpu"))
            manifest = read_json(ctx["historical_run"] / "RUN_MANIFEST.json")
            for name, expected in manifest["source_sha256"].items():
                assert sha(Path(__file__).with_name(name)) == expected, f"historical source mismatch: {name}"
            assert manifest["result_sha256"] == sha(ctx["historical_run"] / "RESULT.json")
            assert manifest["reference_state_sha256"] == sha(ctx["historical_run"] / "REFERENCE_STATE.pt")
            assert ctx["historical"]["config"]["batch"] == 64
            assert ctx["historical"]["config"]["es_draws"] == 32
            selected = {}
            for variant in VARIANTS.values():
                path = checkpoint(root, dataset, seed, variant)
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                assert ckpt["frozen_training_config"] == FROZEN_TRAINING
                assert ckpt["seed"] == seed and ckpt["dataset"] == dataset
                assert ckpt["epochs_trained_in_this_run"] == 80
                history = ctx["historical"]["stage1"]["training"][variant]["all_epoch_history"]
                chosen = min((r for r in history if r["epoch"] >= 21), key=lambda r: r["dev_mae_m"])
                assert chosen["epoch"] == ckpt["selected_epoch"]
                ctx["initial"].load_state_dict(ckpt["state_dict"], strict=True)
                selected[variant] = {"epoch": ckpt["selected_epoch"], "sha256": sha(path)}
            records[f"{dataset}_seed{seed}"] = {
                "roles": ctx["historical"]["data_protocol"],
                "cal_fit_key_sha256": key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
                "cal_eval_key_sha256": key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
                "cal_fit_rows": len(ctx["cal_fit"]), "cal_eval_rows": len(ctx["cal_eval"]),
                "physical_anchors": len(ctx["memory"]["coords"]), "edges": len(ctx["edges"]),
                "checkpoints": selected, "O_A_reuse_exact_source_config_split_epoch_match": True,
            }
            if seed == SEEDS[0]:
                records[f"{dataset}_seed{seed}"]["FIT_only_smoke"] = fit_smoke(
                    dataset, seed, root, torch.device(args.device))
            emit("preflight_run", dataset=dataset, seed=seed, **records[f"{dataset}_seed{seed}"])
    write_json(output / "EXECUTION_MANIFEST.json", {
        "status": "PASS", "input_root": str(root), "source_sha256": source_hashes(),
        "input_sha256": before["input_sha256"], "historical_sha256": historical_sha,
        "runs": records, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "batch": 64, "es_draws_per_stream": 32, "epochs": 80,
        "calibration_grid": {"T_coarse": COARSE_T, "theta_coarse": COARSE_THETA,
                             "T_refinement_count": 21, "theta_refinement_step": .0625},
        "alphas": ALPHAS, "betas": BETAS, "gammas": GAMMAS, **TEST_COUNTS,
        "data_access_note": "Canonical training-container coordinates establish immutable role keys; TEST subsets are never packed, predicted or scored. TUJI only results_train is decoded.",
    })
    emit("preflight_complete", output=str(output))


def fit_smoke(dataset, seed, root, device):
    """Discarded FIT-only numerical/VRAM checks, with zero optimizer steps."""
    ctx = reconstruct(dataset, seed, root, device)
    rows = ctx["roles"]["fit"][:64]
    neighbor, weights = same_floor_neighbors(ctx["memory"]["coords"], ctx["memory"]["building"],
        ctx["memory"]["floor"], count=FROZEN_TRAINING["target_neighbors"],
        temperature_m=FROZEN_TRAINING["spatial_temperature_m"])
    gradient_checks = {}
    for arm in ("C", "D", "E"):
        model = enable_training_arm(ctx["initial"], arm, ctx["graph"])
        model.train()
        terms = model.training_loss(ctx["rss"][rows].to(device), ctx["xy"][rows].to(device),
            torch.from_numpy(ctx["sample_anchor"][rows]).long().to(device), neighbor.to(device), weights.to(device),
            **train_kwargs(ctx["device_ids"][rows].to(device)))
        assert all(torch.isfinite(v).all() for v in terms.values())
        terms["loss"].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        gradient_checks[arm] = {k: float(v.detach()) for k, v in terms.items()
                               if k in ("loss", "nll", "joint_bf_xy_nll", "transport_risk", "alpha")}
        del terms, model
    pack = make_pack(ctx, ctx["initial"], rows[:8], virtual_gamma=.2)
    cell = cell_for_pack(pack, ctx, {"temperature": 1., "theta": [.5, .5]}, .35)
    assert torch.isfinite(cell["q"]).all() and torch.isfinite(cell["xy"]).all()
    assert torch.all(cell["sigma"] > 0)
    assert cell["q"].shape == (8, len(ctx["memory"]["coords"]) + len(ctx["edges"]))
    assert torch.allclose(cell["q"].sum(1), torch.ones(8, device=device), atol=2e-6)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return {"role": "FIT", "training_rows": 64, "field_rows": 8,
            "optimizer_steps": 0, "finite_gradients": gradient_checks,
            "expanded_components": cell["q"].shape[1], "peak_cuda_allocated_bytes": peak}


def enable_training_arm(base, arm, graph=None):
    """Add an opt-in path without editing or monkeypatching historical modules."""
    model = copy.deepcopy(base)
    model.__class__ = SpatialTrainingModel
    model.spatial_arm = arm
    model._spatial_capture = False
    if arm == "E":
        model.register_buffer("spatial_graph", graph.clone())
        model.register_parameter("raw_alpha", torch.nn.Parameter(
            model.anchor_keys.new_tensor(math.log(.1 / .9))))
    return model


class SpatialTrainingModel(AnchorStudentTModel):
    def retrieval_logits(self, rss, ap_dropout=0., neural_scale=None, prototype_distance_weight=None):
        scores, embedding = super().retrieval_logits(
            rss, ap_dropout=ap_dropout, neural_scale=neural_scale,
            prototype_distance_weight=prototype_distance_weight)
        if self.spatial_arm == "E":
            scores = op.graph_diffuse_logits(scores, self.spatial_graph, self.raw_alpha.sigmoid())
        if self._spatial_capture:
            self._spatial_forward_count += 1
            self._spatial_scores, self._spatial_embedding = scores, embedding
        return scores, embedding

    def _candidate_components(self, embedding, candidate_indices):
        if not self._spatial_capture:
            return super()._candidate_components(embedding, candidate_indices)
        # One residual-head call for all components, then gather the original
        # restricted loss candidates. No second augmented/encoder/dropout pass.
        ids = torch.arange(self.n_anchors, device=embedding.device)[None].expand(len(embedding), -1)
        centers, scales = super()._components(embedding, ids)
        self._spatial_centers, self._spatial_scales = centers, scales
        self._spatial_candidate_ids = candidate_indices
        gather = candidate_indices[:, :, None].expand(-1, -1, 2)
        return torch.gather(centers, 1, gather), torch.gather(scales, 1, gather)

    def training_loss(self, rss, target_coords, target_anchor, neighbor_indices, neighbor_weights, **kwargs):
        if self.spatial_arm not in ("C", "D", "E"):
            return super().training_loss(rss, target_coords, target_anchor, neighbor_indices, neighbor_weights, **kwargs)
        assert self.residual_experts == 1 and not self.use_candidate_reranker
        assert kwargs.get("coordinate_weight", 0.) == .25
        self._spatial_capture, self._spatial_forward_count = True, 0
        try:
            terms = super().training_loss(rss, target_coords, target_anchor, neighbor_indices, neighbor_weights, **kwargs)
            assert self._spatial_forward_count == 1, "added loss must reuse the original encoder forward"
            scores, centers, scales = self._spatial_scores, self._spatial_centers, self._spatial_scales
            ids = self._spatial_candidate_ids
            restricted_q = torch.softmax(torch.gather(scores, 1, ids), dim=1)
            gather = ids[:, :, None].expand(-1, -1, 2)
            # Exact differentiable old NLL (the public historical metric is detached).
            old_nll = -torch.logsumexp(
                restricted_q.clamp(min=1e-12).log()
                + self.student_log_prob(target_coords, torch.gather(centers, 1, gather),
                                        torch.gather(scales, 1, gather)), dim=1).mean()
            assert torch.equal(old_nll.detach(), terms["nll"]), "old NLL reconstruction changed"
            q = torch.softmax(scores, dim=1)
            canonical = torch.arange(self.n_anchors, device=rss.device)[None].expand(len(rss), -1)
            joint = op.joint_bf_xy_nll(
                q, canonical, centers, scales, target_coords,
                self.anchor_buildings[target_anchor], self.anchor_floors[target_anchor],
                self.anchor_buildings, self.anchor_floors).mean()
            loss = op.replace_unmasked_nll_with_joint_bf_xy(terms["loss"], old_nll, joint)
            risk = op.component_center_transport_risk(q, centers, target_coords)
            if self.spatial_arm in ("D", "E"):
                loss = op.add_transport_risk_increment(loss, risk)
            terms.update(loss=loss, joint_bf_xy_nll=joint.detach(), transport_risk=risk.mean().detach())
            if self.spatial_arm == "E":
                terms["alpha"] = self.raw_alpha.sigmoid().detach()
            return terms
        finally:
            self._spatial_capture = False
            for name in ("_spatial_scores", "_spatial_embedding", "_spatial_centers", "_spatial_scales", "_spatial_candidate_ids"):
                if hasattr(self, name):
                    delattr(self, name)


def virtual_retrieval_scores(model, embedding, rss, edges):
    """Original neural+prototype law on interpolated stored keys/prototypes."""
    i, j = edges[:, 0], edges[:, 1]
    keys = F.normalize(.5 * (model.anchor_keys[i] + model.anchor_keys[j]), dim=-1)
    neural = model.logit_scale.exp().clamp(max=100.) * embedding @ keys.T
    seen = model.encoder.seen_ap_mask
    query = rss[:, seen].float()
    prototype = .5 * (model.prototype_rss[i][:, seen] + model.prototype_rss[j][:, seen])
    # -1 remains the missing sentinel; no zero-imputation or alternate mask.
    distance = (query.square().mean(1, keepdim=True) + prototype.square().mean(1)[None]
                - 2. * (query @ prototype.T) / max(1, int(seen.sum()))).clamp_min(0.).sqrt()
    return HISTORICAL_NEURAL_SCALE * neural - HISTORICAL_PROTOTYPE_WEIGHT * distance


@torch.no_grad()
def make_pack(ctx, model, rows, *, virtual_gamma=None, evidence=True):
    model.eval()
    device = next(model.parameters()).device
    n = model.n_anchors
    fields = {name: [] for name in ("scores", "xy", "sigma", "z")}
    if virtual_gamma is not None:
        fields.update({name: [] for name in ("virtual_scores", "virtual_z", "virtual_xy", "virtual_sigma", "geometric_xy")})
        obs = ctx["obs"]
        vm = op.interpolate_g13_midpoint_memory(obs["detection_probability"], obs["conditional_mean"],
                                                obs["conditional_variance"], ctx["edges"], gamma_space=virtual_gamma)
        stats = {k: torch.cat((obs[k], vm[k]), dim=0) for k in vm}
    # Small inference blocks bound [B,N+E,AP] RSS evidence memory on 6GB GPUs.
    for start in range(0, len(rows), 8 if virtual_gamma is not None else 32):
        block = rows[start:start + (8 if virtual_gamma is not None else 32)]
        x = ctx["rss"][block].to(device)
        scores, embedding = model.retrieval_logits(x, neural_scale=HISTORICAL_NEURAL_SCALE,
                                                  prototype_distance_weight=HISTORICAL_PROTOTYPE_WEIGHT)
        ids = torch.arange(n, device=device)[None].expand(len(x), -1)
        centers, scales = model._components(embedding, ids)
        if evidence:
            obs = ctx["obs"]
            z = op.g13_continuous_field_features(
                x, model.encoder.seen_ap_mask, obs["detection_probability"],
                obs["conditional_mean"], obs["conditional_variance"], n_physical=n)
        else:
            z = scores.new_zeros((*scores.shape, 2))
        for name, value in (("scores", scores), ("xy", centers), ("sigma", scales), ("z", z)):
            fields[name].append(value)
        if virtual_gamma is not None:
            expanded_z = op.g13_continuous_field_features(
                x, model.encoder.seen_ap_mask, stats["detection_probability"],
                stats["conditional_mean"], stats["conditional_variance"], n_physical=n)
            assert torch.equal(expanded_z[:, :n], z), "virtual evidence altered parent features"
            middle = op.interpolate_midpoint_centers(model.anchor_coords, centers, ctx["edges"])
            for name, value in (("virtual_scores", virtual_retrieval_scores(model, embedding, x, ctx["edges"])),
                                ("virtual_z", expanded_z[:, n:]),
                                ("virtual_xy", middle["residual_interpolated_centers"]),
                                ("virtual_sigma", op.interpolate_midpoint_scales(scales, ctx["edges"])),
                                ("geometric_xy", middle["geometric_centers"])):
                fields[name].append(value)
    pack = {name: torch.cat(value) for name, value in fields.items()}
    pack.update(target=ctx["xy"][rows].to(device), keys=[ctx["keys"][int(i)] for i in rows], rows=np.asarray(rows))
    pack["weight"] = _site_weights(pack["keys"]).to(device)
    pack["building"] = torch.as_tensor(ctx["data"].building[rows], device=device).long()
    pack["floor"] = torch.as_tensor(ctx["data"].floor[rows], device=device).long()
    pack["ids"] = torch.arange(n, device=device)[None].expand(len(rows), -1)
    return pack


def objective_cache(pack, ctx, *, virtual=False):
    centers, scales = pack["xy"], pack["sigma"]
    ids = pack["ids"]
    if virtual:
        centers = torch.cat((centers, pack["virtual_xy"]), dim=1)
        scales = torch.cat((scales, pack["virtual_sigma"]), dim=1)
        ids = torch.cat((ids, ctx["edges"][:, 0][None].expand(len(ids), -1)), dim=1)
    valid = ((ctx["memory"]["building"].to(ids.device)[ids] == pack["building"][:, None])
             & (ctx["memory"]["floor"].to(ids.device)[ids] == pack["floor"][:, None]))
    lp = student_t_logpdf(centers, scales, pack["target"]).masked_fill(~valid, -torch.inf)
    distances = torch.linalg.vector_norm(centers - pack["target"][:, None], dim=-1)
    return lp, distances, pack["weight"]


def objective_from_q(q, cache):
    lp, distances, weights = cache
    valid = (q > 0) & torch.isfinite(lp)
    terms = (q.clamp_min(torch.finfo(q.dtype).tiny).log() + lp).masked_fill(~valid, -torch.inf)
    nll = -torch.logsumexp(terms, dim=-1)
    risk = (q * distances).sum(-1)
    return ((nll + .30 * risk) * weights).sum(-1)


@torch.no_grad()
def grid_calibrate(pack, ctx):
    cache = objective_cache(pack, ctx)
    table = []

    def evaluate_grid(candidates, stage):
        # Parallel parameter blocks, same deterministic grid and reduction order.
        for start in range(0, len(candidates), 8):
            values = candidates[start:start + 8]
            p = torch.tensor(values, dtype=pack["scores"].dtype, device=pack["scores"].device)
            logits = (pack["scores"][None] / p[:, 0, None, None]
                      + pack["z"][None, :, :, 0] * p[:, 1, None, None]
                      + pack["z"][None, :, :, 1] * p[:, 2, None, None])
            losses = objective_from_q(torch.softmax(logits, dim=-1), cache).cpu().tolist()
            table.extend(dict(stage=stage, temperature=t, theta=[v, s], objective=float(loss))
                         for (t, v, s), loss in zip(values, losses))
    evaluate_grid(list(itertools.product(COARSE_T, COARSE_THETA, COARSE_THETA)), "coarse")
    tie = lambda r: (r["objective"], r["temperature"], *r["theta"])
    coarse = min(table, key=tie)
    ti = min(range(len(COARSE_T)), key=lambda i: abs(COARSE_T[i] - coarse["temperature"]))
    ts = [float(x) for x in np.geomspace(COARSE_T[max(0, ti - 1)], COARSE_T[min(24, ti + 1)], 21)]
    tv = [float(x) for x in np.arange(max(0., coarse["theta"][0] - .25), min(1.5, coarse["theta"][0] + .25) + 1e-9, .0625)]
    ss = [float(x) for x in np.arange(max(0., coarse["theta"][1] - .25), min(1.5, coarse["theta"][1] + .25) + 1e-9, .0625)]
    evaluate_grid(list(itertools.product(ts, tv, ss)), "refine")
    # Refinement includes the coarse optimum (or lower objective grid entries).
    best = min(table, key=tie)
    return {"temperature": best["temperature"], "theta": best["theta"]}, table, best["objective"]


def complete_logits(pack, params):
    return op.combine_retrieval_and_g13_logits(
        pack["scores"], pack["z"], params["temperature"], torch.tensor(params["theta"], device=pack["scores"].device))


def fast_cell_mixture(parent_logits, virtual_logits, edges, beta, centers, scales, vcenters, vscales):
    """Vectorized exact Cell LSE; regression-tested against the public operator.

    Global unit cell normalization cancels internal cell normalizers. Each edge
    receives beta*(1/d_i+1/d_j)*exp(ell_e)/Z; isolated parents retain unit mass.
    """
    n, batch = parent_logits.shape[1], len(parent_logits)
    i, j = edges[:, 0], edges[:, 1]
    degree = torch.bincount(edges.flatten(), minlength=n).to(parent_logits.dtype)
    a = torch.where(degree > 0, parent_logits.new_full((n,), 1. - beta), torch.ones_like(degree))
    if beta == 0.:
        weights = torch.cat((torch.softmax(parent_logits, dim=1), parent_logits.new_zeros((batch, len(edges)))), dim=1)
        cell_logits = parent_logits
    else:
        parent_terms = parent_logits + a.log()[None]
        edge_terms = virtual_logits + (beta * (degree[i].reciprocal() + degree[j].reciprocal())).log()[None]
        weights = torch.softmax(torch.cat((parent_terms, edge_terms), dim=1), dim=1)
        # Stable unit-measure cell LSE, with no graph-dependent component padding.
        shift = torch.maximum(parent_logits.max(1).values, virtual_logits.max(1).values)[:, None]
        cell_mass = a[None] * (parent_logits - shift).exp()
        ve = beta * (virtual_logits - shift).exp()
        cell_mass.scatter_add_(1, i[None].expand(batch, -1), ve / degree[i][None])
        cell_mass.scatter_add_(1, j[None].expand(batch, -1), ve / degree[j][None])
        cell_logits = cell_mass.log() + shift
    ids = torch.cat((torch.arange(n, device=edges.device), i))[None].expand(batch, -1)
    return dict(q=weights, xy=torch.cat((centers, vcenters), dim=1),
                sigma=torch.cat((scales, vscales), dim=1), ids=ids, cell_logits=cell_logits)


def zero_mass_safe_energy_score(weights, centers, scales, target, *, draws=32, seed=19602):
    """Exclude globally zero-mass columns inside sampling; retain the full law.

    Appended zero columns can change CUDA float32 cumsum rounding and sampled
    component IDs. Positive weights, centers/scales and RNG streams are unchanged.
    """
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    active = (weights > 0).any(dim=0)
    assert active.any()
    if not bool(active.all()):
        weights, centers, scales = weights[:, active], centers[:, active], scales[:, active]
    return energy_score_mc(weights, centers, scales, target, draws=draws, seed=seed)


@torch.no_grad()
def score_mixture(pack, ctx, q, *, centers=None, scales=None, ids=None, seed):
    centers = pack["xy"] if centers is None else centers
    scales = pack["sigma"] if scales is None else scales
    ids = pack["ids"] if ids is None else ids
    assert torch.isfinite(q).all() and (q >= 0).all()
    assert torch.allclose(q.sum(1), torch.ones(len(q), device=q.device), atol=2e-6, rtol=2e-6)
    metrics, row = spatial_metrics(q, ids, centers, pack["target"], pack["building"], pack["floor"],
                                   ctx["classes"], ctx["class_pairs"], keys=pack["keys"])
    nll = op.joint_bf_xy_nll(q, ids, centers, scales, pack["target"], pack["building"], pack["floor"],
                            ctx["memory"]["building"], ctx["memory"]["floor"])
    es = zero_mass_safe_energy_score(q, centers, scales, pack["target"], draws=32, seed=seed)
    risk = op.component_center_transport_risk(q, centers, pack["target"])
    rows = {"mae_full_m": np.asarray(row["err_full"]),
            "mae_predicted_bf_m": np.asarray(row["err_predicted_bf"]),
            "bf_accuracy": np.asarray(row["bf_correct"]),
            "energy_score_m": es.cpu().numpy(), "joint_nll": nll.cpu().numpy(),
            "transport_risk_m": risk.cpu().numpy()}
    assert all(np.isfinite(x).all() for x in rows.values()), "nonfinite scoring result"
    grouped = {}
    for index, key in enumerate(pack["keys"]):
        grouped.setdefault(key, []).append(index)
    sites = {json.dumps(list(k), separators=(",", ":")): {
        "site": list(k), "n_scans": len(ix), **{name: float(v[ix].mean()) for name, v in rows.items()}
    } for k, ix in grouped.items()}
    for metric, field in (("macro_es_m", "energy_score_m"), ("macro_joint_nll", "joint_nll"),
                           ("macro_transport_risk_m", "transport_risk_m")):
        metrics[metric] = float(np.mean([s[field] for s in sites.values()]))
    values = np.array([s["mae_full_m"] for s in sites.values()])
    metrics.update(site_p95_full_m=float(np.quantile(values, .95)), site_p99_full_m=float(np.quantile(values, .99)),
                   worst_site_mae_m=float(values.max()), n_sites=len(sites), n_scans=len(q))
    return metrics, sites


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def s1_probability(pack, ctx, params, ap, ao, normalization):
    before, logits = op.full_logit_graph_diffusion(
        pack["scores"], pack["z"], torch.tensor(params["theta"], device=pack["scores"].device),
        params["temperature"], ctx["graph"], ap, ao)
    if normalization == "bf_mass":
        q = op.layer_mass_preserving_softmax(pack["scores"], logits, ctx["classes"])
    else:
        q = torch.softmax(logits, dim=1)
    return before, logits, q


@torch.no_grad()
def select_s1(pack, ctx, params):
    cache = objective_cache(pack, ctx)
    selected, tables = {}, {}
    for normalization in ("global", "bf_mass"):
        for arm in ("1A", "1B", "1C", "1D"):
            candidates = list(itertools.product(ALPHAS, ALPHAS)) if arm == "1D" else [
                ((a, 0.) if arm == "1A" else (0., a) if arm == "1B" else (a, a)) for a in ALPHAS]
            table = []
            for ap, ao in candidates:
                before, logits, q = s1_probability(pack, ctx, params, ap, ao, normalization)
                value = float(objective_from_q(q, cache))
                assert math.isfinite(value), f"nonfinite CAL-fit S1 objective: {arm}"
                record = {"alpha_prior": ap, "alpha_obs": ao, "objective": value}
                if arm == "1C":
                    gap = op.log_partition_gap(before, logits)
                    assert float(gap.max()) <= 2e-6, "Jensen partition invariant failed"
                    record.update(log_partition_gap_max=float(gap.max()), log_partition_gap_mean=float(gap.mean()))
                table.append(record)
            name = f"{arm}_{normalization}"
            best = min(table, key=lambda r: (r["objective"], r["alpha_prior"], r["alpha_obs"]))
            selected[name] = {**params, **best, "normalization": normalization, "structure": arm}
            tables[name] = table
    return selected, tables


def cell_for_pack(pack, ctx, params, beta, *, geometric=False):
    parent = complete_logits(pack, params)
    virtual = op.combine_retrieval_and_g13_logits(
        pack["virtual_scores"], pack["virtual_z"], params["temperature"],
        torch.tensor(params["theta"], device=parent.device))
    return fast_cell_mixture(parent, virtual, ctx["edges"], beta, pack["xy"], pack["sigma"],
                             pack["geometric_xy"] if geometric else pack["virtual_xy"], pack["virtual_sigma"])


@torch.no_grad()
def frozen_one(args, dataset, seed, variant, *, resume=False):
    output = Path(args.output).resolve() / "frozen" / f"{dataset}_seed{seed}" / variant
    started = time.perf_counter()
    if resume:
        assert (output / "FROZEN_RUN.lock").is_file() and (output / "CAL_EVAL.lock").is_file()
        assert not (output / "RESULT.json").exists() and not (output / "SITE_DIAGNOSTICS.json").exists()
        write_json(output / "AUTHORIZED_RESUME.lock", {
            "authorization_sha256": sha(Path(args.output) / "USER_AUTHORIZATION_20261019.json"),
            "selection_sha256": sha(output / "CAL_FIT_SELECTION.json"),
            "original_locks_preserved": True, "refit_performed": False, **TEST_COUNTS})
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "FROZEN_RUN.lock", {"dataset": dataset, "seed": seed, "variant": variant,
                   "checkpoint_sha256": sha(checkpoint(args.root, dataset, seed, variant)), **TEST_COUNTS})
    ctx = reconstruct(dataset, seed, args.root, torch.device(args.device))
    ckpt = torch.load(checkpoint(args.root, dataset, seed, variant), map_location=args.device, weights_only=False)
    model = ctx["initial"]
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    if resume:
        selection = read_json(output / "CAL_FIT_SELECTION.json")
        clean, s1, best2, best0 = (selection[k] for k in
                                 ("joint_calibrated_base", "structure1", "2B", "2A_beta_only_control"))
        lock = read_json(output / "CAL_EVAL.lock")
        assert lock["selection_sha256"] == sha(output / "CAL_FIT_SELECTION.json")
        assert lock["cal_eval_key_sha256"] == key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"])
        emit("authorized_frozen_resume", dataset=dataset, seed=seed, variant=variant,
             selection_sha256=lock["selection_sha256"], refit_performed=False)
    else:
        fit = make_pack(ctx, model, ctx["cal_fit"], virtual_gamma=0.)
        clean, grid, cal_objective = grid_calibrate(fit, ctx)
        emit("clean_calibration", dataset=dataset, seed=seed, variant=variant,
             params=clean, objective=cal_objective, grid_count=len(grid))
        s1, s1tables = select_s1(fit, ctx, clean)
        table2 = []
        for gamma in GAMMAS:
            pack = fit if gamma == 0. else make_pack(ctx, model, ctx["cal_fit"], virtual_gamma=gamma)
            cache = objective_cache(pack, ctx, virtual=True)
            for beta in BETAS:
                cell = cell_for_pack(pack, ctx, clean, beta)
                value = float(objective_from_q(cell["q"], cache))
                assert math.isfinite(value), "nonfinite CAL-fit Cell LSE objective"
                table2.append({"beta": beta, "gamma_space": gamma, "objective": value})
            if pack is not fit:
                del pack
        best2 = min(table2, key=lambda r: (r["objective"], r["beta"], r["gamma_space"]))
        best0 = min((r for r in table2 if r["gamma_space"] == 0.), key=lambda r: (r["objective"], r["beta"]))
        selection = {"joint_calibrated_base": clean, "calibration_objective": cal_objective,
                     "calibration_grid": grid, "structure1": s1, "structure1_grids": s1tables,
                     "structure2_grid": table2, "2B": best2,
                     "2A": {"beta": best2["beta"], "gamma_space": 0.},
                     "2A_beta_only_control": best0, "selection_role": "CAL-fit",
                     "tie_break": "objective then lower T/theta; S1 lower alpha_prior/obs; S2 lower beta/gamma"}
        write_json(output / "CAL_FIT_SELECTION.json", selection)
        write_json(output / "CAL_EVAL.lock", {"selection_sha256": sha(output / "CAL_FIT_SELECTION.json"),
                   "cal_eval_key_sha256": key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]), **TEST_COUNTS})
    # All choices are saved before building the reporting-only evaluation pack.
    evaluation = make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=0.)
    results, diagnostics = {}, {}

    def record(name, q, params, centers=None, scales=None, ids=None, extra=None):
        metrics, sites = score_mixture(evaluation, ctx, q, centers=centers, scales=scales, ids=ids, seed=seed + 900)
        results[name] = {"params": params, "metrics": metrics, **(extra or {})}
        diagnostics[name] = sites

    for policy in old.CALIBRATION_POLICIES:
        historical = ctx["historical"]["stage2"]["variants"][variant]["calibration"][policy]["params"]
        q = torch.softmax(complete_logits(evaluation, historical), dim=1)
        record("historical_" + policy, q, historical, extra={"historical_control_only": True})
    logits = complete_logits(evaluation, clean)
    record("joint_calibrated_base", torch.softmax(logits, dim=1), clean)
    record("joint_calibrated_base_bf_mass", op.layer_mass_preserving_softmax(
        evaluation["scores"], logits, ctx["classes"]), {**clean, "normalization": "bf_mass"})
    for name, params in s1.items():
        before, after, q = s1_probability(evaluation, ctx, clean, params["alpha_prior"], params["alpha_obs"], params["normalization"])
        extra = {}
        if params["structure"] == "1C":
            gap = op.log_partition_gap(before, after)
            assert float(gap.max()) <= 2e-6
            extra["log_partition_gap_by_scan"] = gap.cpu().tolist()
        if params["normalization"] == "bf_mass":
            raw_q = torch.softmax(evaluation["scores"], dim=1)
            max_difference = max(float((q[:, ctx["classes"] == c].sum(1) - raw_q[:, ctx["classes"] == c].sum(1)).abs().max())
                                 for c in torch.unique(ctx["classes"]))
            assert max_difference < 2e-6
            extra["max_raw_bf_mass_error"] = max_difference
        record(name, q, params, extra=extra)
    beta0 = cell_for_pack(evaluation, ctx, clean, 0.)
    parent_q = torch.softmax(logits, dim=1)
    n = model.n_anchors
    assert torch.equal(beta0["q"][:, :n], parent_q) and torch.count_nonzero(beta0["q"][:, n:]) == 0
    assert torch.equal(beta0["xy"][:, :n], evaluation["xy"])
    parent_nll = op.joint_bf_xy_nll(parent_q, evaluation["ids"], evaluation["xy"], evaluation["sigma"],
        evaluation["target"], evaluation["building"], evaluation["floor"], model.anchor_buildings, model.anchor_floors)
    expanded_nll = op.joint_bf_xy_nll(beta0["q"], beta0["ids"], beta0["xy"], beta0["sigma"],
        evaluation["target"], evaluation["building"], evaluation["floor"], model.anchor_buildings, model.anchor_floors)
    parent_es = zero_mass_safe_energy_score(parent_q, evaluation["xy"], evaluation["sigma"], evaluation["target"], draws=32, seed=seed + 900)
    expanded_es = zero_mass_safe_energy_score(beta0["q"], beta0["xy"], beta0["sigma"], evaluation["target"], draws=32, seed=seed + 900)
    assert torch.allclose(parent_nll, expanded_nll) and torch.allclose(parent_es, expanded_es), "beta=0 metric identity failed"
    for name, chosen in (("2A", selection["2A"]), ("2B", best2), ("2A_beta_only_control", best0)):
        p = evaluation if chosen["gamma_space"] == 0. else make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=chosen["gamma_space"])
        cell = cell_for_pack(p, ctx, clean, chosen["beta"])
        params = {**clean, **chosen, "normalization": "global", "center_mode": "residual_interpolated"}
        record(name, cell["q"], params, cell["xy"], cell["sigma"], cell["ids"])
        geometric_centers = torch.cat((p["xy"], p["geometric_xy"]), dim=1)
        record(name + "_geometric_diagnostic", cell["q"], {**params, "center_mode": "geometric_diagnostic_only"},
               geometric_centers, cell["sigma"], cell["ids"], {"diagnostic_only": True})
        if p is not evaluation:
            del p
    # Frozen raw point predictions reproduce the original checkpoint exactly up
    # to float32 summation order. ES streams now follow canonical ID order.
    old_raw = ctx["historical"]["stage2"]["variants"][variant]["calibration"]["raw"]["metrics"]
    reproduction = abs(results["historical_raw"]["metrics"]["macro_mae_full_m"] - old_raw["macro_mae_full_m"])
    assert reproduction < 2e-5, f"frozen original raw prediction changed: {reproduction}"
    result = {"phase": "F", "dataset": dataset, "seed": seed, "backbone": variant,
              "selected_epoch": ckpt["selected_epoch"], "checkpoint_sha256": sha(checkpoint(args.root, dataset, seed, variant)),
              "selection_sha256": sha(output / "CAL_FIT_SELECTION.json"), "methods": results,
              "physical_anchors": n, "unique_virtual_edges": len(ctx["edges"]),
              "beta_zero_parent_evidence_exact": True, "beta_zero_joint_nll_ES_allclose": True,
              "raw_MAE_reproduction_abs_m": reproduction, "evaluation_role": "CAL-eval",
              "elapsed_seconds": time.perf_counter() - started, **TEST_COUNTS}
    write_json(output / "SITE_DIAGNOSTICS.json", diagnostics)
    write_json(output / "RESULT.json", result)
    emit("frozen_complete", dataset=dataset, seed=seed, variant=variant,
         elapsed_seconds=result["elapsed_seconds"], methods=len(results), selected_2B=best2)


def frozen(args):
    freeze_guard(args.output)
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS.values():
                folder = Path(args.output) / "frozen" / f"{dataset}_seed{seed}" / variant
                if getattr(args, "resume_authorized", False) and folder.exists():
                    if (folder / "RESULT.json").is_file():
                        result = read_json(folder / "RESULT.json")
                        assert (result["dataset"], result["seed"], result["backbone"]) == (dataset, seed, variant)
                        assert result["checkpoint_sha256"] == sha(checkpoint(args.root, dataset, seed, variant))
                        assert result["selection_sha256"] == sha(folder / "CAL_FIT_SELECTION.json")
                        emit("preserved_completed_frozen_run", dataset=dataset, seed=seed, variant=variant)
                        continue
                    frozen_one(args, dataset, seed, variant, resume=True)
                else:
                    frozen_one(args, dataset, seed, variant)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    # Read-only numerical audit is mandatory before Phase T.
    audit_results(Path(args.output), phase="F", require_complete=True)


def train_kwargs(domain):
    f = FROZEN_TRAINING
    return dict(retrieval_weight=f["retrieval_weight"], hierarchy_weight=f["hierarchy_weight"],
                self_target_weight=f["self_target_weight"], coordinate_weight=f["coordinate_weight"],
                device_ids=domain, domain_adversarial_weight=0., rss_offset_aug_db=f["rss_offset_aug_db"],
                rss_scale_aug=f["rss_scale_aug"], train_candidates=f["train_candidates"], ap_dropout=f["ap_dropout"])


@torch.no_grad()
def dev_metrics(ctx, model):
    pack = make_pack(ctx, model, ctx["roles"]["dev"], evidence=False)
    q = torch.softmax(pack["scores"], dim=1)
    metrics, _ = spatial_metrics(q, pack["ids"], pack["xy"], pack["target"], pack["building"], pack["floor"],
                                  ctx["classes"], ctx["class_pairs"], keys=pack["keys"])
    return {"dev_mae_m": metrics["macro_mae_full_m"], "dev_bf_accuracy": metrics["macro_bf_accuracy"]}


def train_one(args, dataset, seed, arm):
    output = Path(args.output).resolve() / "trained" / f"{dataset}_seed{seed}" / arm
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ctx = reconstruct(dataset, seed, args.root, torch.device(args.device))
    write_json(output / "TRAIN.lock", {"dataset": dataset, "seed": seed, "arm": arm,
               "source_sha256": source_hashes(), "epochs": 80, **TEST_COUNTS})
    model = enable_training_arm(ctx["initial"], arm, ctx["graph"])
    history = []
    if arm in ("O", "A"):
        variant = VARIANTS[arm]
        source = checkpoint(args.root, dataset, seed, variant)
        ckpt = torch.load(source, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        history = ctx["historical"]["stage1"]["training"][variant]["all_epoch_history"]
        epoch = ckpt["selected_epoch"]
        reused = {"reused": True, "path": str(source), "sha256": sha(source),
                  "source_config_split_epoch_verified": True, "new_optimizer_updates": 0}
    else:
        _seed(seed)
        neighbor, weights = same_floor_neighbors(ctx["memory"]["coords"], ctx["memory"]["building"],
            ctx["memory"]["floor"], count=FROZEN_TRAINING["target_neighbors"],
            temperature_m=FROZEN_TRAINING["spatial_temperature_m"])
        neighbor, weights = neighbor.to(args.device), weights.to(args.device)
        fit_rows = ctx["roles"]["fit"]
        ds = TensorDataset(ctx["rss"][fit_rows], ctx["xy"][fit_rows],
            torch.from_numpy(ctx["sample_anchor"][fit_rows]).long(), ctx["device_ids"][fit_rows])
        loader = DataLoader(ds, batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=FROZEN_TRAINING["learning_rate"],
                                      weight_decay=FROZEN_TRAINING["weight_decay"])
        best_score, best_epoch, best_state = math.inf, None, None
        epoch_log = output / "EPOCH_HISTORY.jsonl"
        with epoch_log.open("x", encoding="utf-8") as log:
            for current in range(1, 81):
                model.train()
                sums, count = {}, 0
                for x, y, anchor, domain in loader:
                    x, y, anchor, domain = [v.to(args.device) for v in (x, y, anchor, domain)]
                    optimizer.zero_grad(set_to_none=True)
                    terms = model.training_loss(x, y, anchor, neighbor, weights, **train_kwargs(domain))
                    assert torch.isfinite(torch.stack(list(terms.values()))).all(), "nonfinite training term"
                    terms["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    count += len(x)
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.) + float(value.detach()) * len(x)
                assert count == len(fit_rows)
                model.eval()
                row = {"dataset": dataset, "seed": seed, "arm": arm, "epoch": current,
                       **{f"fit_{k}": v / count for k, v in sums.items()}, **dev_metrics(ctx, model)}
                history.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if current >= 21 and row["dev_mae_m"] < best_score:
                    best_score, best_epoch = row["dev_mae_m"], current
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                emit("epoch_end", **row)
        assert len(history) == 80 and best_state is not None
        epoch = best_epoch
        model.load_state_dict(best_state, strict=True)
        source = output / "best_dev_epoch_21_80.pt"
        torch.save({"state_dict": best_state, "arm": arm, "dataset": dataset, "seed": seed,
                    "selected_epoch": epoch, "dev_mae_m": best_score,
                    "source_sha256": source_hashes(), "frozen_training_config": FROZEN_TRAINING,
                    "alpha": float(model.raw_alpha.sigmoid()) if arm == "E" else None}, source)
        reused = {"reused": False, "path": str(source), "sha256": sha(source),
                  "new_optimizer_updates": 80 * len(loader)}
    write_csv(output / "EPOCH_HISTORY.csv", history)
    write_csv(output / "CHECKPOINT_SELECTION.csv", [{"dataset": dataset, "seed": seed, "arm": arm,
              "policy": old.PRIMARY_CHECKPOINT, "epoch": epoch, "checkpoint_sha256": sha(source)}])
    write_json(output / "CAL_EVAL.lock", {"checkpoint_sha256": sha(source), "epoch": epoch, **TEST_COUNTS})
    pack = make_pack(ctx, model, ctx["cal_eval"], evidence=False)
    q = torch.softmax(pack["scores"], dim=1)
    metrics, sites = score_mixture(pack, ctx, q, seed=seed + 900)
    result = {"phase": "T", "dataset": dataset, "seed": seed, "backbone": arm,
              "selected_epoch": epoch, "checkpoint": reused, "epochs": 80, "batch": 64,
              "methods": {arm: {"params": {"temperature": 1., "theta": [0., 0.],
                      "alpha": float(model.raw_alpha.sigmoid()) if arm == "E" else 0.}, "metrics": metrics}},
              "calibration_performed": False, "evaluation_role": "CAL-eval", **TEST_COUNTS,
              "elapsed_seconds": time.perf_counter() - started}
    write_json(output / "SITE_DIAGNOSTICS.json", {arm: sites})
    write_json(output / "RESULT.json", result)
    emit("trained_arm_complete", dataset=dataset, seed=seed, arm=arm, selected_epoch=epoch,
         reused=reused["reused"], metrics=metrics, elapsed_seconds=result["elapsed_seconds"])


def train(args):
    freeze_guard(args.output)
    audit_results(Path(args.output), phase="F", require_complete=True, write=False)
    for dataset in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                train_one(args, dataset, seed, arm)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    audit_results(Path(args.output), phase="T", require_complete=True)


def audit_results(root, *, phase, require_complete, write=True):
    paths = sorted((Path(root) / ("frozen" if phase == "F" else "trained")).glob("*/*/RESULT.json"))
    expected = 18 if phase == "F" else 45
    if require_complete:
        assert len(paths) == expected, f"Phase {phase} incomplete: {len(paths)}/{expected}"
    max_error, hashes = 0., {}
    for path in paths:
        result, sites = read_json(path), read_json(path.with_name("SITE_DIAGNOSTICS.json"))
        assert all(result[k] == 0 for k in TEST_COUNTS)
        assert result["evaluation_role"] == "CAL-eval"
        for name, entry in result["methods"].items():
            values = list(sites[name].values())
            assert values and sum(s["n_scans"] for s in values) == entry["metrics"]["n_scans"]
            for metric, field in (("macro_mae_full_m", "mae_full_m"), ("macro_mae_predicted_bf_m", "mae_predicted_bf_m"),
                                   ("macro_es_m", "energy_score_m"), ("macro_bf_accuracy", "bf_accuracy"),
                                   ("macro_joint_nll", "joint_nll")):
                error = abs(float(np.mean([s[field] for s in values])) - entry["metrics"][metric])
                assert error < 2e-5, f"site reduction failed: {path}/{name}/{metric}"
                max_error = max(max_error, error)
        if phase == "T":
            with path.with_name("EPOCH_HISTORY.csv").open(encoding="utf-8-sig", newline="") as handle:
                history = list(csv.DictReader(handle))
            assert [int(r["epoch"]) for r in history] == list(range(1, 81))
            best = min((r for r in history if int(r["epoch"]) >= 21), key=lambda r: float(r["dev_mae_m"]))
            assert int(best["epoch"]) == result["selected_epoch"]
        for artifact in path.parent.iterdir():
            if artifact.is_file():
                hashes[str(artifact.relative_to(root)).replace("\\", "/")] = sha(artifact)
    freeze_guard(root)
    record = {"status": "PASS", "phase": phase, "completed": len(paths), "expected": expected,
              "maximum_site_reduction_error": max_error, "artifact_sha256": hashes, **TEST_COUNTS}
    if write:
        write_json(Path(root) / f"PHASE_{phase}_INTEGRITY_REVIEW.json", record)
    emit("phase_audit_complete", phase=phase, completed=len(paths), max_site_reduction_error=max_error)
    return record


def aggregate(args):
    root = Path(args.root).resolve()
    f_audit = audit_results(root, phase="F", require_complete=True, write=False)
    t_audit = audit_results(root, phase="T", require_complete=True, write=False)
    run_rows, paired_rows, site_rows = [], [], []
    grouped = {}
    for path in sorted(root.glob("*/*/*/RESULT.json")):
        r, diagnostics = read_json(path), read_json(path.with_name("SITE_DIAGNOSTICS.json"))
        for method, entry in r["methods"].items():
            params = entry["params"]
            row = {"phase": r["phase"], "dataset": r["dataset"], "seed": r["seed"], "backbone": r["backbone"],
                   "method": method, "selected_epoch": r["selected_epoch"], "params_json": json.dumps(params, sort_keys=True), **entry["metrics"]}
            run_rows.append(row)
            group = (r["phase"], r["dataset"], r["backbone"], method)
            grouped.setdefault(group, []).append(row)
        if r["phase"] == "F":
            comparisons = []
            for method in r["methods"]:
                if method in ("historical_raw", "joint_calibrated_base"):
                    continue
                reference = "historical_raw" if method.startswith("historical_") else "joint_calibrated_base_bf_mass" if method.endswith("bf_mass") else "joint_calibrated_base"
                if method == "joint_calibrated_base_bf_mass":
                    reference = "joint_calibrated_base"
                if method == "2B":
                    comparisons.append(("2A", method))
                comparisons.append((reference, method))
        else:
            if r["backbone"] == "O":
                continue
            previous = {"A": "O", "C": "A", "D": "C", "E": "D"}[r["backbone"]]
            prior_path = path.parent.parent / previous / "RESULT.json"
            prior_sites = read_json(prior_path.with_name("SITE_DIAGNOSTICS.json"))[previous]
            diagnostics[previous] = prior_sites
            r["methods"][previous] = read_json(prior_path)["methods"][previous]
            comparisons = [(previous, r["backbone"])]
        for reference, candidate in comparisons:
            base, cand = r["methods"][reference]["metrics"], r["methods"][candidate]["metrics"]
            prefix = {"phase": r["phase"], "dataset": r["dataset"], "seed": r["seed"],
                      "backbone": r["backbone"], "reference": reference, "candidate": candidate}
            deltas = old._site_deltas(diagnostics[reference], diagnostics[candidate])
            values = np.array([s["mae_delta_m"] for s in deltas.values()])
            paired_rows.append({**prefix, "mae_delta_m": cand["macro_mae_full_m"] - base["macro_mae_full_m"],
                "es_delta_m": cand["macro_es_m"] - base["macro_es_m"],
                "joint_nll_delta": cand["macro_joint_nll"] - base["macro_joint_nll"],
                "bf_accuracy_delta": cand["macro_bf_accuracy"] - base["macro_bf_accuracy"],
                "p95_delta_m": cand["site_p95_full_m"] - base["site_p95_full_m"],
                "p99_delta_m": cand["site_p99_full_m"] - base["site_p99_full_m"],
                "worst_site_delta_m": cand["worst_site_mae_m"] - base["worst_site_mae_m"],
                "improved_sites": int((values < -1e-7).sum()), "harmed_sites": int((values > 1e-7).sum()),
                "max_single_site_harm_m": float(values.max())})
            site_rows.extend({**prefix, "site_key": key, **{k: v for k, v in value.items() if k != "site"}}
                             for key, value in deltas.items())
    summaries = []
    metrics = ("macro_mae_full_m", "macro_mae_predicted_bf_m", "macro_es_m", "macro_joint_nll", "macro_bf_accuracy",
               "row_bf_accuracy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m")
    for group, rows in sorted(grouped.items()):
        assert len(rows) == 3, f"incomplete seed group: {group}"
        summary = dict(zip(("phase", "dataset", "backbone", "method"), group))
        summary["n_seeds"] = 3
        for metric in metrics:
            values = [r[metric] for r in rows]
            summary[metric + "_mean"] = statistics.mean(values)
            summary[metric + "_sd"] = statistics.stdev(values)
        summaries.append(summary)
    for name, rows in (("RUN_LEVEL_METRICS.csv", run_rows), ("AGGREGATE_SUMMARY.csv", summaries),
                       ("PAIRED_DELTAS.csv", paired_rows), ("SITE_DELTAS_ALL_RUNS.csv", site_rows)):
        write_csv(root / name, rows)
    review = {"status": "PASS", "phase_F": f_audit, "phase_T": t_audit,
              "run_metric_rows": len(run_rows), "summary_rows": len(summaries),
              "paired_comparisons": len(paired_rows), "paired_site_rows": len(site_rows),
              "old_source_data_artifacts_unchanged": True, **TEST_COUNTS,
              "aggregation_hashes": {p.name: sha(p) for p in root.glob("*.csv")}}
    write_json(root / "INTEGRITY_REVIEW.json", review)
    write_report(root, summaries, paired_rows, run_rows)
    emit("aggregate_complete", rows=len(run_rows), summary_rows=len(summaries), **TEST_COUNTS)


def write_report(root, summaries, paired, runs):
    manifest = read_json(root / "EXECUTION_MANIFEST.json")
    lines = ["# V17 空间可辨识性实测报告", "",
             "本轮是相同内部 CAL-eval 物理站点上的回顾性描述比较；不是独立盲测或正式晋级。内部 TEST/官方测试评分行数均为 0。",
             "", "## 环境、协议与来源", "",
             f"Python {manifest['python']}；Torch {manifest['torch']}；CUDA {manifest['cuda']}；{manifest['gpu']}。",
             "来源、训练 CSV/MAT 容器、冻结模型及角色键 SHA256 见 EXECUTION_MANIFEST.json。原有源码、历史产物和数据没有修改。",
             "TUJI MAT 仅反序列化 results_train；物理容器包含测试变量，不声称全部测试字节未打开。",
             "CAL-fit 先完成新联合校准、空间参数选择与参数文件冻结，然后才构建 CAL-eval 评分包；所有预声明方案完整报告。",
             "Phase F 不训练；Phase T O/A 在源代码、配置、划分、选中轮次匹配后复用原 80 轮检查点，其余 C/D/E 各训练完整80轮，按21–80轮最低DEV MAE选中。",
             "C 用联合 BF-XY NLL 替换旧 NLL；D 固定增加0.30分量运输风险，保留0.25坐标损失；E对完整检索logit施加可微图扩散，训练和推断使用同一保存alpha。",
             "ES为完整独立轴df4 Student-t混合的2×32样本近似，各方案同seed+900随机流；规范锚点顺序不同于历史Top-K排序，因此历史ES随机采样实现的逐次取样不要求逐位相同。",
             "三种子均值±样本标准差仅描述同一评价集上的训练波动，不能当作独立站点显著性。", "",
             "## 完整指标汇总", "",
             "|阶段|域|骨干|方案|MAE m|ES m|联合NLL|BF准确率|P95 m|P99 m|最差站点 m|",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        fmt = lambda key: f"{s[key+'_mean']:.5f} ± {s[key+'_sd']:.5f}"
        lines.append(f"|{s['phase']}|{s['dataset']}|{s['backbone']}|{s['method']}|" + "|".join(fmt(k) for k in
            ("macro_mae_full_m", "macro_es_m", "macro_joint_nll", "macro_bf_accuracy", "site_p95_full_m", "site_p99_full_m", "worst_site_mae_m")) + "|")
    lines.extend(["", "## 配对结果、负结果与尾部", "",
                  "下表负MAE/ES差表示改善；同一比较保留三个种子。完整站点向量与伤害见SITE_DELTAS_ALL_RUNS.csv；不得根据此表重选方法。", "",
                  "|阶段/域/骨干|对照→候选|ΔMAE 均值±SD|ΔES 均值±SD|单站点最大伤害 m|",
                  "|---|---|---:|---:|---:|"])
    groups = {}
    for r in paired:
        groups.setdefault((r["phase"], r["dataset"], r["backbone"], r["reference"], r["candidate"]), []).append(r)
    for key, rows in sorted(groups.items()):
        fm = lambda k: f"{statistics.mean(r[k] for r in rows):+.5f} ± {statistics.stdev(r[k] for r in rows):.5f}"
        lines.append(f"|{'/'.join(key[:3])}|{key[3]}→{key[4]}|{fm('mae_delta_m')}|{fm('es_delta_m')}|{max(r['max_single_site_harm_m'] for r in rows):.5f}|")
    lines.extend(["", "## 每次运行的固定选择", "",
                  "|阶段/域/种子/骨干|方案|epoch|参数|", "|---|---|---:|---|"])
    for r in runs:
        lines.append(f"|{r['phase']}/{r['dataset']}/{r['seed']}/{r['backbone']}|{r['method']}|{r['selected_epoch']}|{r['params_json']}|")
    lines.extend(["", "## 数值审计与解释限制", "",
        "INTEGRITY_REVIEW.json独立从逐站点数据重算宏指标并回放选中epoch，所有硬断言通过；单次命令、原始stdout/stderr、退出码及耗时见EXECUTION_RECORD.json。",
        "完整图按建筑/楼层隔离、对称且双随机；BF保持模式仅保留未缩放原始scores的类别质量。1C逐扫描logZ差记录在RESULT.json，其非正性不保证MAE改善。",
        "虚拟分量为N个唯一父锚点加E条唯一无向边，无填充；beta=0物理证据/后验严格恒等，扩展NLL/ES allclose；q<=0组件严格排除。",
        "中点RSS原型和Key插值后仍调用原神经/原型距离法则，虚拟G13使用物理参考均值/RMS；gamma以归一化RSS平方计量且不额外上截断。",
        "纯几何中点仅为诊断；2A主对照复用2B选中的beta，额外beta-only控制另列，不能从CAL-eval挑选。",
        "TUJI原O/A邻居ID相同但权重有约3e-8舍入差；CUDA确定性算法未开启，小差异不得解释成跨楼层监督修复的因果作用。",
        "已有训练语料与历史内部评价已经被观察，空间相邻、扫描/设备相关、MC误差及仅三个随机种子限制推论。禁止与历史E5不同划分MAE直接作同协议优越性声明。",
        "本报告不实施新模型或自动开启后续阶段；是否继续应结合均值、尾部、BF错误和跨域一致性判断。", ""])
    with (root / "RUN_REPORT_V17_SPATIAL_STRUCTURE.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "frozen", "train", "all"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(DEFAULT_INPUT))
        p.add_argument("--output", default=str(DEFAULT_OUTPUT))
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "frozen":
            p.add_argument("--resume-authorized", action="store_true")
    p = sub.add_parser("aggregate")
    p.add_argument("--root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "all":
        preflight(args)
        frozen(args)
        train(args)
        aggregate(argparse.Namespace(root=args.output))
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
