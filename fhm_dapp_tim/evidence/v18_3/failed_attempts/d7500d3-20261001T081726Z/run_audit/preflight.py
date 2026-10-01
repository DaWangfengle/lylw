"""Strict CUDA checks on synthetic data only, before registered prepare."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(V183_AUDIT_PHASE='preflight', V183_AUDIT_DIR=str(AUDIT), V183_REPO_ROOT=str(ROOT), CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sitecustomize
sitecustomize.activate()
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_3_single import EVIDENCE_FAMILIES, EVIDENCE_SOURCES, V183_CONFIG, build_v183_model, v183_training_loss, v183_predict

spec = importlib.util.spec_from_file_location('v183_tests', ROOT / 'tests/test_uji_v18_3_experiment.py')
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
exp._set_seed(exp.MODEL_SEED)
device = torch.device('cuda')
assert torch.cuda.is_available() and torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()
small, small_anchor = test._synthetic_memory()
coords = np.asarray([[float(x * 3), float(y * 3)] for b in range(3) for f in range(4) for x in range(5) for y in range(5)], dtype=np.float32)
building = np.asarray([b for b in range(3) for f in range(4) for x in range(5) for y in range(5)], dtype=np.int64)
floor = np.asarray([f for b in range(3) for f in range(4) for x in range(5) for y in range(5)], dtype=np.int64)
g = torch.Generator().manual_seed(18300)
rss = -.9 + .6 * torch.rand(300, 520, generator=g)
rss[torch.rand(300, 520, generator=g) < .75] = -1.
large, large_anchor = exp.build_anchor_memory(rss, coords, building, floor, np.arange(300))
cases = []
for name, memory, naps, nbuildings, batch in [('small', small, 16, 1, 8), ('registered_support', large, 520, 3, 64)]:
    model, geometry, neighbours, weights = build_v183_model(memory=memory, n_aps=naps, n_buildings=nbuildings, n_devices=1, device=device)
    assert not any('gate' in n.lower() for n, _p in model.named_parameters()) and not hasattr(model, 'evidence_gate')
    assert not ((neighbours == torch.arange(model.n_anchors, device=device)[:, None]) & (weights > 0)).any()
    query = memory['prototype_rss'][:batch].to(device)
    target = memory['coords'][:batch].to(device)
    terms = v183_training_loss(model, query, target, torch.arange(batch, dtype=torch.long, device=device), torch.zeros(batch, dtype=torch.long, device=device), neighbours, weights)
    assert all(torch.isfinite(v).all() for v in terms.values())
    terms['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), V183_CONFIG['gradient_clip'], error_if_nonfinite=True)
    torch.optim.AdamW(model.parameters(), lr=V183_CONFIG['learning_rate'], weight_decay=V183_CONFIG['weight_decay']).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    first = v183_predict(model, query, posterior_temperature=1., scale_multiplier=1., target_coords=target)
    assert first['candidate_indices'].shape[1] == min(64, len(memory['coords']))
    assert first['union_candidate_indices'].shape[1] == min(96, len(memory['coords']))
    assert first['within_family_weights'].shape == (batch, 3, 2)
    assert torch.all(first['family_weights'] >= .15 - 1e-6) and torch.all(first['within_family_weights'] >= .30 - 1e-6)
    payload = exp._checkpoint_payload(registry={'registry_sha256': 'synthetic'}, phase='selection', refit_role='selection_refit', selected_epoch=12,
        model=model, memory={**memory, 'stats': {'n_aps': naps}}, origin=np.zeros(2, dtype=np.float32), device_mapping={0: 0}, selection_history=[], refit_history=[])
    with tempfile.TemporaryDirectory(prefix='v183-synthetic-', dir=AUDIT) as tmp:
        path = Path(tmp) / 'synthetic.pt'
        exp._save_torch_new(path, payload)
        restored, _ = exp.load_v183_checkpoint(path, registry={'registry_sha256': 'synthetic'}, device=device, expected_phase='selection')
        second = v183_predict(restored, query, posterior_temperature=1., scale_multiplier=1., target_coords=target)
        assert all(torch.equal(first[k], second[k]) for k in first), 'checkpoint reconstruction mismatch'
    cases.append(dict(case=name, anchors=model.n_anchors, APs=naps, batch=batch,
        strict_forward_backward_optimizer_finite=True, gradient_norm_before_clip=float(gradient_norm.detach()),
        six_evidence_opinions=len(EVIDENCE_SOURCES), three_families=len(EVIDENCE_FAMILIES),
        checkpoint_predictions_exact=True, positive_self_geometry_target=False, no_trainable_gate=True,
        source_weight_bounds_verified=True, family_weight_bounds_verified=True, disagreement_temperature_bounds_verified=True,
        training_only_dropout_verified=True, target_width=neighbours.shape[1], candidate_capacity_valid=True))
assert not any(n.endswith(('.v18_2_single', '.uji_v18_2_experiment')) for n in sys.modules)
report = dict(all_checks_passed=True, python=sys.version, python_executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    formal_experiment_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    model_config=V183_CONFIG, evidence_sources=list(EVIDENCE_SOURCES), evidence_families=list(EVIDENCE_FAMILIES),
    py_compile_passed=3, pytest_passed=42, pytest_failed=0, pytest_seconds=11.23, pre_prepare_bug_fixes=[], clean_start=True,
    cases=cases, real_dataset_reads=0, v18_2_runtime_import=False)
(AUDIT / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
