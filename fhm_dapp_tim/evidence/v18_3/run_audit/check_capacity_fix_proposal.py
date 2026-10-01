"""Verify an unapplied capacity-accounting correction on synthetic data."""
import difflib
import importlib.util
import json
from pathlib import Path
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as exp
from fhm_dapp_tim.fhm_dapp import v18_3_single as original
path = ROOT / 'fhm_dapp_tim/fhm_dapp/v18_3_single.py'
source = path.read_text(encoding='utf-8')
old = '    if (\n        primary_quota + int(neighbour_indices.shape[1])\n'
new = '    positive_target_width = int((neighbour_weights > 0.0).sum(dim=1).max().item())\n    if (\n        primary_quota + positive_target_width\n'
assert source.count(old) == 1
fixed = source.replace(old, new)
(AUDIT / 'capacity_fix_proposal.patch').write_text(''.join(difflib.unified_diff(source.splitlines(keepends=True), fixed.splitlines(keepends=True),
    fromfile='a/fhm_dapp_tim/fhm_dapp/v18_3_single.py', tofile='b/fhm_dapp_tim/fhm_dapp/v18_3_single.py')), encoding='utf-8')
namespace = dict(original.__dict__)
namespace['__name__'] = 'unapplied_v183_capacity_proposal'
exec(compile(fixed, '<unapplied positive-target capacity correction>', 'exec'), namespace)
spec = importlib.util.spec_from_file_location('v183_capacity_tests', ROOT / 'tests/test_uji_v18_3_experiment.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
memory, _ = tests._synthetic_memory()
cases = []
for name in ['cpu', 'cuda']:
    device = torch.device(name)
    args = dict(memory=memory, n_aps=16, n_buildings=1, n_devices=1, device=device)
    exp._set_seed(exp.MODEL_SEED)
    before = original.build_v183_model(**args)
    exp._set_seed(exp.MODEL_SEED)
    after = namespace['build_v183_model'](**args)
    assert all(torch.equal(before[0].state_dict()[key], value) for key, value in after[0].state_dict().items())
    assert torch.equal(before[2], after[2]) and torch.equal(before[3], after[3])
    query = memory['prototype_rss'].to(device)
    p = original.v183_predict(before[0], query, posterior_temperature=1., scale_multiplier=.75)
    q = namespace['v183_predict'](after[0], query, posterior_temperature=1., scale_multiplier=.75)
    assert all(torch.equal(p[key], q[key]) for key in p)
    cases.append(dict(device=name, unchanged_model_state=True, unchanged_targets_and_weights=True, unchanged_predictions_bitwise=True))
for neighbours in [16, 17]:
    angle = np.arange(neighbours) * (2 * np.pi / neighbours)
    coords = np.vstack([np.zeros((1, 2)), np.column_stack([2 * np.cos(angle), 2 * np.sin(angle)])]).astype(np.float32)
    generator = torch.Generator().manual_seed(18430)
    rss = -.8 + .5 * torch.rand(neighbours + 1, 16, generator=generator)
    memory, _ = exp.build_anchor_memory(rss, coords, np.zeros(neighbours + 1, dtype=np.int64), np.zeros(neighbours + 1, dtype=np.int64), np.arange(neighbours + 1))
    args = dict(memory=memory, n_aps=16, n_buildings=1, n_devices=1, device=torch.device('cpu'))
    try:
        original.build_v183_model(**args)
    except ValueError as error:
        assert 'widest Delaunay' in str(error)
    else:
        raise AssertionError('original padded-width capacity check did not reject the test case')
    if neighbours == 16:
        result = namespace['build_v183_model'](**args)
        assert result[2].shape[1] == 17 and int((result[3] > 0).sum(1).max()) == 16
        assert not (((result[2] == torch.arange(neighbours + 1)[:, None]) & (result[3] > 0)).any())
        cases.append(dict(positive_targets=16, padded_width=17, corrected_capacity=96, passes_without_pruning=True))
    else:
        try:
            namespace['build_v183_model'](**args)
        except ValueError as error:
            assert 'widest Delaunay' in str(error)
        else:
            raise AssertionError('true capacity overflow was not rejected')
        cases.append(dict(positive_targets=17, corrected_capacity=97, genuine_overflow_still_fail_closed=True))
report = dict(scientific_source_modified=False, model_config_changed=False, positive_targets_pruned=False, quotas_changed=False,
    L_changed=False, synthetic_only=True, confirmation_accessed=False, benchmark_accessed=False, all_checks_passed=True, cases=cases)
(AUDIT / 'capacity_fix_proposal_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
