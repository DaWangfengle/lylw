"""Synthetic-only reproduction and in-memory verification; frozen files untouched."""
import difflib
import hashlib
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
source_path = ROOT / 'fhm_dapp_tim/fhm_dapp/uji_v18_3_experiment.py'
source = source_path.read_text(encoding='utf-8')
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
marker = '        for name in (\n            "nll",\n            "coords",'
assert source.count(marker) == 1
change = ('        # Prediction diagnostics may carry the radio profiler\'s autograd\n'
          '        # graph. Statistics need values only, including NumPy correlations.\n'
          '        result = {name: value.detach() for name, value in result.items()}\n'
          + marker)
fixed_source = source.replace(marker, change)
patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True), fixed_source.splitlines(keepends=True),
    fromfile='a/fhm_dapp_tim/fhm_dapp/uji_v18_3_experiment.py', tofile='b/fhm_dapp_tim/fhm_dapp/uji_v18_3_experiment.py'))
(AUDIT / 'implementation_fix_proposal.patch').write_text(patch, encoding='utf-8')
spec = importlib.util.spec_from_file_location('v183_synthetic_tests', ROOT / 'tests/test_uji_v18_3_experiment.py')
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
memory, _anchors = test._synthetic_memory()
shard = dict(rss=memory['prototype_rss'], raw_xy=memory['coords'], building_t=memory['building'], floor_t=memory['floor'],
    keys=[exp._encode_key((int(b), int(f), float(x), float(y))) for b, f, (x, y) in zip(memory['building'], memory['floor'], memory['coords'])])
report = dict(real_dataset_reads=0, confirmation_accessed=False, benchmark_accessed=False, scientific_source_modified=False, cases=[])
for name in ['cpu', 'cuda']:
    exp._set_seed(exp.MODEL_SEED)
    device = torch.device(name)
    model, _g, _n, _w = exp.build_v183_model(memory=memory, n_aps=16, n_buildings=1, n_devices=1, device=device)
    try:
        exp.evaluate_shard(model, shard, np.zeros(2, dtype=np.float32), posterior_temperature=1., scale_multiplier=1., device=device)
    except RuntimeError as error:
        assert "Can't call numpy() on Tensor that requires grad" in str(error)
        failure = str(error)
    else:
        raise AssertionError('original implementation did not reproduce the failure')
    namespace = dict(exp.__dict__)
    namespace['__name__'] = 'v183_synthetic_fix_proposal'
    exec(compile(fixed_source, '<unapplied V18.3 implementation fix proposal>', 'exec'), namespace)
    metrics = namespace['evaluate_shard'](model, shard, np.zeros(2, dtype=np.float32), posterior_temperature=1., scale_multiplier=1., device=device)
    assert all(np.isfinite(v) for v in metrics.values())
    prediction = exp.v183_predict(model, shard['rss'].to(device), posterior_temperature=1., scale_multiplier=1., target_coords=shard['raw_xy'].to(device))
    detached = {key: value.detach() for key, value in prediction.items()}
    assert all(torch.equal(prediction[key], detached[key]) for key in prediction)
    assert all(not value.requires_grad for value in detached.values())
    report['cases'].append(dict(device=name, original_failure=failure, proposed_evaluation_completed=True,
        all_statistics_finite=True, detached_prediction_values_bitwise_identical=True, numerical_model_or_config_changed=False))
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
(AUDIT / 'implementation_fix_proposal_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
