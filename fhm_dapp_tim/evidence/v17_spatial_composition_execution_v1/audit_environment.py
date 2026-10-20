"""Read-only input/environment snapshot; does not decode any dataset variables."""
from pathlib import Path
import ast
import hashlib
import json
import platform
import sys

import numpy as np
import scipy
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPLIT = ROOT / 'fhm_dapp_tim/evidence/v17_two_stage_freeze_v1'
OLD = ROOT / 'fhm_dapp_tim/evidence/v17_spatial_structure_v1'
OUTPUT = ROOT / 'fhm_dapp_tim/evidence/v17_spatial_composition_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def verify(mapping, base):
    for name, value in mapping.items():
        assert (base / name).is_file(), str(base / name)
        assert sha(base / name) == value, str(base / name)

assert (ROOT / '.git/HEAD').read_text().strip() == 'ref: refs/heads/v17-spatial-composition-v1'
assert not OUTPUT.exists(), 'Preserve existing output and inspect before execution.'
manifest = read(OLD / 'EXECUTION_MANIFEST.json')
revision = read(OLD / 'ENGINEERING_REVISION.json')
assert sha(OLD / 'EXECUTION_MANIFEST.json') == revision['original_manifest_sha256']
assert sha(OLD / 'USER_AUTHORIZATION_20261019.json') == revision['authorization_sha256']
verify(read(OLD / 'USER_AUTHORIZATION_20261019.json')['preserved_frozen_artifact_sha256'], OLD)
old_sources = dict(manifest['source_sha256'])
for name, item in revision['source_changes'].items():
    assert old_sources[name] == item['before_sha256']
    old_sources[name] = item['after_sha256']
old_sources.update(revision['added_source_sha256'])
verify(old_sources, ROOT)
verify(manifest['input_sha256'], ROOT)
verify(manifest['historical_sha256'], Path(manifest['input_root']))
old_artifacts = read(OLD / 'EXECUTION_RECORD.json')['final_artifact_sha256']
verify(old_artifacts, OLD)
old_artifacts['EXECUTION_RECORD.json'] = sha(OLD / 'EXECUTION_RECORD.json')
checkpoints = sorted(OLD.glob('trained/*/[CDE]/best_dev_epoch_21_80.pt'))
assert len(checkpoints) == 27
pt = read(OLD / 'PHASE_T_INTEGRITY_REVIEW.json')
assert pt['status'] == 'PASS' and pt['completed'] == 45
for p in checkpoints:
    assert sha(p) == pt['artifact_sha256'][p.relative_to(OLD).as_posix()]
checkpoint_inventory = {}
for base in (SPLIT, OLD):
    for p in sorted(base.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.pt', '.pth', '.ckpt'}:
            checkpoint_inventory[p.relative_to(ROOT).as_posix()] = sha(p)
current_sources = {p.relative_to(ROOT).as_posix(): sha(p)
                   for folder in (ROOT / 'fhm_dapp_tim/fhm_dapp', ROOT / 'tests')
                   for p in sorted(folder.glob('*.py'))}
for p in (ROOT / 'V17_SPATIAL_COMPOSITION_DESIGN.md', ROOT / 'V17_SPATIAL_COMPOSITION_CODEX_RUN.txt'):
    current_sources[p.relative_to(ROOT).as_posix()] = sha(p)
source = (ROOT / 'fhm_dapp_tim/fhm_dapp/msep_v17_spatial_composition_v1.py').read_text(encoding='utf-8')
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
        assert name not in {'backward', 'step', 'Adam', 'AdamW', 'SGD', 'save', 'train', 'train_one', 'train_member'}, name
environment = {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
               'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
               'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'vram_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
snapshot = {'environment': environment, 'status': 'PASS', 'branch': 'v17-spatial-composition-v1',
            'old_source_sha256': old_sources, 'current_source_sha256': current_sources,
            'training_input_sha256': manifest['input_sha256'], 'old_artifact_sha256': old_artifacts,
            'historical_artifact_sha256': manifest['historical_sha256'],
            'checkpoint_inventory_sha256': checkpoint_inventory,
            'frozen_checkpoints': [p.relative_to(ROOT).as_posix() for p in checkpoints],
            'static_runner_no_training_optimizer_or_checkpoint_write': True,
            'scientific_code_modified_in_this_run': False,
            'requested_attachment': 'D:/liulanqi/V17_SPATIAL_COMPOSITION_CODEX_RUN (1).txt',
            'requested_attachment_exists': False,
            'execution_protocol': 'V17_SPATIAL_COMPOSITION_CODEX_RUN.txt',
            'protocol_fallback_disclosed_before_execution': True}
with (HERE / 'INPUT_SNAPSHOT.json').open('x', encoding='utf-8') as f:
    json.dump(snapshot, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({'environment': environment, 'status': 'PASS', 'frozen_checkpoints': len(checkpoints),
                  'old_sources_unchanged': len(old_sources), 'old_artifacts_verified': len(old_artifacts),
                  'checkpoint_files_snapshotted': len(checkpoint_inventory),
                  'training_inputs': len(manifest['input_sha256'])}, ensure_ascii=False), flush=True)
