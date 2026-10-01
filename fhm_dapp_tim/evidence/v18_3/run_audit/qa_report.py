"""Verify complete reports and immutable records without importing a model."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
E = AUDIT.parent
PHASES = ['prepare', 'train-selection', 'calibrate-freeze', 'stability', 'train-confirmation', 'confirm', 'train-final', 'benchmark']
HEADINGS = ['GIT', 'TESTS', 'PROTOCOL', 'MODEL', 'TRAINING', 'CALIBRATION', 'STABILITY', 'CONFIRMATION', 'PRE-EXPOSED BENCHMARK', 'COMPUTE', 'AUDIT', 'INTERPRETATION']
FAMILIES = ['representation', 'radio', 'structure']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value, omitted):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != omitted}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)

report = read(E / 'RUN_REPORT_V18_3.json')
markdown = (E / 'RUN_REPORT_V18_3.md').read_text(encoding='utf-8')
assert report['report_sha256'] == digest(report, 'report_sha256')
assert report['report_status'] == 'COMPLETE'
finite(report)
assert re.findall(r'^## (.+)$', markdown, re.M) == HEADINGS
assert not re.search(r'\b(?:TODO|TBD|FIXME|NaN|Infinity)\b', markdown)
assert len(report['INTERPRETATION']) == 8
assert report['TESTS']['py_compile_passed'] == 3
assert report['TESTS']['pytest_passed'] == 43 and report['TESTS']['pytest_failed'] == 0
assert '43 passed' in (AUDIT / 'pytest.log').read_text(encoding='utf-8')
assert (AUDIT / 'py_compile.log').read_text(encoding='utf-8').count('PASS ') == 3
assert report['GIT']['branch'] == 'v18-3-structured-consensus'
assert report['GIT']['formal_experiment_sha'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
snapshot = report['GIT']['source_snapshot']
assert all(sha(ROOT / name) == expected for name, expected in {**snapshot['files'], **snapshot['common_source_files']}.items())
assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *snapshot['files']], cwd=ROOT, text=True).strip()
assert report['AUDIT']['confirm_command_count'] == report['AUDIT']['benchmark_command_count'] == 1
assert report['AUDIT']['denied_accesses'] == report['AUDIT']['per_scan_consensus_boundary_violations'] == 0
assert report['AUDIT']['lock_written_before_confirmation_load']
assert not any(report['AUDIT'][key] for key in ['teacher_used', 'distillation_used', 'ensemble_used', 'best_seed_selected',
    'prediction_averaging', 'trainable_evidence_gate_used', 'v18_2_runtime_import', 'positive_self_anchor_target_used',
    'calibration_labels_used_for_backbone_gradients_before_freeze', 'confirmation_used_for_tuning', 'benchmark_used_for_tuning', 'temperature_below_one_selected'])
assert all(report['PRE_CONFIRMATION_GATE'].values()) and len(report['PRE_CONFIRMATION_GATE']) == 12
assert report['GIT']['restart_authorization']['user_answer'] == '同意最小修复并完整重启（推荐）'
assert not report['GIT']['failed_attempt']['confirmation_lock_created'] and not report['GIT']['failed_attempt']['benchmark_read']
assert report['MODEL']['config'] == report['CALIBRATION']['freeze']['frozen_model_config']
old_preflight = read(E / 'failed_attempts/d7500d3-20261001T081726Z/run_audit/preflight.json')
assert old_preflight['model_config'] == report['MODEL']['config']
assert report['evidence_grade']['independent_external_confirmation'] is False
assert report['evidence_grade']['confirmation'] == 'one-shot sequential internal confirmation'

for section, filename, prefix, metric_key in [('CONFIRMATION', 'confirmation_result_v18_3.json', 'CONFIRMATION', 'confirmation_metrics'),
        ('PRE_EXPOSED_BENCHMARK', 'benchmark_result_v18_3.json', 'BENCHMARK', 'benchmark_metrics')]:
    original = read(E / filename)
    assert original['result_sha256'] == digest(original, 'result_sha256')
    assert report[section] == original
    metrics = original[metric_key]
    for title, field, unit in [('MAE', 'coordinate_macro_mae_m', ' m'), ('ENERGY SCORE', 'coordinate_macro_energy_score_m', ' m'), ('NLL', 'coordinate_macro_mixture_nll', '')]:
        assert f'**{prefix} {title} = {metrics[field]:.6f}{unit}**' in markdown
    assert abs(sum(metrics['mean_family_weight_' + family] for family in FAMILIES) - 1) < 1e-5
    assert all(.15 <= metrics['mean_family_weight_' + family] <= .70 for family in FAMILIES)

bench = report['PRE_EXPOSED_BENCHMARK']
assert bench['benchmark_was_pre_exposed_before_v18_3'] and not bench['blind_confirmation_claim'] and not bench['used_for_model_selection']
assert 'independent external confirmation = NO；pre-exposed benchmark = YES；benchmark used for tuning = NO' in markdown
stab = report['STABILITY']
assert stab['seeds'] == [18431, 18432, 18433, 18434, 18435] == [row['seed'] for row in stab['records']]
assert not stab['ensemble_constructed'] and not stab['best_seed_selected'] and not stab['used_for_model_selection']
for key, summary in stab['summary'].items():
    values = [row['metrics'][key] for row in stab['records']]
    assert math.isclose(summary['mean'], statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(summary['sample_std'], statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-12)
for name, summary in stab['family_weight_seed_summary'].items():
    values = [row['metrics']['mean_family_weight_' + name] for row in stab['records']]
    assert math.isclose(summary['mean'], statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(summary['sample_std'], statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-12)
    assert summary['minimum'] == min(values) and summary['maximum'] == max(values)
epoch = report['MODEL']['selected_epoch']
trajectories = report['TRAINING']['full_trajectories']
for role in ['selection_refit', 'confirmation_refit', 'final_refit']:
    assert len(trajectories[role]) == epoch
    assert [row['epoch'] for row in trajectories[role]] == list(range(1, epoch + 1))
assert [row['seed'] for row in trajectories['stability']] == stab['seeds']
for full, registered in zip(trajectories['stability'], stab['records']):
    assert len(full['history']) == epoch and full['history'][-1] == registered['last_train_metrics']
assert len(report['CALIBRATION']['calibration']['surface']) == 36
assert report['CALIBRATION']['calibration']['posterior_temperature'] >= 1

previous_end = None
for phase in PHASES:
    execution = report['COMPUTE']['executions'][phase]
    runtime = report['COMPUTE']['runtimes'][phase]
    assert execution['exit_code'] == 0 and execution['sources_unchanged'] and execution['runtime_audit_valid']
    assert execution['command'] == [report['TESTS']['python_executable'], '-m', 'fhm_dapp_tim.fhm_dapp.uji_v18_3_experiment', phase] + ([] if phase == 'prepare' else ['--device', 'cuda'])
    start, end = [dt.datetime.fromisoformat(execution[key]) for key in ['started_utc', 'finished_utc']]
    assert end > start and (previous_end is None or start > previous_end)
    previous_end = end
    assert runtime['audit_armed'] and runtime['io_counts']['denied'] == runtime['tensor_checks']['violations'] == 0
    assert not runtime['v18_2_runtime_modules']
    assert execution['audit_instrumentation_sha256'] == report['COMPUTE']['executions']['prepare']['audit_instrumentation_sha256']
    if phase != 'prepare':
        assert runtime['deterministic_algorithms'] and not runtime['deterministic_warn_only']
        assert not runtime['flash_sdpa'] and not runtime['memory_efficient_sdpa'] and runtime['math_sdpa']
        assert runtime['cudnn_deterministic'] and not runtime['cudnn_benchmark'] and runtime['cublas_workspace_config'] == ':4096:8'

result = dict(all_checks_passed=True, model_import_or_inference_performed_by_qa=False, required_sections=HEADINGS,
    report_self_sha256=report['report_sha256'], commands_in_registered_order=PHASES, all_five_seed_histories_complete=True,
    five_seed_sample_sd_valid=True, all_seven_cuda_phases_strict_deterministic=True, one_shot_confirmation_valid=True,
    sequential_internal_evidence_grade_valid=True, pre_exposed_benchmark_label_valid=True,
    original_failed_attempt_and_restart_authorization_preserved=True, model_config_unchanged_from_initial_registration=True,
    formal_scientific_source_unchanged_after_fresh_prepare=True, checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
destination = AUDIT / 'report_qa.json'
assert not destination.exists()
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
