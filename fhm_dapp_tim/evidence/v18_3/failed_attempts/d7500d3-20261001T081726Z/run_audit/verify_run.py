"""Read-only V18.3 checks; sealed confirmation is never manually loaded."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
stage = sys.argv[1]
os.environ.update(V183_AUDIT_PHASE='verify-' + stage, V183_AUDIT_DIR=str(AUDIT), V183_REPO_ROOT=str(ROOT))
import sitecustomize
sitecustomize.activate()
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as exp

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def hashed(payload, key):
    assert payload[key] == exp._canonical_json_sha256(payload, omit=key)
    return payload[key]

def finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)

WITHIN = ['representation_invariant', 'representation_learned', 'radio_prototype', 'radio_marked', 'structure_floor', 'structure_geometry']

def metriccheck(metrics):
    finite(metrics)
    assert abs(sum(metrics['mean_family_weight_' + n] for n in exp.EVIDENCE_FAMILIES) - 1) < 1e-5
    for name in exp.EVIDENCE_FAMILIES:
        assert .15 - 1e-6 <= metrics['mean_family_weight_' + name] <= .70 + 1e-6
        assert 0 <= metrics['mean_family_entropy_' + name] <= 1 + 1e-6
        assert 0 <= metrics['mean_family_disagreement_' + name] <= 1 + 1e-6
    for name in exp.EVIDENCE_SOURCES:
        assert 0 <= metrics['mean_source_entropy_' + name] <= 1 + 1e-6
    for name in WITHIN:
        assert .30 - 1e-6 <= metrics['mean_within_family_weight_' + name] <= .70 + 1e-6
    for pair in [WITHIN[:2], WITHIN[2:4], WITHIN[4:]]:
        assert abs(sum(metrics['mean_within_family_weight_' + n] for n in pair) - 1) < 1e-5
    assert 1 <= metrics['mean_disagreement_temperature'] <= 2
    assert abs(metrics['mean_disagreement_temperature'] - 1 - metrics['mean_global_evidence_disagreement']) < 2e-6
    for name in ['error_evidence_disagreement_correlation', 'radio_mismatch_radio_weight_correlation']:
        assert -1 <= metrics[name] <= 1

def traincheck(record):
    finite(record)
    assert abs(sum(record['family_weight_' + n] for n in exp.EVIDENCE_FAMILIES) - 1) < 1e-5
    assert all(.15 - 1e-6 <= record['family_weight_' + n] <= .70 + 1e-6 for n in exp.EVIDENCE_FAMILIES)
    assert 0 <= record['family_dropout_fraction'] <= 1
    assert 1 <= record['mean_disagreement_temperature'] <= 2

def ckpt(path, phase):
    model, payload = exp.load_v183_checkpoint(path, registry=reg, device=torch.device('cpu'), expected_phase=phase)
    finite(payload)
    assert payload['config'] == exp.V183_CONFIG and payload['model_seed'] == 18431 and payload['selected_epoch'] >= 12
    assert payload['single_member_contract'] == dict(checkpoint_count=1, residual_experts=1, teacher_model=False, ensemble=False)
    assert len(payload['refit_history']) == payload['selected_epoch']
    assert not any('gate' in name.lower() for name, _p in model.named_parameters()) and not hasattr(model, 'evidence_gate')
    for row in payload['refit_history']: traincheck(row)
    return payload, dict(sha256=sha(path), parameters=sum(p.numel() for p in model.parameters()), anchors=model.n_anchors,
        selected_epoch=payload['selected_epoch'], training_seconds=payload['training_seconds'], all_tensors_and_records_finite=True,
        no_trainable_gate=True, checkpoint_count_at_inference=1, residual_experts=1)

def events():
    return [json.loads(line) for path in AUDIT.glob('*_io.jsonl') for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def runtimecheck(phase, no_data=True):
    runtime = read(AUDIT / (phase + '_runtime.json'))
    assert runtime['audit_armed'] and runtime['tensor_checks']['violations'] == 0 and not runtime['v18_2_runtime_modules']
    assert runtime['deterministic_algorithms'] and not runtime['deterministic_warn_only']
    assert not runtime['flash_sdpa'] and not runtime['memory_efficient_sdpa'] and runtime['math_sdpa']
    assert not runtime['cudnn_benchmark'] and runtime['cudnn_deterministic'] and runtime['cublas_workspace_config'] == ':4096:8'
    assert runtime['io_counts']['historical_checkpoint_reads'] == runtime['io_counts']['denied'] == runtime['io_counts']['v18_2_import_attempts'] == 0
    if no_data:
        assert runtime['io_counts']['training_csv_reads'] == runtime['io_counts']['confirmation_reads'] == runtime['io_counts']['benchmark_reads'] == 0
    return runtime

reg = exp.load_registry()
report = dict(stage=stage, registry_sha256=reg['registry_sha256'])
if stage == 'prepare':
    expected = dict(member_fit=(506, 10601), early_stop=(89, 2064), selection_refit=(595, 12665), calibration=(151, 3139), outer_refit=(746, 15804), confirmation=(187, 4133))
    assert reg['training_rows'] == 19937 and reg['training_coordinate_count'] == 933 and reg['n_aps'] == 520
    assert sha(reg['train_data']) == reg['train_data_sha256']
    roles = {name: {exp._decode_key(k) for k in role['coordinate_keys']} for name, role in reg['roles'].items()}
    allkeys = roles['outer_refit'] | roles['confirmation']
    diagnostics = exp.audit_coordinate_roles(roles, allkeys)
    assert diagnostics == reg['split_diagnostics'] and len(exp._strata(allkeys)) == 13
    strata_counts = {}
    for name, role in reg['roles'].items():
        assert (role['coordinate_count'], role['row_count']) == expected[name]
        assert exp._sha256_keys(roles[name]) == role['coordinate_key_sha256']
        assert exp._sha256_indices(np.asarray(role['row_indices'], dtype=np.int64)) == role['row_index_sha256']
        assert len(exp._strata(roles[name])) == 13
        strata_counts[name] = {str(k): len(v) for k, v in exp._strata(roles[name]).items()}
        if name != 'confirmation':
            payload = exp.load_role_shard(reg, name)
            finite(payload)
            assert len(payload['keys']) == role['row_count'] and {exp._decode_key(k) for k in payload['keys']} == roles[name]
    registered = reg['role_shards']['confirmation']
    assert Path(registered['path']).is_file() and len(registered['sha256']) == 64
    reads = [e for e in events() if e['kind'] == 'role_shard' and e['reading'] and e['path'].endswith('/confirmation.pt')]
    assert len(reads) == 1 and reads[0]['phase'] == 'prepare' and reads[0]['purpose'] == 'registered_prepare_hash_only'
    assert reg['single_member_contract'] == dict(checkpoint_count_at_inference=1, residual_experts=1, teacher_model=False,
        knowledge_distillation=False, member_axis=False, ensemble=False, trainable_evidence_gate=False,
        evidence_families=list(exp.EVIDENCE_FAMILIES), inference_family_weight_floor=.15,
        within_family_source_weight_floor=.30, family_dropout_training_only=True, calibration_temperature_can_sharpen=False)
    assert all(reg['evaluation_contract'].values()) and reg['evaluation_contract']['confirmation_is_sequential_not_independent_external']
    report.update(training_rows=19937, physical_coordinates=933, building_floor_strata=13, n_aps=520,
        roles={n: dict(coordinates=a, rows=b) for n, (a, b) in expected.items()}, role_strata_coordinate_counts=strata_counts,
        train_data_sha256=reg['train_data_sha256'], split_diagnostics=diagnostics, role_shards=reg['role_shards'],
        independent_loaded_shard_hash_checks=5, confirmation_hash_fixed_by_prepare=True,
        confirmation_contents_not_manually_loaded=True, confirmation_hash_checked_by_protocol_after_lock=False,
        confirmation_evidence_grade='one-shot sequential internal confirmation', official_evidence_grade='PRE-EXPOSED DESCRIPTIVE BENCHMARK')
elif stage == 'selection':
    payload, checkpoint = ckpt(exp.DEFAULT_SELECTION_CHECKPOINT, 'selection')
    history = payload['selection_history']
    for row in history:
        traincheck(row['train'])
        metriccheck(row['early_stop'])
    eligible = [r for r in history if r['epoch'] >= 12]
    best = min(eligible, key=lambda r: (r['early_stop']['coordinate_macro_energy_score_m'], r['epoch']))
    shortlist = [r for r in eligible if r['early_stop']['coordinate_macro_energy_score_m'] <= best['early_stop']['coordinate_macro_energy_score_m'] + .05]
    winner = min(shortlist, key=lambda r: (r['early_stop']['coordinate_macro_mae_m'], r['early_stop']['coordinate_macro_energy_score_m'], r['epoch']))
    assert winner['epoch'] == payload['selected_epoch'] == exp.select_epoch_from_history(history) and checkpoint['anchors'] == 595
    assert all(bool(r['eligible_for_selection']) == (r['epoch'] >= 12) for r in history)
    runtimecheck('train-selection')
    report.update(checkpoint=checkpoint, selection_epochs=len(history), best_eligible_energy_epoch=best['epoch'],
        best_eligible_energy_score_m=best['early_stop']['coordinate_macro_energy_score_m'],
        shortlist=[dict(epoch=r['epoch'], mae_m=r['early_stop']['coordinate_macro_mae_m'], energy_m=r['early_stop']['coordinate_macro_energy_score_m']) for r in shortlist],
        selected_early_stop=winner['early_stop'], selection_history=history, selection_refit_history=payload['refit_history'])
else:
    freeze = exp.load_freeze(registry=reg)
    cal = read(exp.DEFAULT_CALIBRATION_RESULT)
    hashed(cal, 'result_sha256')
    finite(cal)
    assert freeze['calibration_result_sha256'] == cal['result_sha256'] and freeze['frozen_selected_epoch'] == cal['selected_epoch'] >= 12
    assert cal['registry_sha256'] == reg['registry_sha256'] and not cal['confirmation_accessed']
    assert freeze['frozen_model_config'] == exp.V183_CONFIG
    assert freeze['selection_checkpoint_sha256'] == cal['selection_checkpoint_sha256'] == sha(exp.DEFAULT_SELECTION_CHECKPOINT) == read(AUDIT / 'selection_verification.json')['checkpoint']['sha256']
    cc = cal['calibration']
    surface = cc['surface']
    assert len(surface) == 36
    assert {(r['posterior_temperature'], r['scale_multiplier']) for r in surface} == {(t, s) for t in exp.CALIBRATION_TEMPERATURE_GRID for s in exp.CALIBRATION_SCALE_GRID}
    winner = min(surface, key=lambda r: (r['coordinate_macro_mixture_nll'], abs(math.log(r['posterior_temperature'])) + abs(math.log(r['scale_multiplier']))))
    assert 1 <= cc['posterior_temperature'] == freeze['frozen_posterior_temperature'] == winner['posterior_temperature']
    assert cc['scale_multiplier'] == freeze['frozen_scale_multiplier'] == winner['scale_multiplier']
    assert cc['best_coordinate_macro_mixture_nll'] == winner['coordinate_macro_mixture_nll']
    metriccheck(cal['calibration_metrics'])
    runtimecheck('calibrate-freeze')
    report.update(freeze_sha256=freeze['freeze_sha256'], selection_checkpoint_unchanged=True,
        calibration_result_sha256=cal['result_sha256'], calibration=cc, calibration_metrics=cal['calibration_metrics'])
    if stage != 'freeze':
        stab = read(exp.DEFAULT_STABILITY_RESULT)
        hashed(stab, 'result_sha256')
        finite(stab)
        assert stab['registry_sha256'] == reg['registry_sha256'] and stab['freeze_sha256'] == freeze['freeze_sha256']
        assert stab['seeds'] == list(exp.STABILITY_SEEDS) and [r['seed'] for r in stab['records']] == list(exp.STABILITY_SEEDS)
        assert not stab['ensemble_constructed'] and not stab['best_seed_selected'] and not stab['used_for_model_selection']
        assert all(r['last_train_metrics']['epoch'] == freeze['frozen_selected_epoch'] for r in stab['records'])
        for row in stab['records']:
            metriccheck(row['metrics'])
            traincheck(row['last_train_metrics'])
        for name, values in stab['summary'].items():
            observed = [r['metrics'][name] for r in stab['records']]
            assert math.isclose(values['mean'], statistics.mean(observed), rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(values['sample_std'], statistics.stdev(observed), rel_tol=1e-12, abs_tol=1e-12)
            assert values['minimum'] == min(observed) and values['maximum'] == max(observed)
        runtimecheck('stability')
        report.update(stability_result_sha256=stab['result_sha256'], stability_records=stab['records'],
            stability_summary=stab['summary'], all_five_independent_seeds_verified=True,
            stability_family_mean_ranges={name: [min(r['metrics']['mean_family_weight_' + name] for r in stab['records']), max(r['metrics']['mean_family_weight_' + name] for r in stab['records'])] for name in exp.EVIDENCE_FAMILIES})
    if stage in {'confirmation-refit', 'gate-confirmation', 'confirmation-result', 'final', 'gate-benchmark', 'benchmark-result'}:
        payload, checkpoint = ckpt(exp.DEFAULT_CONFIRMATION_CHECKPOINT, 'confirmation')
        assert payload['freeze_sha256'] == freeze['freeze_sha256'] and payload['selected_epoch'] == freeze['frozen_selected_epoch']
        assert payload['refit_role'] == 'outer_refit' and checkpoint['anchors'] == 746
        runtimecheck('train-confirmation')
        report.update(confirmation_checkpoint=checkpoint, confirmation_refit_history=payload['refit_history'])
    if stage == 'gate-confirmation':
        assert not exp.DEFAULT_CONFIRMATION_LOCK.exists() and not exp.DEFAULT_CONFIRMATION_RESULT.exists()
        assert not any((AUDIT / n).exists() for n in ['confirm_invocation.lock', 'confirm_execution.json', 'confirm.log'])
        seen = events()
        assert not any(not e['permitted'] for e in seen)
        assert not any(e['kind'] == 'pre_exposed_benchmark' and e['reading'] for e in seen)
        assert not any(e['kind'] == 'role_shard' and e['path'].endswith('/confirmation.pt') and e['reading'] and e['phase'] != 'prepare' for e in seen)
        assert read(AUDIT / 'confirmation-refit_verification.json')['confirmation_checkpoint']['sha256'] == checkpoint['sha256']
        runtimecheck('preflight')
        names = ['registry self-hash valid', 'calibration result self-hash valid', 'freeze self-hash valid',
                 'confirmation checkpoint registry SHA matches', 'confirmation checkpoint freeze SHA matches',
                 'selected epoch matches freeze', 'config matches freeze', 'posterior temperature matches freeze and >=1',
                 'scale multiplier matches freeze', 'confirmation lock absent', 'confirmation result absent',
                 'confirmation shard never manually loaded or evaluated after prepare']
        report.update(gate_checks={str(i): True for i in range(1, 13)}, gate_check_names={str(i): n for i, n in enumerate(names, 1)},
            prior_confirm_invocations=0, confirmation_contents_loaded_before_lock=False, prepare_registration_hash_only=True,
            no_trainable_gate=True, v18_2_runtime_import=False)
    if stage in {'confirmation-result', 'final', 'gate-benchmark', 'benchmark-result'}:
        conf = exp.load_confirmation_result(exp.DEFAULT_CONFIRMATION_RESULT, registry=reg, freeze=freeze)
        finite(conf)
        metriccheck(conf['confirmation_metrics'])
        lock = read(exp.DEFAULT_CONFIRMATION_LOCK)
        hashed(lock, 'lock_sha256')
        assert lock['confirmation_consumed'] and conf['confirmation_lock_sha256'] == lock['lock_sha256']
        assert conf['checkpoint_sha256'] == lock['checkpoint_sha256'] == checkpoint['sha256'] and conf['single_member']
        assert conf['confirmation_metrics']['rows'] == 4133 and conf['confirmation_metrics']['coordinate_groups'] == 187
        seen = events()
        reads = [e for e in seen if e['kind'] == 'role_shard' and e['reading'] and e['path'].endswith('/confirmation.pt') and e['phase'] == 'confirm']
        writes = [e for e in seen if e['kind'] == 'confirmation_lock' and not e['reading']]
        assert len(writes) == 1 and len(reads) == 2 and writes[0]['time_ns'] < min(e['time_ns'] for e in reads)
        assert all(e['confirmation_lock_present'] and e['permitted'] for e in reads)
        runtimecheck('confirm', no_data=False)
        report.update(confirmation_result_sha256=conf['result_sha256'], confirmation_lock_sha256=lock['lock_sha256'],
            confirmation_metrics=conf['confirmation_metrics'], confirmation_shard_sha_verified_after_lock=True,
            lock_written_before_confirmation_read=True, confirm_command_count=1,
            confirmation_first_read_time_ns=min(e['time_ns'] for e in reads), lock_write_open_time_ns=writes[0]['time_ns'],
            confirmation_evidence_grade='one-shot sequential internal confirmation', independent_external_confirmation=False)
    if stage in {'final', 'gate-benchmark', 'benchmark-result'}:
        final, final_check = ckpt(exp.DEFAULT_FINAL_CHECKPOINT, 'final')
        assert final_check['anchors'] == 933 and final['refit_role'] == 'full_train' and final['freeze_sha256'] == freeze['freeze_sha256']
        assert final['selected_epoch'] == freeze['frozen_selected_epoch'] and final['confirmation_result_sha256'] == conf['result_sha256']
        assert final['confirmation_result_file_sha256'] == sha(exp.DEFAULT_CONFIRMATION_RESULT)
        runtimecheck('train-final', no_data=False)
        report.update(final_checkpoint=final_check, final_refit_history=final['refit_history'])
    if stage == 'gate-benchmark':
        assert not exp.DEFAULT_BENCHMARK_RESULT.exists() and not any((AUDIT / n).exists() for n in ['benchmark_invocation.lock', 'benchmark_execution.json', 'benchmark.log'])
        assert not any(e['kind'] == 'pre_exposed_benchmark' and e['reading'] for e in events())
        assert read(AUDIT / 'final_verification.json')['final_checkpoint']['sha256'] == final_check['sha256']
        report.update(pre_exposed_descriptive_benchmark=True, blind_confirmation_claim=False,
            used_for_model_selection=False, final_checkpoint_verified=True, prior_benchmark_invocations=0)
    if stage == 'benchmark-result':
        bench = read(exp.DEFAULT_BENCHMARK_RESULT)
        hashed(bench, 'result_sha256')
        finite(bench)
        metriccheck(bench['benchmark_metrics'])
        assert bench['registry_sha256'] == reg['registry_sha256'] and bench['freeze_sha256'] == freeze['freeze_sha256']
        assert bench['final_checkpoint_sha256'] == final_check['sha256']
        assert bench['benchmark_was_pre_exposed_before_v18_3'] and not bench['blind_confirmation_claim'] and not bench['used_for_model_selection']
        reads = [e for e in events() if e['kind'] == 'pre_exposed_benchmark' and e['reading']]
        assert len(reads) == 2 and all(e['phase'] == 'benchmark' and e['permitted'] for e in reads)
        runtimecheck('benchmark', no_data=False)
        report.update(benchmark_result_sha256=bench['result_sha256'], benchmark_data_sha256=bench['benchmark_data_sha256'],
            benchmark_metrics=bench['benchmark_metrics'], benchmark_command_count=1)

destination = AUDIT / ('pre_confirmation_gate.json' if stage == 'gate-confirmation' else 'pre_benchmark_gate.json' if stage == 'gate-benchmark' else stage + '_verification.json')
assert not destination.exists()
destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
large = {'role_shards', 'selection_history', 'selection_refit_history', 'confirmation_refit_history', 'final_refit_history',
         'calibration', 'calibration_metrics', 'stability_records', 'selected_early_stop', 'confirmation_metrics', 'benchmark_metrics', 'role_strata_coordinate_counts'}
print(json.dumps({k: v for k, v in report.items() if k not in large}, indent=2))
