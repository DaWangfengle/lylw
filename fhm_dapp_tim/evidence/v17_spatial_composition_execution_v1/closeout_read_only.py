"""Independent artifact-only closeout. No model import, forward pass, or scoring."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'fhm_dapp_tim/evidence/v17_spatial_composition_v1'
OLD = ROOT / 'fhm_dapp_tim/evidence/v17_spatial_structure_v1'
SPLIT = ROOT / 'fhm_dapp_tim/evidence/v17_two_stage_freeze_v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)

def load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    finite(value)
    return value

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def verify(mapping, base):
    for name, expected in mapping.items():
        assert sha(base / name) == expected, str(base / name)

snapshot = load(HERE / 'INPUT_SNAPSHOT.json')
verify(snapshot['old_source_sha256'], ROOT)
verify(snapshot['current_source_sha256'], ROOT)
verify(snapshot['training_input_sha256'], ROOT)
verify(snapshot['old_artifact_sha256'], OLD)
verify(snapshot['historical_artifact_sha256'], SPLIT)
verify(snapshot['checkpoint_inventory_sha256'], ROOT)
inventory = {p.relative_to(ROOT).as_posix()
             for base in (SPLIT, OLD) for p in base.rglob('*')
             if p.is_file() and p.suffix.lower() in {'.pt', '.pth', '.ckpt'}}
assert inventory == set(snapshot['checkpoint_inventory_sha256'])
assert not any(p.is_file() and p.suffix.lower() in {'.pt', '.pth', '.ckpt'} for p in OUT.rglob('*'))
manifest = load(OUT / 'EXECUTION_MANIFEST.json')
review = load(OUT / 'INTEGRITY_REVIEW.json')
assert manifest['status'] == review['status'] == 'PASS'
verify(manifest['source_sha256'], ROOT)
verify(manifest['checkpoint_sha256'], OLD)
verify(review['artifact_sha256'], OUT)
expected_methods = {'raw_backbone','joint_calibrated_base','1A_global','1D_global','2A','2B',
                    '2A_beta_only_control','combo_1A_2A','combo_1A_2B','combo_1A_2A_beta_only_control',
                    'combo_1D_2A','combo_1D_2B','combo_1D_2A_beta_only_control'}
expected_ids = {(d,s,b) for d in ('tuji','uts','uji') for s in (17301,17302,17303) for b in ('C','D','E')}
results = {}; site_data = {}; max_error = 0.0; selections_checked = 0
for path in sorted(OUT.glob('runs/*/*/RESULT.json')):
    result = load(path); sites = load(path.with_name('SITE_DIAGNOSTICS.json'))
    lock = load(path.with_name('CAL_EVAL.lock')); run_lock = load(path.with_name('RUN.lock'))
    selection = load(path.with_name('CAL_FIT_SELECTION.json'))
    key = result['dataset'],int(result['seed']),result['backbone']
    assert key in expected_ids and key not in results
    results[key] = result; site_data[key] = sites
    assert set(result['methods']) == set(sites) == expected_methods
    assert lock['selection_sha256'] == result['selection_sha256'] == sha(path.with_name('CAL_FIT_SELECTION.json'))
    assert lock['checkpoint_sha256'] == run_lock['checkpoint_sha256'] == result['checkpoint_sha256']
    assert lock['source_sha256'] == run_lock['source_sha256'] == manifest['source_sha256']
    assert lock['cal_fit_key_sha256'] == result['cal_fit_key_sha256']
    assert lock['cal_eval_key_sha256'] == result['cal_eval_key_sha256']
    assert lock['selection_frozen_before_cal_eval'] is True
    assert result['optimizer_steps'] == 0 and result['retraining_performed'] is False
    for obj in (result,lock,run_lock,selection):
        assert obj['internal_test_rows_scored'] == obj['official_test_rows_scored'] == 0
    clean_best = min(selection['calibration_grid'],key=lambda r:(r['objective'],r['temperature'],*r['theta']))
    assert selection['joint_calibrated_base'] == {'temperature':clean_best['temperature'],'theta':clean_best['theta']}
    for name in ('1A_global','1D_global'):
        best = min(selection['structure1_grids'][name],key=lambda r:(r['objective'],r['alpha_prior'],r['alpha_obs']))
        for field in ('alpha_prior','alpha_obs','objective'):
            assert selection['structure1'][name][field] == best[field]
        for field in ('alpha_prior','alpha_obs'):
            assert selection['combined'][name][field] == selection['structure1'][name][field]
    tables = [(selection['structure2_grid'],selection)] + [(selection['combined_grid'][n],selection['combined'][n]) for n in ('1A_global','1D_global')]
    for table,chosen in tables:
        assert len(table) == 49
        assert chosen['2B'] == min(table,key=lambda r:(r['objective'],r['beta'],r['gamma_space']))
        assert chosen['2A_beta_only_control'] == min((r for r in table if r['gamma_space']==0),key=lambda r:(r['objective'],r['beta']))
        assert chosen['2A']['beta'] == chosen['2B']['beta'] and chosen['2A']['gamma_space'] == 0
    selections_checked += 1
    for method,entry in result['methods'].items():
        values=list(sites[method].values()); metrics=entry['metrics']
        assert sum(v['n_scans'] for v in values)==metrics['n_scans']==result['cal_eval_rows']
        assert len(values)==metrics['n_sites']
        for metric,field in [('macro_mae_full_m','mae_full_m'),('macro_mae_predicted_bf_m','mae_predicted_bf_m'),('macro_es_m','energy_score_m'),('macro_joint_nll','joint_nll'),('macro_bf_accuracy','bf_accuracy'),('macro_transport_risk_m','transport_risk_m')]:
            error=abs(statistics.mean(v[field] for v in values)-metrics[metric])
            assert error<2e-5,(key,method,metric,error)
            max_error=max(max_error,error)
        mae=np.array([v['mae_full_m'] for v in values])
        for metric,value in [('site_p95_full_m',np.quantile(mae,.95)),('site_p99_full_m',np.quantile(mae,.99)),('worst_site_mae_m',max(mae))]:
            assert abs(metrics[metric]-value)<2e-5
assert set(results)==expected_ids
run_rows=rows(OUT/'RUN_LEVEL_METRICS.csv'); summary_rows=rows(OUT/'AGGREGATE_SUMMARY.csv')
paired_rows=rows(OUT/'PAIRED_DELTAS.csv'); paired_sites=rows(OUT/'SITE_DELTAS_ALL_RUNS.csv')
assert len(run_rows)==351 and len(summary_rows)==117
grouped=defaultdict(list)
for row in run_rows:
    key=row['dataset'],int(row['seed']),row['backbone']
    actual=results[key]['methods'][row['method']]['metrics']
    for metric,value in actual.items(): assert float(row[metric])==float(value)
    grouped[(row['dataset'],row['backbone'],row['method'])].append(actual)
for row in summary_rows:
    group=grouped[row['dataset'],row['backbone'],row['method']]
    assert len(group)==int(row['n_seeds'])==3
    for field in row:
        if field.endswith('_mean') or field.endswith('_sd'):
            metric=field.rsplit('_',1)[0]; values=[g[metric] for g in group]
            expected=statistics.mean(values) if field.endswith('_mean') else statistics.stdev(values)
            assert float(row[field])==expected
for row in paired_rows:
    key=row['dataset'],int(row['seed']),row['backbone']; methods=results[key]['methods']
    ref=methods[row['reference']]['metrics']; cand=methods[row['candidate']]['metrics']
    for field,metric in [('mae_delta_m','macro_mae_full_m'),('es_delta_m','macro_es_m'),('joint_nll_delta','macro_joint_nll'),('bf_accuracy_delta','macro_bf_accuracy'),('p95_delta_m','site_p95_full_m'),('p99_delta_m','site_p99_full_m'),('worst_site_delta_m','worst_site_mae_m')]:
        assert float(row[field])==cand[metric]-ref[metric]
for row in paired_sites:
    key=row['dataset'],int(row['seed']),row['backbone']; sites=site_data[key]
    ref=sites[row['reference']][row['site_key']]; cand=sites[row['candidate']][row['site_key']]
    assert ref['site']==cand['site']==json.loads(row['site']) and ref['n_scans']==cand['n_scans']==int(row['n_scans'])
    for field,metric in [('mae_delta_m','mae_full_m'),('es_delta_m','energy_score_m'),('joint_nll_delta','joint_nll'),('bf_accuracy_delta','bf_accuracy')]:
        assert float(row[field])==cand[metric]-ref[metric]
for path in OUT.rglob('*.json'): load(path)
record=load(HERE/'EXECUTION_RECORD.json')
for cmd in record['commands']:
    if cmd['status']=='COMPLETE':
        for stream in ('stdout','stderr'): assert sha(HERE/cmd[stream])==cmd[stream+'_sha256']
report={'status':'PASS','model_inference_performed':False,'run_count':len(results),'methods_per_run':13,
        'run_metric_rows':len(run_rows),'summary_rows':len(summary_rows),'paired_rows':len(paired_rows),
        'paired_site_rows':len(paired_sites),'cal_fit_selections_recomputed_from_saved_grids':selections_checked,
        'max_site_macro_reduction_difference':max_error,'frozen_checkpoint_count':27,
        'checkpoint_inventory_unchanged':len(inventory),'old_sources_unchanged':len(snapshot['old_source_sha256']),
        'old_artifacts_unchanged':len(snapshot['old_artifact_sha256']),
        'internal_test_rows_scored':0,'official_test_rows_scored':0,
        'artifact_sha256':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}}
with (HERE/'CLOSEOUT_REVIEW.json').open('x',encoding='utf-8') as f:
    json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k!='artifact_sha256'}),flush=True)
