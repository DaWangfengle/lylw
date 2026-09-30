"""Read-only verification of registered V18.1 artifacts; never evaluates a model."""
import os
import sys
from pathlib import Path
AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
stage = sys.argv[1]
os.environ.update(V181_AUDIT_PHASE='verify-'+stage, V181_AUDIT_DIR=str(AUDIT), V181_REPO_ROOT=str(ROOT))
import sitecustomize
sys.path.insert(0, str(ROOT))
import hashlib
import json
import math
import statistics
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_1_experiment as exp

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def selfcheck(p,key):
    assert p[key] == exp._canonical_json_sha256(p, omit=key)
    return p[key]
def finite(p):
    if torch.is_tensor(p):
        assert torch.isfinite(p).all(), 'nonfinite tensor'
    elif isinstance(p, dict):
        for v in p.values(): finite(v)
    elif isinstance(p, (tuple,list)):
        for v in p: finite(v)
    elif isinstance(p,float): assert math.isfinite(p), 'nonfinite scalar'
def ckpt(path,phase):
    model,p=exp.load_v181_checkpoint(path,registry=reg,device=torch.device('cpu'),expected_phase=phase)
    finite(p)
    assert p['config']==exp.V181_CONFIG and p['single_member_contract']=={'checkpoint_count':1,'member_axis':False,'residual_experts':1}
    assert p['selected_epoch']>=12 and p['model_seed']==18311
    return p, {'sha256':sha(path),'parameters':sum(t.numel() for t in model.parameters()),'anchors':int(model.n_anchors),'selected_epoch':p['selected_epoch'],'all_tensors_and_records_finite':True,'training_seconds':p['training_seconds']}
reg=exp.load_registry()
report={'stage':stage,'registry_sha256':reg['registry_sha256']}
if stage=='prepare':
    expected={'member_fit':(635,13523),'early_stop':(111,2281),'selection_refit':(746,15804),'calibration':(187,4133),'full_train':(933,19937)}
    assert reg['training_rows']==19937 and reg['training_coordinate_count']==933 and reg['n_aps']==520
    assert sha(reg['train_data'])==reg['train_data_sha256']
    roles={n:{exp._decode_key(k) for k in r['coordinate_keys']} for n,r in reg['roles'].items()}
    diagnostics=exp.audit_coordinate_roles(roles,roles['full_train'])
    assert diagnostics==reg['split_diagnostics'] and len(exp._strata(roles['full_train']))==13
    for n,r in reg['roles'].items():
        assert (r['coordinate_count'],r['row_count'])==expected[n]
        assert r['coordinate_key_sha256']==exp._sha256_keys(roles[n])
        assert r['row_index_sha256']==exp._sha256_indices(__import__('numpy').asarray(r['row_indices'],dtype='int64'))
    shards={}
    for n in ('member_fit','early_stop','selection_refit','calibration'):
        p=exp.load_role_shard(reg,n)
        finite(p)
        assert {exp._decode_key(k) for k in p['keys']}==roles[n]
        assert list(p['row_indices'].tolist())==reg['roles'][n]['row_indices']
        assert len(p['keys'])==reg['roles'][n]['row_count']
        shards[n]=reg['role_shards'][n]
    assert not (exp.DEFAULT_SHARDS/'full_train.pt').exists()
    assert reg['single_member_contract']['checkpoint_count_at_inference']==1 and reg['single_member_contract']['residual_experts']==1
    report.update(training_rows=19937,physical_coordinates=933,n_aps=520,building_floor_strata=13,roles={n:{'coordinates':v[0],'rows':v[1]} for n,v in expected.items()},split_diagnostics=diagnostics,train_data_sha256=reg['train_data_sha256'],shards=shards,training_sha_valid=True,shard_sha_valid=True,no_full_train_pt=True)
elif stage=='selection':
    p,c=ckpt(exp.DEFAULT_SELECTION_CHECKPOINT,'selection')
    history=p['selection_history']; eligible=[r for r in history if r['epoch']>=12]
    best=min(eligible,key=lambda r:(r['early_stop']['coordinate_macro_energy_score_m'],r['epoch']))
    shortlist=[r for r in eligible if r['early_stop']['coordinate_macro_energy_score_m']<=best['early_stop']['coordinate_macro_energy_score_m']+.05]
    winner=min(shortlist,key=lambda r:(r['early_stop']['coordinate_macro_mae_m'],r['early_stop']['coordinate_macro_energy_score_m'],r['epoch']))
    assert winner['epoch']==p['selected_epoch']==exp.select_epoch_from_history(history)
    assert len(p['refit_history'])==p['selected_epoch'] and c['anchors']==746
    report.update(checkpoint=c,selection_epochs=len(history),best_eligible_energy_epoch=best['epoch'],best_eligible_energy_score_m=best['early_stop']['coordinate_macro_energy_score_m'],shortlist=[{'epoch':r['epoch'],'mae_m':r['early_stop']['coordinate_macro_mae_m'],'energy_m':r['early_stop']['coordinate_macro_energy_score_m']} for r in shortlist],selected_early_stop=winner['early_stop'],selection_history=history,refit_history=p['refit_history'])
elif stage in ('freeze','stability','final','gate'):
    freeze=exp.load_freeze(registry=reg)
    cal=load(exp.DEFAULT_CALIBRATION_RESULT); selfcheck(cal,'result_sha256'); finite(cal)
    assert cal['registry_sha256']==reg['registry_sha256']
    assert cal['result_sha256']==freeze['calibration_result_sha256']
    assert cal['selection_checkpoint_sha256']==freeze['selection_checkpoint_sha256']==sha(exp.DEFAULT_SELECTION_CHECKPOINT)
    assert cal['selected_epoch']==freeze['frozen_selected_epoch']
    cc=cal['calibration']; surface=cc['surface']
    pairs={(r['posterior_temperature'],r['scale_multiplier']) for r in surface}
    assert len(surface)==48 and pairs=={(t,s) for t in exp.CALIBRATION_TEMPERATURE_GRID for s in exp.CALIBRATION_SCALE_GRID}
    winner=min(surface,key=lambda r:(r['coordinate_macro_mixture_nll'],abs(math.log(r['posterior_temperature']))+abs(math.log(r['scale_multiplier'])),r['posterior_temperature'],r['scale_multiplier']))
    assert winner['posterior_temperature']==cc['posterior_temperature']==freeze['frozen_posterior_temperature']
    assert winner['scale_multiplier']==cc['scale_multiplier']==freeze['frozen_scale_multiplier']
    assert cc['best_coordinate_macro_mixture_nll']==winner['coordinate_macro_mixture_nll']
    previous=load(AUDIT/'selection_verification.json')
    assert previous['checkpoint']['sha256']==sha(exp.DEFAULT_SELECTION_CHECKPOINT)
    report.update(freeze_sha256=freeze['freeze_sha256'],calibration_result_sha256=cal['result_sha256'],selection_checkpoint_unchanged=True,grid_cases=48,calibration=cc,calibration_metrics=cal['calibration_metrics'])
    if stage in ('stability','final','gate'):
        stab=load(exp.DEFAULT_STABILITY_RESULT); selfcheck(stab,'result_sha256'); finite(stab)
        assert stab['freeze_sha256']==freeze['freeze_sha256'] and stab['registry_sha256']==reg['registry_sha256']
        assert stab['seeds']==list(exp.STABILITY_SEEDS) and len(stab['records'])==5
        assert not stab['ensemble_constructed'] and not stab['best_seed_selected'] and not stab['used_for_formal_model_selection']
        assert [r['seed'] for r in stab['records']]==list(exp.STABILITY_SEEDS)
        assert all(r['epochs']==freeze['frozen_selected_epoch'] for r in stab['records'])
        for name,s in stab['summary'].items():
            vals=[r['metrics'][name] for r in stab['records']]
            assert math.isclose(s['mean'],statistics.mean(vals),rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(s['sample_std'],statistics.stdev(vals),rel_tol=1e-12,abs_tol=1e-12)
        report.update(stability_result_sha256=stab['result_sha256'],stability_records=stab['records'],stability_summary=stab['summary'],all_five_independent_seeds_verified=True)
    if stage in ('final','gate'):
        p,c=ckpt(exp.DEFAULT_FINAL_CHECKPOINT,'final')
        assert p['freeze_sha256']==freeze['freeze_sha256'] and p['selected_epoch']==freeze['frozen_selected_epoch'] and p['config']==freeze['frozen_model_config']
        assert c['anchors']==933 and p['refit_role']=='full_train' and len(p['refit_history'])==p['selected_epoch']
        assert not (exp.DEFAULT_SHARDS/'full_train.pt').exists()
        report.update(final_checkpoint=c,final_history=p['refit_history'])
    if stage=='gate':
        audit_events=[]
        for path in AUDIT.glob('*_io.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if line.strip(): audit_events.append(json.loads(line))
        assert not any(e['kind']=='validation' for e in audit_events)
        assert not any(e['kind']=='historical_evidence' for e in audit_events)
        assert not any(not e['permitted'] for e in audit_events)
        assert not exp.DEFAULT_EXTERNAL_LOCK.exists() and not exp.DEFAULT_EXTERNAL_RESULT.exists()
        assert not any((AUDIT/n).exists() for n in ('external_invocation.lock','external_execution.json','external.log'))
        assert all(load(AUDIT/(s+'_execution.json'))['exit_code']==0 and load(AUDIT/(s+'_execution.json'))['sources_unchanged'] for s in ('prepare','train-selection','calibrate-freeze','stability','train-final'))
        for phase in ('train-selection','stability','train-final'):
            runtime=load(AUDIT/(phase+'_runtime.json'))
            assert runtime['deterministic_algorithms'] and not runtime['deterministic_warn_only']
            assert not runtime['flash_sdpa'] and not runtime['memory_efficient_sdpa'] and runtime['math_sdpa']
            assert not runtime['cudnn_benchmark'] and runtime['cudnn_deterministic']
            assert runtime['cublas_workspace_config']==':4096:8'
            assert runtime['io_counts']=={'validation_reads':0,'training_csv_reads':0,'denied':0}
        assert load(AUDIT/'final_verification.json')['final_checkpoint']['sha256']==c['sha256']
        names=['registry self-hash valid','freeze self-hash valid','calibration result SHA matches freeze','final checkpoint SHA recorded','final checkpoint freeze SHA matches','selected epoch matches freeze','final config matches freeze','posterior temperature matches freeze','scale multiplier matches freeze','external consumption lock absent','external result absent','no previous external invocation','no validation read/hash/preview in current run']
        report.update(gate_checks={str(i):True for i in range(1,14)},gate_check_names={str(i):name for i,name in enumerate(names,1)},validation_read_hash_preview_before_external=False,historical_outcome_opened_in_current_run=False,prior_external_invocations=0)
else: raise ValueError(stage)
destination=AUDIT/('pre_external_gate.json' if stage=='gate' else stage+'_verification.json')
assert not destination.exists()
destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('selection_history','refit_history','final_history','calibration','calibration_metrics','stability_records','shards','selected_early_stop')},indent=2))
