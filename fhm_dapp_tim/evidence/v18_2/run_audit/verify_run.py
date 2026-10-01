"""Read-only artifact checks, with confirmation shard kept sealed until confirm."""
import hashlib,json,math,os,statistics,sys
from pathlib import Path
AUDIT=Path(__file__).resolve().parent; ROOT=AUDIT.parents[3]; sys.path.insert(0,str(ROOT))
stage=sys.argv[1]; os.environ.update(V182_AUDIT_PHASE='verify-'+stage,V182_AUDIT_DIR=str(AUDIT),V182_REPO_ROOT=str(ROOT))
import sitecustomize
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_2_experiment as exp
E=exp.DEFAULT_EVIDENCE

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def hashed(p,key): assert p[key]==exp._canonical_json_sha256(p,omit=key); return p[key]
def finite(v):
    if torch.is_tensor(v): assert torch.isfinite(v).all()
    elif isinstance(v,dict):
        for x in v.values(): finite(x)
    elif isinstance(v,(list,tuple)):
        for x in v: finite(x)
    elif isinstance(v,float): assert math.isfinite(v)
def metriccheck(m):
    finite(m); assert abs(sum(m['mean_gate_weight_'+n] for n in exp.EVIDENCE_SOURCES)-1)<1e-5
    assert all(0<=m['mean_gate_weight_'+n]<=1 and 0<=m['mean_source_entropy_'+n]<=1+1e-6 for n in exp.EVIDENCE_SOURCES)
def ckpt(path,phase):
    model,p=exp.load_v182_checkpoint(path,registry=reg,device=torch.device('cpu'),expected_phase=phase); finite(p)
    assert p['config']==exp.V182_CONFIG and p['model_seed']==18321 and p['selected_epoch']>=12
    assert p['single_member_contract']=={'checkpoint_count':1,'residual_experts':1,'teacher_model':False,'ensemble':False}
    assert len(p['refit_history'])==p['selected_epoch']
    return p,{'sha256':sha(path),'parameters':sum(v.numel() for v in model.parameters()),'anchors':model.n_anchors,'selected_epoch':p['selected_epoch'],'training_seconds':p['training_seconds'],'all_tensors_and_records_finite':True}
def events():
    return [json.loads(line) for p in AUDIT.glob('*_io.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
reg=exp.load_registry(); report={'stage':stage,'registry_sha256':reg['registry_sha256']}
if stage=='prepare':
    expected={'member_fit':(506,10708),'early_stop':(89,1957),'selection_refit':(595,12665),'calibration':(151,3139),'outer_refit':(746,15804),'confirmation':(187,4133)}
    assert reg['training_rows']==19937 and reg['training_coordinate_count']==933 and reg['n_aps']==520
    assert sha(reg['train_data'])==reg['train_data_sha256']
    roles={n:{exp._decode_key(k) for k in r['coordinate_keys']} for n,r in reg['roles'].items()}
    allkeys=roles['outer_refit']|roles['confirmation']; diag=exp.audit_coordinate_roles(roles,allkeys)
    assert diag==reg['split_diagnostics'] and len(exp._strata(allkeys))==13
    for n,r in reg['roles'].items():
        assert (r['coordinate_count'],r['row_count'])==expected[n]
        assert exp._sha256_keys(roles[n])==r['coordinate_key_sha256']
        assert exp._sha256_indices(np.asarray(r['row_indices'],dtype=np.int64))==r['row_index_sha256']
        if n!='confirmation':
            p=exp.load_role_shard(reg,n); finite(p)
            assert len(p['keys'])==r['row_count'] and {exp._decode_key(k) for k in p['keys']}==roles[n]
    conf_record=reg['role_shards']['confirmation']; assert Path(conf_record['path']).is_file() and len(conf_record['sha256'])==64
    conf_reads=[e for e in events() if e['kind']=='role_shard' and e['reading'] and e['path'].endswith('/confirmation.pt')]
    assert len(conf_reads)==1 and conf_reads[0]['phase']=='prepare' and conf_reads[0]['purpose']=='registered_prepare_hash_only'
    assert reg['single_member_contract']=={'checkpoint_count_at_inference':1,'residual_experts':1,'teacher_model':False,'knowledge_distillation':False,'member_axis':False,'ensemble':False}
    report.update(training_rows=19937,physical_coordinates=933,building_floor_strata=13,n_aps=520,roles={n:{'coordinates':a,'rows':b} for n,(a,b) in expected.items()},train_data_sha256=reg['train_data_sha256'],split_diagnostics=diag,role_shards=reg['role_shards'],independent_loaded_shard_hash_checks=5,confirmation_hash_fixed_by_prepare=True,confirmation_contents_not_manually_loaded=True,confirmation_hash_checked_by_protocol_after_lock=False)
elif stage=='selection':
    p,c=ckpt(exp.DEFAULT_SELECTION_CHECKPOINT,'selection'); history=p['selection_history']
    eligible=[r for r in history if r['epoch']>=12]; best=min(eligible,key=lambda r:(r['early_stop']['coordinate_macro_energy_score_m'],r['epoch']))
    shortlist=[r for r in eligible if r['early_stop']['coordinate_macro_energy_score_m']<=best['early_stop']['coordinate_macro_energy_score_m']+.05]
    winner=min(shortlist,key=lambda r:(r['early_stop']['coordinate_macro_mae_m'],r['early_stop']['coordinate_macro_energy_score_m'],r['epoch']))
    assert winner['epoch']==p['selected_epoch']==exp.select_epoch_from_history(history) and c['anchors']==595
    assert all(bool(r['eligible_for_selection'])==(r['epoch']>=12) for r in history)
    report.update(checkpoint=c,selection_epochs=len(history),best_eligible_energy_epoch=best['epoch'],best_eligible_energy_score_m=best['early_stop']['coordinate_macro_energy_score_m'],shortlist=[{'epoch':r['epoch'],'mae_m':r['early_stop']['coordinate_macro_mae_m'],'energy_m':r['early_stop']['coordinate_macro_energy_score_m']} for r in shortlist],selected_early_stop=winner['early_stop'],selection_history=history,selection_refit_history=p['refit_history'])
else:
    freeze=exp.load_freeze(registry=reg); cal=read(exp.DEFAULT_CALIBRATION_RESULT); hashed(cal,'result_sha256'); finite(cal)
    assert freeze['calibration_result_sha256']==cal['result_sha256'] and freeze['frozen_selected_epoch']==cal['selected_epoch']>=12
    assert cal['registry_sha256']==reg['registry_sha256']
    assert freeze['selection_checkpoint_sha256']==cal['selection_checkpoint_sha256']==sha(exp.DEFAULT_SELECTION_CHECKPOINT)==read(AUDIT/'selection_verification.json')['checkpoint']['sha256']
    cc=cal['calibration']; surface=cc['surface']; assert len(surface)==48
    assert {(r['posterior_temperature'],r['scale_multiplier']) for r in surface}=={(t,s) for t in exp.CALIBRATION_TEMPERATURE_GRID for s in exp.CALIBRATION_SCALE_GRID}
    winner=min(surface,key=lambda r:(r['coordinate_macro_mixture_nll'],abs(math.log(r['posterior_temperature']))+abs(math.log(r['scale_multiplier']))))
    assert cc['posterior_temperature']==freeze['frozen_posterior_temperature']==winner['posterior_temperature']
    assert cc['scale_multiplier']==freeze['frozen_scale_multiplier']==winner['scale_multiplier']
    assert cc['best_coordinate_macro_mixture_nll']==winner['coordinate_macro_mixture_nll']; metriccheck(cal['calibration_metrics'])
    report.update(freeze_sha256=freeze['freeze_sha256'],selection_checkpoint_unchanged=True,calibration_result_sha256=cal['result_sha256'],calibration=cc,calibration_metrics=cal['calibration_metrics'])
    if stage not in {'freeze'}:
        stab=read(exp.DEFAULT_STABILITY_RESULT); hashed(stab,'result_sha256'); finite(stab)
        assert stab['registry_sha256']==reg['registry_sha256'] and stab['freeze_sha256']==freeze['freeze_sha256']
        assert stab['seeds']==list(exp.STABILITY_SEEDS) and [r['seed'] for r in stab['records']]==list(exp.STABILITY_SEEDS)
        assert not stab['ensemble_constructed'] and not stab['best_seed_selected'] and not stab['used_for_model_selection']
        assert all(r['last_train_metrics']['epoch']==freeze['frozen_selected_epoch'] for r in stab['records'])
        for name,v in stab['summary'].items():
            x=[r['metrics'][name] for r in stab['records']]
            assert math.isclose(v['mean'],statistics.mean(x),rel_tol=1e-12,abs_tol=1e-12) and math.isclose(v['sample_std'],statistics.stdev(x),rel_tol=1e-12,abs_tol=1e-12)
        report.update(stability_result_sha256=stab['result_sha256'],stability_records=stab['records'],stability_summary=stab['summary'],all_five_independent_seeds_verified=True)
    if stage in {'confirmation-refit','gate-confirmation','confirmation-result','final','gate-benchmark','benchmark-result'}:
        p,c=ckpt(exp.DEFAULT_CONFIRMATION_CHECKPOINT,'confirmation')
        assert p['freeze_sha256']==freeze['freeze_sha256'] and p['selected_epoch']==freeze['frozen_selected_epoch'] and p['refit_role']=='outer_refit' and c['anchors']==746
        report.update(confirmation_checkpoint=c,confirmation_refit_history=p['refit_history'])
    if stage=='gate-confirmation':
        assert not exp.DEFAULT_CONFIRMATION_LOCK.exists() and not exp.DEFAULT_CONFIRMATION_RESULT.exists()
        assert not any((AUDIT/n).exists() for n in ['confirm_invocation.lock','confirm_execution.json','confirm.log'])
        seen=events(); assert not any(not e['permitted'] for e in seen)
        assert not any(e['kind']=='pre_exposed_benchmark' and e['reading'] for e in seen)
        assert not any(e['kind']=='role_shard' and e['path'].endswith('/confirmation.pt') and e['reading'] and e['phase']!='prepare' for e in seen)
        assert read(AUDIT/'confirmation-refit_verification.json')['confirmation_checkpoint']['sha256']==c['sha256']
        for phase in ['train-selection','stability','train-confirmation']:
            r=read(AUDIT/(phase+'_runtime.json'))
            assert r['deterministic_algorithms'] and not r['deterministic_warn_only'] and not r['flash_sdpa'] and not r['memory_efficient_sdpa'] and r['math_sdpa']
            assert not r['cudnn_benchmark'] and r['cudnn_deterministic'] and r['cublas_workspace_config']==':4096:8'
            assert r['io_counts']=={'training_csv_reads':0,'confirmation_reads':0,'benchmark_reads':0,'historical_checkpoint_reads':0,'denied':0}
        names=['registry self-hash valid','calibration result self-hash valid','freeze self-hash valid','confirmation checkpoint registry SHA matches','confirmation checkpoint freeze SHA matches','selected epoch matches freeze','config matches freeze','posterior temperature matches freeze','scale multiplier matches freeze','confirmation lock absent','confirmation result absent','confirmation shard never manually loaded or evaluated after prepare']
        report.update(gate_checks={str(i):True for i in range(1,13)},gate_check_names={str(i):n for i,n in enumerate(names,1)},prior_confirm_invocations=0,confirmation_contents_loaded_before_lock=False,prepare_registration_hash_only=True)
    if stage in {'confirmation-result','final','gate-benchmark','benchmark-result'}:
        conf=exp.load_confirmation_result(exp.DEFAULT_CONFIRMATION_RESULT,registry=reg,freeze=freeze); finite(conf); metriccheck(conf['confirmation_metrics'])
        lock=read(exp.DEFAULT_CONFIRMATION_LOCK); hashed(lock,'lock_sha256')
        assert lock['confirmation_consumed'] and conf['confirmation_lock_sha256']==lock['lock_sha256']
        assert conf['checkpoint_sha256']==lock['checkpoint_sha256']==c['sha256'] and conf['single_member']
        assert conf['confirmation_metrics']['rows']==4133 and conf['confirmation_metrics']['coordinate_groups']==187
        seen=events(); reads=[e for e in seen if e['kind']=='role_shard' and e['reading'] and e['path'].endswith('/confirmation.pt') and e['phase']=='confirm']
        writes=[e for e in seen if e['kind']=='confirmation_lock' and not e['reading']]
        assert len(writes)==1 and len(reads)==2 and writes[0]['time_ns']<min(e['time_ns'] for e in reads)
        assert all(e['confirmation_lock_present'] and e['permitted'] for e in reads)
        report.update(confirmation_result_sha256=conf['result_sha256'],confirmation_lock_sha256=lock['lock_sha256'],confirmation_metrics=conf['confirmation_metrics'],confirmation_shard_sha_verified_after_lock=True,lock_written_before_confirmation_read=True,confirm_command_count=1,confirmation_first_read_time_ns=min(e['time_ns'] for e in reads),lock_write_open_time_ns=writes[0]['time_ns'])
    if stage in {'final','gate-benchmark','benchmark-result'}:
        final,fc=ckpt(exp.DEFAULT_FINAL_CHECKPOINT,'final')
        assert fc['anchors']==933 and final['refit_role']=='full_train' and final['freeze_sha256']==freeze['freeze_sha256'] and final['selected_epoch']==freeze['frozen_selected_epoch']
        assert final['confirmation_result_sha256']==conf['result_sha256'] and final['confirmation_result_file_sha256']==sha(exp.DEFAULT_CONFIRMATION_RESULT)
        report.update(final_checkpoint=fc,final_refit_history=final['refit_history'])
    if stage=='gate-benchmark':
        assert not exp.DEFAULT_BENCHMARK_RESULT.exists() and not any((AUDIT/n).exists() for n in ['benchmark_invocation.lock','benchmark_execution.json','benchmark.log'])
        assert not any(e['kind']=='pre_exposed_benchmark' and e['reading'] for e in events())
        assert read(AUDIT/'final_verification.json')['final_checkpoint']['sha256']==fc['sha256']
        r=read(AUDIT/'train-final_runtime.json')
        assert r['deterministic_algorithms'] and not r['deterministic_warn_only'] and not r['flash_sdpa'] and not r['memory_efficient_sdpa'] and r['math_sdpa'] and r['cudnn_deterministic'] and not r['cudnn_benchmark']
        assert r['io_counts']['training_csv_reads']==r['io_counts']['benchmark_reads']==r['io_counts']['denied']==0
        report.update(pre_exposed_descriptive_benchmark=True,blind_confirmation_claim=False,used_for_model_selection=False,final_checkpoint_verified=True,prior_benchmark_invocations=0)
    if stage=='benchmark-result':
        bench=read(exp.DEFAULT_BENCHMARK_RESULT); hashed(bench,'result_sha256'); finite(bench); metriccheck(bench['benchmark_metrics'])
        assert bench['registry_sha256']==reg['registry_sha256'] and bench['freeze_sha256']==freeze['freeze_sha256'] and bench['final_checkpoint_sha256']==fc['sha256']
        assert bench['benchmark_was_pre_exposed_before_v18_2'] and not bench['blind_confirmation_claim'] and not bench['used_for_model_selection']
        reads=[e for e in events() if e['kind']=='pre_exposed_benchmark' and e['reading']]; assert len(reads)==2 and all(e['phase']=='benchmark' and e['permitted'] for e in reads)
        report.update(benchmark_result_sha256=bench['result_sha256'],benchmark_data_sha256=bench['benchmark_data_sha256'],benchmark_metrics=bench['benchmark_metrics'],benchmark_command_count=1)
destination=AUDIT/('pre_confirmation_gate.json' if stage=='gate-confirmation' else 'pre_benchmark_gate.json' if stage=='gate-benchmark' else stage+'_verification.json')
assert not destination.exists(); destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
large={'role_shards','selection_history','selection_refit_history','confirmation_refit_history','final_refit_history','calibration','calibration_metrics','stability_records','selected_early_stop','confirmation_metrics','benchmark_metrics'}
print(json.dumps({k:v for k,v in report.items() if k not in large},indent=2))
