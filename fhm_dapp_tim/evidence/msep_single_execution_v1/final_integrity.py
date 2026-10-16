"""Final byte/hash and command-count audit; no prediction or model fitting."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON token {value}: {path}')
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=reject)


def main():
    freeze = read(AUDIT/'METHOD_FREEZE.json')
    record = read(AUDIT/'EXECUTION_RECORD.json')
    scientific = dict(freeze['historical_source_sha256'],**freeze['source_sha256'])
    changed = [p for p,h in scientific.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    assert not changed, changed
    assert (ROOT/'.git/HEAD').read_text().strip() == freeze['branch_ref']
    assert sha(AUDIT/'DATA_AUDIT.json') == freeze['data_audit_sha256']
    assert sha(AUDIT/'SMOKE_RESULT_AUDIT.json') == freeze['smoke_audit_sha256']
    inputs = {p:h for info in freeze['inputs'].values() for p,h in info['input_sha256'].items()}
    assert all(sha(ROOT/p)==h for p,h in inputs.items())
    for command in record['commands']:
        for metadata in command['logs'].values():
            path = ROOT/metadata['path']
            assert sha(path)==metadata['sha256'] and path.stat().st_size==metadata['bytes']
    outputs = {}
    for dataset,relative in freeze['formal_outputs'].items():
        directory=ROOT/relative
        result=read(directory/'RESULT.json')
        sites=read(directory/'SITE_DIAGNOSTICS.json')
        lock=read(directory/'TEST_INVOCATION.lock')
        assert result['evaluation_role']==sites['evaluation_role']=='test'
        assert len(result['test_metrics'])==len(sites['by_model'])==8
        assert result['checkpoint_sha256']==sha(directory/'CHECKPOINTS.pt')==lock['checkpoint_sha256']
        for key in ('seed','split_seed','support','base_epochs','joint_epochs','batch','device','patience','threads'):
            assert result['config'][key]==freeze['formal_config'][key], (dataset,key)
        assert result['config']['es_draws']==freeze['formal_config']['es_draws_per_stream']
        assert len(result['base_training']['history'])<=80
        assert len(result['joint_fit_history'])==10
        for label in (f'smoke_{dataset}', f'full_{dataset}'):
            commands=[c for c in record['commands'] if c['label']==label]
            assert len(commands)==1 and commands[0]['exit_code']==0
            if label.startswith('full_'):
                assert commands[0]['started_at']<=lock['created_at']<=commands[0]['ended_at']
        outputs[dataset]={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    audit=read(AUDIT/'FULL_RESULT_AUDIT.json')
    assert all(v['status']=='PASS' for v in audit.values())
    read(AUDIT/'PAIRED_SITE_ANALYSIS.json')
    review={'status':'PASS','reviewed_at':datetime.datetime.now().astimezone().isoformat(),
            'frozen_scientific_source_files_unchanged':len(scientific),
            'historical_modules_unchanged':len(freeze['historical_source_sha256']),
            'input_training_files_unchanged':len(inputs),
            'original_command_logs_hashes_verified':sum(len(c['logs']) for c in record['commands']),
            'formal_train_and_test_invocations_per_dataset':1,
            'smoke_invocations_per_dataset':1,'official_test_inference_invocations':0,
            'data_boundary_evidence':'Reviewed loader/call graph and train-variable-only MATLAB adapter; not a system-wide OS I/O trace.',
            'scientific_command_warnings':{c['label']:c['logs']['stderr']['bytes'] for c in record['commands'] if c['label'].startswith(('smoke_','full_'))},
            'retained_failures':[{'label':c['label'],'exit_code':c['exit_code']} for c in record['commands'] if c['exit_code']!=0],
            'formal_output_sha256':outputs,
            'audit_artifact_sha256':{name:sha(AUDIT/name) for name in ('METHOD_FREEZE.json','DATA_AUDIT.json','SMOKE_RESULT_AUDIT.json','FULL_RESULT_AUDIT.json','PAIRED_SITE_ANALYSIS.json','PARAMETER_REVIEW.json','FIX_RECORD.md','RUN_REPORT_MSEP_SINGLE_POSTERIOR.md')}}
    with (AUDIT/'INTEGRITY_REVIEW.json').open('x',encoding='utf-8') as handle:
        json.dump(review,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in review.items() if not k.endswith('sha256')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
