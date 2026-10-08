import datetime, hashlib, importlib.metadata, json, platform, sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    head='ref: refs/heads/v17-g2-support-risk-mechanism-audit'
    assert (ROOT/'.git/HEAD').read_text().strip()==head
    assert not (AUDIT.parent/'G2_MECHANISM_AUDIT.json').exists()
    assert not (AUDIT.parent/'RUN_REPORT_V17_G2.md').exists()
    inputs=['data/UJIndoorLoc/trainingData.csv','fhm_dapp_tim/evidence/historical_reproduction/v17/registry_v17_reproduction.json']+['fhm_dapp_tim/evidence/v17_g1_geometry_support/'+x+'_v17_g1.json' for x in ('registry','calibration','freeze','confirmation')]
    hashes={n:sha(ROOT/n) for n in inputs}
    assert hashes[inputs[0]]=='45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168'
    protected={}
    for directory in (ROOT/'fhm_dapp_tim/fhm_dapp',ROOT/'tests',ROOT/'fhm_dapp_tim/evidence'):
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or p.is_relative_to(AUDIT.parent) or '__pycache__' in p.parts or 'runtime_cache' in p.parts: continue
            if p.name in {'uji_v17_g2_mechanism.py','test_uji_v17_g2_mechanism.py'}: continue
            if p.suffix in {'.py','.json','.md','.txt','.csv','.log','.diff'}:
                protected[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
    weights={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'checkpoint_content_not_opened':True} for p in sorted((ROOT/'fhm_dapp_tim/models').rglob('*')) if p.is_file() and p.suffix in {'.pt','.pth','.ckpt','.onnx','.safetensors'}}
    prior=json.loads((ROOT/'fhm_dapp_tim/evidence/v17_g1_geometry_support/run_audit/INITIAL_EXECUTION_AUDIT.json').read_text(encoding='utf-8'))
    for name,entry in prior['protected_historical_assets'].items():
        p=ROOT/name
        if 'sha256' in entry: assert sha(p)==entry['sha256'],name
        else: assert p.stat().st_size==entry['bytes'] and p.stat().st_mtime_ns==entry['mtime_ns'],name
    sources=['fhm_dapp_tim/fhm_dapp/uji_v17_g2_mechanism.py','tests/test_uji_v17_g2_mechanism.py','fhm_dapp_tim/evidence/v17_g2_mechanism/DESIGN_V17_G2.md','V17_G2_CODEX_RUN.txt']
    initial_sources={}
    (AUDIT/'initial_sources').mkdir(exist_ok=False)
    for name in sources:
        p=ROOT/name
        initial_sources[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
        with (AUDIT/'initial_sources'/p.name).open('xb') as f: f.write(p.read_bytes())
    result={'status':'INITIAL_READ_ONLY_AUDIT_PASS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'client_timezone':'Asia/Shanghai','python_executable':sys.executable,'python_version':sys.version,'platform':platform.platform(),'versions':{n:importlib.metadata.version(n) for n in ('numpy','pandas','scipy','pytest','torch')},'branch_marker_read_without_git_command':head,'input_sha256':hashes,'initial_G2_sources':initial_sources,'protected_historical_text_assets':protected,'protected_model_metadata':weights,'previous_G1_historical_assets_unchanged':len(prior['protected_historical_assets']),'G2_outputs_absent_at_start':True,'checkpoint_content_opened':0,'model_verification_scope':'Size and mtime only; no checkpoint-content hashing or torch loading.','Git_commands_executed':0,'operational_exceptions':[{'action':'apply_patch audit-helper creation','result':'Failed to write initialize_audit.py; no scientific source or result modified.','resolution':'Created exclusively under G2 run_audit using reviewed filesystem execution.'}]}
    with (AUDIT/'INITIAL_EXECUTION_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    print(json.dumps({'status':result['status'],'input_sha256':hashes,'versions':result['versions'],'protected_text_assets':len(protected),'protected_model_metadata':len(weights),'previous_G1_historical_assets_unchanged':result['previous_G1_historical_assets_unchanged']},indent=2))
if __name__=='__main__': main()
