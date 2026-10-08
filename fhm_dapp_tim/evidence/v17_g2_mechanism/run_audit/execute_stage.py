import argparse, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parent
PYTHON=Path(r'D:\miniminiconda\envs\liuyan\python.exe')
MODULE='fhm_dapp_tim.fhm_dapp.uji_v17_g2_mechanism'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('compile','pytest','preflight','analyze','check')); parser.add_argument('--attempt',type=int,default=1); args=parser.parse_args()
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    assert (ROOT/'.git/HEAD').read_text().strip()=='ref: refs/heads/v17-g2-support-risk-mechanism-audit'
    if args.stage=='compile': command=[str(PYTHON),'-m','compileall','-q','fhm_dapp_tim/fhm_dapp/uji_v17_g2_mechanism.py','tests/test_uji_v17_g2_mechanism.py']
    elif args.stage=='pytest': command=[str(PYTHON),'-m','pytest','-q','tests/test_uji_v17_g2_mechanism.py']
    else: command=[str(PYTHON),'-m',MODULE,args.stage]
    label=f'{args.stage}_attempt{args.attempt:02d}'
    record_path=AUDIT/(label+'.json'); out_path=AUDIT/(label+'.stdout.log'); err_path=AUDIT/(label+'.stderr.log'); trace_path=AUDIT/(label+'.access.json')
    assert not any(p.exists() for p in (record_path,out_path,err_path,trace_path)),label
    if args.stage=='analyze':
        assert not (AUDIT.parent/'G2_MECHANISM_AUDIT.json').exists() and not (AUDIT.parent/'RUN_REPORT_V17_G2.md').exists()
        with (AUDIT/'ANALYZE_COMMAND_STARTED.json').open('x',encoding='utf-8',newline='\n') as f: json.dump({'command':command,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':args.attempt},f,indent=2); f.write('\n')
    cache=AUDIT/'runtime_cache'; cache.mkdir(exist_ok=True)
    env=dict(os.environ); env.update({'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPYCACHEPREFIX':str(cache/'pycache'),'PYTEST_ADDOPTS':'-p no:cacheprovider','PYTHONPATH':os.pathsep.join([str(AUDIT/'access_guard'),str(ROOT)]),'G2_AUDIT_ROOT':str(ROOT),'G2_ACCESS_TRACE':str(trace_path),'CUDA_VISIBLE_DEVICES':'-1','MPLCONFIGDIR':str(cache/'matplotlib')})
    record={'stage':args.stage,'attempt':args.attempt,'command':command,'working_directory':str(ROOT),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_path':str(out_path),'stderr_path':str(err_path),'access_trace_path':str(trace_path),'cpu_only_environment':True,'scientific_arguments_unchanged':True,'guard_sha256':sha(AUDIT/'access_guard/sitecustomize.py')}
    started=time.perf_counter()
    with out_path.open('xb') as out,err_path.open('xb') as err:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err); record['child_pid']=process.pid
        print(json.dumps({'stage':label,'child_pid':process.pid,'command':command}),flush=True)
        code=process.wait()
    record.update({'exit_code':code,'runtime_seconds':time.perf_counter()-started,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_sha256':sha(out_path),'stderr_sha256':sha(err_path),'access_trace_sha256':sha(trace_path) if trace_path.exists() else None})
    with record_path.open('x',encoding='utf-8',newline='\n') as f: json.dump(record,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'stage':label,'exit_code':code,'runtime_seconds':record['runtime_seconds']}),flush=True)
    print(out_path.read_text(encoding='utf-8',errors='replace')[-2400:],flush=True)
    if code: print(err_path.read_text(encoding='utf-8',errors='replace')[-2400:],flush=True)
    return code
if __name__=='__main__': raise SystemExit(main())
