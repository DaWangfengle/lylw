"""Invoke each exact registered CLI once with immutable source and I/O audit."""
import datetime as dt,hashlib,json,os,subprocess,sys,time
from pathlib import Path
AUDIT=Path(__file__).resolve().parent; ROOT=AUDIT.parents[3]
PHASES=['prepare','train-selection','calibrate-freeze','stability','train-confirmation','confirm','train-final','benchmark']
phase=sys.argv[1]; index=PHASES.index(phase)
files=['fhm_dapp_tim/fhm_dapp/v18_2_single.py','fhm_dapp_tim/fhm_dapp/uji_v18_2_experiment.py','fhm_dapp_tim/evidence/v18_2/DESIGN_V18_2.md','tests/test_uji_v18_2_experiment.py','V18_2_CODEX_RUN.txt']
def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def snapshot(): return {'formal_experiment_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
original=snapshot(); saved=AUDIT/'source_snapshot.json'
if saved.exists(): assert json.loads(saved.read_text(encoding='utf-8'))==original
else:
    assert phase=='prepare'; saved.write_text(json.dumps(original,indent=2),encoding='utf-8')
for previous in PHASES[:index]: assert json.loads((AUDIT/(previous+'_execution.json')).read_text(encoding='utf-8'))['exit_code']==0
log_path=AUDIT/(phase+'.log'); result_path=AUDIT/(phase+'_execution.json')
assert not log_path.exists() and not result_path.exists(),'phase previously invoked'
if phase in {'confirm','benchmark'}:
    assert (AUDIT/('pre_confirmation_gate.json' if phase=='confirm' else 'pre_benchmark_gate.json')).is_file()
    with (AUDIT/(phase+'_invocation.lock')).open('x',encoding='utf-8') as f: f.write(now()+'\n')
command=[sys.executable,'-m','fhm_dapp_tim.fhm_dapp.uji_v18_2_experiment',phase]
if phase!='prepare': command+=['--device','cuda']
env=dict(os.environ); env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',V182_AUDIT_PHASE=phase,V182_AUDIT_DIR=str(AUDIT),V182_REPO_ROOT=str(ROOT)); env['PYTHONPATH']=str(AUDIT)+os.pathsep+env.get('PYTHONPATH','')
record={'phase':phase,'command':command,'started_utc':now(),'formal_experiment_sha':original['formal_experiment_sha']}
print(json.dumps({'event':'started',**record}),flush=True); started=time.perf_counter()
with log_path.open('x',encoding='utf-8') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    record['python_process_id']=child.pid
    for line in child.stdout:
        log.write(line); log.flush()
        try:
            item=json.loads(line)
            if 'epoch' in item or 'refit_epoch' in item:
                train=item.get('train',item); stop=item.get('early_stop',{})
                print(json.dumps({'epoch':item.get('epoch'),'refit_epoch':item.get('refit_epoch'),'total_epochs':item.get('total_epochs'),'joint_geometry_loss':train['joint_geometry_loss'],'ranking_loss':train['ranking_loss'],'union_recall_mass':train['union_recall_mass'],'early_stop_mae_m':stop.get('coordinate_macro_mae_m'),'early_stop_energy_m':stop.get('coordinate_macro_energy_score_m')}),flush=True)
        except (json.JSONDecodeError,KeyError,TypeError):
            if any(word in line for word in ['Traceback','Error:','Warning:']): print(line,end='',flush=True)
    code=child.wait()
record.update(finished_utc=now(),wall_seconds=time.perf_counter()-started,exit_code=code,sources_unchanged=snapshot()==original)
result_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({'event':'finished',**record}),flush=True); sys.exit(code)
