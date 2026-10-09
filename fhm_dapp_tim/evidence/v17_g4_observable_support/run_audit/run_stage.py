"""Run one prescribed G4 command; retain raw logs and exit/time/hash metadata."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
root=Path(__file__).resolve().parents[4];audit=Path(__file__).resolve().parent
label=sys.argv[1];command=[sys.executable,*sys.argv[2:]]
(audit/'runtime_cache/pytest').mkdir(parents=True,exist_ok=True)
stdout=audit/(label+'.stdout.log');stderr=audit/(label+'.stderr.log');metadata=audit/(label+'.json')
assert not any(p.exists() for p in (stdout,stderr,metadata,audit/(label+'.access.json')))
env=os.environ.copy()
env.update({'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
 'PYTHONPYCACHEPREFIX':str(audit/'runtime_cache/pycache'),
 'PYTHONPATH':str(audit/'access_guard')+os.pathsep+str(root),
 'G4_REPO':str(root),'G4_AUDIT':str(audit),'G4_STAGE':label,
 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
 'PYTEST_ADDOPTS':'-p no:cacheprovider --basetemp="'+(audit/'runtime_cache/pytest'/label).as_posix()+'"'})
started=datetime.now(timezone.utc).isoformat();tick=time.perf_counter()
print(json.dumps({'stage':label,'started_utc':started,'command':command}),flush=True)
with stdout.open('xb') as out,stderr.open('xb') as err:
 proc=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err)
 print(json.dumps({'stage':label,'child_pid':proc.pid}),flush=True)
 code=proc.wait()
rec={'stage':label,'command':command,'cwd':str(root),'started_utc':started,
 'ended_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-tick,
 'exit_code':code,'stdout':str(stdout.relative_to(root)),'stderr':str(stderr.relative_to(root)),
 'stdout_sha256':hashlib.sha256(stdout.read_bytes()).hexdigest(),
 'stderr_sha256':hashlib.sha256(stderr.read_bytes()).hexdigest()}
access=audit/(label+'.access.json')
if access.exists():rec['access_guard_sha256']=hashlib.sha256(access.read_bytes()).hexdigest()
with metadata.open('x',encoding='utf-8') as f:json.dump(rec,f,indent=2);f.write('\n')
print(json.dumps(rec),flush=True)
print(stdout.read_text(encoding='utf-8',errors='replace')[-4500:],flush=True)
if code:print(stderr.read_text(encoding='utf-8',errors='replace')[-4500:],file=sys.stderr,flush=True)
sys.exit(code)
