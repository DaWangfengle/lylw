"""Run one prescribed command with separate logs, exit code and SHA256."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time
root=Path(__file__).resolve().parents[4]
audit=Path(__file__).resolve().parent
label=sys.argv[1]
command=[sys.executable,*sys.argv[2:]]
stdout_path=audit/(label+'.stdout.log')
stderr_path=audit/(label+'.stderr.log')
metadata_path=audit/(label+'.json')
assert not any(p.exists() for p in (stdout_path,stderr_path,metadata_path,audit/(label+'.access.json')))
env=os.environ.copy()
env.update({
    'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1',
    'PYTHONPYCACHEPREFIX':str(audit/'runtime_cache/pycache'),
    'PYTHONPATH':str(audit/'access_guard')+os.pathsep+str(root),
    'G3_REPO':str(root),'G3_AUDIT':str(audit),'G3_STAGE':label,
    'CUBLAS_WORKSPACE_CONFIG':':4096:8',
    'PYTEST_ADDOPTS':'-p no:cacheprovider --basetemp='+str(audit/'runtime_cache/pytest'/label),
})
started=datetime.now(timezone.utc).isoformat()
tick=time.perf_counter()
print(json.dumps({'stage':label,'started_utc':started,'command':command}),flush=True)
with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
    proc=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err)
    print(json.dumps({'stage':label,'child_pid':proc.pid}),flush=True)
    exit_code=proc.wait()
record={'stage':label,'command':command,'cwd':str(root),'started_utc':started,
        'ended_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-tick,
        'exit_code':exit_code,'stdout':str(stdout_path.relative_to(root)),'stderr':str(stderr_path.relative_to(root)),
        'stdout_sha256':hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
        'stderr_sha256':hashlib.sha256(stderr_path.read_bytes()).hexdigest()}
access=audit/(label+'.access.json')
if access.exists():
    record['access_guard_sha256']=hashlib.sha256(access.read_bytes()).hexdigest()
with metadata_path.open('x',encoding='utf-8') as f:
    json.dump(record,f,indent=2)
    f.write('\n')
print(json.dumps(record),flush=True)
print(stdout_path.read_text(encoding='utf-8',errors='replace')[-6000:],flush=True)
if exit_code:
    print(stderr_path.read_text(encoding='utf-8',errors='replace')[-6000:],file=sys.stderr,flush=True)
sys.exit(exit_code)
