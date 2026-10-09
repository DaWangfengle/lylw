"""Compact Python-open guard: UTS train only until frozen, locked evaluation."""
import atexit, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root=Path(os.environ['G3_REPO'])
audit=Path(os.environ['G3_AUDIT'])
label=os.environ['G3_STAGE']
train=os.path.normcase(os.path.abspath(root/'data/UTSIndoorLoc/UTS_training.csv'))
test=os.path.normcase(os.path.abspath(root/'data/UTSIndoorLoc/UTS_test.csv'))
data_prefix=os.path.normcase(os.path.abspath(root/'data'))+os.sep
records={'stage':label,'python_data_open_counts':{},'first_UTS_test_open':None,'denied_attempts':[]}
def hook(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
        return
    path=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
    if not path.startswith(data_prefix):
        return
    mode=str(args[1] or '')
    flags=int(args[2] or 0)
    write=any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    permit=not write and path==train
    if path==test and not write and label=='evaluate_test_attempt01':
        lock=audit.parent/'external_test_invocation.lock'
        freeze=audit.parent/'freeze_v17_g3_uts.json'
        permit=lock.is_file() and freeze.is_file()
        if permit and records['first_UTS_test_open'] is None:
            records['first_UTS_test_open']={
                'utc':datetime.now(timezone.utc).isoformat(),
                'lock_sha256_before_first_open':hashlib.sha256(lock.read_bytes()).hexdigest(),
                'freeze_sha256_before_first_open':hashlib.sha256(freeze.read_bytes()).hexdigest(),
                'lock_and_freeze_existed_before_first_open':True,
            }
    if not permit:
        records['denied_attempts'].append({'path':path,'mode':mode,'flags':flags})
        raise PermissionError('G3 data access boundary: '+path)
    records['python_data_open_counts'][path]=records['python_data_open_counts'].get(path,0)+1
sys.addaudithook(hook)
def finish():
    with (audit/(label+'.access.json')).open('x',encoding='utf-8') as f:
        json.dump(records,f,indent=2,ensure_ascii=False)
        f.write('\n')
atexit.register(finish)
