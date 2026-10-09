"""G4 Python-open boundary: UTS training only; no old models/outcomes or old writes."""
import atexit,json,os,sys
from pathlib import Path
root=Path(os.environ['G4_REPO']);audit=Path(os.environ['G4_AUDIT']);label=os.environ['G4_STAGE']
def norm(p):return os.path.normcase(os.path.abspath(p))
train=norm(root/'data/UTSIndoorLoc/UTS_training.csv');data_prefix=norm(root/'data')+os.sep
old=[]
for name in ('v17_g1_geometry_support','v17_g2_mechanism','v17_g3_uts_frozen_transfer'):
 old += [norm(root/'fhm_dapp_tim/evidence'/name)+os.sep,norm(root/'fhm_dapp_tim/models'/name)+os.sep]
record={'stage':label,'python_data_open_counts':{},'prior_G1_G2_G3_artifact_open_counts':{},'denied_attempts':[]}
def hook(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=norm(os.fsdecode(args[0]));mode=str(args[1] or '');flags=int(args[2] or 0)
 write=any(c in mode for c in 'wax+') or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
 forbidden=(path.startswith(data_prefix) and (write or path!=train)) or any(path.startswith(p) for p in old)
 if write and path.startswith(norm(root/'fhm_dapp_tim/fhm_dapp')+os.sep):forbidden=True
 if forbidden:
  record['denied_attempts'].append({'path':path,'mode':mode,'flags':flags})
  raise PermissionError('G4 train-only data and immutable-source boundary: '+path)
 if path.startswith(data_prefix):record['python_data_open_counts'][path]=record['python_data_open_counts'].get(path,0)+1
sys.addaudithook(hook)
def finish():
 with (audit/(label+'.access.json')).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,ensure_ascii=False);f.write('\n')
atexit.register(finish)
