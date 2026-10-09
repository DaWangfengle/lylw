"""Train-only pre-prepare checks; no model prediction or existing checkpoint loads."""
from pathlib import Path
import hashlib,json,platform,shutil,sys
from collections import defaultdict
import numpy as np,pandas as pd,scipy,pytest,torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g4_observable_support as g4
from fhm_dapp_tim.fhm_dapp.anchor import location_keys
root=g4.REPO;audit=g4.EVIDENCE/'run_audit'
assert not g4.MANIFEST.exists()
assert hashlib.sha256(g4.DATA_TRAIN.read_bytes()).hexdigest()=='a7cc7d12169be4d6cd9197f5761e463ecdd9e0e798b4f67790ca468d3028785b'
data,roles,rowmap=g4._data_roles()
fit=rowmap['fit'];origin=data['raw_xy'][fit].mean(axis=0).astype(np.float32)
centered=(data['raw_xy']-origin[None,:]).astype(np.float32)
centerkeys=location_keys(centered,data['building'],data['floor'])
forward=defaultdict(set);reverse=defaultdict(set)
for i in fit:
 forward[data['keys'][int(i)]].add(centerkeys[int(i)])
 reverse[centerkeys[int(i)]].add(data['keys'][int(i)])
assert len(forward)==len(reverse)==len(roles['fit'])
assert all(len(v)==1 for v in forward.values()) and all(len(v)==1 for v in reverse.values())
strata={}
for role,sites in roles.items():
 counts=defaultdict(int)
 for k in sites:counts[f'B{k[0]}_F{k[1]}']+=1
 strata[role]=dict(sorted(counts.items()))
assert all(set(strata[role])==set(strata['fit']) for role in roles)
assert len(set(int(x) for values in rowmap.values() for x in values))==len(data['keys'])
assert torch.cuda.is_available(),'Requested CUDA environment unavailable'
properties=torch.cuda.get_device_properties(0)
keyhash={r:hashlib.sha256(json.dumps([list(k) for k in sorted(sites)],separators=(',',':')).encode()).hexdigest() for r,sites in roles.items()}
rec={'status':'PREPARE_CONTRACT_VERIFIED','python_executable':sys.executable,'python_version':sys.version,
 'platform':platform.platform(),'versions':{'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'pytest':pytest.__version__},
 'cuda':torch.version.cuda,'gpu':properties.name,'gpu_total_bytes':properties.total_memory,'gpu_free_total_bytes':list(torch.cuda.mem_get_info()),'disk_free_bytes':shutil.disk_usage(root).free,
 'role_groups':{r:len(v) for r,v in roles.items()},'role_rows':{r:len(v) for r,v in rowmap.items()},
 'role_sorted_keys_sha256':keyhash,'stratum_group_counts':strata,
 'pairwise_physical_group_overlap':{a+'__'+b:len(roles[a]&roles[b]) for a,b in [('fit','calibration'),('fit','confirmation'),('calibration','confirmation')]},
 'fit_R0_centered_float32_four_decimal_keys_bijective':True,'fit_origin_float32':origin.tolist(),
 'only_training_CSV_read':True,'existing_checkpoints_loaded':False,'model_predictions_computed':False,
 'training_loss_code_review':'Original R0 _fit_refit is passed only rows[fit]; original loader and anchor memory use these indices',
 'current_test_sha256':hashlib.sha256((root/'tests/test_uji_v17_g4_observable_support.py').read_bytes()).hexdigest(),
 'source_sha256':g4._source_hashes()}
print(json.dumps(rec,indent=2,allow_nan=False))
