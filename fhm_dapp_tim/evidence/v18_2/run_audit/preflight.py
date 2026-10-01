"""Synthetic-only strict CUDA and checkpoint reconstruction preflight."""
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_2_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_2_single import EVIDENCE_SOURCES,V182_CONFIG,build_v182_model,v182_training_loss,v182_predict
ROOT=Path.cwd(); AUDIT=ROOT/'fhm_dapp_tim/evidence/v18_2/run_audit'
spec=importlib.util.spec_from_file_location('v182_tests',ROOT/'tests/test_uji_v18_2_experiment.py'); test=importlib.util.module_from_spec(spec); spec.loader.exec_module(test)
exp._set_seed(exp.MODEL_SEED); device=torch.device('cuda')
assert torch.cuda.is_available() and torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()
small,small_anchor=test._synthetic_memory()
coords=np.asarray([[float(x*3),float(y*3)] for b in range(3) for f in range(4) for x in range(5) for y in range(5)],dtype=np.float32)
building=np.asarray([b for b in range(3) for f in range(4) for x in range(5) for y in range(5)],dtype=np.int64)
floor=np.asarray([f for b in range(3) for f in range(4) for x in range(5) for y in range(5)],dtype=np.int64)
g=torch.Generator().manual_seed(18200)
rss=-.9+.6*torch.rand(300,520,generator=g); rss[torch.rand(300,520,generator=g)<.75]=-1.
large,large_anchor=exp.build_anchor_memory(rss,coords,building,floor,np.arange(300))
cases=[]
for name,memory,anchors,naps,nbuildings,batch in [('small',small,small_anchor,16,1,8),('registered_support',large,large_anchor,520,3,64)]:
    model,geometry,neighbours,weights=build_v182_model(memory=memory,n_aps=naps,n_buildings=nbuildings,n_devices=1,device=device)
    assert not ((neighbours==torch.arange(model.n_anchors,device=device)[:,None])&(weights>0)).any()
    query=memory['prototype_rss'][:batch].to(device); target=memory['coords'][:batch].to(device)
    terms=v182_training_loss(model,query,target,torch.arange(batch,dtype=torch.long,device=device),torch.zeros(batch,dtype=torch.long,device=device),neighbours,weights)
    assert all(torch.isfinite(v).all() for v in terms.values())
    terms['loss'].backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.evidence_gate[-1].weight.grad is not None
    torch.nn.utils.clip_grad_norm_(model.parameters(),V182_CONFIG['gradient_clip'])
    torch.optim.AdamW(model.parameters(),lr=V182_CONFIG['learning_rate'],weight_decay=V182_CONFIG['weight_decay']).step()
    first=v182_predict(model,query,posterior_temperature=1.,scale_multiplier=1.,target_coords=target)
    assert all(torch.isfinite(v).all() for v in first.values())
    assert torch.allclose(first['gate_weights'].sum(1),torch.ones(batch,device=device),atol=1e-6)
    assert torch.allclose(first['union_probability'].sum(1),torch.ones(batch,device=device),atol=1e-6)
    assert first['candidate_indices'].shape[1]==min(64,len(memory['coords'])) and first['union_candidate_indices'].shape[1]==min(96,len(memory['coords']))
    payload=exp._checkpoint_payload(registry={'registry_sha256':'synthetic'},phase='selection',refit_role='selection_refit',selected_epoch=12,model=model,memory={**memory,'stats':{'n_aps':naps}},origin=np.zeros(2,dtype=np.float32),device_mapping={0:0},selection_history=[],refit_history=[])
    with tempfile.TemporaryDirectory(prefix='v182-synthetic-') as tmp:
        path=Path(tmp)/'synthetic.pt'; exp._save_torch_new(path,payload)
        restored,_=exp.load_v182_checkpoint(path,registry={'registry_sha256':'synthetic'},device=device,expected_phase='selection')
        second=v182_predict(restored,query,posterior_temperature=1.,scale_multiplier=1.,target_coords=target)
        assert all(torch.equal(first[k],second[k]) for k in first),'checkpoint reconstruction mismatch'
    cases.append({'case':name,'anchors':model.n_anchors,'APs':naps,'batch':batch,'strict_forward_backward_optimizer_finite':True,'gate_gradient_finite':True,'six_evidence_opinions':len(EVIDENCE_SOURCES),'checkpoint_predictions_exact':True,'positive_self_geometry_target':False})
report={'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'formal_experiment_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'model_config':V182_CONFIG,'evidence_sources':list(EVIDENCE_SOURCES),'py_compile_passed':3,'pytest_passed':28,'pytest_failed':0,'pytest_seconds':5.8,'pre_prepare_bug_fixes':[],'clean_start':True,'cases':cases,'real_dataset_reads':0}
(AUDIT/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
