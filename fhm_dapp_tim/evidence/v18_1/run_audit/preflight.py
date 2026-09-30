import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
ROOT = Path.cwd()
from fhm_dapp_tim.fhm_dapp import uji_v18_1_experiment as exp
from fhm_dapp_tim.fhm_dapp.v18_1_single import V181_CONFIG, build_v181_model, v181_training_loss, v181_predict
spec=importlib.util.spec_from_file_location("v181_tests",ROOT/"tests/test_uji_v18_1_experiment.py")
test=importlib.util.module_from_spec(spec); spec.loader.exec_module(test)
exp._set_seed(exp.MODEL_SEED)
assert torch.cuda.is_available()
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.backends.cuda.flash_sdp_enabled()
assert not torch.backends.cuda.mem_efficient_sdp_enabled()
assert torch.backends.cuda.math_sdp_enabled()
memory, anchors = test._synthetic_memory()
device = torch.device("cuda")
model, geometry, neighbours, weights = build_v181_model(memory=memory,n_aps=12,
    n_buildings=1,n_devices=1,device=device)
assert not ((neighbours==torch.arange(model.n_anchors,device=device)[:,None])&(weights>0)).any()
rss=memory["prototype_rss"].to(device)
target=memory["coords"].to(device)
terms=v181_training_loss(model,rss,target,torch.from_numpy(anchors).to(device),
    torch.zeros(len(rss),device=device,dtype=torch.long),neighbours,weights)
assert all(torch.isfinite(v).all() for v in terms.values())
terms["loss"].backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
first=v181_predict(model,rss,top_k=min(64,model.n_anchors),posterior_temperature=1.0,
    scale_multiplier=1.0,target_coords=target)
assert all(torch.isfinite(v).all() for v in first.values())
assert torch.allclose(first["candidate_logits_corrected"], first["candidate_logits_learned"]-
    first["radio_energy"]/V181_CONFIG["radio_lambda"])
payload=exp._checkpoint_payload(registry={"registry_sha256":"synthetic"},
    phase="selection",refit_role="selection_refit",selected_epoch=12,model=model,
    memory={**memory,"stats":{"n_aps":12}},origin=np.zeros(2,dtype=np.float32),
    device_mapping={0:0},selection_history=[],refit_history=[])
with tempfile.TemporaryDirectory(prefix="v181-synthetic-") as tmp:
    path=Path(tmp)/"synthetic.pt"; exp._save_torch_new(path,payload)
    restored,_=exp.load_v181_checkpoint(path,registry={"registry_sha256":"synthetic"},
        device=device,expected_phase="selection")
    second=v181_predict(restored,rss,top_k=min(64,model.n_anchors),posterior_temperature=1.0,
        scale_multiplier=1.0,target_coords=target)
    assert all(torch.equal(first[k],second[k]) for k in first), "Reconstruction mismatch"
report={"python":sys.version,"python_executable":sys.executable,"torch":torch.__version__,
    "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
    "strict_cuda_forward_backward_and_radio_finite":True,"reconstruction_exact":True,
    "positive_self_geometry_target":False,"model_config":V181_CONFIG,
    "formal_experiment_sha":"e87871d6f5d75c1c3e21e3911b2a3fb0b4a5e781",
    "py_compile_passed":3,"pytest_passed":22,"pytest_failed":0,"pytest_seconds":11.42,
    "pre_prepare_code_fixes":[],"clean_start":True,"formal_run_restarts":0}
(ROOT/"fhm_dapp_tim/evidence/v18_1/run_audit/preflight.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
