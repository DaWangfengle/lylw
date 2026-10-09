"""Checkpoint-only pre-test panel integrity review; no prediction or test access."""
import json
import torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _sha256_file
manifest=g3._load_json(g3.MANIFEST,"manifest_sha256")
assert not g3.LOCK.exists() and not g3.RESULT.exists()
ref=None
members=[]
for i in range(1,26):
 p=g3.member_path(i)
 saved=torch.load(p,map_location="cpu",weights_only=True)
 assert saved["manifest_sha256"]==manifest["manifest_sha256"]
 assert saved["member_index"]==i and saved["member_seed"]==g3.SEEDS[i-1]
 assert saved["epoch"]==g3.EPOCHS[i-1]
 contract=saved["split_contract"]
 assert contract["training_source"]=="UTS_training.csv"
 assert contract["training_rows"]==manifest["train_geometry"]["rows"]
 assert contract["test_rows_accessed"] is False and contract["test_labels_consumed"] is False
 assert contract["test_metrics_consumed"] is False and contract["outer_uji_consumed"] is False
 assert contract["g1_development_consumed"] is False
 assert len(saved["refit_history"])==g3.EPOCHS[i-1]
 if ref is None: ref=saved
 assert saved["anchor_coordinate_keys"]==ref["anchor_coordinate_keys"]
 assert torch.equal(saved["coord_origin"],ref["coord_origin"])
 for key in ["anchor_coords","anchor_buildings","anchor_floors","prototype_rss"]:
  assert torch.equal(saved["model_state"][key],ref["model_state"][key]),(i,key)
 members.append({"member":i,"seed":saved["member_seed"],"epoch":saved["epoch"],"checkpoint_sha256":_sha256_file(p)})
assert len(set(m["checkpoint_sha256"] for m in members))==25
print(json.dumps({"status":"PRETEST_PANEL_INTEGRITY_VERIFIED","manifest_sha256":manifest["manifest_sha256"],"members":members,"same_anchor_dictionaries":True,"same_origin_and_prototype_memory":True,"unique_checkpoint_sha256_count":25,"checkpoint_training_contract_and_epoch_history_verified":True,"test_file_accessed":False,"prediction_run":False},indent=2))
