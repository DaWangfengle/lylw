"""Training-file-only coordinate and RSS contract review, before prepare."""
import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _safe_tree
raw=pd.read_csv(g3.DATA_TRAIN,usecols=["Pos_x","Pos_y","Building_ID","Floor_ID"])
data, reference=g3.training_data()
geometry=g3._training_geometry(data)
physical=[tuple(row) for row in raw[["Building_ID","Floor_ID","Pos_x","Pos_y"]].itertuples(index=False,name=None)]
forward,reverse={},{}
for original,legacy in zip(physical,data["keys"]):
    assert original not in forward or forward[original]==legacy
    assert legacy not in reverse or reverse[legacy]==original, "original CSV sites merged in legacy key"
    forward[original]=legacy
    reverse[legacy]=original
assert len(forward)==geometry["anchor_count"]==1466
canonical,_=ds.load_uts("train")
visible=np.isfinite(canonical.rss_dbm)
rss=data["rss"].numpy()
assert np.all(rss[~visible]==np.float32(-1.0))
assert rss.dtype==np.float32 and np.isfinite(rss).all()
origin=np.asarray(geometry["origin_mean_xy"],dtype=np.float32)
buffer=io.BytesIO()
torch.save(_safe_tree({"coord_origin":origin,"array":np.arange(3,dtype=np.float32)}),buffer)
buffer.seek(0)
saved=torch.load(buffer,map_location="cpu",weights_only=True)
assert torch.equal(saved["coord_origin"],torch.from_numpy(origin))
print(json.dumps({"status":"TRAINING_CONTRACT_VERIFIED","training_rows":len(raw),"exact_raw_training_physical_sites":len(forward),"legacy_training_anchors":geometry["anchor_count"],"coordinate_group_collisions":0,"coordinate_units_contract":"metres as specified by UTS Pos_x/Pos_y adapter and frozen design; numeric range checked, no external unit verification claimed","raw_x_range":[float(raw.Pos_x.min()),float(raw.Pos_x.max())],"raw_y_range":[float(raw.Pos_y.min()),float(raw.Pos_y.max())],"building_mapping":reference.building_mapping,"floor_mapping":reference.floor_mapping,"AP_count":len(reference.ap_ids),"missing_cells_correctly_minus_one":int((~visible).sum()),"visible_RSS_range_dbm":[float(canonical.rss_dbm[visible].min()),float(canonical.rss_dbm[visible].max())],"float32_RSS_valid":True,"restricted_checkpoint_origin_roundtrip_bitwise_equal":True,"provided_test_accessed":False},indent=2,allow_nan=False))
