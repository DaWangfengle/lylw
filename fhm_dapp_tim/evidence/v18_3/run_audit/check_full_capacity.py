"""Authorized coordinate-only full-refit feasibility check; never evaluates a model."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
sys.path.insert(0, str(ROOT))
authorization_path = AUDIT / 'capacity_preflight_authorization.json'
assert authorization_path.is_file(), 'explicit user authorization is required before reading coordinate columns'
authorization = json.loads(authorization_path.read_text(encoding='utf-8-sig'))
assert authorization['coordinate_only_full_capacity_preflight_authorized']
import numpy as np
import pandas as pd
import torch
from fhm_dapp_tim.fhm_dapp import uji_v18_3_experiment as exp
from fhm_dapp_tim.fhm_dapp.anchor import location_keys
from fhm_dapp_tim.fhm_dapp.data import denorm_coords
from fhm_dapp_tim.fhm_dapp.coherent_marked_v9 import build_delaunay_geometry
from fhm_dapp_tim.fhm_dapp.v18_single import delaunay_neighbor_targets

reg = exp.load_registry()
csv = Path(reg['train_data'])
with csv.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
assert digest == reg['train_data_sha256']
frame = pd.read_csv(csv, usecols=['LONGITUDE', 'LATITUDE', 'BUILDINGID', 'FLOOR'])
raw = frame[['LONGITUDE', 'LATITUDE']].to_numpy(dtype=np.float32)
building = frame['BUILDINGID'].to_numpy(dtype=np.int64)
floor = frame['FLOOR'].to_numpy(dtype=np.int64)
building_stats = {}
x_norm, y_norm = np.zeros(len(raw), dtype=np.float32), np.zeros(len(raw), dtype=np.float32)
for bi in range(int(building.max()) + 1):
    mask = building == bi
    x, y = raw[mask, 0], raw[mask, 1]
    bounds = dict(x_min=float(x.min()), x_max=float(x.max()), y_min=float(y.min()), y_max=float(y.max()))
    building_stats[bi] = bounds
    dx, dy = bounds['x_max'] - bounds['x_min'], bounds['y_max'] - bounds['y_min']
    x_norm[mask] = (x - bounds['x_min']) / (dx if dx > 0 else 1) * 2.0 - 1.0
    y_norm[mask] = (y - bounds['y_min']) / (dy if dy > 0 else 1) * 2.0 - 1.0
# Reproduce the registered load_dataset -> raw_coords float32 round trip.
x_meter, y_meter = denorm_coords(x_norm, y_norm, building, {'building_stats': building_stats})
raw = np.stack([x_meter, y_meter], axis=1).astype(np.float32)
outer = exp.load_role_shard(reg, 'outer_refit')
assert np.array_equal(raw[reg['roles']['outer_refit']['row_indices']], outer['raw_xy'].numpy())
records = []
for role in ['outer_refit', 'full_train']:
    rows = list(reg['roles']['outer_refit']['row_indices'])
    if role == 'full_train':
        # train_final combines outer_refit then confirmation without reordering.
        rows += list(reg['roles']['confirmation']['row_indices'])
        assert sorted(rows) == list(range(reg['training_rows']))
    xy = raw[rows]
    b, f = building[rows], floor[rows]
    origin = xy.mean(axis=0).astype(np.float32)
    centered = (xy - origin[None, :]).astype(np.float32)
    grouped = {}
    for index, key in enumerate(location_keys(centered, b, f)):
        grouped.setdefault(key, []).append(index)
    coords, ab, af = [], [], []
    for key in sorted(grouped):
        indices = grouped[key]
        coords.append(centered[indices].mean(axis=0))
        ab.append(int(b[indices[0]]))
        af.append(int(f[indices[0]]))
    geometry = build_delaunay_geometry(torch.tensor(np.asarray(coords), dtype=torch.float32), torch.tensor(ab), torch.tensor(af))
    neighbours, weights = delaunay_neighbor_targets(geometry)
    width = int((weights > 0).sum(1).max())
    widest = torch.nonzero((weights > 0).sum(1) == width).flatten().tolist()
    record = dict(role=role, rows=len(rows), anchors=len(coords), padded_target_width=neighbours.shape[1],
        maximum_positive_nonself_targets=width, primary_quotas=80, required_positive_target_capacity=80 + width,
        registered_L=96, genuine_capacity_overflow=80 + width > 96,
        original_padded_width_check_passes=80 + neighbours.shape[1] <= 96,
        coordinate_origin=origin.tolist(), widest_anchor_indices=widest,
        widest_anchor_positive_target_counts=[int((weights[i] > 0).sum()) for i in widest])
    records.append(record)
assert records[0]['padded_target_width'] == 17 and records[0]['maximum_positive_nonself_targets'] == 16
report = dict(status='STATIC FULL-CAPACITY CHECK COMPLETE', roles=records,
    model_instantiated=False, model_training_or_prediction_performed=False, confirmation_shard_opened=False,
    coordinate_columns_parsed_only=['LONGITUDE', 'LATITUDE', 'BUILDINGID', 'FLOOR'], RSS_columns_parsed=False,
    benchmark_opened=False, scientific_source_changed=False, registered_config_changed=False,
    full_coordinate_row_order_matches_registered_train_final=True,
    loader_normalization_roundtrip_matches_registered_outer_shard_bitwise=True,
    sufficient_full_capacity=not records[-1]['genuine_capacity_overflow'],
    checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
(AUDIT / 'full_capacity_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
