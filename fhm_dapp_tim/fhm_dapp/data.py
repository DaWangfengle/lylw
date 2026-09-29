# 从 rf_diffuse_data.py 内联，自包含 fhm_dapp_tim 实验
"""
RF-Diffuse Multi-Dataset Loader.

Unified data loading for all supported datasets:
  - UJIIndoorLoc (520 APs, 3 buildings, 5 floors)
  - UTSIndoorLoc (589 APs, 1 building, 16 floors)
  - SODIndoorLoc (52-175 APs, 1 building per site)
  - HDLC (59 APs, BLE+WIFI, layout change)
  - Tampere (992 APs, separate RSS/coord files)

All datasets normalized to common format:
  - RSSI: [-1, 1], invisible = -1.0 (clipped -110~-30 dBm)
  - Coords: per-building [x, y] normalized to [-1, 1]
  - Building/Floor: integer indices from 0
"""
import torch
import numpy as np
import pandas as pd
import os as _os


DATASET_REGISTRY = {
    'UJI': {
        'detect': lambda cols: 'LONGITUDE' in cols and 'WAP' in str(cols[0]).upper(),
        'ap_prefix': 'WAP',
        'col_x': 'LONGITUDE', 'col_y': 'LATITUDE', 'col_floor': 'FLOOR',
        'col_building': 'BUILDINGID', 'col_space': 'SPACEID',
        'col_user': 'USERID', 'col_phone': 'PHONEID', 'col_timestamp': 'TIMESTAMP',
        'rss_invisible': 100,
        'floor_offset': 0, 'building_offset': 0, 'space_offset': 1,
        'user_offset': 1, 'phone_offset': 1,
        'default_n_aps': 520, 'default_n_buildings': 3, 'default_n_floors': 5, 'default_n_spaces': 254,
    },
    'UTS': {
        'detect': lambda cols: 'Floor_ID' in cols and 'WAP' in str(cols[0]).upper(),
        'ap_prefix': 'WAP',
        'col_x': 'Pos_x', 'col_y': 'Pos_y', 'col_floor': 'Floor_ID',
        'col_building': 'Building_ID', 'col_space': None,
        'col_user': 'User_ID', 'col_phone': 'Phone_type', 'col_timestamp': 'Time',
        'rss_invisible': 100,
        'floor_offset': 3, 'building_offset': 0,
        'space_offset': 0, 'user_offset': 0, 'phone_offset': 0,
        'default_n_aps': 589, 'default_n_buildings': 1, 'default_n_floors': 16, 'default_n_spaces': 0,
    },
    'SOD': {
        'detect': lambda cols: 'MAC1' in cols or 'ECoord' in cols,
        'ap_prefix': 'MAC',
        'col_x': 'ECoord', 'col_y': 'NCoord', 'col_floor': 'FloorID',
        'col_building': 'BuildingID', 'col_space': 'SceneID',
        'col_user': 'UserID', 'col_phone': 'PhoneID', 'col_timestamp': 'SampleTimes',
        'rss_invisible': 100,
        'floor_offset': -1, 'building_offset': -1,
        'space_offset': 0, 'user_offset': 0, 'phone_offset': 0,
        'default_n_aps': 52, 'default_n_buildings': 1, 'default_n_floors': 3, 'default_n_spaces': 0,
    },
    'HDLC': {
        'detect': lambda cols: 'BLE 1' in cols and 'Floor' in cols and 'x' in cols and 'y' in cols,
        'ap_prefix': ('BLE', 'WAP', 'WIFI'),
        'col_x': 'x', 'col_y': 'y', 'col_floor': 'Floor',
        'col_building': None, 'col_space': None,
        'col_user': None, 'col_phone': None, 'col_timestamp': None,
        'rss_invisible': 100,
        'floor_offset': 0, 'building_offset': 0,
        'space_offset': 0, 'user_offset': 0, 'phone_offset': 0,
        'default_n_aps': 59, 'default_n_buildings': 1, 'default_n_floors': 3, 'default_n_spaces': 0,
    },
    'Tampere': {
        'detect': lambda cols: len(cols) > 100 and all(str(c).replace('.','').isdigit() for c in cols[:5]),
        'ap_prefix': None,
        'col_x': None, 'col_y': None, 'col_floor': None,
        'col_building': None, 'col_space': None,
        'col_user': None, 'col_phone': None, 'col_timestamp': None,
        'rss_invisible': 100,
        'floor_offset': 0, 'building_offset': 0,
        'space_offset': 0, 'user_offset': 0, 'phone_offset': 0,
        'default_n_aps': 992, 'default_n_buildings': 1, 'default_n_floors': 5, 'default_n_spaces': 0,
    },
}


def detect_dataset(df):
    cols = list(df.columns)
    for name, spec in DATASET_REGISTRY.items():
        if spec['detect'](cols):
            return name
    raise ValueError(f"Unknown dataset. Cols: {cols[:10]}...")


def load_dataset(csv_path, dataset_type='auto', config=None, device='cpu',
                 reference_stats=None):
    """Universal dataset loader for UJI/UTS/SOD/HDLC/Tampere.

    When ``reference_stats`` is provided, AP ordering and coordinate
    normalization are reused from the training set. Evaluation data must not
    fit its own coordinate bounds because that changes the target space seen
    by the trained model.
    """

    # ── Tampere: separate RSS/coord files ──
    if 'Tampere' in str(csv_path) or dataset_type == 'Tampere':
        dataset_type = 'Tampere'
        df = pd.read_csv(csv_path, header=None)
        data_dir = _os.path.dirname(csv_path)
        base_name = _os.path.basename(str(csv_path))
        if base_name.startswith('Test_'):
            split_prefix = 'Test'
            coord_file = _os.path.join(data_dir, 'Test_coordinates_21Aug17.csv')
        elif base_name.startswith('Training_'):
            split_prefix = 'Training'
            coord_file = _os.path.join(data_dir, 'Training_coordinates_21Aug17.csv')
        else:
            split_prefix = base_name.split('_', 1)[0]
            coord_file = str(csv_path).replace('_rss_', '_coordinates_')
        if not _os.path.exists(coord_file):
            raise FileNotFoundError(f"Tampere coordinate file not found: {coord_file}")
        dfc = pd.read_csv(coord_file, header=None)
        if len(dfc) != len(df):
            raise ValueError(
                f"Tampere RSS/coordinate row mismatch: {len(df)} vs {len(dfc)}")
        x = dfc.iloc[:, 0].values.astype(np.float32)
        y = dfc.iloc[:, 1].values.astype(np.float32)
        z = dfc.iloc[:, 2].values.astype(np.float32)
        floor = np.round(z / 3.7).astype(np.int64).clip(min=0)
        n_aps = df.shape[1]
        rss = df.fillna(100).replace(100, -110).values.astype(np.float32).clip(-110, -30)
        rss_norm = (rss + 110) / 80.0 * 2.0 - 1.0
        if reference_stats is None:
            bs = {0: {'x_min': float(x.min()), 'x_max': float(x.max()),
                      'y_min': float(y.min()), 'y_max': float(y.max())}}
        else:
            raw_bs = reference_stats['building_stats']
            ref = raw_bs.get(0, raw_bs.get('0'))
            if ref is None:
                raise ValueError("reference_stats has no building 0")
            bs = {0: dict(ref)}
        x_n = ((x - bs[0]['x_min']) /
               (bs[0]['x_max'] - bs[0]['x_min'] + 1e-8) * 2.0 - 1.0)
        y_n = ((y - bs[0]['y_min']) /
               (bs[0]['y_max'] - bs[0]['y_min'] + 1e-8) * 2.0 - 1.0)
        coords_n = np.stack([x_n, y_n], axis=1).astype(np.float32)
        device_file = _os.path.join(
            data_dir, f'{split_prefix}_device_21Aug17.csv')
        date_file = _os.path.join(
            data_dir, f'{split_prefix}_date_21Aug17.csv')
        if not _os.path.exists(device_file):
            raise FileNotFoundError(f"Tampere device file not found: {device_file}")
        if not _os.path.exists(date_file):
            raise FileNotFoundError(f"Tampere date file not found: {date_file}")
        device_names = (
            pd.read_csv(device_file, header=None).iloc[:, 0].astype(str).str.strip()
        )
        date_values = pd.read_csv(date_file, header=None).iloc[:, 0]
        if len(device_names) != len(df) or len(date_values) != len(df):
            raise ValueError("Tampere RSS/device/date row mismatch")
        if reference_stats is None:
            phone_mapping = {
                name: index
                for index, name in enumerate(sorted(device_names.unique()))
            }
            training_phone_names = list(sorted(device_names.unique()))
        else:
            raw_mapping = reference_stats.get('phone_mapping')
            if not isinstance(raw_mapping, dict) or not raw_mapping:
                raise ValueError(
                    "Tampere reference_stats has no training-derived phone mapping")
            phone_mapping = {
                str(name): int(identifier)
                for name, identifier in raw_mapping.items()
            }
            training_phone_names = list(
                reference_stats.get('training_phone_names', phone_mapping)
            )
            next_identifier = max(phone_mapping.values()) + 1
            for name in sorted(set(device_names) - set(phone_mapping)):
                phone_mapping[name] = next_identifier
                next_identifier += 1
        phone_ids = np.asarray(
            [phone_mapping[name] for name in device_names], dtype=np.int64)
        parsed_dates = pd.to_datetime(date_values, errors='coerce')
        if bool(parsed_dates.isna().any()):
            raise ValueError("Tampere date file contains an invalid timestamp")
        timestamps = (
            (parsed_dates - pd.Timestamp('1970-01-01'))
            // pd.Timedelta(seconds=1)
        ).to_numpy(dtype=np.int64, copy=False)
        stats = {'wap_cols': list(range(n_aps)), 'n_aps': n_aps,
                 'building_stats': bs, 'n_buildings': 1,
                 'n_floors': int(floor.max()) + 1,
                 'dataset_type': 'Tampere',
                 'phone_mapping': phone_mapping,
                 'training_phone_names': training_phone_names,
                 'observed_phone_names': list(sorted(device_names.unique()))}
        space = torch.zeros(len(df), dtype=torch.long)
        user = torch.zeros(len(df), dtype=torch.long)
        return (torch.tensor(rss_norm), torch.tensor(coords_n),
                torch.zeros(len(df), dtype=torch.long),
                torch.tensor(floor, dtype=torch.long), space, user,
                torch.tensor(phone_ids, dtype=torch.long),
                torch.tensor(timestamps, dtype=torch.long), stats)

    # ── Standard CSV datasets ──
    df = pd.read_csv(
        csv_path,
        usecols=lambda c: not str(c).startswith('Unnamed'),
        low_memory=False,
    )
    if dataset_type == 'auto':
        dataset_type = detect_dataset(df)
    spec = DATASET_REGISTRY[dataset_type]

    # Drop rows where coordinates are not numeric (HDLC has duplicate header row)
    if spec['col_x'] and spec['col_x'] in df.columns:
        df = df[pd.to_numeric(df[spec['col_x']], errors='coerce').notna()]
    df = df.reset_index(drop=True)

    # AP columns (handle tuple prefix for HDLC)
    ap_prefix = spec['ap_prefix']
    if reference_stats is not None and reference_stats.get('wap_cols'):
        wap_cols = list(reference_stats['wap_cols'])
    elif isinstance(ap_prefix, tuple):
        wap_cols = [c for c in df.columns
                    if any(str(c).upper().startswith(p.upper())
                           for p in ap_prefix)]
    else:
        wap_cols = [c for c in df.columns if str(c).upper().startswith(ap_prefix)]

    # RSSI processing (replace invisible BEFORE clip to preserve invalid values)
    rss_inv = spec['rss_invisible']
    rss_frame = df.reindex(columns=wap_cols, fill_value=rss_inv)
    rss = rss_frame.apply(
        pd.to_numeric, errors='coerce').fillna(rss_inv).values.astype(np.float32)
    rss[rss == rss_inv] = -110  # Must happen BEFORE clip
    rss = np.clip(rss, -110, -30)
    rss[rss == rss_inv] = -110
    rss_norm = (rss + 110) / 80.0 * 2.0 - 1.0

    # Coordinates
    if spec['col_x'] and spec['col_x'] in df.columns:
        x = df[spec['col_x']].values.astype(np.float32)
        y = df[spec['col_y']].values.astype(np.float32)
    else:
        x = np.arange(len(df), dtype=np.float32)
        y = np.zeros(len(df), dtype=np.float32)

    # Building
    if spec['col_building'] and spec['col_building'] in df.columns:
        raw_bldg = df[spec['col_building']].values
        if raw_bldg.dtype.kind in ('U', 'O', 'S'):
            names = sorted(set(raw_bldg))
            bldg = np.array([{n:i for i,n in enumerate(names)}[v] for v in raw_bldg], dtype=np.int64)
        else:
            bldg = (raw_bldg.astype(np.int64) + spec['building_offset']).clip(min=0)
    else:
        bldg = np.zeros(len(df), dtype=np.int64)

    # Floor
    if spec['col_floor'] and spec['col_floor'] in df.columns:
        floor = (df[spec['col_floor']].values.astype(np.int64) + spec['floor_offset']).clip(min=0)
    else:
        floor = np.zeros(len(df), dtype=np.int64)

    # Optional fields with safe defaults
    def _get_col(name, offset=0):
        if name and name in df.columns:
            v = df[name].values
            if v.dtype.kind in ('U', 'O', 'S'):
                names = sorted(set(v))
                return np.array([{n:i for i,n in enumerate(names)}[x] for x in v], dtype=np.int64)
            return (v.astype(np.int64) + offset).clip(min=0)
        return np.zeros(len(df), dtype=np.int64)

    spaceid = _get_col(spec['col_space'], spec['space_offset'])
    userid = _get_col(spec['col_user'], spec['user_offset'])
    phoneid = _get_col(spec['col_phone'], spec['phone_offset'])

    if spec['col_timestamp'] and spec['col_timestamp'] in df.columns:
        raw_ts = df[spec['col_timestamp']].values
        if raw_ts.dtype.kind in ('U', 'O', 'S'):
            # Historical loaders treated non-numeric timestamps as missing.
            # Keep that behavior, but use an exact integer representation so
            # downstream splitters never lose precision merely by loading.
            timestamp = np.zeros(len(df), dtype=np.int64)
        elif raw_ts.dtype.kind in ('i', 'u'):
            timestamp = raw_ts.astype(np.int64, copy=False)
        else:
            # Fractional timestamps are uncommon but valid.  Float64 retains
            # sub-second ordering without the ~128-second quantization that
            # float32 introduces for Unix timestamps around 1.37e9.
            timestamp = raw_ts.astype(np.float64)
    else:
        timestamp = np.zeros(len(df), dtype=np.int64)

    # Per-building normalization
    n_bldg = int(bldg.max()) + 1
    building_stats = {}
    x_norm = np.zeros_like(x, dtype=np.float32)
    y_norm = np.zeros_like(y, dtype=np.float32)
    for bi in range(n_bldg):
        bm = bldg == bi
        if bm.sum() == 0:
            continue
        x_b, y_b = x[bm], y[bm]
        if reference_stats is None:
            stats_b = {
                'x_min': float(x_b.min()), 'x_max': float(x_b.max()),
                'y_min': float(y_b.min()), 'y_max': float(y_b.max()),
            }
        else:
            raw_bs = reference_stats['building_stats']
            stats_b = raw_bs.get(bi, raw_bs.get(str(bi)))
            if stats_b is None:
                raise ValueError(
                    f"reference_stats has no coordinate bounds for building {bi}")
            stats_b = dict(stats_b)
        building_stats[bi] = stats_b
        dx = stats_b['x_max'] - stats_b['x_min']
        dy = stats_b['y_max'] - stats_b['y_min']
        x_norm[bm] = ((x_b - stats_b['x_min']) /
                      (dx if dx > 0 else 1) * 2.0 - 1.0)
        y_norm[bm] = ((y_b - stats_b['y_min']) /
                      (dy if dy > 0 else 1) * 2.0 - 1.0)

    coords_norm = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    all_stats = {'wap_cols': wap_cols, 'n_aps': len(wap_cols),
                 'building_stats': building_stats,
                 'n_buildings': n_bldg, 'n_floors': int(floor.max()) + 1,
                 'n_spaces': int(spaceid.max()) + 1 if spaceid.max() > 0 else 0,
                 'dataset_type': dataset_type}

    return (torch.tensor(rss_norm), torch.tensor(coords_norm),
            torch.tensor(bldg, dtype=torch.long), torch.tensor(floor, dtype=torch.long),
            torch.tensor(spaceid, dtype=torch.long), torch.tensor(userid, dtype=torch.long),
            torch.tensor(phoneid, dtype=torch.long), torch.tensor(timestamp), all_stats)


def denorm_coords(x_norm, y_norm, building, all_stats):
    """Convert normalized coords [-1,1] back to meters."""
    bs = all_stats['building_stats']
    x_m = np.zeros_like(x_norm, dtype=np.float32)
    y_m = np.zeros_like(y_norm, dtype=np.float32)
    for bi, stats in bs.items():
        bm = building == int(bi)
        if bm.sum() == 0:
            continue
        x_m[bm] = (x_norm[bm] + 1.0) / 2.0 * (stats['x_max'] - stats['x_min']) + stats['x_min']
        y_m[bm] = (y_norm[bm] + 1.0) / 2.0 * (stats['y_max'] - stats['y_min']) + stats['y_min']
    return x_m, y_m


