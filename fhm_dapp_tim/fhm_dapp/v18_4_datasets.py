"""V18.4 canonical adapters for UJI, UTS, Tampere and TUJI1.

This module is deliberately independent of the legacy data.py loader.  Old
experiments therefore keep their historical preprocessing unchanged.

Canonical contract
------------------
* rss_dbm: float32 [rows, APs], NaN means AP not observed.
* coords_m: float32 [rows, 2] in the dataset's metric/local coordinate system.
* building/floor/device/user: contiguous training-derived integer IDs; -1 is
  reserved for missing or unseen evaluation categories.
* timestamp_s: float64 seconds where available, NaN otherwise.
* AP ordering and categorical mappings are always learned from training data
  and reused for evaluation.  Evaluation files never expand the feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RSS_MISSING_VALUE = 100.0
RSS_CLIP_MIN_DBM = -110.0
RSS_CLIP_MAX_DBM = -30.0


@dataclass(frozen=True)
class DatasetReference:
    dataset: str
    ap_ids: tuple[str, ...]
    building_mapping: dict[str, int]
    floor_mapping: dict[str, int]
    device_mapping: dict[str, int]
    user_mapping: dict[str, int]
    extra: dict[str, object]


@dataclass(frozen=True)
class CanonicalFingerprintData:
    dataset: str
    split: str
    rss_dbm: np.ndarray
    coords_m: np.ndarray
    building: np.ndarray
    floor: np.ndarray
    device: np.ndarray
    user: np.ndarray
    timestamp_s: np.ndarray
    ap_ids: tuple[str, ...]
    metadata: dict[str, object]

    def validate(self) -> None:
        n = int(self.rss_dbm.shape[0])
        if self.rss_dbm.ndim != 2:
            raise ValueError("rss_dbm must be [rows, APs]")
        if self.rss_dbm.shape[1] != len(self.ap_ids):
            raise ValueError("AP dictionary width mismatch")
        if len(set(self.ap_ids)) != len(self.ap_ids):
            raise ValueError("AP identifiers must be unique")
        if self.coords_m.shape != (n, 2):
            raise ValueError("coords_m must be [rows,2]")
        for name, value in (
            ("building", self.building),
            ("floor", self.floor),
            ("device", self.device),
            ("user", self.user),
            ("timestamp_s", self.timestamp_s),
        ):
            if np.asarray(value).shape != (n,):
                raise ValueError(f"{name} must contain one value per row")
        if not np.isfinite(self.coords_m).all():
            raise ValueError("coordinates must be finite")
        observed = self.rss_dbm[np.isfinite(self.rss_dbm)]
        if observed.size and (
            float(observed.max()) > 0.0 or float(observed.min()) < -150.0
        ):
            raise ValueError("observed RSS values are outside plausible dBm range")
        if np.any(self.building < -1) or np.any(self.floor < -1):
            raise ValueError("building/floor IDs must be >=-1")

    def model_rss(self) -> np.ndarray:
        """Return the historical model scale [-1,1], missing exactly -1."""

        out = np.full(self.rss_dbm.shape, -1.0, dtype=np.float32)
        visible = np.isfinite(self.rss_dbm)
        clipped = np.clip(
            self.rss_dbm[visible],
            RSS_CLIP_MIN_DBM,
            RSS_CLIP_MAX_DBM,
        )
        out[visible] = (
            (clipped - RSS_CLIP_MIN_DBM)
            / (RSS_CLIP_MAX_DBM - RSS_CLIP_MIN_DBM)
            * 2.0
            - 1.0
        ).astype(np.float32)
        return out


def _mapping(values: Iterable[object]) -> dict[str, int]:
    keys = sorted({str(value) for value in values})
    return {key: index for index, key in enumerate(keys)}


def _encode(
    values: Sequence[object],
    mapping: dict[str, int],
    *,
    allow_unseen: bool,
) -> tuple[np.ndarray, list[str]]:
    encoded = np.empty(len(values), dtype=np.int64)
    unseen: set[str] = set()
    for index, value in enumerate(values):
        key = str(value)
        if key in mapping:
            encoded[index] = int(mapping[key])
        elif allow_unseen:
            encoded[index] = -1
            unseen.add(key)
        else:
            raise ValueError(f"unseen training category: {key}")
    return encoded, sorted(unseen)


def _dense_rss(
    frame: pd.DataFrame,
    ap_ids: Sequence[str],
    *,
    missing_value: float = RSS_MISSING_VALUE,
) -> np.ndarray:
    rss = frame.reindex(columns=list(ap_ids), fill_value=missing_value)
    rss = rss.apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32, copy=True
    )
    missing = (~np.isfinite(rss)) | np.isclose(rss, float(missing_value))
    rss[missing] = np.nan
    return rss


def _timestamp_seconds(values: Sequence[object]) -> np.ndarray:
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float64)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    result = np.full(len(series), np.nan, dtype=np.float64)
    valid = parsed.notna().to_numpy()
    if valid.any():
        result[valid] = (
            parsed[valid].astype("int64").to_numpy(dtype=np.float64)
            / 1_000_000_000.0
        )
    return result


def _coordinate_group_count(data: CanonicalFingerprintData) -> int:
    keys = {
        (
            int(b),
            int(f),
            round(float(x), 4),
            round(float(y), 4),
        )
        for (x, y), b, f in zip(
            data.coords_m,
            data.building,
            data.floor,
        )
    }
    return len(keys)


def dataset_summary(data: CanonicalFingerprintData) -> dict[str, object]:
    data.validate()
    observed = np.isfinite(data.rss_dbm)
    visible_per_scan = observed.sum(axis=1)
    summary = {
        "dataset": data.dataset,
        "split": data.split,
        "rows": int(len(data.coords_m)),
        "aps": int(len(data.ap_ids)),
        "coordinate_groups": int(_coordinate_group_count(data)),
        "building_ids": sorted(map(int, np.unique(data.building))),
        "floor_ids": sorted(map(int, np.unique(data.floor))),
        "device_ids": sorted(map(int, np.unique(data.device))),
        "user_ids": sorted(map(int, np.unique(data.user))),
        "observed_fraction": float(observed.mean()),
        "visible_ap_mean": float(visible_per_scan.mean()),
        "visible_ap_min": int(visible_per_scan.min()),
        "visible_ap_max": int(visible_per_scan.max()),
        "coordinate_min": data.coords_m.min(axis=0).astype(float).tolist(),
        "coordinate_max": data.coords_m.max(axis=0).astype(float).tolist(),
        "metadata": data.metadata,
    }
    return summary


def _reference(
    *,
    dataset: str,
    ap_ids: Sequence[str],
    building_values: Sequence[object],
    floor_values: Sequence[object],
    device_values: Sequence[object],
    user_values: Sequence[object],
    extra: dict[str, object] | None = None,
) -> DatasetReference:
    return DatasetReference(
        dataset=dataset,
        ap_ids=tuple(map(str, ap_ids)),
        building_mapping=_mapping(building_values),
        floor_mapping=_mapping(floor_values),
        device_mapping=_mapping(device_values),
        user_mapping=_mapping(user_values),
        extra=dict(extra or {}),
    )


def _load_standard_csv(
    path: Path,
    *,
    dataset: str,
    split: str,
    x_col: str,
    y_col: str,
    building_col: str,
    floor_col: str,
    device_col: str,
    user_col: str,
    timestamp_col: str,
    reference: DatasetReference | None,
    ap_prefix: str = "WAP",
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    frame = pd.read_csv(path, low_memory=False)
    raw_ap_ids = [
        str(column)
        for column in frame.columns
        if str(column).upper().startswith(ap_prefix.upper())
    ]
    if reference is None:
        ref = _reference(
            dataset=dataset,
            ap_ids=raw_ap_ids,
            building_values=frame[building_col].astype(str).tolist(),
            floor_values=frame[floor_col].astype(str).tolist(),
            device_values=frame[device_col].astype(str).tolist(),
            user_values=frame[user_col].astype(str).tolist(),
            extra={"training_file": str(path.relative_to(REPO_ROOT))},
        )
        allow_unseen = False
    else:
        if reference.dataset != dataset:
            raise ValueError("reference belongs to a different dataset")
        ref = reference
        allow_unseen = True

    rss_dbm = _dense_rss(frame, ref.ap_ids)
    building, unseen_building = _encode(
        frame[building_col].astype(str).tolist(),
        ref.building_mapping,
        allow_unseen=allow_unseen,
    )
    floor, unseen_floor = _encode(
        frame[floor_col].astype(str).tolist(),
        ref.floor_mapping,
        allow_unseen=allow_unseen,
    )
    device, unseen_device = _encode(
        frame[device_col].astype(str).tolist(),
        ref.device_mapping,
        allow_unseen=True,
    )
    user, unseen_user = _encode(
        frame[user_col].astype(str).tolist(),
        ref.user_mapping,
        allow_unseen=True,
    )
    coords = frame[[x_col, y_col]].to_numpy(dtype=np.float32)
    timestamp = _timestamp_seconds(frame[timestamp_col].tolist())

    extra_ap_columns = sorted(set(raw_ap_ids) - set(ref.ap_ids))
    missing_training_ap_columns = sorted(set(ref.ap_ids) - set(raw_ap_ids))
    data = CanonicalFingerprintData(
        dataset=dataset,
        split=split,
        rss_dbm=rss_dbm,
        coords_m=coords,
        building=building,
        floor=floor,
        device=device,
        user=user,
        timestamp_s=timestamp,
        ap_ids=ref.ap_ids,
        metadata={
            "source_file": str(path.relative_to(REPO_ROOT)),
            "extra_evaluation_aps_ignored": extra_ap_columns,
            "training_aps_absent_from_split": missing_training_ap_columns,
            "unseen_buildings": unseen_building,
            "unseen_floors": unseen_floor,
            "unseen_devices": unseen_device,
            "unseen_users": unseen_user,
        },
    )
    data.validate()
    return data, ref


def load_uji(
    split: str,
    *,
    reference: DatasetReference | None = None,
    data_root: Path = DATA_ROOT,
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    if split == "train":
        path = data_root / "UJIndoorLoc" / "trainingData.csv"
        if reference is not None:
            raise ValueError("UJI train must fit its own reference")
    elif split in {"official_validation", "preexposed_official"}:
        path = data_root / "UJIndoorLoc" / "validationData.csv"
        if reference is None:
            raise ValueError("UJI validation requires training reference")
        split = "preexposed_official"
    else:
        raise ValueError(f"unknown UJI split: {split}")
    return _load_standard_csv(
        path,
        dataset="UJIIndoorLoc",
        split=split,
        x_col="LONGITUDE",
        y_col="LATITUDE",
        building_col="BUILDINGID",
        floor_col="FLOOR",
        device_col="PHONEID",
        user_col="USERID",
        timestamp_col="TIMESTAMP",
        reference=reference,
    )


def load_uts(
    split: str,
    *,
    reference: DatasetReference | None = None,
    data_root: Path = DATA_ROOT,
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    if split == "train":
        path = data_root / "UTSIndoorLoc" / "UTS_training.csv"
        if reference is not None:
            raise ValueError("UTS train must fit its own reference")
    elif split == "test":
        path = data_root / "UTSIndoorLoc" / "UTS_test.csv"
        if reference is None:
            raise ValueError("UTS test requires training reference")
    else:
        raise ValueError(f"unknown UTS split: {split}")
    return _load_standard_csv(
        path,
        dataset="UTSIndoorLoc",
        split=split,
        x_col="Pos_x",
        y_col="Pos_y",
        building_col="Building_ID",
        floor_col="Floor_ID",
        device_col="Phone_type",
        user_col="User_ID",
        timestamp_col="Time",
        reference=reference,
    )


def load_tampere(
    split: str,
    *,
    reference: DatasetReference | None = None,
    data_root: Path = DATA_ROOT,
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    directory = data_root / "Tampere"
    if split == "train":
        rss_path = directory / "Training_rss.csv"
        coord_path = directory / "Training_coordinates.csv"
        if reference is not None:
            raise ValueError("Tampere train must fit its own reference")
    elif split == "test":
        rss_path = directory / "Test_rss.csv"
        coord_path = directory / "Test_coordinates.csv"
        if reference is None:
            raise ValueError("Tampere test requires training reference")
    else:
        raise ValueError(f"unknown Tampere split: {split}")

    rss_frame = pd.read_csv(rss_path, header=None)
    coord_frame = pd.read_csv(coord_path, header=None)
    if len(rss_frame) != len(coord_frame):
        raise ValueError("Tampere RSS/coordinate row mismatch")
    if coord_frame.shape[1] != 3:
        raise ValueError("Tampere coordinates must have x,y,z columns")

    raw_ap_ids = tuple(f"AP{index:04d}" for index in range(rss_frame.shape[1]))
    z_values = coord_frame.iloc[:, 2].to_numpy(dtype=np.float64)
    if reference is None:
        ref = _reference(
            dataset="Tampere",
            ap_ids=raw_ap_ids,
            building_values=["0"] * len(rss_frame),
            floor_values=[repr(float(value)) for value in z_values],
            device_values=["unknown"] * len(rss_frame),
            user_values=["unknown"] * len(rss_frame),
            extra={
                "floor_z_values_m": sorted(map(float, np.unique(z_values))),
                "training_rss_file": str(rss_path.relative_to(REPO_ROOT)),
                "training_coordinate_file": str(
                    coord_path.relative_to(REPO_ROOT)
                ),
            },
        )
    else:
        if reference.dataset != "Tampere":
            raise ValueError("reference belongs to a different dataset")
        ref = reference
        if len(ref.ap_ids) != rss_frame.shape[1]:
            raise ValueError("Tampere AP width differs from training")

    raw_rss = rss_frame.to_numpy(dtype=np.float32, copy=True)
    raw_rss[
        (~np.isfinite(raw_rss))
        | np.isclose(raw_rss, RSS_MISSING_VALUE)
    ] = np.nan
    floor, unseen_floor = _encode(
        [repr(float(value)) for value in z_values],
        ref.floor_mapping,
        allow_unseen=split != "train",
    )
    building = np.zeros(len(rss_frame), dtype=np.int64)
    device = np.full(len(rss_frame), -1, dtype=np.int64)
    user = np.full(len(rss_frame), -1, dtype=np.int64)
    timestamp = np.full(len(rss_frame), np.nan, dtype=np.float64)
    coords = coord_frame.iloc[:, :2].to_numpy(dtype=np.float32)

    data = CanonicalFingerprintData(
        dataset="Tampere",
        split=split,
        rss_dbm=raw_rss,
        coords_m=coords,
        building=building,
        floor=floor,
        device=device,
        user=user,
        timestamp_s=timestamp,
        ap_ids=ref.ap_ids,
        metadata={
            "rss_file": str(rss_path.relative_to(REPO_ROOT)),
            "coordinate_file": str(coord_path.relative_to(REPO_ROOT)),
            "unseen_floor_heights": unseen_floor,
            "floor_height_mapping": dict(ref.floor_mapping),
        },
    )
    data.validate()
    return data, ref


TUJI_DEVICE_FILES = {
    "A12": "Formatted_data_A12.mat",
    "POCO": "Formatted_data_POCO.mat",
    "S20": "Formatted_data_S20.mat",
    "S7": "Formatted_data_S7.mat",
    "tabS7": "Formatted_data_tabS7.mat",
}


def _tuji_scans(path: Path, split: str) -> list[object]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = "results_train" if split == "train" else "results_test"
    result = payload[key]
    scans = np.asarray(result.Wifi, dtype=object).reshape(-1).tolist()
    return scans


def _tuji_measurements(scan: object) -> np.ndarray:
    measured = np.asarray(scan.measured, dtype=np.float64)
    if measured.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if measured.ndim == 1:
        if measured.shape[0] != 2:
            raise ValueError("TUJI measured vector must contain AP ID and RSS")
        measured = measured.reshape(1, 2)
    if measured.ndim != 2 or measured.shape[1] != 2:
        raise ValueError("TUJI measured must be [visible_APs,2]")
    return measured


def _tuji_ap_id(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("TUJI AP identifier must be finite")
    integer = int(round(float(value)))
    if abs(float(value) - integer) > 0.25:
        raise ValueError("TUJI AP identifier is not integer-valued")
    return f"MAC{integer:d}"


def _tuji_collect(
    directory: Path,
    split: str,
    devices: Sequence[str],
) -> tuple[list[tuple[str, object]], set[str]]:
    rows: list[tuple[str, object]] = []
    ap_ids: set[str] = set()
    for device_name in devices:
        path = directory / TUJI_DEVICE_FILES[device_name]
        for scan in _tuji_scans(path, split):
            rows.append((device_name, scan))
            for ap_value, _rss in _tuji_measurements(scan):
                ap_ids.add(_tuji_ap_id(ap_value))
    return rows, ap_ids


def _tuji_ap_sort_key(value: str) -> int:
    return int(value.removeprefix("MAC"))


def load_tuji1(
    split: str,
    *,
    devices: Sequence[str] | None = None,
    reference: DatasetReference | None = None,
    data_root: Path = DATA_ROOT,
) -> tuple[CanonicalFingerprintData, DatasetReference]:
    if split not in {"train", "test"}:
        raise ValueError(f"unknown TUJI1 split: {split}")
    directory = data_root / "TUJI1"
    selected_devices = tuple(
        devices if devices is not None else TUJI_DEVICE_FILES.keys()
    )
    unknown = sorted(set(selected_devices) - set(TUJI_DEVICE_FILES))
    if unknown:
        raise ValueError(f"unknown TUJI1 devices: {unknown}")

    rows, observed_aps = _tuji_collect(
        directory,
        split,
        selected_devices,
    )
    if reference is None:
        if split != "train":
            raise ValueError("TUJI1 test requires training reference")
        ap_ids = tuple(sorted(observed_aps, key=_tuji_ap_sort_key))
        ref = _reference(
            dataset="TUJI1",
            ap_ids=ap_ids,
            building_values=["0"] * len(rows),
            floor_values=["0"] * len(rows),
            device_values=list(selected_devices),
            user_values=["unknown"] * len(rows),
            extra={
                "training_devices": list(selected_devices),
                "device_files": {
                    name: TUJI_DEVICE_FILES[name]
                    for name in selected_devices
                },
            },
        )
    else:
        if reference.dataset != "TUJI1":
            raise ValueError("reference belongs to a different dataset")
        ref = reference

    ap_index = {name: index for index, name in enumerate(ref.ap_ids)}
    rss = np.full((len(rows), len(ref.ap_ids)), np.nan, dtype=np.float32)
    coords = np.empty((len(rows), 2), dtype=np.float32)
    device_names = []
    ignored_observations = 0
    ignored_ap_ids: set[str] = set()
    duplicate_observations = 0

    for row_index, (device_name, scan) in enumerate(rows):
        coords[row_index] = np.asarray(scan.coords, dtype=np.float32).reshape(2)
        device_names.append(device_name)
        seen_in_row: set[int] = set()
        for ap_value, rssi in _tuji_measurements(scan):
            ap_name = _tuji_ap_id(ap_value)
            column = ap_index.get(ap_name)
            if column is None:
                ignored_observations += 1
                ignored_ap_ids.add(ap_name)
                continue
            if column in seen_in_row:
                duplicate_observations += 1
                rss[row_index, column] = max(
                    float(rss[row_index, column]),
                    float(rssi),
                )
            else:
                rss[row_index, column] = float(rssi)
                seen_in_row.add(column)

    device, unseen_devices = _encode(
        device_names,
        ref.device_mapping,
        allow_unseen=split != "train",
    )
    data = CanonicalFingerprintData(
        dataset="TUJI1",
        split=split,
        rss_dbm=rss,
        coords_m=coords,
        building=np.zeros(len(rows), dtype=np.int64),
        floor=np.zeros(len(rows), dtype=np.int64),
        device=device,
        user=np.full(len(rows), -1, dtype=np.int64),
        timestamp_s=np.full(len(rows), np.nan, dtype=np.float64),
        ap_ids=ref.ap_ids,
        metadata={
            "devices": list(selected_devices),
            "unseen_devices": unseen_devices,
            "ignored_evaluation_ap_ids": sorted(
                ignored_ap_ids,
                key=_tuji_ap_sort_key,
            ),
            "ignored_evaluation_observations": int(ignored_observations),
            "duplicate_observations_combined_by_max_rss": int(
                duplicate_observations
            ),
        },
    )
    data.validate()
    return data, ref


def load_all_training_summaries(
    data_root: Path = DATA_ROOT,
) -> dict[str, object]:
    uji_train, uji_ref = load_uji("train", data_root=data_root)
    uji_eval, _ = load_uji(
        "preexposed_official",
        reference=uji_ref,
        data_root=data_root,
    )
    uts_train, uts_ref = load_uts("train", data_root=data_root)
    uts_test, _ = load_uts("test", reference=uts_ref, data_root=data_root)
    tampere_train, tampere_ref = load_tampere("train", data_root=data_root)
    tampere_test, _ = load_tampere(
        "test",
        reference=tampere_ref,
        data_root=data_root,
    )
    tuji_train, tuji_ref = load_tuji1("train", data_root=data_root)
    tuji_test, _ = load_tuji1(
        "test",
        reference=tuji_ref,
        data_root=data_root,
    )
    return {
        "UJIIndoorLoc": {
            "train": dataset_summary(uji_train),
            "preexposed_official": dataset_summary(uji_eval),
        },
        "UTSIndoorLoc": {
            "train": dataset_summary(uts_train),
            "test": dataset_summary(uts_test),
        },
        "Tampere": {
            "train": dataset_summary(tampere_train),
            "test": dataset_summary(tampere_test),
        },
        "TUJI1": {
            "train": dataset_summary(tuji_train),
            "test": dataset_summary(tuji_test),
        },
    }


def main() -> None:
    summaries = load_all_training_summaries()
    print(json.dumps(summaries, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
