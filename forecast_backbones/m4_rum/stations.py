"""M4-Rum station keys and capacity sources; legacy table behavior stays default."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.forecasting.stations import (
    StationMeta,
    norm_station_name,
    station_cap_mw,
)


def numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )


def first_capacity(value):
    """A scalar or 192-point list: use exactly the first value, never a mean."""
    values = numpy(value)
    if values.ndim == 0:
        cap = float(values)
    elif values.ndim == 1 and len(values) in (1, 192):
        cap = float(values[0])
    else:
        raise ValueError("Capacity must be a scalar or a 192-point list")
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Capacity first value must be finite and positive")
    return cap


def stable_station_id(name):
    # Deterministic across processes, CSV row order, subsets and new stations.
    # Keep IDs exactly representable in float64 for existing report machinery.
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (1 << 52) + (int.from_bytes(digest[:8], "big") & ((1 << 52) - 1))


class RumStationMeta(StationMeta):
    def __init__(self, dataset):
        self.dataset = dataset
        self.from_data = dataset["station_meta"] is None
        if self.from_data:
            self._rows, self._by_name = {}, {}
            return
        path = Path(dataset["station_meta"])
        if not path.exists() and not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.is_file():
            raise FileNotFoundError(f"station_meta {path} does not exist")
        try:
            pd.read_csv(path, nrows=0, encoding="utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            encoding = "gbk"
        columns = pd.read_csv(path, nrows=0, encoding=encoding).columns
        mapping = self.detect_cols(columns, dataset["station_meta_cols"]) or {}
        name_col = mapping.get("station") or "station"
        frame = pd.read_csv(path, encoding=encoding, dtype={name_col: str})
        frame.columns = frame.columns.str.strip()
        id_col = mapping.get("station_id") or "station_id"
        # station + capacity is sufficient; coordinate fields are unused by Rum.
        if id_col not in frame:
            if name_col not in frame:
                raise ValueError("Station CSV requires a station name column")
            names = frame[name_col].map(norm_station_name)
            if names.isna().any():
                raise ValueError("Station CSV contains empty station names")
            frame[id_col] = names.map(stable_station_id)
        mapping = dict(mapping)
        mapping["station_id"] = id_col
        for key in self.OPTIONAL:
            if mapping.get(key, key) not in frame:
                mapping[key] = None
        super().__init__(frame, mapping)

    def _register(self, name, sid):
        existing = self._rows.get(sid)
        if existing is not None and existing["station"] != name:
            raise ValueError(f"Station ID collision: {name!r}, {existing['station']!r}")
        if name in self._by_name and self._by_name[name] != sid:
            raise ValueError(f"Conflicting IDs for station {name!r}")
        self._by_name[name] = sid
        self._rows[sid] = {
            "station_id": sid,
            "station": name,
            "cap_mw": np.nan,
            "lat": np.nan,
            "lon": np.nan,
            "alt_m": 0.0,
        }

    def resolve(self, values):
        if not self.from_data and self.dataset["station_key"] is None:
            return super().resolve(values)
        mapping, unknown = {}, []
        for value in values:
            name = norm_station_name(value)
            if name is None:
                unknown.append("<empty station>")
            elif self.from_data:
                sid = stable_station_id(name)
                self._register(name, sid)
                mapping[name] = sid
            elif name in self._by_name:
                mapping[name] = self._by_name[name]
            else:
                unknown.append(name)
        return mapping, unknown, []

    def identity(self, batch):
        ds = self.dataset
        key, id_col = ds["station_key"], ds["station_id_col"]
        if key is not None:
            if key not in batch:
                raise ValueError(f"Missing configured station key {key!r}")
            names = numpy(batch[key]).reshape(-1)
        elif id_col in batch:
            raw = numpy(batch[id_col]).reshape(-1)
            if raw.dtype.kind in "iuf":
                if (
                    not np.isfinite(raw).all()
                    or not np.equal(raw, raw.astype(np.int64)).all()
                ):
                    raise ValueError("station_id must contain integers")
                ids = raw.astype(np.int64)
                if self.from_data:
                    names = numpy(batch.get("station", ids)).reshape(-1)
                    for sid, name in zip(ids, names):
                        name = norm_station_name(name)
                        if name is None:
                            raise ValueError("Empty station name")
                        self._register(name, int(sid))
                return ids
            # Preserve the legacy loader's textual integer-ID convention.
            names = np.array([norm_station_name(v) for v in raw], dtype=object)
            numeric = pd.to_numeric(pd.Series(names), errors="coerce")
            if numeric.notna().all():
                return self.identity({id_col: numeric.to_numpy()})
        elif "station" in batch:
            names = numpy(batch["station"]).reshape(-1)
        else:
            raise ValueError("station_id or station is required")
        mapping, unknown, conflicts = self.resolve(names)
        if unknown or conflicts:
            raise ValueError(
                f"Unknown/conflicting station names: {unknown}, {conflicts}"
            )
        return np.array([mapping[norm_station_name(n)] for n in names], dtype=np.int64)

    def capacities(self, batch, ids):
        if not self.from_data:
            return np.array([station_cap_mw(self, sid) for sid in ids], np.float32)
        col = self.dataset["capacity_col"]
        if col not in batch:
            raise ValueError(f"Missing configured capacity column {col!r}")
        values = batch[col]
        if len(values) != len(ids):
            raise ValueError("Capacity and station batch sizes differ")
        return np.array([first_capacity(v) for v in values], np.float32)

    def prepare_frame(self, frame):
        frame = frame.copy()
        ids = self.identity(frame)
        frame[self.dataset["station_id_col"]] = ids
        if self.from_data:
            if "station" not in frame:
                frame["station"] = [self.name(sid) for sid in ids]
            col = self.dataset["capacity_col"]
            frame[col] = self.capacities(frame, ids)
        return frame

    def prepare_parquets(self, paths, cfg, scratch_dir, reuse=False):
        if not self.from_data and self.dataset["station_key"] is None:
            return paths
        scratch = Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        prepared = []
        for i, source in enumerate(paths):
            digest = hashlib.sha256(str(Path(source).resolve()).encode()).hexdigest()[
                :16
            ]
            target = scratch / f"rum_{i}_{digest}.parquet"
            if not reuse or not target.exists():
                self.prepare_frame(pd.read_parquet(source)).to_parquet(
                    target, index=False
                )
            prepared.append(str(target))
        return prepared
