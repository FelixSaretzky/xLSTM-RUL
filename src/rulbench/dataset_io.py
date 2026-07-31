"""
Dataset persistence -- HDF5, streaming writes and lazy reads.

This module is the shared DATA LAYER: it owns the ``Unit`` record and the
generator reference grid, so that data generation never has to import the
model code (and therefore never pulls in torch).

Why HDF5 instead of pickle shards:
  * STREAMING WRITES  -- the generator never holds the whole dataset in RAM
    (previously: a list of all units, several GB at 200k units).
  * LAZY READS with random access -- the sampler only reads the slices it
    actually needs, not the entire dataset.
  * No pickle -> no code-execution risk, portable, self-describing.

LAYOUT ("packed", not ragged):
  Variable-length time series are stored back-to-back in one large array plus
  an offset index. A unit is therefore a slice -- O(1) random access, without
  creating one HDF5 group per unit (which would be metadata-slow at 1e5 units).

      /sensors     (sum_T, n_channels) float32   chunked
      /hi          (sum_T,)            float32   normalised state X'
      /rul         (sum_T,)            float32   supervised baseline only
      /offset      (N+1,)              int64     cumulative start indices
      /onset, /t_fail, /censored, /unit_id, /n_process        (N,)
      /mu_grid, /sigma_grid  (N, GEN_GRID_N) float32  operator curves
      /params      (N, 4)              float32   best linear fit (ablation)
      /shape_drift, /shape_diff  (N,)  strings   diagnostics
      attrs: config (JSON), n_units, gen_grid_n, n_sensors

COMPRESSION: sensor data is essentially noise and barely compresses losslessly
(gzip buys a few percent over lzf at noticeable CPU cost). Default is "lzf"
(fast, moderate). If space matters: float16 halves the size at ~1e-3 relative
precision -- lossy, but harmless for instance-normalised sensor data.
Deliberately optional, not the default.

Requires only numpy and h5py.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
import numpy as np
import h5py


# =====================================================================
# SHARED DATA STRUCTURES
# =====================================================================

GEN_GRID_N = 65                    # fine reference grid used by the generator


def gen_grid() -> np.ndarray:
    """Fixed fine grid on which the generator stores the SDE operators.
    The model interpolates from it onto its own (coarser) grid, so generator
    resolution and model resolution stay decoupled."""
    return np.linspace(0.0, 1.0, GEN_GRID_N, dtype=np.float64)


@dataclass
class Unit:
    """One simulated machine life, from start of observation to failure or
    censoring. Carries what the model sees (sensors) and what it is trained
    to infer (state trajectory and operator curves)."""
    sensors: np.ndarray            # (T, n_channels) process sensors + load channels
    hi: np.ndarray                 # (T,)  normalised latent state X', failure at 1
    onset: int                     # start of degradation (regime switch)
    t_fail: int                    # first-passage time; beyond T if censored
    rul: np.ndarray                # (T,)  piecewise-linear label, BASELINE only
    censored: bool
    unit_id: int = -1
    mu_grid: np.ndarray = None     # (GEN_GRID_N,) drift     mu(x'), reference load s=1
    sigma_grid: np.ndarray = None  # (GEN_GRID_N,) diffusion sigma(x')
    params: np.ndarray = None      # (4,) best LINEAR fit -- target of the ablation
    shapes: tuple = ()             # (drift_shape, diff_shape) for diagnostics
    n_process: int = 0             # number of PROCESS sensors; load channels follow


# =====================================================================
# WRITING (streaming)
# =====================================================================

class HDF5Writer:
    """Writes units one at a time -- constant memory footprint.

    Usage:
        with HDF5Writer("data/train.h5", cfg) as w:
            for _ in range(n):
                w.append(gen.sample_unit())
    """

    def __init__(self, path: str, cfg, n_sensors: int | None = None,
                 compression: str | None = "lzf", dtype=np.float32,
                 chunk_rows: int = 4096):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        self.dtype = dtype
        self.n_sensors = n_sensors if n_sensors is not None else cfg.n_sensors
        self.f = h5py.File(path, "w")
        self.f.attrs["config"] = json.dumps(asdict(cfg))
        self.f.attrs["gen_grid_n"] = GEN_GRID_N
        self.f.attrs["n_sensors"] = self.n_sensors

        kw = dict(compression=compression) if compression else {}
        mk = self.f.create_dataset
        # --- time-resolved (packed) ---
        self.ds_sensors = mk("sensors", shape=(0, self.n_sensors),
                             maxshape=(None, self.n_sensors), dtype=dtype,
                             chunks=(chunk_rows, self.n_sensors), **kw)
        self.ds_hi = mk("hi", shape=(0,), maxshape=(None,), dtype=np.float32,
                        chunks=(chunk_rows,), **kw)
        self.ds_rul = mk("rul", shape=(0,), maxshape=(None,), dtype=np.float32,
                         chunks=(chunk_rows,), **kw)
        # --- per unit ---
        self.ds_offset = mk("offset", shape=(1,), maxshape=(None,), dtype=np.int64,
                            chunks=(1024,))
        self.ds_offset[0] = 0
        self._scalar = {}
        for name, dt in [("onset", np.int32), ("t_fail", np.int32),
                         ("censored", np.bool_), ("unit_id", np.int64),
                         ("n_process", np.int32)]:
            self._scalar[name] = mk(name, shape=(0,), maxshape=(None,), dtype=dt,
                                    chunks=(1024,))
        self.ds_mu = mk("mu_grid", shape=(0, GEN_GRID_N), maxshape=(None, GEN_GRID_N),
                        dtype=np.float32, chunks=(256, GEN_GRID_N), **kw)
        self.ds_sg = mk("sigma_grid", shape=(0, GEN_GRID_N), maxshape=(None, GEN_GRID_N),
                        dtype=np.float32, chunks=(256, GEN_GRID_N), **kw)
        self.ds_par = mk("params", shape=(0, 4), maxshape=(None, 4),
                         dtype=np.float32, chunks=(1024, 4))
        st = h5py.string_dtype()
        self.ds_shd = mk("shape_drift", shape=(0,), maxshape=(None,), dtype=st,
                         chunks=(1024,))
        self.ds_shg = mk("shape_diff", shape=(0,), maxshape=(None,), dtype=st,
                         chunks=(1024,))
        self.n = 0
        self.total_t = 0

    def append(self, u: Unit):
        T = len(u.sensors)
        if u.sensors.shape[1] != self.n_sensors:
            raise ValueError(f"channel count mismatch: {u.sensors.shape[1]} "
                             f"!= {self.n_sensors}")
        a, b = self.total_t, self.total_t + T
        for ds, val in [(self.ds_sensors, u.sensors), (self.ds_hi, u.hi),
                        (self.ds_rul, u.rul)]:
            ds.resize(b, axis=0)
            ds[a:b] = val
        self.total_t = b

        i = self.n
        self.ds_offset.resize(i + 2, axis=0); self.ds_offset[i + 1] = b
        for name, val in [("onset", u.onset), ("t_fail", u.t_fail),
                          ("censored", u.censored), ("unit_id", u.unit_id),
                          ("n_process", u.n_process)]:
            ds = self._scalar[name]; ds.resize(i + 1, axis=0); ds[i] = val
        for ds, val in [(self.ds_mu, u.mu_grid), (self.ds_sg, u.sigma_grid),
                        (self.ds_par, u.params)]:
            ds.resize(i + 1, axis=0); ds[i] = val
        sd, sg = (u.shapes if len(u.shapes) == 2 else ("", ""))
        self.ds_shd.resize(i + 1, axis=0); self.ds_shd[i] = sd
        self.ds_shg.resize(i + 1, axis=0); self.ds_shg[i] = sg
        self.n += 1

    def close(self):
        self.f.attrs["n_units"] = self.n
        self.f.attrs["total_timesteps"] = self.total_t
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# =====================================================================
# READING
# =====================================================================

class InMemoryStore:
    """Trivial wrapper around a list of units -- same interface as the HDF5
    store, so samplers and scalers accept either."""

    def __init__(self, units: list[Unit]):
        self.units = units
        self.lengths = np.array([len(u.sensors) for u in units], dtype=np.int64)

    def __len__(self):
        return len(self.units)

    def __getitem__(self, i) -> Unit:
        return self.units[i]

    def close(self):
        pass


class HDF5UnitStore:
    """Lazy access. ``store[i]`` materialises EXACTLY one unit (tens of kB).

    ``lengths`` is available without touching the data (derived from the
    offset index), so filtering by sequence length needs no full scan.
    A small LRU cache absorbs repeated access to the same unit.
    """

    def __init__(self, path: str, cache_size: int = 512):
        self.path = path
        self.f = h5py.File(path, "r")
        self.offset = self.f["offset"][:]                  # (N+1,) klein
        self.lengths = np.diff(self.offset)
        self.n_sensors = int(self.f.attrs["n_sensors"])
        self.config = json.loads(self.f.attrs["config"])
        self._cache: dict[int, Unit] = {}
        self._order: list[int] = []
        self._cache_size = cache_size

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i) -> Unit:
        i = int(i)
        hit = self._cache.get(i)
        if hit is not None:
            return hit
        a, b = int(self.offset[i]), int(self.offset[i + 1])
        f = self.f
        u = Unit(sensors=np.asarray(f["sensors"][a:b], dtype=np.float32),
                 hi=np.asarray(f["hi"][a:b], dtype=np.float32),
                 onset=int(f["onset"][i]), t_fail=int(f["t_fail"][i]),
                 rul=np.asarray(f["rul"][a:b], dtype=np.float32),
                 censored=bool(f["censored"][i]), unit_id=int(f["unit_id"][i]),
                 mu_grid=np.asarray(f["mu_grid"][i], dtype=np.float32),
                 sigma_grid=np.asarray(f["sigma_grid"][i], dtype=np.float32),
                 params=np.asarray(f["params"][i], dtype=np.float32),
                 n_process=int(f["n_process"][i]),
                 shapes=(_s(f["shape_drift"][i]), _s(f["shape_diff"][i])))
        self._cache[i] = u
        self._order.append(i)
        if len(self._order) > self._cache_size:
            self._cache.pop(self._order.pop(0), None)
        return u

    def sample_units(self, n: int, seed: int = 0) -> list[Unit]:
        """Random subset -- for sanity checks and scaler fitting."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self), size=min(n, len(self)), replace=False)
        return [self[i] for i in idx]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _s(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def open_dataset(path: str, **kw):
    """Open a dataset. HDF5 file -> lazy store.
    Directory with meta.json -> legacy pickle shards (loaded into RAM)."""
    if os.path.isdir(path):
        return InMemoryStore(_load_pickle_shards(path))
    return HDF5UnitStore(path, **kw)


def _load_pickle_shards(out_dir: str) -> list[Unit]:
    """Backwards compatibility for old shard directories."""
    import pickle
    with open(os.path.join(out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    units = []
    for name in meta["shards"]:
        with open(os.path.join(out_dir, name), "rb") as fh:
            units.extend(pickle.load(fh))
    return units


def dataset_info(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return dict(n_units=int(f.attrs["n_units"]),
                    total_timesteps=int(f.attrs["total_timesteps"]),
                    n_sensors=int(f.attrs["n_sensors"]),
                    size_mb=os.path.getsize(path) / 1e6,
                    config=json.loads(f.attrs["config"]))