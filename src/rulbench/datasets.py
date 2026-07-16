"""Uniform query API over the six-benchmark suite — one file, all datasets.

    units, meta = load(name, fd=1, split="dev")

Every dataset is returned in one canonical structure:

    Unit.features : np.ndarray (T, L, C) - T degradation steps per unit, each an
                    L-sample signal snapshot over C channels
    Unit.rul      : np.ndarray (T,)      - remaining useful life at each step,
                    in the dataset's native time unit (see meta["time_unit"])
    Unit.censored : True when the run ends by observation cutoff instead of
                    failure - rul then counts down to the cutoff, so it is a
                    lower bound on the true RUL, not the true RUL

C-MAPSS, N-CMAPSS, FEMTO, and XJTU-SY delegate to the rul-datasets readers
(auto-download + cache under ~/.rul-datasets; first N-CMAPSS load pulls ~16 GB).
Milling is parsed directly from NASA's mill.mat because PyPHM's loader discards
the continuous VB wear target. IMS ships no labels of any kind and raises until
a labeling convention is adopted. PHME20 (an extra loader, not part of the
benchmark suite) is parsed from the legacy challenge zips into ~/.rulbench.

Semantics note: for the bearing sets and milling, one step is a physical
acquisition snapshot (burst/cut); for C-MAPSS and N-CMAPSS it is a sliding or
per-cycle window over consecutive cycles, per the rul-datasets convention.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS = ("cmapss", "ncmapss", "femto", "xjtu", "milling", "phme20", "ims")


@dataclass(eq=False)
class Unit:
    dataset: str
    unit_id: str
    features: np.ndarray  # (T, L, C)
    rul: np.ndarray  # (T,)
    censored: bool = False  # True: rul counts down to observation cutoff, not failure


_META = {
    "cmapss": {
        "time_unit": "flight cycle",
        "channels": "14 of 21 sensors (Ragab selection)",
        "rul_cap": 125,
        "step": "sliding window over cycles (stride 1); test split = last window only",
    },
    "ncmapss": {
        "time_unit": "flight cycle",
        "channels": "4 operating conditions + 14 physical + 14 virtual sensors",
        "rul_cap": 65,
        "step": "one window per flight cycle (padded/cropped)",
    },
    "femto": {
        "time_unit": "burst (~10 s apart)",
        "channels": "2 accelerometer axes @ 25.6 kHz",
        "rul_cap": None,
        "step": "one 0.1 s acquisition burst (2560 samples)",
    },
    "xjtu": {
        "time_unit": "burst (~1 min apart)",
        "channels": "2 accelerometer axes @ 25.6 kHz",
        "rul_cap": None,
        "step": "one 1.28 s acquisition burst (32768 samples)",
    },
}


def _from_rul_datasets(reader, name: str, split: str) -> list[Unit]:
    reader.prepare_data()
    features, targets = reader.load_split(split)
    return [
        Unit(name, f"{name}-fd{reader.fd}-{split}-{i:03d}", f, np.asarray(t))
        for i, (f, t) in enumerate(zip(features, targets))
    ]


def load(name: str, fd: int = 1, split: str = "dev", **kwargs):
    """Load one benchmark in the canonical (units, meta) form.

    fd selects the sub-dataset where one exists (C-MAPSS FD1-4, N-CMAPSS 1-7,
    FEMTO/XJTU-SY condition 1-3, PHME20 particle size 1-2); split is
    dev/val/test for the rul-datasets benchmarks, dev/val for PHME20. Milling
    has neither and rejects non-default fd/split. Extra kwargs go to the
    underlying rul-datasets reader (max_rul, window_size, feature_select,
    run_split_dist, ...) or, for milling and phme20, include_censored.
    """
    if name == "cmapss":
        from rul_datasets.reader import CmapssReader

        reader = CmapssReader(fd, **kwargs)
    elif name == "ncmapss":
        from rul_datasets.reader import NCmapssReader

        reader = NCmapssReader(fd, **kwargs)
    elif name == "femto":
        from rul_datasets.reader import FemtoReader

        reader = FemtoReader(fd, **kwargs)
    elif name == "xjtu":
        from rul_datasets.reader import XjtuSyReader

        reader = XjtuSyReader(fd, **kwargs)
    elif name == "milling":
        if fd != 1 or split != "dev":
            raise ValueError(
                "milling has no sub-datasets or splits; call load('milling') "
                "without fd/split"
            )
        return _load_milling(**kwargs)
    elif name == "phme20":
        return _load_phme20(fd=fd, split=split, **kwargs)
    elif name == "ims":
        raise NotImplementedError(
            "IMS ships no RUL/failure labels; adopt a labeling convention "
            "(failure-onset times per run) before it can enter the benchmark."
        )
    else:
        raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
    units = _from_rul_datasets(reader, name, split)
    # rul_cap comes off the live reader so a max_rul override is reflected
    return units, dict(_META[name], rul_cap=reader.max_rul, fd=fd, split=split)


# --------------------------------------------------------------------------
# PHME20 — PHM Society European Conf. 2020 data challenge (filtration clogging)
#
# Custom path: the dataset is in no loader library, and the original challenge
# page is offline. The 2020 zips are still served from the legacy domain
# (verified live 2026-07-17):
#   https://phmeurope.org/2020/wp-content/uploads/sites/3/2020/06/Training.zip
#   https://phmeurope.org/2020/wp-content/uploads/sites/3/2020/06/Validation.zip
# 32 run-to-failure experiments (24 train, 8 validation; the 16-run challenge
# test set was never released). CSV schema (verbatim header, all files):
# Time(s),Flow_Rate(ml/m),Upstream_Pressure(psi),Downstream_Pressure(psi) @10Hz.
#
# Target construction (official challenge rule, quoted in Ince et al. 2020 and
# Beirami et al. 2020, PHME20 solution papers): the filter is CLOGGED when the
# pressure drop (upstream - downstream) first exceeds 20 psi. Raw logs continue
# past the crossing; the loader truncates each run at the first crossing
# (Beirami convention) so rul >= 0, with rul in seconds until clogging.
#
# fd selects the particle-size condition: 1 = "Small" (45-53 micron),
# 2 = "Large" (63-75 micron). split: dev = Training, val = Validation.
# Solid-ratio profile per sample id (read from the shipped Operation Profiles
# workbooks, hardcoded here to avoid an xlsx dependency): train 0.4% (1-4,
# 33-36), 0.425% (5-8, 37-40), 0.45% (9-12, 41-44); validation 0.475% (all).
# --------------------------------------------------------------------------

_PHME20_URLS = {
    "dev": "https://phmeurope.org/2020/wp-content/uploads/sites/3/2020/06/Training.zip",
    "val": "https://phmeurope.org/2020/wp-content/uploads/sites/3/2020/06/Validation.zip",
}
_PHME20_CACHE = Path.home() / ".rulbench" / "phme20"
_PHME20_SIZE_DIR = {1: "Small", 2: "Large"}
_PHME20_CLOG_PSI = 20.0


def _phme20_extract(split: str) -> Path:
    name = "Training" if split == "dev" else "Validation"
    target = _PHME20_CACHE / name
    if target.exists():
        return target
    _PHME20_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = _PHME20_CACHE / f"{name}.zip"
    if not zip_path.exists():
        # the legacy phmeurope.org WAF 406es python's default UA and even a
        # bare "Mozilla/5.0"; a full browser UA string passes
        req = urllib.request.Request(
            _PHME20_URLS[split],
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            },
        )
        part = zip_path.with_name(zip_path.name + ".part")
        with urllib.request.urlopen(req) as resp:
            part.write_bytes(resp.read())
        part.rename(zip_path)
    # extract to a per-run staging dir and rename into place, so neither an
    # interrupted run nor a concurrent one can leave a half-populated target
    # that the exists() check trusts
    staging = Path(tempfile.mkdtemp(dir=_PHME20_CACHE, prefix=f"{name}.extracting-"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            for n in z.namelist():
                if n.endswith(".csv") and "__MACOSX" not in n:
                    z.extract(n, staging)
        if not (staging / name).is_dir():
            raise FileNotFoundError(f"no {name}/ folder with CSVs inside {zip_path.name}")
        try:
            (staging / name).rename(target)
        except OSError:
            if not target.exists():  # a concurrent extraction winning is fine
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _load_phme20(fd: int = 1, split: str = "dev", include_censored: bool = False):
    if split not in ("dev", "val"):
        raise ValueError(
            "phme20 has splits 'dev' (24 training runs) and 'val' (8 validation "
            "runs); the challenge's 16-run test set was never publicly released."
        )
    if fd not in _PHME20_SIZE_DIR:
        raise ValueError("phme20 fd must be 1 (small particles) or 2 (large)")

    csv_dir = _phme20_extract(split) / _PHME20_SIZE_DIR[fd]
    units = []
    for path in sorted(csv_dir.glob("Sample*.csv")):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        time, flow, up, down = data.T
        crossed = np.flatnonzero((up - down) > _PHME20_CLOG_PSI)
        if len(crossed) == 0:
            if not include_censored:
                continue
            end = len(time) - 1
        else:
            end = crossed[0]
        feats = data[: end + 1, 1:4].astype(np.float32)[:, None, :]  # (T, 1, 3)
        rul = (time[end] - time[: end + 1]).astype(np.float32)
        unit_id = f"phme20-fd{fd}-{split}-{path.stem.lower()}"
        units.append(Unit("phme20", unit_id, feats, rul, censored=len(crossed) == 0))

    meta = {
        "time_unit": "second (10 Hz sampling)",
        "channels": "flow rate [ml/min], upstream pressure [psi], downstream pressure [psi]",
        "rul_cap": None,
        "step": "one 0.1 s reading (L=1)",
        "target": f"seconds until pressure drop first exceeds {_PHME20_CLOG_PSI} psi",
        "particle_size": "45-53 micron" if fd == 1 else "63-75 micron",
        "fd": fd,
        "split": split,
        "censored_included": include_censored,
    }
    return units, meta


# --------------------------------------------------------------------------
# NASA Milling (custom path — no library provides the continuous RUL target)
#
# PyPHM's milling loader discretizes flank wear into 3 classes and discards
# the continuous VB values, so mill.mat is parsed directly (download verified
# 2026-07-14: phm-datasets S3, ~15 MB, nested zip).
#
# Target construction (the dataset ships no per-cut RUL):
#   1. Within each case (one tool / parameter setting), VB is measured only
#      for some cuts. VB is linearly interpolated over the cut index between
#      labeled cuts; cuts after the last labeled cut are dropped.
#   2. Tool life ends at the first cut whose (interpolated) VB reaches
#      _VB_FAIL = 0.45 mm (RUL-literature convention).
#   3. rul[t] = failure_cut_index - t, in cuts. Cases that never reach
#      _VB_FAIL are censored and excluded unless include_censored=True.
#
# Known data caveats (kept, not silently fixed): cuts 18/95/106 show anomalous
# signals per community analysis; per-cut signal lengths vary slightly and are
# trimmed to the shortest length within a case.
# --------------------------------------------------------------------------

_MILL_URL = "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
_MILL_CACHE = Path.home() / ".rulbench" / "milling"
_MILL_CHANNELS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]
_VB_FAIL = 0.45  # mm


def _mill_mat_path() -> Path:
    mat = _MILL_CACHE / "mill.mat"
    if mat.exists():
        return mat
    _MILL_CACHE.mkdir(parents=True, exist_ok=True)
    outer_zip = _MILL_CACHE / "milling.zip"
    if not outer_zip.exists():
        part = outer_zip.with_name(outer_zip.name + ".part")
        urllib.request.urlretrieve(_MILL_URL, part)
        part.rename(outer_zip)
    # write to a staging file and rename, so an interrupted run never leaves
    # a truncated mill.mat that the exists() check trusts
    staged = mat.with_name(mat.name + ".part")
    staged.unlink(missing_ok=True)  # a stale part file would defeat the guard below
    with zipfile.ZipFile(outer_zip) as outer:
        for n in outer.namelist():
            if n.endswith(".zip"):
                outer.extract(n, _MILL_CACHE)
                with zipfile.ZipFile(_MILL_CACHE / n) as inner:
                    for m in inner.namelist():
                        if m.endswith(".mat"):
                            staged.write_bytes(inner.read(m))
    if not staged.exists():
        raise FileNotFoundError("mill.mat not found inside the NASA archive")
    staged.rename(mat)
    return mat


def _load_milling(include_censored: bool = False):
    import scipy.io

    records = scipy.io.loadmat(_mill_mat_path(), simplify_cells=True)["mill"]

    cases: dict[int, list] = {}
    for rec in records:
        cases.setdefault(int(rec["case"]), []).append(rec)
    for recs in cases.values():
        recs.sort(key=lambda r: int(r["run"]))

    units = []
    for case_id, recs in cases.items():
        vb = np.array(
            [float(r["VB"]) if np.size(r["VB"]) == 1 and not np.isnan(r["VB"]) else np.nan for r in recs]
        )
        labeled = np.flatnonzero(~np.isnan(vb))
        if len(labeled) < 2:
            continue
        idx = np.arange(len(recs))
        vb_i = np.interp(idx, labeled, vb[labeled])
        last = labeled[-1]
        crossed = np.flatnonzero(vb_i[: last + 1] >= _VB_FAIL)
        if len(crossed) == 0 and not include_censored:
            continue
        end = crossed[0] if len(crossed) else last
        rul = end - idx[: end + 1]

        length = min(min(len(np.atleast_1d(r[c])) for c in _MILL_CHANNELS) for r in recs[: end + 1])
        feats = np.stack(
            [
                np.stack([np.atleast_1d(r[c])[:length] for c in _MILL_CHANNELS], axis=-1)
                for r in recs[: end + 1]
            ]
        )  # (T, L, 6)
        unit = Unit(
            "milling",
            f"milling-case{case_id:02d}",
            feats,
            rul.astype(float),
            censored=len(crossed) == 0,
        )
        units.append(unit)

    meta = {
        "time_unit": "cut",
        "channels": "spindle current x2, vibration x2, AE x2 @ 250 Hz",
        "rul_cap": None,
        "step": "one milling cut (~9000 samples)",
        "target": f"cuts until interpolated VB >= {_VB_FAIL} mm",
        "censored_included": include_censored,
    }
    return units, meta
