"""RULPrior -- synthetic run-to-failure / RUL data from dotime's SCM generator.

Adapter over ``CausalTimePrior``. It reuses the SCM sampler and observational
rollout unchanged, applies the :mod:`rul_mechanism` degradation-and-failure
layer, and exposes the result three ways:

* :meth:`RULPrior.generate`        -> raw list of units (failed + censored),
* :meth:`RULPrior.to_dataframe`    -> a tidy table (the analyst-facing view),
* :meth:`RULPrior.generate_batch`  -> padded arrays for a PFN trainer
                                      (``target_key='RUL'``).

Task shapes
-----------
* ``task_mode="fleet"`` (default): one SCM == one *machine family*. Extra
  *pilot* rollouts calibrate the family's failure threshold, stressor
  normalization and RUL cap; the SCM is then rolled out several times to give
  a fleet of units that share a failure mechanism but differ in ageing rate,
  threshold and noise. Units whose health never reaches the threshold within
  the horizon are emitted as right-censored suspensions (``censored=True``,
  all-NaN RUL), like the suspended units real fleets contain. This matches
  the in-context deployment story (context = units of the target family,
  query = a held-out unit of the same family).
* ``task_mode="trajectory"``: each unit is an independent SCM with its own
  mechanism and calibration. Censored draws are resampled here -- a
  single-unit task needs a labeled trajectory. Simpler, but does not
  reproduce cross-unit-within-family transfer.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:                                                 # package-relative or flat import
    from .rul_mechanism import (FamilyCalibration, Mechanism, RulConfig,
                                RulSample, calibrate_family,
                                induce_run_to_failure, sample_mechanism,
                                valid_trajectory)
except ImportError:                                  # pragma: no cover
    from rul_mechanism import (FamilyCalibration, Mechanism, RulConfig,
                               RulSample, calibrate_family,
                               induce_run_to_failure, sample_mechanism,
                               valid_trajectory)

# A generated dataset is a flat list of (family_id, unit_id, RulSample).
Row = Tuple[int, int, RulSample]


def _scm_lag(scm) -> Optional[int]:
    """Max lag ``K`` if the SCM variant exposes it (metadata only)."""
    return int(scm._K) if hasattr(scm, "_K") else None


@contextlib.contextmanager
def _scoped_rng(seed: int):
    """Make all global-RNG draws inside the block reproducible, then restore
    the process's global RNG state on exit.

    CausalTimePrior threads a seeded ``torch.Generator`` through almost every
    draw, but a few escape it and fall through to *global* RNGs:

    * ``temporal_scm.py`` pre-samples per-step noise via
      ``self.noise[v].distribution.sample(...)`` (bypasses the generator-aware
      ``sample_n``) -> global **torch** RNG, on the default/chain SCM branch;
    * two ``torch.distributions.Beta(...).sample()`` edge-probability draws
      (``prior.py``, ``regime_switching_builder.py``) -> global **torch** RNG;
    * ``regime_switching.py`` draws regime transitions via ``np.random.*``
      -> global **numpy** RNG.

    Reseeding both global RNGs for the duration of generation makes the whole
    pipeline deterministic given ``seed``. ``torch.random.fork_rng`` restores
    the torch CPU state afterward; we save/restore numpy's global state by
    hand. So unrelated *CPU* torch/numpy code before or after is unaffected.
    Caveat: ``torch.manual_seed`` also reseeds the global CUDA/MPS generators
    and ``fork_rng(devices=[])`` does not restore those -- irrelevant to
    generation itself (CPU-only), but code relying on an accelerator's global
    RNG stream would see it reseeded after this block.
    """
    import torch
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)


class RULPrior:
    def __init__(self, ctp, cfg: Optional[RulConfig] = None, T_max: int = 200,
                 task_mode: str = "fleet", fleet_size: Tuple[int, int] = (3, 8),
                 n_pilot: int = 3, max_resample: int = 50, seed: int = 0):
        """
        Parameters
        ----------
        ctp : CausalTimePrior
            An already-configured generator; SCM-family knobs (chain /
            regime-switching probabilities, ``N_max``, ``K_max``) live there.
        cfg : RulConfig
            RUL mechanism configuration (degradation law, surface, thresholds).
        T_max : int
            Horizon length rolled out per unit. With the family-calibrated
            threshold, a longer horizon only lowers the censoring rate; it
            does not stretch the failure times. Must be at least
            ``cfg.threshold_ref_time`` -- shorter horizons would silently
            anchor the threshold at the horizon itself and re-couple
            lifetimes to it.
        task_mode : {"fleet", "trajectory"}
        fleet_size : (int, int)
            Inclusive range for the number of units drawn per family.
        n_pilot : int
            Pilot rollouts per family used to calibrate threshold scale,
            stressor normalization and RUL cap (never emitted as units).
        max_resample : int
            Cap on SCM redraws per family/unit before giving up
            (divergence / failed calibration / too-short lives).
        """
        self.ctp = ctp
        self.cfg = cfg or RulConfig()
        if T_max < self.cfg.threshold_ref_time:
            raise ValueError(
                f"T_max={T_max} is shorter than threshold_ref_time="
                f"{self.cfg.threshold_ref_time}: the calibration anchor would "
                f"clamp to the horizon and failure times would scale with "
                f"T_max again. Raise T_max or lower cfg.threshold_ref_time.")
        self.T_max = T_max
        self.task_mode = task_mode
        self.fleet_size = fleet_size
        self.n_pilot = n_pilot
        self.max_resample = max_resample
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- core sampling ------------------------------------------------------ #
    def _rollout(self, scm):
        # dotime prints a divergence warning and returns zeros for unstable SCMs;
        # those units are rejected downstream, so silence the noise here.
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return scm.sample_observational(
                T=self.T_max, burn_in=self.ctp.config["burn_in"],
                generator=self.ctp.generator)

    def _calibrated_mechanism(
            self, scm) -> Optional[Tuple[Mechanism, FamilyCalibration]]:
        """Draw a mechanism and calibrate it on pilot rollouts of ``scm``."""
        mech = sample_mechanism(len(scm._topo), self.rng, self.cfg)
        pilots: List[np.ndarray] = []
        for _ in range(self.n_pilot * 4):            # allow for divergent rollouts
            X = valid_trajectory(self._rollout(scm), self.cfg)
            if X is not None:
                pilots.append(X)
            if len(pilots) == self.n_pilot:
                break
        if len(pilots) < self.n_pilot:
            return None                              # SCM too unstable to calibrate
        calib = calibrate_family(pilots, mech, self.rng, self.cfg)
        return None if calib is None else (mech, calib)

    def sample_family(self) -> Tuple[Optional[List[RulSample]], Optional[dict]]:
        """One SCM family -> list of units sharing a mechanism + calibration.

        The list mixes failed and censored units as they occur; a family is
        accepted once it has at least two *failed* units (an in-context RUL
        task needs labeled examples).
        """
        for _ in range(self.max_resample):
            scm = self.ctp.sample_scm()
            mc = self._calibrated_mechanism(scm)
            if mc is None:
                continue
            mech, calib = mc
            target = int(self.rng.integers(self.fleet_size[0], self.fleet_size[1] + 1))
            units: List[RulSample] = []
            for _try in range(target * 4):            # allow for divergent/short rollouts
                s = induce_run_to_failure(self._rollout(scm), self.rng, self.cfg,
                                          mech, calib)
                if s is not None:
                    units.append(s)
                if len(units) >= target:
                    break
            n_failed = sum(not s.censored for s in units)
            if n_failed >= 2:
                return units, dict(N=len(scm._topo), K=_scm_lag(scm),
                                   n_units=len(units), n_failed=n_failed,
                                   n_censored=len(units) - n_failed,
                                   stressor=mech.stressor, rul_cap=calib.rul_cap)
        return None, None

    def sample_unit(self) -> Tuple[Optional[RulSample], Optional[dict]]:
        """One independent labeled unit (own SCM, mechanism and calibration).

        Censored draws are resampled: a standalone unit without a label is
        useless outside the fleet context.
        """
        for _ in range(self.max_resample):
            scm = self.ctp.sample_scm()
            mc = self._calibrated_mechanism(scm)
            if mc is None:
                continue
            mech, calib = mc
            for _try in range(4):
                s = induce_run_to_failure(self._rollout(scm), self.rng, self.cfg,
                                          mech, calib)
                if s is not None and not s.censored:
                    return s, dict(N=len(scm._topo), K=_scm_lag(scm),
                                   rul_cap=calib.rul_cap)
        return None, None

    # -- public dataset API ------------------------------------------------- #
    def generate(self, n: int = 5) -> List[Row]:
        """Generate a dataset.

        ``n`` families (fleet mode) or ``n`` independent units (trajectory
        mode). Returns a flat list of ``(family_id, unit_id, RulSample)``.
        Raises ``RuntimeError`` if a family/unit cannot be sampled within
        ``max_resample`` SCM redraws (rather than retrying forever).

        Reproducible in ``self.seed``: the wrapper's own numpy Generator is
        reset here, and generation runs inside :func:`_scoped_rng` so the
        global torch/numpy draws that CausalTimePrior leaks are seeded too
        (see that function). Two calls with the same seed return identical data.
        """
        self.rng = np.random.default_rng(self.seed)   # reset per call for reproducibility
        if hasattr(self.ctp, "generator"):            # reset dotime's seeded torch.Generator too
            self.ctp.generator.manual_seed(self.seed)
        rows: List[Row] = []
        no_sample = (
            f"no viable sample within max_resample={self.max_resample} SCM "
            f"redraws -- check RulConfig (min_life vs threshold_ref_time/"
            f"T_max, diverge_abs_max) against the CausalTimePrior settings")
        with _scoped_rng(self.seed):
            if self.task_mode == "fleet":
                for fam in range(n):
                    units, _ = self.sample_family()
                    if units is None:
                        raise RuntimeError(no_sample)
                    rows.extend((fam, u, s) for u, s in enumerate(units))
            elif self.task_mode == "trajectory":
                for u in range(n):
                    s, _ = self.sample_unit()
                    if s is None:
                        raise RuntimeError(no_sample)
                    rows.append((u, 0, s))
            else:
                raise ValueError(f"unknown task_mode {self.task_mode!r}")
        return rows

    # -- views -------------------------------------------------------------- #
    def to_dataframe(self, rows: List[Row], long: bool = True) -> pd.DataFrame:
        """Tidy table of a generated dataset.

        ``long=True``  : one row per (family, unit, cycle), sensors + health +
                         RUL (NaN for censored units).
        ``long=False`` : one row per unit (censored, T_fail, cap, threshold, ...).
        """
        if not rows:
            raise ValueError("rows is empty -- nothing to tabulate")
        if not long:
            recs = [dict(family_id=f, unit_id=u, n_sensors=s.sensors.shape[1],
                         censored=s.censored, T_end=s.sensors.shape[0],
                         T_fail=s.T_fail, rul_cap=s.rul_cap,
                         threshold=round(s.threshold, 4),
                         n_responsive=len(s.responsive))
                    for f, u, s in rows]
            return pd.DataFrame.from_records(recs)

        n_max = max(s.sensors.shape[1] for _, _, s in rows)
        recs = []
        for f, u, s in rows:
            T, N = s.sensors.shape
            for t in range(T):
                rec = {"family_id": f, "unit_id": u, "cycle": t,
                       "censored": s.censored}
                for j in range(n_max):
                    rec[f"sensor_{j:02d}"] = s.sensors[t, j] if j < N else np.nan
                rec["health"] = s.health[t]
                rec["RUL"] = s.rul[t]
                recs.append(rec)
        return pd.DataFrame.from_records(recs)

    def generate_batch(self, rows: Optional[List[Row]] = None, n: int = 5) -> dict:
        """Model-ready padded arrays for a PFN trainer.

        Returns a dict with ``sensors`` (B, T, N), ``RUL`` (B, T),
        ``step_mask`` / ``variable_mask`` / ``label_mask``, ``censored`` (B,),
        ``T_fail`` (B, -1 for censored units) and ``target_key='RUL'``.
        ``RUL`` is NaN wherever no label exists -- both at pad positions and
        at the observed steps of censored units; ``label_mask`` is True
        exactly where a loss can be computed.
        """
        if rows is None:
            rows = self.generate(n=n)
        if not rows:
            raise ValueError("rows is empty -- nothing to batch")
        samples = [s for _, _, s in rows]
        B = len(samples)
        Tb = max(s.sensors.shape[0] for s in samples)
        Nb = max(s.sensors.shape[1] for s in samples)

        sensors = np.zeros((B, Tb, Nb))
        rul = np.full((B, Tb), np.nan)
        step_mask = np.zeros((B, Tb), dtype=bool)
        var_mask = np.zeros((B, Nb), dtype=bool)
        censored = np.zeros(B, dtype=bool)
        t_fail = np.full(B, -1, dtype=int)
        for i, s in enumerate(samples):
            T, N = s.sensors.shape
            sensors[i, :T, :N] = s.sensors
            rul[i, :T] = s.rul
            step_mask[i, :T] = True
            var_mask[i, :N] = True
            censored[i] = s.censored
            if not s.censored:
                t_fail[i] = s.T_fail
        label_mask = step_mask & ~np.isnan(rul)
        return dict(sensors=sensors, RUL=rul, step_mask=step_mask,
                    variable_mask=var_mask, label_mask=label_mask,
                    censored=censored, T_fail=t_fail, target_key="RUL")


# --------------------------------------------------------------------------- #
# One-call convenience                                                        #
# --------------------------------------------------------------------------- #
def generate_rul_dataset(n: int = 5, cfg: Optional[RulConfig] = None, seed: int = 0,
                         T_max: int = 200, task_mode: str = "fleet", ctp=None,
                         **prior_kw):
    """Build a generator, sample a dataset, return tidy tables + handles.

    Returns
    -------
    long_df : pd.DataFrame     per-cycle table (sensors + health + RUL)
    summary_df : pd.DataFrame  per-unit table (censored, T_fail, cap, ...)
    rows : list                raw (family_id, unit_id, RulSample)
    prior : RULPrior           the generator (for generate_batch, reuse, ...)
    """
    if ctp is None:
        from causal_time_prior.prior import CausalTimePrior
        ctp = CausalTimePrior(seed=seed)
    prior = RULPrior(ctp, cfg=cfg, T_max=T_max, task_mode=task_mode, seed=seed, **prior_kw)
    rows = prior.generate(n=n)
    return prior.to_dataframe(rows, long=True), prior.to_dataframe(rows, long=False), rows, prior
