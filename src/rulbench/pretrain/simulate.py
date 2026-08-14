"""
First-passage RUL from predicted state and operator curves (route B).

Euler-Maruyama rollout of dX = load * mu(X) dt + sigma(X) dW from the
window-end state until X >= 1, with dt = 1 -- the generators integrate
with exactly this scheme and step size, so the rollout matches the
generative process rather than approximating it.  mu, sigma come from
the predicted grids by linear interpolation (constant beyond the grid
ends).

Protocol, fixed here rather than left to the caller:
  * the point summary is the MEDIAN crossing time, CAPPED at rul_cap --
    the labels are capped, an uncapped summary would systematically
    overshoot every early-life window;
  * paths that have not crossed by the horizon count as horizon (the
    cap absorbs them); their fraction is reported, not hidden;
  * the future load is held at 1.0 (the reference load the operator
    grids are stated at).  The true trajectories were driven by a
    varying s(t), so this is a named approximation of the evaluation,
    not of the model;
  * pre-onset windows are the caller's concern: the operator curves
    describe post-onset dynamics only, and the remaining healthy
    waiting time is not represented -- report that stratum separately
    (``pre_onset`` in the sampler batches).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    n_paths: int = 256
    horizon: int = 160          # > rul_cap; longer buys nothing under the cap
    rul_cap: float = 125.0
    load: float = 1.0
    threshold: float = 1.0


def first_passage(x0: float, mu_grid: np.ndarray, sigma_grid: np.ndarray,
                  grid: np.ndarray, cfg: SimConfig | None = None,
                  rng: np.random.Generator | None = None) -> dict:
    """Monte-Carlo first-passage summary from state ``x0``.

    mu_grid, sigma_grid are RAW operator values on ``grid`` (exponentiate
    log-space predictions before calling).
    """
    cfg = cfg or SimConfig()
    rng = rng or np.random.default_rng(0)
    x = np.full(cfg.n_paths, float(x0))
    t_cross = np.full(cfg.n_paths, cfg.horizon, dtype=np.float64)
    alive = np.ones(cfg.n_paths, dtype=bool)
    if x0 >= cfg.threshold:
        t_cross[:] = 0.0
        alive[:] = False
    for t in range(1, cfg.horizon + 1):
        if not alive.any():
            break
        xa = x[alive]
        mu = np.interp(xa, grid, mu_grid)
        sg = np.interp(xa, grid, sigma_grid)
        xa = xa + cfg.load * mu + sg * rng.standard_normal(xa.shape)
        xa = np.maximum(xa, 0.0)
        x[alive] = xa
        crossed = xa >= cfg.threshold
        if crossed.any():
            idx = np.flatnonzero(alive)[crossed]
            t_cross[idx] = t
            alive[idx] = False
    capped = np.minimum(t_cross, cfg.rul_cap)
    return dict(rul=float(np.median(capped)),
                rul_mean=float(capped.mean()),
                frac_not_crossed=float(alive.mean()),
                samples=capped)


def route_b_from_predictions(hi_end: np.ndarray, dyn: np.ndarray,
                             grid: np.ndarray, cfg: SimConfig | None = None,
                             dyn_log_mean: float = -5.0,
                             dyn_log_std: float = 2.0,
                             seed: int = 0) -> np.ndarray:
    """Vector of capped median first-passage RULs, one per window.

    hi_end (B,) predicted window-end state; dyn (B, G, 2) dynamics-head
    output in the sampler's STANDARDISED log space -- it is inverted here
    with the same constants (``WindowConfig.dyn_log_mean/std``) before
    exponentiating.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(seed)
    log_dyn = dyn * dyn_log_std + dyn_log_mean
    out = np.empty(len(hi_end))
    for j in range(len(hi_end)):
        mu = np.exp(log_dyn[j, :, 0])
        sg = np.exp(log_dyn[j, :, 1])
        out[j] = first_passage(hi_end[j], mu, sg, grid, cfg, rng)["rul"]
    return out
