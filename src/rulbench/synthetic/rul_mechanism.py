"""RUL mechanism layer.

Turn an already-generated multivariate observational trajectory into a
run-to-failure sample (or a right-censored suspension) with a
piecewise-linear RUL label.

These are pure functions of trajectories. The module imports nothing from
``dotime`` / ``causal_time_prior``, so it is unit-testable in isolation and
reusable if the trajectory source is ever swapped. Randomness is threaded
through an explicit ``numpy`` ``Generator``; there is no global state.

Concepts
--------
* **degradation state** ``d(t)`` -- a monotone, non-negative health index built
  by accumulating a per-step wear increment. Monotone => failure is absorbing.
* **family calibration** -- family-level constants estimated once from *pilot*
  rollouts that never become labeled units: the wear scale that anchors the
  failure threshold, the stressor normalization, and the RUL cap. Fixing these
  before any unit is drawn keeps the failure model first-passage: the
  threshold is independent of the trajectory it is applied to, so a unit that
  wears slowly genuinely fails later (or not at all), and the lifetime
  distribution does not depend on the rollout horizon.
* **failure time** ``T_fail`` -- the first crossing of the unit's threshold
  ``theta = frac * health_ref``. A trajectory that never crosses within the
  horizon is emitted as a right-censored suspension (``censored=True``,
  RUL label unknown).
* **response surface** -- a random smooth map ``health -> per-sensor offset``,
  built the way ``caussim`` builds its outcome surfaces mu(x): a basis
  expansion (spline / Nystroem) followed by random Gaussian affine
  coefficients. The surface is normalized on a fixed reference grid and the
  health/channel scales come from the family calibration, never from the unit
  itself -- so the sensor value at time ``t`` depends only on the wear
  accumulated up to ``t``, a censored unit shows only the part of the
  signature its wear actually reached, and ``strength`` sets the amplitude.
  Each sensor gets its own random response, so a synthetic fleet shows
  diverse failure signatures.
* **RUL label** -- piecewise-linear and capped (Heimes 2008):
  ``RUL(t) = min(T_fail - 1 - t, C)`` with ``C`` a *constant per family*, as
  in the C-MAPSS convention, so the plateau value does not encode the unit's
  own total life.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import SplineTransformer


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class RulConfig:
    # -- degradation dynamics -------------------------------------------------
    degradation_law: str = "cumulative_stress"     # "cumulative_stress" | "gamma" | "linear"
    base_rate_log10_range: Tuple[float, float] = (-2.3, -1.3)  # per-unit ageing speed

    # -- failure (first-passage against a family-calibrated threshold) --------
    threshold_ref_time: int = 100                   # reference age for the wear scale
    threshold_frac_range: Tuple[float, float] = (0.4, 0.9)     # theta = frac * health_ref
    min_life: int = 20                              # discard units that fail too fast

    # -- RUL label ------------------------------------------------------------
    rul_cap_frac: float = 0.6                       # C = round(frac * median pilot life), per family

    # -- sensor degradation signature (caussim-style response surface) --------
    surface_family: str = "spline"                  # "spline" | "nystroem" | "linear" | "none"
    responsive_frac: float = 0.5                    # fraction of channels carrying the signature
    surface_n_knots: int = 4                        # spline
    surface_degree: int = 3                         # spline
    nystroem_components: int = 10                   # nystroem
    nystroem_gamma: float = 1.0                     # nystroem
    surface_strength_range: Tuple[float, float] = (0.5, 2.0)   # amplitude, in units of sensor std

    # -- sanity ---------------------------------------------------------------
    diverge_abs_max: float = 400.0                  # reject SCM draws that blew up


@dataclass
class Mechanism:
    """Family-level failure mechanism, shared by every unit of one SCM family.

    Fixing this per family (not per unit) is what makes a "fleet" coherent: all
    units of a family fail through the same mechanism and expose the same sensor
    response to wear, while differing in ageing rate, threshold and noise.
    """
    stressor: int                                   # channel whose volatility drives wear
    basis: object                                   # fitted spline/Nystroem transformer, or None
    responsive: List[int]                           # channels carrying a degradation signature
    betas: List[np.ndarray]                         # random affine coefficients, one per channel
    strengths: List[float]                          # signature amplitude, one per channel
    surf_mu: List[float]                            # surface normalization on the fixed grid
    surf_sd: List[float]                            # (per responsive channel)


@dataclass
class FamilyCalibration:
    """Family-level constants estimated from pilot rollouts.

    Pilots are extra rollouts of the same SCM that never become labeled units,
    so every constant here is fixed *before* any unit's trajectory is seen.
    """
    health_ref: float                               # typical wear at the reference age
    stressor_mu: float                              # stressor normalization (mean)
    stressor_sd: float                              # stressor normalization (std)
    rul_cap: int                                    # constant per-family RUL cap, in cycles
    health_mu: float                                # health normalization for the response
    health_sd: float                                # surface (pooled pilot wear scale)
    channel_sd: np.ndarray                          # (N,) per-channel amplitude scale


@dataclass
class RulSample:
    sensors: np.ndarray                             # (T, N) sensor block, with signature
    rul: np.ndarray                                 # (T,) capped RUL; all-NaN when censored
    health: np.ndarray                              # (T,) latent degradation state
    censored: bool                                  # True = suspension (no failure observed)
    T_fail: Optional[int]                           # None when censored
    threshold: float                                # theta (un-crossed if censored)
    rul_cap: int                                    # the family's cap (label plateau)
    responsive: List[int]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):                        # torch.Tensor -> numpy
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def valid_trajectory(X, cfg: RulConfig) -> Optional[np.ndarray]:
    """Convert to numpy and reject divergent / degenerate SCM rollouts.

    dotime returns an all-zeros block for diverged simulations, hence the
    lower bound on the peak amplitude.
    """
    X = _as_numpy(X)
    peak = float(np.abs(X).max()) if X.size else 0.0
    if not np.isfinite(X).all() or peak > cfg.diverge_abs_max or peak < 1e-6:
        return None
    return X


# --------------------------------------------------------------------------- #
# Mechanism                                                                   #
# --------------------------------------------------------------------------- #
def _make_basis(cfg: RulConfig, rng: np.random.Generator):
    """Fit the response-surface basis on a fixed reference grid.

    Fitting on a shared grid (rather than each unit's own health) lets the
    family's random coefficients define one surface that every unit's
    standardized health is evaluated against.
    """
    grid = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    if cfg.surface_family == "spline":
        basis = SplineTransformer(
            n_knots=cfg.surface_n_knots, degree=cfg.surface_degree,
            knots="uniform", extrapolation="constant",
        ).fit(grid)
    elif cfg.surface_family == "nystroem":
        basis = Nystroem(
            n_components=cfg.nystroem_components, gamma=cfg.nystroem_gamma,
            random_state=int(rng.integers(2**31 - 1)),
        ).fit(grid)
    elif cfg.surface_family in ("linear", "none"):
        basis = None
    else:
        raise ValueError(f"unknown surface_family {cfg.surface_family!r}")
    phi_grid = grid if basis is None else basis.transform(grid)
    return basis, phi_grid


def sample_mechanism(N: int, rng: np.random.Generator, cfg: RulConfig) -> Mechanism:
    """Draw a family-level failure mechanism for an ``N``-channel SCM.

    Each responsive channel's surface is normalized on the fixed reference
    grid (``surf_mu`` / ``surf_sd``), so its scale is a family constant rather
    than a statistic of any unit's own trajectory.
    """
    stressor = int(rng.integers(0, N))
    basis, phi_grid = _make_basis(cfg, rng)
    p = phi_grid.shape[1]
    n_resp = 0 if cfg.surface_family == "none" else max(1, round(cfg.responsive_frac * N))
    responsive = sorted(rng.choice(N, size=n_resp, replace=False).tolist()) if n_resp else []
    betas = [rng.normal(0.0, 1.0, size=p) for _ in responsive]
    strengths = [float(rng.uniform(*cfg.surface_strength_range)) for _ in responsive]
    surf_grid = [phi_grid @ b for b in betas]
    surf_mu = [float(s.mean()) for s in surf_grid]
    surf_sd = [float(s.std()) for s in surf_grid]
    return Mechanism(stressor=stressor, basis=basis, responsive=responsive,
                     betas=betas, strengths=strengths,
                     surf_mu=surf_mu, surf_sd=surf_sd)


# --------------------------------------------------------------------------- #
# Degradation, calibration, failure, label, signature                         #
# --------------------------------------------------------------------------- #
def degradation_state(X: np.ndarray, mech: Mechanism, rng: np.random.Generator,
                      cfg: RulConfig, stressor_mu: float,
                      stressor_sd: float) -> np.ndarray:
    """Monotone, non-negative health ``d(t)`` from the trajectory. Shape ``(T,)``.

    A per-unit ageing rate is drawn here so units of one family age at
    different speeds; because the threshold is calibrated at family level,
    that rate genuinely shifts the failure time. Wear is driven by the
    severity of the family's stressor channel, normalized with the *family*
    constants (not the unit's own trajectory), so wear at time ``t`` depends
    only on the past and present of the unit.
    """
    T = X.shape[0]
    base_rate = 10.0 ** rng.uniform(*cfg.base_rate_log10_range)
    severity = np.abs((X[:, mech.stressor] - stressor_mu) / stressor_sd)
    if cfg.degradation_law == "cumulative_stress":
        incr = base_rate * (0.5 + severity)
    elif cfg.degradation_law == "gamma":                           # monotone gamma process
        incr = rng.gamma(shape=1.0 + severity, scale=base_rate)
    elif cfg.degradation_law == "linear":                          # near-constant drift
        incr = base_rate * (0.5 + 0.1 * np.abs(rng.normal(size=T)))
    else:
        raise ValueError(f"unknown degradation_law {cfg.degradation_law!r}")
    return np.cumsum(np.clip(incr, 0.0, None))


def calibrate_family(pilots: List[np.ndarray], mech: Mechanism,
                     rng: np.random.Generator,
                     cfg: RulConfig) -> Optional[FamilyCalibration]:
    """Estimate the family constants from pilot trajectories.

    ``health_ref`` is the median pilot wear at the reference age
    ``threshold_ref_time``. Anchoring the threshold there (rather than at the
    end of the horizon) keeps the lifetime distribution intrinsic: a longer
    rollout horizon only converts would-be-censored units into observed
    failures, it does not move the failure times. This requires pilots at
    least ``threshold_ref_time`` steps long (``RULPrior`` enforces
    ``T_max >= threshold_ref_time``); shorter pilots fall back to anchoring at
    their own horizon, which re-couples lifetimes to it.

    The RUL cap is the median pilot life at the midpoint threshold, scaled by
    ``rul_cap_frac`` -- one constant per family, in cycles. Pilots that do not
    cross within their horizon enter the median as infinite lives, so the cap
    does not drift with the horizon; if the median pilot itself does not
    cross, the reference age is used instead.

    ``health_mu`` / ``health_sd`` / ``channel_sd`` are the pooled-pilot scales
    used to normalize the response surface, so a unit's sensors never depend
    on that unit's own future.

    Returns ``None`` if the pilots cannot support a calibration (degenerate
    wear scale).
    """
    stress = np.concatenate([p[:, mech.stressor] for p in pilots])
    mu, sd = float(stress.mean()), float(stress.std())
    if sd < 1e-12:
        sd = 1.0                                    # constant stressor: severity -> 0
    healths = [degradation_state(p, mech, rng, cfg, mu, sd) for p in pilots]
    ref_t = min(cfg.threshold_ref_time, min(len(h) for h in healths))
    health_ref = float(np.median([h[ref_t - 1] for h in healths]))
    if not np.isfinite(health_ref) or health_ref <= 1e-12:
        return None
    theta_mid = float(np.mean(cfg.threshold_frac_range)) * health_ref
    lives = [float(np.argmax(h >= theta_mid)) + 1 if h[-1] >= theta_mid else np.inf
             for h in healths]
    life_ref = float(np.median(lives))
    if not np.isfinite(life_ref):
        life_ref = float(ref_t)
    cap = max(1, round(cfg.rul_cap_frac * life_ref))
    pooled_h = np.concatenate(healths)
    pooled_X = np.concatenate(pilots, axis=0)
    return FamilyCalibration(health_ref=health_ref, stressor_mu=mu,
                             stressor_sd=sd, rul_cap=cap,
                             health_mu=float(pooled_h.mean()),
                             health_sd=float(pooled_h.std()) + 1e-8,
                             channel_sd=pooled_X.std(axis=0) + 1e-8)


def failure_time(health: np.ndarray, rng: np.random.Generator, cfg: RulConfig,
                 calib: FamilyCalibration) -> Tuple[Optional[int], float]:
    """First crossing of ``theta = frac * health_ref``.

    Returns ``(T_fail, theta)``, with ``T_fail = None`` when the trajectory
    never crosses within its horizon (right-censored suspension). Because
    ``theta`` is fixed by the family calibration, not by this trajectory,
    censoring is a real outcome, not dead code.
    """
    theta = float(rng.uniform(*cfg.threshold_frac_range)) * calib.health_ref
    hit = health >= theta
    if not hit.any():
        return None, theta
    return int(np.argmax(hit)) + 1, theta


def rul_label(T_fail: int, cap: int) -> np.ndarray:
    """Piecewise-linear capped RUL (Heimes 2008): ``min(T_fail - 1 - t, cap)``.

    ``cap`` is the family constant from :func:`calibrate_family`, shared by
    all units of the fleet as in the C-MAPSS convention.
    """
    rul = (T_fail - 1 - np.arange(T_fail)).astype(float)
    return np.minimum(rul, cap)


def apply_response_surface(sensors: np.ndarray, health: np.ndarray,
                           mech: Mechanism, cfg: RulConfig,
                           calib: FamilyCalibration) -> np.ndarray:
    """Add each responsive channel's random smooth response to ``health``.

    caussim recipe: normalized health -> basis expansion -> affine(beta) ->
    grid-normalized surface, scaled to the family's channel scale. Every
    normalization constant is a family constant (calibration or reference
    grid), never a statistic of this unit's own trajectory -- so the sensor
    value at time ``t`` is a function of the wear reached at ``t`` alone, and
    a censored unit only shows the part of the signature its wear covered.
    """
    if not mech.responsive:
        return sensors
    h = ((health - calib.health_mu) / calib.health_sd).reshape(-1, 1)
    phi = h if mech.basis is None else mech.basis.transform(h)      # (T, p)
    out = sensors.copy()
    for k, (j, beta, strength) in enumerate(zip(mech.responsive, mech.betas,
                                                mech.strengths)):
        surf = (phi @ beta - mech.surf_mu[k]) / (mech.surf_sd[k] + 1e-8)
        out[:, j] = out[:, j] + strength * calib.channel_sd[j] * surf
    return out


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def induce_run_to_failure(X, rng: np.random.Generator, cfg: RulConfig,
                          mech: Mechanism,
                          calib: FamilyCalibration) -> Optional[RulSample]:
    """Trajectory -> ``RulSample``, or ``None`` on divergence / too-short life.

    A trajectory whose health never reaches the threshold is returned as a
    right-censored suspension: full-horizon sensors, ``censored=True``, and an
    all-NaN RUL label (the true RUL is unknown, only bounded below).
    """
    X = valid_trajectory(X, cfg)
    if X is None:
        return None                                                # divergent / degenerate SCM

    health = degradation_state(X, mech, rng, cfg,
                               calib.stressor_mu, calib.stressor_sd)
    T_fail, theta = failure_time(health, rng, cfg, calib)
    if T_fail is None:                                             # right-censored suspension
        sensors = apply_response_surface(X.copy(), health, mech, cfg, calib)
        return RulSample(sensors=sensors, rul=np.full(X.shape[0], np.nan),
                         health=health, censored=True, T_fail=None,
                         threshold=theta, rul_cap=calib.rul_cap,
                         responsive=list(mech.responsive))
    if T_fail < cfg.min_life:
        return None

    sensors = apply_response_surface(X[:T_fail].copy(), health[:T_fail], mech, cfg, calib)
    return RulSample(sensors=sensors, rul=rul_label(T_fail, calib.rul_cap),
                     health=health[:T_fail], censored=False, T_fail=T_fail,
                     threshold=theta, rul_cap=calib.rul_cap,
                     responsive=list(mech.responsive))
