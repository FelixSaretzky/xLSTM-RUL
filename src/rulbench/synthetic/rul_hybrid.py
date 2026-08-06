"""Hybrid RUL prior: dotime-sampled sensors driven by a latent health SDE.

The generator runs in three stages, top to bottom of this file:

1. **Injection** -- a precomputed health trajectory X'(t) and load s(t)
   are fed into a sampled dotime ``TemporalSCM`` as two extra root nodes
   ("H", "S") with outgoing edges only, so the sensors *respond* to
   health and load while the sensor-side structure (graph, lagged
   edges, hidden nodes) stays the sampled dotime prior.  A rollout of
   the injected SCM has column 0 = H, column 1 = S, columns 2.. = the
   original sensors; training data must only ever use columns 2..
   (column 0 is the label-defining latent).
2. **Wiring + calibration** -- :func:`sample_calibrated_emission` picks
   which sensors respond, gives them monotone activations, and scales
   their weights so the noise-free sensor excursion from X'=0 to X'=1
   has norm 1.0, with the total sensor noise drawn on the same SNR axis
   as the sibling generator (``rul_sde``).
3. **Sampling** -- :func:`sample_hybrid_unit` draws the shared latent
   block via ``rul_sde`` (onset, SDE operators, load), rolls out the
   sensors, and emits ``dataset_io.Unit`` records; ``__main__`` is the
   CLI.

dotime is pinned to exactly 0.1.2: the injection rides upstream
internals (the private noise path, the rollout's unused ``generator``
argument), verified against that version.  Rollout noise comes from the
global torch RNG; :func:`sample_hybrid_unit` seeds it itself, while
:func:`observational_rollout` leaves seeding to the caller.

The upstream contracts and traps live at their definition sites:
carrier exactness (:class:`DeterministicCarrier`), silently dropped
parents and intervention index shifts (:func:`inject_drivers`),
divergence handling (:func:`observational_rollout`).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

import torch.distributions as tdist

from dotime._activations import Tanh, TanhReLU, TanhX2
from dotime._sampling import ShiftedExponentialSampler, TorchDistributionSampler
from dotime.prior import Abs, Cos, Sin, Square
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_scm import TemporalSCM
from dotime.temporal_scm_builder import TemporalSCMBuilder

from rulbench.dataset_io import HDF5Writer, Unit, gen_grid
from rulbench.synthetic.rul_sde import TSCMGenerator, TSCMPriorConfig

DRIVER_HEALTH = "H"
DRIVER_LOAD = "S"

# dotime's early-abort threshold (temporal_scm.py, |x| > 500 at checkpoints);
# values are additionally clamped to +/-1000 on every write.  Signals must
# stay strictly inside the tighter bound or rollouts zero out / clip silently.
_DOTIME_ABORT_ABS = 500.0


class RolloutDiverged(RuntimeError):
    """The dotime rollout diverged (upstream returned an all-zeros buffer)."""


class DeterministicCarrier:
    """Noise sampler that deterministically emits a fixed signal.

    Implements the one interface the dotime rollout consumes
    (``.distribution.sample(shape)``) and returns the stored ``(total_T,)``
    signal.  The requested length is checked against the stored one so a
    horizon mismatch fails loudly instead of silently misaligning the
    latent state and the emitted sensors.  The signal is copied on
    construction: later mutation of the caller's array cannot reach into
    rollouts, and the finiteness/range checks stay binding.
    """

    def __init__(self, values) -> None:
        values = torch.as_tensor(values, dtype=torch.float32).clone()
        if values.ndim != 1:
            raise ValueError(f"carrier signal must be 1-D, got shape {tuple(values.shape)}")
        if not torch.isfinite(values).all():
            raise ValueError("carrier signal contains non-finite values")
        peak = float(values.abs().max()) if values.numel() else 0.0
        if peak >= _DOTIME_ABORT_ABS:
            raise ValueError(
                f"carrier signal peaks at |{peak:.3g}| >= {_DOTIME_ABORT_ABS:g}: dotime "
                f"declares divergence above that bound and would zero the rollout")
        self._values = values
        self.distribution = self  # consumed upstream as `.distribution.sample(...)`

    @property
    def values(self) -> torch.Tensor:
        return self._values

    def sample(self, sample_shape=()) -> torch.Tensor:
        n = int(sample_shape[0]) if len(sample_shape) else 1
        if n != self._values.shape[0]:
            raise ValueError(
                f"carrier holds {int(self._values.shape[0])} steps but the rollout "
                f"requested {n} (= T + burn_in); pass signals of exactly that length")
        return self._values


class _ScaledNoise:
    """Wrap a dotime noise sampler, scaling every draw by a constant factor.

    Rides the same ``.distribution.sample(shape)`` path as the carrier;
    preserves the inner distribution family (Normal/Uniform/Laplace) while
    setting the total sensor-noise norm to the drawn SNR-axis target.
    """

    def __init__(self, inner, k: float) -> None:
        self._inner = inner
        self._k = float(k)
        self.distribution = self

    def sample(self, sample_shape=()) -> torch.Tensor:
        return self._inner.distribution.sample(sample_shape) * self._k


class _Passthrough(nn.Module):
    """Mechanism of a driver node: returns the carrier value verbatim.

    Equivalent to the no-parents branch of ``TemporalMechanism.forward``,
    but without initialising unused weight/bias parameters from the global
    RNG.
    """

    def forward(self, parent_values_instant, parent_values_lagged, eps):
        return eps


def inject_drivers(
    scm: TemporalSCM,
    health,
    load,
    h_weights: dict[str, float],
    s_weights: dict[str, float],
) -> TemporalSCM:
    """Rebuild ``scm`` with the root driver nodes "H" and "S" wired in.

    Parameters
    ----------
    scm : TemporalSCM
        A sampled dotime SCM (e.g. from ``TemporalSCMBuilder.sample``).
        It is left untouched: wired child mechanisms are deep-copied
        before receiving their driver weight entries, so neither the input
        SCM nor previously injected SCMs are affected by later injections.
    health, load : array-like, shape (T + burn_in,)
        Driver signals over the *full* simulation horizon of the intended
        rollout, burn-in prefix included (the health prefix is the healthy
        baseline the labels assume -- typically zeros, or a constant
        initial-health offset).
    h_weights, s_weights : dict
        Instantaneous driver->sensor weights, keyed by sensor node name.

    Returns
    -------
    TemporalSCM
        A fresh SCM; roll it out with :func:`observational_rollout`.

    Notes
    -----
    The weight entries are what make a driver edge live: dotime
    mechanisms silently ignore parents they hold no weight for.  Driver
    nodes carry ``hidden=True, driver=True`` node attributes so
    attribute-based sensor filters classify them correctly.  If the
    injected SCM is ever used with an ``InterventionSpec`` (v1 does
    not): targets are positional topo indices, which all shift by +2
    here, and indices 0/1 would overwrite the carriers -- never
    intervene on them.
    """
    topo = scm.dag.topo_order
    for d in (DRIVER_HEALTH, DRIVER_LOAD):
        if d in topo:
            raise ValueError(f"node name {d!r} already exists in the SCM")
    unknown = (set(h_weights) | set(s_weights)) - set(topo)
    if unknown:
        raise ValueError(f"wiring targets not in the SCM: {sorted(unknown)}")

    health_t = torch.as_tensor(health, dtype=torch.float32)
    load_t = torch.as_tensor(load, dtype=torch.float32)
    if health_t.shape != load_t.shape:
        raise ValueError(
            f"health and load must have equal length, got "
            f"{tuple(health_t.shape)} vs {tuple(load_t.shape)}")

    G_0 = scm.dag.G_0.copy()
    G_0.add_node(DRIVER_HEALTH, hidden=True, driver=True)
    G_0.add_node(DRIVER_LOAD, hidden=True, driver=True)
    for v in h_weights:
        G_0.add_edge(DRIVER_HEALTH, v)
    for v in s_weights:
        G_0.add_edge(DRIVER_LOAD, v)

    n = len(topo)
    G_lags = []
    for G_k in scm.dag.G_lags:
        G_new = np.zeros((n + 2, n + 2), dtype=G_k.dtype)
        # topo-position coordinates; sensor block unchanged.  Driver rows stay
        # zero: a lagged driver edge would need BOTH a matrix entry here AND a
        # weights_lagged parameter on the child (else it is silently dropped).
        G_new[2:, 2:] = G_k
        G_lags.append(G_new)

    dag = TemporalDAG(
        G_0=G_0, G_lags=G_lags, K=scm.dag.K,
        topo_order=[DRIVER_HEALTH, DRIVER_LOAD, *topo])

    # deep-copy only the wired children, then add their driver weights; all
    # other mechanisms are shared read-only (forward never mutates them)
    mechanisms = dict(scm.mechanisms)
    wired = set(h_weights) | set(s_weights)
    for v in wired:
        mechanisms[v] = copy.deepcopy(scm.mechanisms[v])
    with torch.no_grad():
        for driver, weights in ((DRIVER_HEALTH, h_weights), (DRIVER_LOAD, s_weights)):
            for v, w in weights.items():
                mechanisms[v].weights_instant[driver] = nn.Parameter(
                    torch.tensor([float(w)], dtype=torch.float32, device=scm.device),
                    requires_grad=False)
    mechanisms[DRIVER_HEALTH] = _Passthrough()
    mechanisms[DRIVER_LOAD] = _Passthrough()

    noise = dict(scm.noise)
    noise[DRIVER_HEALTH] = DeterministicCarrier(health_t)
    noise[DRIVER_LOAD] = DeterministicCarrier(load_t)

    return TemporalSCM(dag, mechanisms, noise, device=scm.device, dtype=scm.dtype)


def observational_rollout(scm: TemporalSCM, burn_in: int) -> torch.Tensor:
    """Roll out an injected SCM and enforce the driver-column contract.

    Derives ``T`` from the carrier length (so the T/burn-in split cannot
    be double-booked against the signals), raises :class:`RolloutDiverged`
    instead of accepting dotime's silent all-zeros divergence return, and
    verifies both driver columns bitwise against the stored signals --
    which also catches any future upstream change to the noise path the
    carrier rides.

    The caller still owns global-RNG seeding (``torch.manual_seed``) and
    the reject-and-resample policy on :class:`RolloutDiverged`.
    """
    h_carrier = scm.noise.get(DRIVER_HEALTH)
    s_carrier = scm.noise.get(DRIVER_LOAD)
    if not isinstance(h_carrier, DeterministicCarrier) or not isinstance(
            s_carrier, DeterministicCarrier):
        raise ValueError("not an injected SCM: driver carriers missing")
    total_T = int(h_carrier.values.shape[0])
    if not 0 <= burn_in < total_T:
        raise ValueError(f"burn_in={burn_in} outside [0, {total_T})")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            out = scm.sample_observational(T=total_T - burn_in, burn_in=burn_in)
        except RuntimeWarning as w:
            raise RolloutDiverged(str(w)) from None

    for col, carrier, name in ((0, h_carrier, DRIVER_HEALTH), (1, s_carrier, DRIVER_LOAD)):
        if not torch.equal(out[:, col], carrier.values[burn_in:]):
            raise RolloutDiverged(
                f"emitted {name} column deviates from the injected signal -- "
                f"divergence, clipping, or an upstream noise-path change")
    return out


# --------------------------------------------------------------------------- #
# Step 2: wiring policy + excursion calibration                               #
# --------------------------------------------------------------------------- #
# dotime's full activation pool defines the emission diversity of UNWIRED
# sensors.  Wired (responsive) sensors get their activation REASSIGNED to a
# strictly monotone one (Identity or Tanh) after selection -- the same
# order-of-operations as the sibling generator's ``signature_nl``
# restriction (wire first, then constrain the nonlinearity), which keeps
# the expected responsive-sensor count comparable across the two arms
# (p_signature over all observed sensors) instead of capping it at the
# monotone share of the pool.  TanhReLU/ReLU are monotone but gate
# negative pre-activations to a constant (annihilates the signature at the
# settled operating point); Sin/Cos/Abs/Square/TanhX2 can fold it -- hence
# the reassignment set is exactly (Identity, Tanh).
_FULL_ACTIVATION_POOL = (
    nn.Identity(), Tanh(), TanhX2(), TanhReLU(), nn.ReLU(),
    Sin(), Cos(), Abs(), Square())
_MONOTONE_ACTIVATIONS = (nn.Identity, Tanh)


@dataclass
class EmissionConfig:
    """Sampling and calibration knobs for one calibrated emission SCM.

    v1 defaults implement the agreed ablation constraints: 8 observed
    process sensors (``dropout_prob=0`` -- hidden nodes are a later
    ablation axis), excursion norm calibrated to exactly 1.0, and the
    dotime diverse-branch hyperpriors (Beta edge probability, lag decay,
    noise-scale distributions) reproduced with our own seeded generator so
    no draw leaks to the global RNG.
    """

    n_nodes: int = 8
    k_max: int = 3
    edge_beta: tuple = (2.0, 5.0)
    dropout_prob: float = 0.0
    gamma: float = 0.7
    sigma_w: float = 1.0
    sigma_b: float = 0.5
    activations: tuple = _FULL_ACTIVATION_POOL
    # wiring
    p_signature: float = 0.6      # per eligible sensor, H -> sensor
    p_load_leak: float = 0.3      # per observed sensor, S -> sensor
    load_leak_scale: float = 0.3
    # calibration (noise-free, hence deterministic given the seed)
    calib_T: int = 80             # settle horizon of the frozen rollouts
    settle_tol: float = 1e-4      # max step-to-step drift over the last rows
    norm_tol: float = 1e-3        # |excursion norm - 1| acceptance
    max_calib_iter: int = 8
    midramp_band: tuple = (0.35, 0.65)  # excursion at X'=0.5 (linearity check)
    max_tries: int = 50
    # post-calibration SNR axis: total sensor-noise norm vs the unit-norm
    # signature, log-uniform.  Same range as the sibling generator's
    # noise_norm_range, so end-of-life SNR = 1/||sigma|| in ~2.5..50 is a
    # matched prior axis across both arms (dotime's raw noise scales sit
    # near ||sigma|| ~ 1.3, which measurably drowns the signature).
    noise_norm_range: tuple = (0.02, 0.4)


@dataclass
class CalibratedEmission:
    """A wired, excursion-calibrated emission SCM ready for injection."""

    scm: TemporalSCM              # the UN-injected base SCM
    h_weights: dict[str, float]   # calibrated: excursion norm 1.0
    s_weights: dict[str, float]
    meta: dict = field(default_factory=dict)


def observed_sensor_nodes(scm: TemporalSCM) -> list[str]:
    """Non-hidden nodes of a base SCM, in topological order."""
    return [v for v in scm.dag.topo_order
            if not scm.dag.G_0.nodes[v].get("hidden", False)]


def _settled_row(scm, h_weights, s_weights, x_const, cfg) -> torch.Tensor | None:
    """Noise-free settled sensor state at constant X'=x_const, s=1.

    Reuses the real simulator (no bespoke fixed-point code): the base SCM
    is injected with constant carriers and every sensor node's noise is
    replaced by a zero carrier, so the rollout is the deterministic sensor
    map itself.  Returns the last row, or None if the dynamics have not
    settled (no usable fixed point) or the rollout diverged.
    """
    const = np.full(cfg.calib_T, x_const, dtype=np.float32)
    injected = inject_drivers(scm, const, np.ones_like(const), h_weights, s_weights)
    zero = np.zeros(cfg.calib_T, dtype=np.float32)
    noise = {v: (injected.noise[v] if v in (DRIVER_HEALTH, DRIVER_LOAD)
                 else DeterministicCarrier(zero))
             for v in injected.dag.topo_order}
    frozen = TemporalSCM(injected.dag, injected.mechanisms, noise,
                         device=injected.device, dtype=injected.dtype)
    try:
        out = observational_rollout(frozen, burn_in=0)
    except RolloutDiverged:
        return None
    if float((out[-5:] - out[-6:-1]).abs().max()) > cfg.settle_tol:
        return None
    return out[-1]


def sample_calibrated_emission(seed: int, cfg: EmissionConfig | None = None) -> CalibratedEmission:
    """Sample a dotime emission SCM, wire the drivers, calibrate the signature.

    Policy (from the design reviews, wiring-count-aligned with the sibling
    generator): every observed sensor is an H-candidate and is wired with
    ``p_signature`` (at least one always).  Wired sensors then get their
    activation reassigned to Identity or Tanh (the sibling's
    ``signature_nl`` move), and a wired G_0 root gets its noise re-drawn
    from the non-root scale so the SNR axis stays calibrated instead of
    inheriting the ~10x root noise.  S leaks into observed sensors as an
    operating-condition effect and is not calibrated.  The H weights are
    iteratively rescaled through the real frozen simulator until the
    noise-free sensor excursion between X'=0 and X'=1 has norm 1.0 (over
    observed sensor columns), then a mid-ramp linearity check guards
    against saturation-flattened signatures.  SCMs without a usable fixed
    point, with an unreachable norm, or outside the mid-ramp band are
    rejected and redrawn; rejection counts are reported in ``meta``
    because they shape the effective prior.

    Fully deterministic in ``seed``: policy draws use a local numpy
    Generator, the builder a local torch Generator, and the calibration
    rollouts are noise-free, so the global-RNG leaks of dotime's noisy
    rollouts never enter.
    """
    cfg = cfg or EmissionConfig()
    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(int(rng.integers(2**31 - 1)))
    rejections: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejections[reason] = rejections.get(reason, 0) + 1

    for _try in range(cfg.max_tries):
        edge_prob = float(rng.beta(*cfg.edge_beta))
        K = int(rng.integers(1, cfg.k_max + 1))
        builder = TemporalSCMBuilder(
            num_nodes=cfg.n_nodes, max_lag=K, edge_prob=edge_prob,
            dropout_prob=cfg.dropout_prob, gamma=cfg.gamma,
            activations=list(cfg.activations),
            root_std_dist=ShiftedExponentialSampler(rate=1.0, shift=0.1),
            non_root_std_dist=ShiftedExponentialSampler(rate=10.0, shift=0.01),
            sigma_w=cfg.sigma_w, sigma_b=cfg.sigma_b)
        scm = builder.sample(tgen)

        observed = observed_sensor_nodes(scm)
        if not observed:
            _reject("no_observed_sensor")
            continue
        wired_h = [v for v in observed if rng.random() < cfg.p_signature]
        if not wired_h:
            wired_h = [observed[int(rng.integers(len(observed)))]]
        n_root_wired = 0
        for v in wired_h:
            # sibling-parity: responsive sensors carry a monotone nonlinearity
            scm.mechanisms[v].activation = (
                nn.Identity() if rng.random() < 0.5 else Tanh())
            if scm.dag.G_0.in_degree(v) == 0:
                # wired root: re-draw its noise at non-root scale, else the
                # ~10x root noise drowns the calibrated signature
                std = float(ShiftedExponentialSampler(rate=10.0, shift=0.01).sample(tgen))
                scm.noise[v] = TorchDistributionSampler(tdist.Normal(loc=0.0, scale=std))
                n_root_wired += 1
        h_weights = {v: float(rng.normal(0.0, 1.0)) for v in wired_h}
        s_weights = {v: float(rng.normal(0.0, cfg.load_leak_scale))
                     for v in observed if rng.random() < cfg.p_load_leak}

        # excursion columns: observed sensors, at their injected positions
        inj_topo = [DRIVER_HEALTH, DRIVER_LOAD, *scm.dag.topo_order]
        obs_cols = [inj_topo.index(v) for v in observed]

        base = _settled_row(scm, h_weights, s_weights, 0.0, cfg)
        if base is None:
            _reject("no_fixed_point")
            continue

        norm = None
        for _it in range(cfg.max_calib_iter):
            top = _settled_row(scm, h_weights, s_weights, 1.0, cfg)
            if top is None:
                _reject("no_fixed_point")
                norm = None
                break
            norm = float(torch.linalg.vector_norm(top[obs_cols] - base[obs_cols]))
            if norm < 1e-9:
                _reject("zero_excursion")
                norm = None
                break
            if abs(norm - 1.0) <= cfg.norm_tol:
                break
            h_weights = {v: w / norm for v, w in h_weights.items()}
        else:
            _reject("norm_unreachable")
            norm = None
        if norm is None:
            continue

        mid_row = _settled_row(scm, h_weights, s_weights, 0.5, cfg)
        if mid_row is None:
            _reject("no_fixed_point")
            continue
        midramp = float(torch.linalg.vector_norm(mid_row[obs_cols] - base[obs_cols]))
        if not cfg.midramp_band[0] <= midramp <= cfg.midramp_band[1]:
            _reject("midramp_saturation")
            continue

        # SNR axis: rescale the observed sensors' noise so the total noise
        # norm hits the drawn target (calibration above is noise-free, so
        # this cannot invalidate it).  torch distributions expose .stddev
        # uniformly across Normal/Uniform/Laplace.
        lo, hi = np.log(cfg.noise_norm_range[0]), np.log(cfg.noise_norm_range[1])
        target_norm = float(np.exp(rng.uniform(lo, hi)))
        stds = np.array([float(scm.noise[v].distribution.stddev) for v in observed])
        k = target_norm / float(np.sqrt((stds**2).sum()))
        for v in observed:
            scm.noise[v] = _ScaledNoise(scm.noise[v], k)

        meta = dict(seed=seed, tries=_try + 1, K=K, edge_prob=edge_prob,
                    wired_h=sorted(h_weights), wired_s=sorted(s_weights),
                    n_root_wired=n_root_wired, excursion_norm=norm,
                    midramp=midramp, noise_norm=target_norm,
                    rejections=rejections)
        return CalibratedEmission(scm=scm, h_weights=h_weights,
                                  s_weights=s_weights, meta=meta)

    raise RuntimeError(
        f"no calibratable emission SCM within max_tries={cfg.max_tries}; "
        f"rejections: {rejections}")


# --------------------------------------------------------------------------- #
# Step 3: sampling loop -> Unit records                                       #
# --------------------------------------------------------------------------- #
@dataclass
class HybridConfig:
    """Unit-level knobs of the hybrid prior.

    ``latent`` is the SHARED latent-block configuration (the sibling
    generator's ``TSCMPriorConfig``): only its SDE/load/validity fields are
    consumed here (drift/diffusion ranges and shapes, ``healthy_range``,
    ``stressor_*``/seasonality, ``hard_cap``, ramp/length bounds,
    ``x_ceiling``, ``max_retries``); its sensor-side fields are ignored --
    the emission world is ``emission``.  Both prior arms must consume the
    same ``latent`` instance for the controlled comparison to hold.

    Deviations from the sibling generator, both review-mandated: the load
    AR(1) starts at stationarity (z0 ~ N(0,1), not z=0 -- a z=0 start
    leaves a deterministic window-start compression for ~1/(1-rho) steps),
    and units start at X'(0) = x0 ~ U(*x0_range*) instead of exactly 0
    (N-CMAPSS-style unknown initial health; the burn-in prefix is held at
    x0 so sensors settle at the degraded baseline).  ``load_noise_range``
    reaches down to 1e-3 so C-MAPSS/N-CMAPSS-grade near-exact load
    observability is inside the prior support.
    """

    latent: TSCMPriorConfig = field(default_factory=TSCMPriorConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    burn_in: int = 50
    rul_cap: float = 125.0
    censor_prob: float = 0.3
    x0_range: tuple = (0.0, 0.1)
    n_load_channels: int = 2
    load_gain_log_sigma: float = 0.3
    load_noise_range: tuple = (1e-3, 0.15)
    max_emission_scms: int = 3       # fresh emission SCMs per latent draw
    max_emission_retries: int = 5    # noise re-rolls per emission SCM (F12)


def _stationary_load(total_T: int, rng, lcfg) -> np.ndarray:
    """The sibling generator's load process with a stationary AR(1) start."""
    rho = float(rng.uniform(*lcfg.stressor_rho_range))
    z = float(rng.standard_normal())  # stationary init (not z=0)
    seasonal = rng.random() < lcfg.p_seasonal
    per = float(rng.uniform(*lcfg.season_period_range)) if seasonal else 1.0
    amp = float(rng.uniform(*lcfg.season_amp_range)) if seasonal else 0.0
    pha = float(rng.uniform(0.0, 2.0 * np.pi))
    s = np.empty(total_T, dtype=np.float32)
    for t in range(total_T):
        z = rho * z + np.sqrt(1.0 - rho**2) * rng.standard_normal()
        season = amp * np.sin(2.0 * np.pi * t / per + pha) if seasonal else 0.0
        s[t] = max(0.1, 1.0 + 0.4 * z + season)
    return s


def _integrate_latent(x0, onset, ops, s, rng, lcfg):
    """Euler-Maruyama from X'(0..onset) = x0, first passage at X' >= 1.

    ``s`` is the load in emitted coordinates (post burn-in); mirrors the
    sibling's ``_integrate_to_failure`` (dt = 1, load-modulated drift,
    clipped state) plus the initial-health offset.
    """
    X = np.full(len(s), np.float32(x0))
    for t in range(onset + 1, len(s)):
        x = float(X[t - 1])
        x = x + s[t - 1] * float(ops["mu"](x)) + float(ops["sigma"](x)) * rng.standard_normal()
        x = min(max(x, 0.0), lcfg.x_ceiling)
        X[t] = np.float32(x)
        if x >= 1.0:
            return X[:t + 1], t
    return X, -1


def _sample_latent(gen: TSCMGenerator, hcfg: HybridConfig):
    """One valid latent draw: (X'_window, s_full, onset, t_fail, censored, x0, ops)."""
    lcfg = gen.cfg
    for _ in range(lcfg.max_retries):
        x0 = float(gen.rng.uniform(*hcfg.x0_range))
        onset = int(gen.rng.integers(*lcfg.healthy_range))
        s_full = _stationary_load(hcfg.burn_in + lcfg.hard_cap, gen.rng, lcfg)
        ops = gen._sample_operators()
        X, t_fail = _integrate_latent(x0, onset, ops, s_full[hcfg.burn_in:], gen.rng, lcfg)
        if t_fail < 0 or not np.all(np.isfinite(X)):
            continue
        if float(X.max()) >= lcfg.x_ceiling * 0.99:
            continue
        ramp = t_fail - onset
        if not (lcfg.min_ramp <= ramp <= lcfg.max_ramp):
            continue
        if not (lcfg.min_length <= t_fail + 1 <= lcfg.max_length):
            continue
        T = t_fail + 1
        censored = False
        if gen.rng.random() < hcfg.censor_prob and (onset + lcfg.min_ramp) < t_fail:
            lo = max(onset + lcfg.min_ramp, lcfg.min_length)
            if lo < t_fail:
                T = int(gen.rng.integers(lo, t_fail))
                censored = True
        return X[:T], s_full[:hcfg.burn_in + T], onset, t_fail, censored, x0, ops
    raise RuntimeError(
        "no valid latent draw within max_retries -- check the latent config "
        "(drift/diffusion ranges vs ramp/length bounds)")


def sample_hybrid_unit(seed: int, hcfg: HybridConfig | None = None,
                       unit_id: int = -1) -> Unit:
    """One hybrid-prior unit: shared latent block -> calibrated dotime emission.

    Emission failures never touch the latent draw (the review's F12 rule):
    a diverged rollout re-rolls only the emission noise, and a repeatedly
    diverging emission SCM is replaced by a freshly calibrated one while
    X'(t) and s(t) stay fixed, so rejection cannot filter the latent prior
    through emission stability.
    """
    hcfg = hcfg or HybridConfig()
    rng = np.random.default_rng(seed)
    gen = TSCMGenerator(hcfg.latent, seed=int(rng.integers(2**31 - 1)))
    X_win, s_full, onset, t_fail, censored, x0, ops = _sample_latent(gen, hcfg)
    T = len(X_win)

    health_carrier = np.concatenate(
        [np.full(hcfg.burn_in, np.float32(x0)), X_win])
    out = None
    for _scm_try in range(hcfg.max_emission_scms):
        try:
            ce = sample_calibrated_emission(int(rng.integers(2**31 - 1)), hcfg.emission)
        except RuntimeError:
            continue
        injected = inject_drivers(ce.scm, health_carrier, s_full,
                                  ce.h_weights, ce.s_weights)
        for _roll_try in range(hcfg.max_emission_retries):
            torch.manual_seed(int(rng.integers(2**31 - 1)))
            try:
                out = observational_rollout(injected, burn_in=hcfg.burn_in)
                break
            except RolloutDiverged:
                continue
        if out is not None:
            break
    if out is None:
        raise RuntimeError("emission failed for this latent draw "
                           "(all SCMs/noise re-rolls diverged)")

    # emitted sensors: observed SCM nodes only (never the driver columns)
    inj_topo = injected.dag.topo_order
    obs_cols = [inj_topo.index(v) for v in observed_sensor_nodes(ce.scm)]
    sensors = out[:, obs_cols].numpy().astype(np.float32)

    # dedicated noisy load channels (multiplicative gain, no offset)
    s_win = s_full[hcfg.burn_in:]
    gain = np.exp(rng.normal(0.0, hcfg.load_gain_log_sigma, hcfg.n_load_channels))
    lo, hi = np.log(hcfg.load_noise_range[0]), np.log(hcfg.load_noise_range[1])
    lnz = np.exp(rng.uniform(lo, hi, hcfg.n_load_channels))
    L = gain * s_win[:, None] + lnz * rng.standard_normal((T, hcfg.n_load_channels))
    sensors = np.concatenate([sensors, L.astype(np.float32)], axis=1)

    # piecewise-linear label, family-free GLOBAL cap (C-MAPSS convention,
    # no pre-onset plateau special-casing)
    rul = np.clip(t_fail - np.arange(T), 0, hcfg.rul_cap).astype(np.float32)

    gg = gen_grid()
    mu_grid = np.asarray(ops["mu"](gg), dtype=np.float32)
    sg_grid = np.asarray(ops["sigma"](gg), dtype=np.float32)
    A = np.stack([np.ones_like(gg), gg], 1)
    (a0, a1), *_ = np.linalg.lstsq(A, mu_grid, rcond=None)
    (b0, b1), *_ = np.linalg.lstsq(A, sg_grid, rcond=None)
    params = np.array([max(a0, 1e-6), max(a1, 1e-6),
                       max(b0, 1e-6), max(b1, 1e-6)], dtype=np.float32)

    return Unit(sensors=sensors, hi=X_win.astype(np.float32), onset=onset,
                t_fail=int(t_fail), rul=rul, censored=bool(censored),
                unit_id=unit_id, mu_grid=mu_grid, sigma_grid=sg_grid,
                params=params, shapes=ops["shapes"],
                n_process=len(obs_cols))


def generate_units(n: int, seed: int, hcfg: HybridConfig | None = None):
    """Yield ``n`` units; failed draws are skipped and counted, not retried."""
    hcfg = hcfg or HybridConfig()
    rng = np.random.default_rng(seed)
    made = dropped = 0
    while made < n:
        unit_seed = int(rng.integers(2**31 - 1))
        try:
            u = sample_hybrid_unit(unit_seed, hcfg, unit_id=made)
        except RuntimeError:
            dropped += 1
            if dropped > 10 * n + 50:
                raise RuntimeError(
                    f"dropped {dropped} draws for {made}/{n} units -- "
                    f"config likely infeasible")
            continue
        made += 1
        yield u


def _json_cfg(hcfg: HybridConfig) -> HybridConfig:
    """Copy of the config with activation modules replaced by class names,
    so the HDF5 writer can JSON-serialise it into the file attrs."""
    from dataclasses import replace
    return replace(hcfg, emission=replace(
        hcfg.emission,
        activations=tuple(type(a).__name__ for a in hcfg.emission.activations)))


def write_hybrid_dataset(path: str, n: int, seed: int,
                         hcfg: HybridConfig | None = None,
                         progress_every: int = 0) -> list[Unit]:
    """Stream ``n`` hybrid units into the shared HDF5 layout.

    Returns the first 500 units as an in-memory probe for diagnostics.
    """
    hcfg = hcfg or HybridConfig()
    n_channels = hcfg.emission.n_nodes + hcfg.n_load_channels
    probe: list[Unit] = []
    with HDF5Writer(path, _json_cfg(hcfg), n_sensors=n_channels) as w:
        for k, u in enumerate(generate_units(n, seed, hcfg)):
            w.append(u)
            if len(probe) < 500:
                probe.append(u)
            if progress_every and (k + 1) % progress_every == 0:
                print(f"  {k + 1}/{n} ...")
    return probe


if __name__ == "__main__":
    import argparse

    from rulbench.dataset_io import dataset_info
    from rulbench.synthetic.rul_sde import sanity_check

    ap = argparse.ArgumentParser(
        description="Generate hybrid-prior RUL units (dotime emission + shared SDE latent block)")
    ap.add_argument("--out", required=True, help="target file, e.g. data/hybrid_train.h5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-every", type=int, default=100)
    a = ap.parse_args()

    hcfg = HybridConfig()
    print(f"Generating {a.n} hybrid units -> {a.out}  "
          f"({hcfg.emission.n_nodes} sensors + {hcfg.n_load_channels} load channels, "
          f"seed={a.seed})")
    probe = write_hybrid_dataset(a.out, a.n, a.seed, hcfg,
                                 progress_every=a.check_every)

    print("\nDiagnostics on the first 500 units:")
    sanity_check(probe)
    info = dataset_info(a.out)
    print(f"\n{info['n_units']} units, {info['total_timesteps']} timesteps, "
          f"{info['size_mb']:.1f} MB "
          f"({1000 * info['size_mb'] / max(info['n_units'], 1):.1f} kB/unit)")
