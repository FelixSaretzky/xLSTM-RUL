"""
Offline generation of a prior dataset -- TSCM with SDE degradation.

STANDALONE: depends only on numpy, h5py and dataset_io. It does NOT import the
model code, so generating data does not require torch.

    python generate_dataset.py --out data/train.h5 --n 20000 --seed 0
    python generate_dataset.py --out data/val.h5   --n 2000  --seed 1
    python generate_dataset.py --out data/train_ablation.h5 --n 20000 --no-couple

Produces units for BOTH model variants:
  (A) baseline: piecewise-linear RUL label -> supervised regression
  (B) main path: ground-truth LATENT STATE + SDE OPERATORS
                 -> amortised posterior inference, label-free at inference

STRUCTURE (dynamic Bayesian network / temporal SCM):
  A fixed template graph per unit, evaluated over time. Two edge types inside
  the sensor network:
      intra-slice : instantaneous sensor->sensor edges (acyclic, topo order)
      inter-slice : self-AR  x[t-1,i] -> x[t,i]
  Two exogenous, time-varying drivers feed in:
      s(t)   stressor / load  (AR(1), rho per unit, optionally seasonal)
      X'(t)  degradation      (SDE, regime switch at onset)

NORMALISATION:
  The latent state lives in units of the failure threshold: failure at X' = 1.
  The population spread of the tolerance is absorbed into the base-rate prior
  -- equivalent, since only the state-to-threshold ratio is identifiable.

  MODELLING CHOICE: the sensor signature is defined over X', and the amplitude
  vector is calibrated so that the deterministic sensor excursion between
  X'=0 and X'=1 has norm 1. Hence  ||x(t) - baseline|| ~ X'(t): the health
  index becomes a quantity that can be READ OFF the sensors, and the failure
  threshold lives on an observable scale.

TWO INDEPENDENT DEGREES OF FREEDOM:
  * onset     -- WHEN degradation starts (absolute healthy duration)
  * operators -- HOW FAST and IN WHAT SHAPE it proceeds
  The ramp length is EMERGENT (integration until first passage), not tied to
  the onset.

Ablation switches: --no-couple (no d->sensor edges), --signature-norm free.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import argparse
import numpy as np

from rulbench.dataset_io import Unit, gen_grid, GEN_GRID_N, HDF5Writer, dataset_info


def _loguniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def total_channels(cfg) -> int:
    """Observed dimensionality = process sensors + load channels."""
    return cfg.n_sensors + cfg.n_load_channels


# =====================================================================
# SHAPE LIBRARY for the SDE operators
# =====================================================================
# Every shape f satisfies f(0) = 1 -- the absolute rate sits separately in a
# base factor. SCALE and SHAPE are therefore decoupled prior axes.
#
# Why nonlinear? The functional head can represent arbitrary mu(x), sigma(x);
# a purely linear prior would teach it that all operators are linear. The
# shapes below are deliberately modelled on degradation physics:
#   linear       classic Wiener drift
#   power        super-proportional, Paris-law-like crack growth (p>1)
#   sublinear    degressive, run-in wear (p<1)
#   exponential  self-accelerating, bearing degradation
#   saturating   rise with plateau
#   knee         long flat, then a sharp bend  <- the "20 minute" case
#   exppoly      generic smooth shapes (FIM-ODE style, polynomial in exponent)

DRIFT_SHAPES = ("linear", "power", "sublinear", "exponential",
                "saturating", "knee", "exppoly")
DIFF_SHAPES = ("constant", "linear", "power", "exppoly")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def sample_shape(rng, kind: str):
    """Draw a shape function f with f(0)=1. Returns (name, callable)."""
    if kind == "linear":
        a = _loguniform(rng, 0.1, 30.0)
        return kind, (lambda x, a=a: 1.0 + a * x)
    if kind == "power":
        a, p = _loguniform(rng, 0.1, 30.0), rng.uniform(1.5, 4.0)
        return kind, (lambda x, a=a, p=p: 1.0 + a * np.power(np.clip(x, 0, None), p))
    if kind == "sublinear":
        a, p = _loguniform(rng, 0.1, 20.0), rng.uniform(0.2, 0.8)
        return kind, (lambda x, a=a, p=p: 1.0 + a * np.power(np.clip(x, 0, None), p))
    if kind == "exponential":
        k = rng.uniform(0.5, 5.0)
        return kind, (lambda x, k=k: np.exp(k * x))
    if kind == "saturating":
        a, h = _loguniform(rng, 0.5, 20.0), rng.uniform(0.05, 0.5)
        return kind, (lambda x, a=a, h=h: 1.0 + a * x / (x + h))
    if kind == "knee":
        a = _loguniform(rng, 1.0, 50.0)
        xk, w = rng.uniform(0.4, 0.9), rng.uniform(0.02, 0.12)
        base = _sigmoid(-xk / w)
        return kind, (lambda x, a=a, xk=xk, w=w, b=base:
                      1.0 + a * (_sigmoid((x - xk) / w) - b))
    # exppoly: f = exp(c1 x + c2 x^2 + c3 x^3) -> f(0)=1, strictly positive
    c = rng.normal(0, 1.5, size=3)
    return "exppoly", (lambda x, c=c: np.exp(np.clip(
        c[0] * x + c[1] * x ** 2 + c[2] * x ** 3, -3.0, 5.0)))


def constant_shape():
    return "constant", (lambda x: np.ones_like(np.asarray(x, dtype=float)))


@dataclass
class TSCMPriorConfig:
    n_sensors: int = 8
    # --- sensor graph ---
    graph_type: str = "mixed"                  # none | chain | dag | hub | mixed
    edge_density_range: tuple = (0.05, 0.5)    # nur "dag": Dichte pro Unit
    edge_weight_scale: float = 0.5
    self_ar_range: tuple = (0.0, 0.6)
    # --- node distributions (observation model) ---
    noise_families: tuple = ("gauss", "student_t", "skewed")
    student_df_range: tuple = (2.5, 15.0)
    skew_range: tuple = (0.2, 0.7)   # sg=1.0 would be extreme (kurtosis ~50)
    # NOISE LEVEL as a calibrated quantity: what is drawn is the TOTAL NORM of
    # the noise vector. With signature_norm="unit" (signature norm = 1) this
    # makes  SNR at end of life = 1 / ||sigma||  an explicit, log-uniform prior
    # axis (roughly 2.5 .. 50). Previously noise_scale was a fixed 0.3 per
    # channel -> ||sigma|| ~ 0.85 at 8 sensors, i.e. SNR ~ 1: the signature
    # drowned in the noise (measured).
    noise_norm_range: tuple = (0.02, 0.4)
    noise_scale: float = 0.3                   # only used for signature_norm="free"
    sensor_clip: float = 50.0
    nonlinearities: tuple = ("tanh", "relu", "sin", "identity")
    # --- signature (degradation -> sensor) ---
    p_signature: float = 0.6
    signature_modes: tuple = ("location", "scale", "both")
    sig_scale_gain_range: tuple = (0.1, 3.0)   # variance factor at end of life
    # SIGNATURE NORMALISATION -- makes the state READABLE from the sensors:
    #   "unit": amplitudes are scaled so that the deterministic sensor
    #           excursion between X'=0 and X'=1 has norm 1. Then
    #           ||x(t) - baseline|| = X'(t) -- the health index is computable
    #           from the sensors, and the threshold X'=1 lives on an
    #           observable scale.
    #   "free": earlier behaviour (A ~ N(0,1) independently) -> X' only weakly
    #           identifiable through the prior. Kept as an ablation.
    signature_norm: str = "unit"               # "unit" | "free"
    # Signature sensors only get shape-preserving nonlinearities so the norm
    # relation is not destroyed inside the generator (relu can clip the
    # signature away, sin makes it ambiguous).
    signature_nl: tuple = ("identity", "tanh")
    # --- stressor / load ---
    # DEDICATED LOAD CHANNELS: operating load is recorded on real machines
    # (speed, setpoints, power). Treating it as unknown would be needlessly
    # pessimistic -- and load otherwise distorts the drift without the model
    # being able to correct for it. The channels sit at the BACK of the sensor
    # matrix: sensors[:, :n_sensors] = process, [:, n_sensors:] = load.
    n_load_channels: int = 2
    load_gain_log_sigma: float = 0.3           # channel gain (log-normal around 1)
    load_noise_range: tuple = (0.01, 0.15)     # measurement noise of load channels
    p_stressor_obs: float = 0.3                # additionally: load leaking into process sensors
    stressor_rho_range: tuple = (0.5, 0.98)    # PER UNIT
    p_seasonal: float = 0.5
    season_period_range: tuple = (20, 300)
    season_amp_range: tuple = (0.05, 0.5)
    # --- degree of freedom 1: WHEN degradation starts ---
    healthy_range: tuple = (20, 400)
    # --- degree of freedom 2: HOW FAST / IN WHAT SHAPE degradation proceeds ---
    # theta is normalised away (failure at X'=1). Its population spread is
    # absorbed into the base-rate distribution -- mathematically equivalent,
    # since only the state-to-threshold ratio is identifiable anyway.
    dt: float = 1.0
    drift_base_range: tuple = (3e-4, 0.08)     # mu(0)
    diff_base_range: tuple = (3e-4, 0.10)      # sigma(0)
    drift_shapes: tuple = DRIFT_SHAPES
    diff_shapes: tuple = DIFF_SHAPES
    shape_clip: tuple = (0.05, 200.0)          # numerical bounds on f(x)
    # --- safeguards / rejection ---
    hard_cap: int = 3000
    x_ceiling: float = 5.0                     # normalised ceiling (failure at 1)
    min_ramp: int = 8
    max_ramp: int = 1500
    min_length: int = 120
    max_length: int = 2000
    max_retries: int = 50
    # --- observation / label ---
    censor_prob: float = 0.3
    rul_cap: float = 125.0
    couple_hi: bool = True
    x_0_range: tuple = (0.0, 0.05)


class TSCMGenerator:
    def __init__(self, cfg: TSCMPriorConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._next_id = 0
        self.n_rejected = 0

    # ---------------- node mechanism ----------------
    @staticmethod
    def _nl(kind, v):
        if kind == "tanh": return np.tanh(v)
        if kind == "relu": return v if v > 0 else 0.0
        if kind == "sin":  return np.sin(v)
        return v

    def _noise_draw(self, fam, par):
        """Standardised noise (mean 0, variance ~1) per family."""
        rng = self.rng
        if fam == "student_t":
            df = par
            return rng.standard_t(df) / np.sqrt(df / (df - 2.0))
        if fam == "skewed":
            sg = par                                   # right-skewed, standardised
            z = rng.normal()
            m = np.exp(sg ** 2 / 2)
            v = np.sqrt((np.exp(sg ** 2) - 1) * np.exp(sg ** 2))
            return (np.exp(sg * z) - m) / v
        return rng.normal()

    # ---------------- graph ----------------
    def _sample_graph(self):
        c, rng = self.cfg, self.rng
        n = c.n_sensors
        order = rng.permutation(n)
        W = np.zeros((n, n))                            # W[i,j]: j -> i (instantan)

        gtype = c.graph_type
        if gtype == "mixed":
            gtype = str(rng.choice(["chain", "dag", "hub"]))

        if gtype == "none":
            # No sensor-sensor edges at all. Sensors are then conditionally
            # independent given X' and s -- the textbook observation model of a
            # state-space model. Ablation for "does the prior need the graph?":
            # measured cross-sensor correlation is dominated by the shared
            # drivers anyway (0.161/0.170/0.178 for chain/dag/hub).
            pass
        elif gtype == "chain":

            for k in range(n - 1):
                W[order[k + 1], order[k]] = rng.normal(0, c.edge_weight_scale)
        elif gtype == "hub":                            # order[0] drives all others
            hub = order[0]
            for i in order[1:]:
                W[i, hub] = rng.normal(0, c.edge_weight_scale)
        else:                                           # "dag": density per unit
            dens = rng.uniform(*c.edge_density_range)
            for pi, i in enumerate(order):
                for j in order[:pi]:
                    if rng.random() < dens:
                        W[i, j] = rng.normal(0, c.edge_weight_scale)

        # --- signature: which sensors see the degradation, and HOW ---
        has_sig = (rng.random(n) < c.p_signature) if c.couple_hi else np.zeros(n, bool)
        modes = [str(rng.choice(c.signature_modes)) if h else None for h in has_sig]
        # With "unit" at least one location sensor is required, otherwise
        # there is no excursion to scale to norm 1.
        if c.couple_hi and c.signature_norm == "unit" and not any(
                m in ("location", "both") for m in modes):
            j = int(rng.integers(n))
            has_sig[j] = True
            modes[j] = str(rng.choice(("location", "both")))
        A_loc = np.array([rng.normal(0, 1) if m in ("location", "both") else 0.0
                          for m in modes])              # Amplitude am Lebensende
        g_scl = np.array([float(_loguniform(rng, *c.sig_scale_gain_range))
                          if m in ("scale", "both") else 0.0 for m in modes])

        # --- node distributions ---
        fams, pars = [], []
        for _ in range(n):
            f = str(rng.choice(c.noise_families))
            if f == "student_t":   p = float(rng.uniform(*c.student_df_range))
            elif f == "skewed":    p = float(rng.uniform(*c.skew_range))
            else:                  p = 0.0
            fams.append(f); pars.append(p)

        # Nonlinearities: shape-preserving for signature sensors, free otherwise
        nls = []
        for i in range(n):
            pool = (c.signature_nl if (modes[i] is not None
                                       and c.signature_norm == "unit")
                    else c.nonlinearities)
            nls.append(str(rng.choice(pool)))

        g = dict(order=order, W=W, gtype=gtype,
                 self_ar=rng.uniform(*c.self_ar_range, size=n),
                 A_loc=A_loc, g_scl=g_scl, modes=modes,
                 w_str=rng.normal(0, 0.3, size=n) * (rng.random(n) < c.p_stressor_obs),
                 nl=nls,
                 noise=self._sample_noise(n),
                 fam=fams, fpar=pars,
                 n_sig=int(has_sig.sum()))
        if c.couple_hi and c.signature_norm == "unit":
            self._calibrate_signature(g)
        return g

    def _sample_noise(self, n):
        """Per-channel noise levels. With "unit" the TOTAL NORM is drawn, so that
        the signal-to-noise ratio at end of life is a controlled prior axis
        instead of a by-product of fixed constants."""
        c, rng = self.cfg, self.rng
        w = 0.5 + rng.random(n)                       # relative weighting
        if c.signature_norm != "unit":
            return c.noise_scale * w
        target = float(_loguniform(rng, *c.noise_norm_range))
        return w / np.linalg.norm(w) * target

    # ---------------- calibrate the signature to unit norm ----------------
    def _steady_state(self, g, xval, s_ref=1.0, n_steps=80):
        """Noise-free fixed point of the sensor dynamics at constant X' and s.
        Accounts for graph propagation, self-AR and nonlinearities -- the
        calibration is therefore numerically exact, not merely formal."""
        n = self.cfg.n_sensors
        x = np.zeros(n)
        for _ in range(n_steps):
            xn = np.zeros(n)
            for i in g["order"]:
                mean = (g["W"][i] @ xn + g["self_ar"][i] * x[i]
                        + g["A_loc"][i] * xval + g["w_str"][i] * s_ref)
                xn[i] = self._nl(g["nl"][i], mean)
            x = xn
        return x

    def _calibrate_signature(self, g, n_iter=4):
        """Scale A_loc so that ||excursion(X'=1) - excursion(X'=0)|| = 1. Then
        ||x(t) - baseline|| ~ X'(t) and the health index is computable directly
        from the sensors. Exact for linear nodes, iterative under tanh."""
        for _ in range(n_iter):
            d = self._steady_state(g, 1.0) - self._steady_state(g, 0.0)
            nrm = float(np.linalg.norm(d))
            if not np.isfinite(nrm) or nrm < 1e-9:
                return g
            g["A_loc"] = g["A_loc"] / nrm
            if abs(nrm - 1.0) < 1e-3:
                break
        return g

    # ---------------- stressor: rho per unit + seasonality ----------------
    # External pressure on the system, sampled per sensor group
    def _stressor(self, T):
        c, rng = self.cfg, self.rng
        rho = float(rng.uniform(*c.stressor_rho_range))
        z = float(rng.standard_normal())
        s = np.empty(T)
        seasonal = rng.random() < c.p_seasonal
        per = float(rng.uniform(*c.season_period_range)) if seasonal else 0.0
        amp = float(rng.uniform(*c.season_amp_range)) if seasonal else 0.0
        pha = float(rng.uniform(0, 2 * np.pi)) if seasonal else 0.0
        for t in range(T):
            z = rho * z + np.sqrt(1 - rho ** 2) * rng.normal()
            season = amp * np.sin(2 * np.pi * t / per + pha) if seasonal else 0.0
            s[t] = max(0.1, 1.0 + 0.4 * z + season)
        return s, dict(rho=rho, seasonal=seasonal, period=per, amp=amp)

    # ---------------- SDE ----------------
    def _sample_operators(self):
        """Draw base rates AND shapes -> two separate prior axes."""
        c, rng = self.cfg, self.rng
        mu0 = float(_loguniform(rng, *c.drift_base_range))
        sg0 = float(_loguniform(rng, *c.diff_base_range))
        dname, f = sample_shape(rng, str(rng.choice(c.drift_shapes)))
        gk = str(rng.choice(c.diff_shapes))
        gname, g = constant_shape() if gk == "constant" else sample_shape(rng, gk)
        lo, hi = c.shape_clip
        mu = lambda x, f=f, mu0=mu0: mu0 * np.clip(f(x), lo, hi)
        sg = lambda x, g=g, sg0=sg0: sg0 * np.clip(g(x), lo, hi)
        return dict(mu=mu, sigma=sg, shapes=(dname, gname))

    def _integrate_to_failure(self, onset, ops, s):
        """Euler-Maruyama in the NORMALISED state space: failure at X' >= 1."""
        c, rng = self.cfg, self.rng
        dt, sq = c.dt, np.sqrt(c.dt) 
        x0 = float(rng.uniform(*c.x0_range))       # e.g. (0.0, 0.1) in threshold units
        X = np.full(c.hard_cap, x0)
        #X = np.zeros(c.hard_cap)
        for t in range(onset + 1, c.hard_cap):
            x = X[t - 1]
            mu = s[t-1] * float(ops["mu"](x))
            sig = float(ops["sigma"](x))
            x = x + mu * dt + sig * sq * rng.normal()
            # Protection from exploding errors
            x = min(max(x, 0.0), c.x_ceiling)
            X[t] = x
            # Integrate until First-Passage Step
            if x >= 1.0:
                return X[:t + 1], t
        return X, -1

    def _valid(self, X, onset, t_fail):
        c = self.cfg
        if t_fail < 0:                                    return False
        if not np.all(np.isfinite(X)):                    return False
        if X.max() >= c.x_ceiling * 0.99:                 return False
        ramp = t_fail - onset
        if ramp < c.min_ramp or ramp > c.max_ramp:        return False
        if (t_fail + 1) < c.min_length:                   return False
        if (t_fail + 1) > c.max_length:                   return False
        return True

    # ---------------- unit ----------------
    def sample_unit(self) -> Unit:
        c, rng = self.cfg, self.rng
        n = c.n_sensors

        # Sample Stressor, Health Index and integrate it till it fails
        X = None; t_fail = -1; onset = 0; s = None; ops = None
        for _ in range(c.max_retries):
            onset = int(rng.integers(*c.healthy_range))
            s_full, _ = self._stressor(c.hard_cap)
            ops = self._sample_operators()
            Xc, tf = self._integrate_to_failure(onset, ops, s_full)
            if self._valid(Xc, onset, tf):
                X, t_fail, s = Xc, tf, s_full[:len(Xc)]
                break
            self.n_rejected += 1
        if X is None:
            raise RuntimeError("No valid trajectory after max_retries -- "
                               "check hyperprior ranges (base rates vs. shapes?).")

        # --- observation window: post-failure is discarded ---
        # Censoring the synthesized set
        T = t_fail + 1
        censored = False
        if rng.random() < c.censor_prob and (onset + c.min_ramp) < t_fail:
            lo = max(onset + c.min_ramp, c.min_length)
            if lo < t_fail:
                T = int(rng.integers(lo, t_fail)); censored = True
        X = X[:T]; s = s[:T]

        # --- sensors: DBN evaluation, signature via X' ---
        # Signature is used as an influence on the 
        g = self._sample_graph()
        x = np.zeros((T, n)); clip = c.sensor_clip
        for t in range(T):
            xp = x[t - 1] if t > 0 else np.zeros(n)
            for i in g["order"]:                          # topology order
                # Here is the topology used to get the location of the node in the graph
                mean = (g["W"][i] @ x[t]                  # intra-slice
                        + g["self_ar"][i] * xp[i]         # inter-slice
                        + g["A_loc"][i] * X[t]            # location signature
                        + g["w_str"][i] * s[t])           # stressor
                scale = g["noise"][i] * (1.0 + g["g_scl"][i] * X[t])   # scale signature
                eps = self._noise_draw(g["fam"][i], g["fpar"][i])
                x[t, i] = np.clip(self._nl(g["nl"][i], mean) + scale * eps, -clip, clip)
        # --- load channels: direct, noisy observation of s(t) ---
        # Purely multiplicative (no offset) so that L/mean(L) ~ s/mean(s) and a
        # model can read the relative load without calibration.
        if c.n_load_channels > 0:
            gain = np.exp(rng.normal(0.0, c.load_gain_log_sigma, c.n_load_channels))
            lnz = _loguniform(rng, *c.load_noise_range, size=c.n_load_channels)
            L = (gain * s[:, None]
                 + lnz * rng.normal(size=(T, c.n_load_channels)))
            x = np.concatenate([x, L.astype(np.float64)], axis=1)

        if not np.all(np.isfinite(x)):
            return self.sample_unit()

        # --- store the operators on the fine generator grid (targets) ---
        gg = gen_grid()
        mu_grid = np.asarray(ops["mu"](gg), dtype=np.float32)
        sg_grid = np.asarray(ops["sigma"](gg), dtype=np.float32)

        # --- best LINEAR fit -> parametric ablation (deliberately
        #     misspecified as soon as the shape is nonlinear) ---
        A = np.stack([np.ones_like(gg), gg], 1)
        (a0, a1), *_ = np.linalg.lstsq(A, mu_grid, rcond=None)
        (b0, b1), *_ = np.linalg.lstsq(A, sg_grid, rcond=None)
        params = np.array([max(a0, 1e-6), max(a1, 1e-6),
                           max(b0, 1e-6), max(b1, 1e-6)], dtype=np.float32)

        # --- (A) piecewise-linear RUL label for the baseline ---
        tt = np.arange(T)
        rul = np.clip(t_fail - tt, 0, c.rul_cap).astype(np.float32)
        # rul[:onset] = min(t_fail - onset, c.rul_cap) # Could be a leakage

        uid = self._next_id; self._next_id += 1
        return Unit(sensors=x.astype(np.float32), hi=X.astype(np.float32),
                    onset=int(onset), t_fail=int(t_fail), rul=rul,
                    censored=bool(censored), unit_id=uid,
                    mu_grid=mu_grid, sigma_grid=sg_grid,
                    params=params, shapes=ops["shapes"],
                    n_process=int(c.n_sensors))


def sanity_check(units, gen=None):
    import collections
    tf = np.array([u.t_fail for u in units]); on = np.array([u.onset for u in units])
    ln = np.array([len(u.sensors) for u in units]); cen = np.mean([u.censored for u in units])
    ramp = tf - on
    hi_end = np.array([u.hi[-1] for u in units])
    ratio = np.array([u.mu_grid[-1] / max(u.mu_grid[0], 1e-9) for u in units])
    # Curvature measure: relative deviation from the best-fit straight line
    curv = []
    for u in units:
        gg = gen_grid(); A = np.stack([np.ones_like(gg), gg], 1)
        fit = A @ np.linalg.lstsq(A, u.mu_grid, rcond=None)[0]
        curv.append(np.abs(u.mu_grid - fit).max() / max(np.abs(u.mu_grid).max(), 1e-9))
    curv = np.array(curv)
    print(f"  t_fail:  {tf.min()}..{tf.max()}   (std {tf.std():.0f})")
    print(f"  onset:   {on.min()}..{on.max()}   (std {on.std():.0f})")
    print(f"  ramp:    {ramp.min()}..{ramp.max()}   (std {ramp.std():.0f})   <- emergent")
    print(f"  length:  {ln.min()}..{ln.max()}   censored: {cen:.1%}")
    print(f"  X' end:  {hi_end.min():.2f}..{hi_end.max():.2f}")
    print(f"  corr(onset, ramp) = {np.corrcoef(on, ramp)[0,1]:+.2f}  (near 0 = decoupled)")
    print(f"  mu(1)/mu(0): median {np.median(ratio):.1f}  "
          f"({np.percentile(ratio,5):.1f}..{np.percentile(ratio,95):.1f})")
    print(f"  nonlinearity of mu (rel. dev. from a straight line): "
          f"median {np.median(curv):.3f}, fraction >0.05: {np.mean(curv>0.05):.0%}")
    dsh = collections.Counter(u.shapes[0] for u in units)
    gsh = collections.Counter(u.shapes[1] for u in units)
    # Empirical health index: ||x - baseline|| at end of life (should be ~1)
    dev = []
    n_proc = units[0].n_process if hasattr(units[0], "n_process") else None
    for u in units:
        if u.censored or u.onset < 20:
            continue
        p = u.sensors[:, :u.n_process] if u.n_process else u.sensors
        base = p[:u.onset].mean(0)
        dev.append(float(np.linalg.norm(p[-1] - base)))
    if dev:
        print(f"  ||x - baseline|| at end of life: median {np.median(dev):.2f} "
              f"(should be ~1 when signature_norm='unit')")
    print(f"  drift shapes:     {dict(dsh.most_common())}")
    print(f"  diffusion shapes:{dict(gsh.most_common())}")
    if gen is not None:
        print(f"  rejected trajectories: {gen.n_rejected}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="target file, e.g. data/train.h5")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sensors", type=int, default=8, help="process sensors")
    ap.add_argument("--n-load", type=int, default=2, help="dedicated load channels")
    ap.add_argument("--graph-type", default="mixed",
                    choices=["none", "chain", "dag", "hub", "mixed"])
    ap.add_argument("--no-couple", action="store_true")
    ap.add_argument("--signature-norm", default="unit", choices=["unit", "free"])
    ap.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    ap.add_argument("--float16", action="store_true",
                    help="store sensors as float16 (half the size, lossy)")
    ap.add_argument("--check-every", type=int, default=2000)
    a = ap.parse_args()

    cfg = TSCMPriorConfig(n_sensors=a.n_sensors, n_load_channels=a.n_load,
                          graph_type=a.graph_type, couple_hi=not a.no_couple,
                          signature_norm=a.signature_norm)
    gen = TSCMGenerator(cfg, seed=a.seed)
    comp = None if a.compression == "none" else a.compression
    dtype = np.float16 if a.float16 else np.float32

    print(f"Generating {a.n} units -> {a.out}  "
          f"({cfg.n_sensors} process + {cfg.n_load_channels} load = "
          f"{total_channels(cfg)} channels, graph={cfg.graph_type}, "
          f"couple_hi={cfg.couple_hi}, "
          f"compression={a.compression}, dtype={np.dtype(dtype).name})")
    # STREAMING: constant memory footprint, only a small diagnostics buffer
    probe = []
    with HDF5Writer(a.out, cfg, n_sensors=total_channels(cfg),
                    compression=comp, dtype=dtype) as w:
        for k in range(a.n):
            u = gen.sample_unit()
            w.append(u)
            if len(probe) < 500:
                probe.append(u)
            if a.check_every and (k + 1) % a.check_every == 0:
                print(f"  {k+1}/{a.n} ...")

    print("\nDiagnostics on the first 500 units:")
    sanity_check(probe, gen)
    info = dataset_info(a.out)
    print(f"\n{info['n_units']} units, {info['total_timesteps']} timesteps, "
          f"{info['size_mb']:.1f} MB "
          f"({1000*info['size_mb']/max(info['n_units'],1):.1f} kB/unit)")