"""
Window sampling and input normalisation for pretraining.

Consumes the packed HDF5 datasets through :func:`rulbench.dataset_io.
open_dataset` and keeps every unit in memory (the datasets are tens of
MB).  A training example is a right-padded window of ``window`` cycles
ending at a sampled position ``e``:

  * content is LEFT-aligned: real steps at ``0..n_real-1``, zero pads
    after, ``mask`` True on real steps.  With a causal encoder the
    trailing pads never reach a real step's hidden state; the RUL
    readout uses the masked pooling / the last real index.
  * labels: ``hi`` per real step (clamped -- the first-passage overshoot
    beyond ``hi_clamp`` occurs only at terminal steps and is Euler
    discretisation, not signal), ``rul[e] / rul_cap`` at the window end,
    and the unit's operator curves in log space, standardised with the
    fixed constants ``dyn_log_mean``/``dyn_log_std`` so that every
    training target is O(1) and the head losses start balanced (the raw
    log operators have E[y^2] ~ 26 and would dominate the shared
    encoder's gradient ~30x).  Invert via ``y * std + mean`` before
    exponentiating (``simulate.route_b_from_predictions`` does).

WINDOW-END SAMPLING.  Uniform end positions under-represent the windows
a zero-shot benchmark actually queries: with median T well below the
window length, most windows reach step 0 and contain the healthy
plateau, and terminal windows contain the failure step.  The sampler
therefore draws three explicit strata -- terminal (``p_terminal``),
anchor-free (``p_anchor_free``: start after onset, end before failure,
only units long enough to host one), and uniform otherwise -- so the
mix is a measured, configurable quantity instead of a side effect.

NORMALISATION.  The input half of per-instance normalisation (RevIN,
Kim et al. 2022), as a stateless function: masked per-window,
per-channel statistics, never across the batch.  No inverse transform
exists because every training target lives outside sensor space
(threshold units resp. cycles).  Process channels are z-scored with a
std floor; the load channels are divided by their window mean instead,
which cancels the unknown per-unit gain while preserving the relative
load L/mean(L) ~ s/mean(s) that modulates the drift.  A near-constant
guard replaces a load slot by 1.0 when the window mean is not
identified (|mean| <= c * std), which real operating-condition channels
can trigger.  Pads are re-zeroed after normalisation.

CHANNEL LAYOUT.  The model input is a fixed ``max_channels``-wide frame
(TabPFN-style fixed maximum width; FIM-SDE does the same with
``max_dimension`` + dimension mask): units may carry ANY number of
process sensors up to ``max_channels - n_load_slots``.  The normalised
process channels are scattered into a random subset of the first
``max_channels - n_load_slots`` slots (``permute_slots``, training
only -- eval placement is the identity), unused slots stay exactly
zero, and the load channels always occupy the LAST ``n_load_slots``
slots so their semantics are positional.  Mixing datasets of different
sensor counts is the point: the prior emits the width distribution and
the projection learns to be slot-agnostic.  What this buys is the
INTERFACE -- any channel count up to 30 process + 2 load, in any order
(real sensor ordering is meaningless, so slot-agnosticism is necessary,
not cosmetic): C-MAPSS feeds 14 or 21 process sensors + 2 op settings
in the load slots (the third setting lands in a process slot and is
merely z-scored), N-CMAPSS fits as exactly 30 process + 2 load.
Distributional transfer is NOT bought here; it is carried by prior
realism, and widths outside the trained set are extrapolation.

Two measured properties to keep in mind (they shape the evaluation
protocol, not the code): the per-window information about X' grows
~sqrt(k) with the sensor count -- the excursion-norm calibration fixes
the total norm, not the Fisher information -- so performance-vs-width
curves need width-matched references; and ``path_weights`` exists
because uniform-over-units sampling makes the arm/width marginal an
accident of the file sizes rather than a declared prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rulbench.dataset_io import gen_grid, open_dataset


@dataclass
class WindowConfig:
    window: int = 512
    p_terminal: float = 0.15
    p_anchor_free: float = 0.25
    min_real: int = 64          # shortest real content of a uniform window
    hi_clamp: float = 1.2
    std_floor: float = 1e-4
    load_guard_c: float = 1.0   # load slot identified iff |mean| > c * std
                                # (measured margins: prior load windows reach
                                # |mean|/std >= 1.2, min-max scaled C-MAPSS
                                # setting jitter stays <= 0.7)
    log_eps: float = 1e-6       # offset under the log of the operator targets
    dyn_log_mean: float = -5.0  # fixed standardisation of the log operators
    dyn_log_std: float = 2.0    # (measured ~N(-5, 2^2) across both arms)
    max_channels: int = 32      # model input width (N-CMAPSS default = 32)
    n_load_slots: int = 2       # load channels, always the LAST slots
    permute_slots: bool = True  # scatter process channels (training only)


def normalize_windows(x: torch.Tensor, mask: torch.Tensor, n_process: int,
                      cfg: WindowConfig) -> torch.Tensor:
    """Masked per-window, per-channel input normalisation (see module doc).

    x (B, T, C) float32, mask (B, T) bool with True = real step.  Returns a
    new tensor; pads are exactly zero.
    """
    m = mask.unsqueeze(-1).float()                       # (B, T, 1)
    n = m.sum(dim=1).clamp(min=1.0)                      # (B, 1)
    mean = (x * m).sum(dim=1) / n                        # (B, C)
    var = (((x - mean.unsqueeze(1)) * m) ** 2).sum(dim=1) / n
    std = var.sqrt()

    proc = (x[..., :n_process] - mean[:, None, :n_process]) \
        / std[:, None, :n_process].clamp(min=cfg.std_floor)

    lmean, lstd = mean[:, n_process:], std[:, n_process:]
    identified = lmean.abs() > cfg.load_guard_c * lstd   # (B, C_load)
    denom = torch.where(identified, lmean, torch.ones_like(lmean))
    load = torch.where(identified.unsqueeze(1),
                       x[..., n_process:] / denom.unsqueeze(1),
                       torch.ones_like(x[..., n_process:]))

    out = torch.cat([proc, load], dim=-1)
    return out * m


class WindowSampler:
    """In-memory window sampler over one or more packed HDF5 datasets."""

    def __init__(self, paths: list[str], cfg: WindowConfig | None = None,
                 seed: int = 0, path_weights: list[float] | None = None):
        """``path_weights``: relative sampling mass per dataset file
        (uniform within a file).  None = uniform over the pooled units,
        which makes the arm/width marginal proportional to file sizes."""
        self.cfg = cfg or WindowConfig()
        self.rng = np.random.default_rng(seed)
        # Slot permutation draws come from a separate stream so the strata
        # sequence is reproducible independent of batch size.
        self.perm_rng = np.random.default_rng(seed + 1_000_003)
        if path_weights is not None and len(path_weights) != len(paths):
            raise ValueError("path_weights must match the number of paths")
        self.sensors, self.hi, self.rul = [], [], []
        self.onset, self.t_fail, self.censored = [], [], []
        self.mu_grid, self.sigma_grid, self.n_process = [], [], []
        self.rul_cap = None
        self.n_channels = self.cfg.max_channels     # model input width
        max_proc = self.cfg.max_channels - self.cfg.n_load_slots
        units_per_path = []
        for path in paths:
            store = open_dataset(path)
            units_per_path.append(len(store))
            cap = float(store.config.get("rul_cap", 125.0))
            if self.rul_cap is None:
                self.rul_cap = cap
            elif cap != self.rul_cap:
                raise ValueError(f"rul_cap mismatch across datasets: "
                                 f"{cap} vs {self.rul_cap}")
            for i in range(len(store)):
                u = store[i]
                n_load = u.sensors.shape[1] - u.n_process
                if n_load != self.cfg.n_load_slots:
                    raise ValueError(
                        f"{path}: unit has {n_load} load channels, "
                        f"expected {self.cfg.n_load_slots}")
                if u.n_process > max_proc:
                    raise ValueError(
                        f"{path}: {u.n_process} process sensors exceed the "
                        f"{max_proc} process slots of max_channels="
                        f"{self.cfg.max_channels}")
                self.n_process.append(u.n_process)
                self.sensors.append(u.sensors)
                self.hi.append(u.hi)
                self.rul.append(u.rul)
                self.onset.append(u.onset)
                self.t_fail.append(u.t_fail)
                self.censored.append(u.censored)
                self.mu_grid.append(u.mu_grid)
                self.sigma_grid.append(u.sigma_grid)
            store.close()
        self.n_units = len(self.sensors)
        self.lengths = np.array([len(s) for s in self.sensors])
        self.onset = np.array(self.onset)
        self.censored = np.array(self.censored)
        self.n_process = np.array(self.n_process)
        if path_weights is None:
            self.p_unit = None
        else:
            p = np.concatenate([
                np.full(n, w / max(n, 1))
                for n, w in zip(units_per_path, path_weights)])
            self.p_unit = p / p.sum()
        self.grid = gen_grid().astype(np.float32)
        w = self.cfg.window
        self.log_dyn = [
            (np.stack([np.log(m + self.cfg.log_eps),
                       np.log(s + self.cfg.log_eps)], axis=-1)
             - self.cfg.dyn_log_mean) / self.cfg.dyn_log_std
            for m, s in zip(self.mu_grid, self.sigma_grid)]
        # Anchor-free stratum: window start after onset, failure step (the
        # last step of an uncensored unit) excluded.
        self._af_units, self._af_lo, self._af_hi = [], [], []
        for i in range(self.n_units):
            lo = self.onset[i] + w                       # start = lo-w+1 > onset
            hi = self.lengths[i] - (1 if self.censored[i] else 2)
            if lo <= hi:
                self._af_units.append(i)
                self._af_lo.append(lo)
                self._af_hi.append(hi)
        self._af_units = np.array(self._af_units, dtype=np.int64)
        self._af_lo = np.array(self._af_lo, dtype=np.int64)
        self._af_hi = np.array(self._af_hi, dtype=np.int64)

    def _draw_unit(self) -> int:
        if self.p_unit is None:
            return int(self.rng.integers(self.n_units))
        return int(self.rng.choice(self.n_units, p=self.p_unit))

    def _sample_end(self) -> tuple[int, int]:
        """One (unit, end) draw according to the three strata."""
        c, rng = self.cfg, self.rng
        r = rng.random()
        if r < c.p_terminal:
            i = self._draw_unit()
            return i, int(self.lengths[i] - 1)
        if r < c.p_terminal + c.p_anchor_free and len(self._af_units) > 0:
            if self.p_unit is None:
                k = int(rng.integers(len(self._af_units)))
            else:
                p_af = self.p_unit[self._af_units]
                k = int(rng.choice(len(self._af_units), p=p_af / p_af.sum()))
            e = int(rng.integers(self._af_lo[k], self._af_hi[k] + 1))
            return int(self._af_units[k]), e
        i = self._draw_unit()
        lo = min(c.min_real - 1, self.lengths[i] - 1)
        return i, int(rng.integers(lo, self.lengths[i]))

    def _assemble(self, draws: list[tuple[int, int]],
                  permute: bool = False) -> dict:
        c = self.cfg
        B, W = len(draws), c.window
        n_slots = c.max_channels - c.n_load_slots
        x = torch.zeros(B, W, c.max_channels)
        mask = torch.zeros(B, W, dtype=torch.bool)
        y_health = torch.zeros(B, W)
        y_rul = torch.zeros(B)
        y_dyn = torch.zeros(B, len(self.grid), 2)
        last_idx = torch.zeros(B, dtype=torch.long)
        pre_onset = torch.zeros(B, dtype=torch.bool)
        hi_end = torch.zeros(B)
        for j, (i, e) in enumerate(draws):
            s = max(0, e - W + 1)
            n_real = e - s + 1
            npi = int(self.n_process[i])
            compact = torch.zeros(W, npi + c.n_load_slots)
            compact[:n_real] = torch.from_numpy(self.sensors[i][s:e + 1])
            mask[j, :n_real] = True
            compact = normalize_windows(
                compact[None], mask[j][None], npi, c)[0]
            if permute:
                slots = torch.from_numpy(
                    self.perm_rng.choice(n_slots, size=npi, replace=False))
            else:
                slots = torch.arange(npi)
            # NOTE x[j][:, slots]: with `x[j, :, slots]` the advanced index
            # jumps to the front (shape (npi, W)) and the assign transposes.
            x[j][:, slots] = compact[:, :npi]
            x[j][:, n_slots:] = compact[:, npi:]
            y_health[j, :n_real] = torch.from_numpy(
                np.minimum(self.hi[i][s:e + 1], c.hi_clamp))
            y_rul[j] = float(self.rul[i][e]) / self.rul_cap
            y_dyn[j] = torch.from_numpy(self.log_dyn[i])
            last_idx[j] = n_real - 1
            pre_onset[j] = bool(e < self.onset[i])
            hi_end[j] = float(self.hi[i][e])
        return dict(x=x, mask=mask, y_health=y_health, y_rul=y_rul,
                    y_dyn=y_dyn, last_idx=last_idx, pre_onset=pre_onset,
                    hi_end=hi_end,
                    units=torch.tensor([i for i, _ in draws]),
                    ends=torch.tensor([e for _, e in draws]))

    def sample_batch(self, batch_size: int) -> dict:
        return self._assemble([self._sample_end() for _ in range(batch_size)],
                              permute=self.cfg.permute_slots)

    def fixed_eval_draws(self, per_unit: int = 2, seed: int = 0
                         ) -> list[tuple[int, int]]:
        """Deterministic validation draws: the terminal window of every unit
        plus ``per_unit - 1`` seeded random ends."""
        rng = np.random.default_rng(seed)
        draws = []
        for i in range(self.n_units):
            draws.append((i, int(self.lengths[i] - 1)))
            for _ in range(per_unit - 1):
                lo = min(self.cfg.min_real - 1, self.lengths[i] - 1)
                draws.append((i, int(rng.integers(lo, self.lengths[i]))))
        return draws

    def eval_batches(self, draws: list[tuple[int, int]], batch_size: int):
        for a in range(0, len(draws), batch_size):
            yield self._assemble(draws[a:a + batch_size])
