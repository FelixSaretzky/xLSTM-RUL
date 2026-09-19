"""
Does the encoder state carry the OPERATORS -- and does the dyn head use it?

The same probe that settled the health head, applied to y_dyn.  Freezes a
trained model, collects the encoder state, and fits a ridge regression onto
the operator targets.  Three numbers on held-out windows:

  * R^2 of a linear probe on the pooled encoder state -- is it there?
  * R^2 of the dyn head's own mean                    -- does the head use it?
  * R^2 of a probe on two hand-built window features  -- what a formula gets

R^2 is computed PER GRID POINT and against the per-point mean, never pooled.
A pooled R^2 would credit the model for knowing that mu grows with x across
the grid, which is prior structure and not sensor information -- the same
mistake that made the flat reference (0.187) look beatable when the
per-point one (0.049) was not.

The hand-built baseline uses what a formula can read off one window: the OLS
slope of the health-index proxy over time, and the residual spread around
that trend.  Measured model-free, that slope tracks the true drift with
log-correlation up to 0.44 under ideal conditions, so a probe scoring far
below it means the encoder discarded something a formula keeps.

    python analyze_encoder_dyn.py [ckpt] [data] [--zero-x|--zero-stats]

``--zero-x`` blanks the window but keeps the instance statistics, ``--zero-stats``
does the reverse.  The old ``--zero-inputs`` control blanked both at once and so
could not say which pathway carried the signal.

The calibration block at the end answers the other half: a head whose mean is
informative (R^2 > 0) but whose NLL is worse than the uninformed reference is
mispredicting its own uncertainty, not its location.  ``sd(residual)`` versus
``mean exp(log_std)`` makes that visible, and ``corr`` says whether the
predicted spread tracks the actual error at all or is effectively constant.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from rulbench.pretrain.model import load_checkpoint
from rulbench.pretrain.windows import WindowConfig, WindowSampler

N_FIT, N_TEST, ALPHA, BATCH = 3000, 1000, 1.0, 64


def r2_per_point(pred, y):
    """Mean over grid points and channels of 1 - SSE/SST, each against its
    OWN mean.  pred, y: (N, G, 2)."""
    sse = ((y - pred) ** 2).sum(0)
    sst = ((y - y.mean(0, keepdims=True)) ** 2).sum(0)
    return float((1.0 - sse / np.maximum(sst, 1e-12)).mean())


def ridge_fit(X, Y, alpha=ALPHA):
    """Multi-output ridge with an intercept.  X (N, D), Y (N, K)."""
    X = np.column_stack([X, np.ones(len(X))])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def ridge_predict(W, X):
    return np.column_stack([X, np.ones(len(X))]) @ W


def window_features(x, mask, n_slots):
    """Two hand-built features per window: the OLS slope of the health-index
    proxy over time, and the residual spread around that trend.

    The proxy is the norm over the process slots of the NORMALISED window,
    which is what a formula can see -- not the calibrated ||x - baseline||,
    since the baseline is not in the window.
    """
    B, T, _ = x.shape
    hi = x[:, :, :n_slots].norm(dim=-1)
    m = mask.float()
    t = torch.arange(T, device=x.device, dtype=x.dtype)[None, :].expand(B, -1)
    n = m.sum(1).clamp(min=2.0)
    tbar = (t * m).sum(1) / n
    ybar = (hi * m).sum(1) / n
    tc, yc = (t - tbar[:, None]) * m, (hi - ybar[:, None]) * m
    slope = (tc * yc).sum(1) / (tc ** 2).sum(1).clamp(min=1e-9)
    resid = yc - slope[:, None] * tc
    spread = ((resid ** 2 * m).sum(1) / n).sqrt()
    return torch.stack([slope, spread], dim=-1)


def main(ckpt, data, zero_x=False, zero_stats=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(ckpt, map_location=device).to(device).eval()
    W = model.cfg.context_length
    cfg = WindowConfig(window=W, p_terminal=0.0)
    s = WindowSampler([data], cfg, seed=3)
    grid = torch.from_numpy(s.grid).to(device)
    n_slots = cfg.max_channels - cfg.n_load_slots
    need = N_FIT + N_TEST

    H, HEAD, LSD, FEAT, Y = [], [], [], [], []
    got = 0
    with torch.no_grad():
        while got < need:
            b = s.sample_batch(BATCH)
            b = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            if zero_x:
                b["x"] = torch.zeros_like(b["x"])
            if zero_stats:
                b["mean"] = torch.zeros_like(b["mean"])
                b["std"] = torch.zeros_like(b["std"])
            stats = model.scale_proj(torch.cat([b["mean"], b["std"]], dim=-1))
            h = model.encoder(model.in_proj(b["x"]) + stats.unsqueeze(1))

            # the dyn head pools the whole window, so the probe gets the same
            # view: masked mean plus the state at the query time
            m = b["mask"].float().unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            idx = b["last_idx"][:, None, None].expand(-1, 1, h.shape[-1])
            last = h.gather(1, idx).squeeze(1)

            H.append(torch.cat([pooled, last], dim=-1).float().cpu())
            dyn = model.dyn_head(h, b["mask"], grid)
            HEAD.append(dyn[..., :2].float().cpu())
            LSD.append(dyn[..., 2:].clamp(model.cfg.min_logstd,
                                          model.cfg.max_logstd).float().cpu())
            FEAT.append(window_features(b["x"], b["mask"], n_slots).float().cpu())
            Y.append(b["y_dyn"].float().cpu())
            got += BATCH

    H = torch.cat(H).numpy()[:need]
    HEAD = torch.cat(HEAD).numpy()[:need]
    LSD = torch.cat(LSD).numpy()[:need]
    FEAT = torch.cat(FEAT).numpy()[:need]
    Y = torch.cat(Y).numpy()[:need]
    G = Y.shape[1]
    Yf = Y.reshape(len(Y), -1)

    tag = " [zero-x]" if zero_x else " [zero-stats]" if zero_stats else ""
    print(f"{ckpt}  on  {data}{tag}")
    print(f"n = {len(Y)} windows (fit {N_FIT}, test {len(Y) - N_FIT}), "
          f"window {W}, grid points {G}\n")

    Wgt = ridge_fit(H[:N_FIT], Yf[:N_FIT])
    pred = ridge_predict(Wgt, H[N_FIT:]).reshape(-1, G, 2)
    print(f"  probe on encoder state  : R^2 = {r2_per_point(pred, Y[N_FIT:]):+.3f}")

    print(f"  dyn head itself         : R^2 = "
          f"{r2_per_point(HEAD[N_FIT:], Y[N_FIT:]):+.3f}")

    Wgt = ridge_fit(FEAT[:N_FIT], Yf[:N_FIT])
    pred = ridge_predict(Wgt, FEAT[N_FIT:]).reshape(-1, G, 2)
    print(f"  probe on slope + spread : R^2 = {r2_per_point(pred, Y[N_FIT:]):+.3f}")

    # split the head's R^2 by operator: mu (channel 0) vs sigma (channel 1).
    # The model-free measurement said drift is unbiased but very noisy from a
    # single window, and diffusion is not separable from measurement noise at
    # all -- so a head that learns anything should learn it on mu first.
    for k, name in ((0, "mu"), (1, "sigma")):
        yk = Y[N_FIT:, :, k:k + 1]
        print(f"    head on log {name:5s}     : R^2 = "
              f"{r2_per_point(HEAD[N_FIT:, :, k:k + 1], yk):+.3f}")

    # Is the predicted spread calibrated?  An informative mean paired with an
    # NLL worse than the uninformed reference means the log_std branch is the
    # binding constraint.  Three numbers per operator: the actual residual sd,
    # the sd the head claims, and how well the claim tracks the |residual| it
    # should have anticipated.  A near-zero corr means log_std is effectively a
    # learned constant, and the marginal sd is then the level it should sit at.
    print("\n  calibration of sde_log_std (test split)")
    res, sd_hat = Y[N_FIT:] - HEAD[N_FIT:], np.exp(LSD[N_FIT:])
    for k, name in ((0, "mu"), (1, "sigma")):
        r, p = res[:, :, k].ravel(), sd_hat[:, :, k].ravel()
        marg = Y[N_FIT:, :, k].std(0).mean()
        c = float(np.corrcoef(np.abs(r), p)[0, 1]) if p.std() > 1e-9 else 0.0
        print(f"    log {name:5s}: sd(resid) {r.std():.3f}   "
              f"mean sd_hat {p.mean():.3f} (sd {p.std():.3f})   "
              f"marginal {marg:.3f}   corr(|resid|, sd_hat) {c:+.3f}")
        print(f"             NLL now {(LSD[N_FIT:, :, k] + 0.5 * (r.reshape(-1, G) / p.reshape(-1, G)) ** 2).mean():+.4f}"
              f"   best constant sd {np.log(r.std()) + 0.5:+.4f}"
              f"   uninformed {np.log(marg) + 0.5:+.4f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    main(args[0] if args else "runs/v3_weighted_health/last.pt",
         args[1] if len(args) > 1 else "data3/base_val.h5",
         zero_x="--zero-x" in flags, zero_stats="--zero-stats" in flags)