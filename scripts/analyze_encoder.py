"""Does the encoder state carry X' at all -- and does the head use it?

Freezes a trained model, collects the encoder state at the query time, and
fits a ridge regression onto hi_end.  Compares three numbers on held-out
windows:

  * R^2 of a linear probe on the encoder state -- is the information there?
  * R^2 of the health head's own mean          -- does the head use it?
  * R^2 of a probe on the raw ||x - baseline|| -- what a formula gets

If the probe scores high and the head does not, the encoder keeps the
information and the head fails to read it (an optimisation problem).  If
the probe scores low too, the encoder discards it (an architecture problem).
"""
import numpy as np
import torch

from rulbench.pretrain.model import load_checkpoint
from rulbench.pretrain.windows import WindowSampler, WindowConfig

CKPT = "runs/v3_base/last.pt"
DATA = "data3/base_val.h5"
N_FIT, N_TEST, ALPHA = 3000, 1000, 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_checkpoint(CKPT, map_location=device).to(device).eval()
cfg = WindowConfig(window=model.cfg.context_length, p_terminal=0.0)
s = WindowSampler([DATA], cfg, seed=3)
grid = torch.from_numpy(s.grid).to(device)

H, HEAD, HI, Y = [], [], [], []
with torch.no_grad():
    for _ in range((N_FIT + N_TEST) // 64 + 1):
        b = s.sample_batch(64)
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        h = model.encoder(model.in_proj(b["x"]) + model.scale_proj(
            torch.cat([b["mean"], b["std"]], dim=-1)).unsqueeze(1))
        idx = b["last_idx"][:, None, None].expand(-1, 1, h.shape[-1])
        H.append(h.gather(1, idx).squeeze(1).cpu())
        HEAD.append(model.health_head(h)[..., 0]
                    .gather(1, b["last_idx"][:, None]).squeeze(1).cpu())
        # what a formula gets from the same window: the last normalised
        # process values, whose norm is the empirical health index
        n_slots = cfg.max_channels - cfg.n_load_slots
        xw = b["x"].gather(
            1, b["last_idx"][:, None, None].expand(-1, 1, b["x"].shape[-1]))
        HI.append(xw.squeeze(1)[:, :n_slots].norm(dim=-1).cpu())
        Y.append(b["hi_end"].cpu())

H = torch.cat(H).numpy(); HEAD = torch.cat(HEAD).numpy()
HI = torch.cat(HI).numpy(); Y = torch.cat(Y).numpy()


def r2(pred, y):
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ridge_r2(X, y, n_fit=N_FIT, alpha=ALPHA):
    X = np.column_stack([X, np.ones(len(X))]) if X.ndim > 1 else \
        np.column_stack([X, np.ones(len(X))])
    Xf, yf, Xt, yt = X[:n_fit], y[:n_fit], X[n_fit:], y[n_fit:]
    A = Xf.T @ Xf + alpha * np.eye(Xf.shape[1])
    w = np.linalg.solve(A, Xf.T @ yf)
    return r2(Xt @ w, yt)


print(f"n = {len(Y)}  (fit {N_FIT}, test {len(Y) - N_FIT})")
print(f"  probe on encoder state : R^2 = {ridge_r2(H, Y):+.3f}")
print(f"  health head itself     : R^2 = {r2(HEAD[N_FIT:], Y[N_FIT:]):+.3f}")
print(f"  probe on ||x|| at query: R^2 = {ridge_r2(HI.reshape(-1, 1), Y):+.3f}")
