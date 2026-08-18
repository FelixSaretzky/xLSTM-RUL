"""
Zero-shot C-MAPSS evaluation of a pretrained checkpoint (issue #8).

Both readouts of the one model, side by side:

  direct     RUL head at the window end, times the training cap -- the
             point estimate comparable against published numbers.
  simulated  health head gives the window-end state X'(t), the dynamics
             head gives mu(x), sigma(x); Euler-Maruyama Monte Carlo to
             first passage at X' = 1 (``simulate.first_passage``).
             Reported as capped median plus quantiles -- the readout
             that carries uncertainty.

Data comes through rul-datasets (``eval`` dependency group).  With
``window_size=1`` the reader returns stride-1 windows of length one,
i.e. the raw per-unit series, scaled min-max on the dev split and with
the literature's 14-sensor selection and 125-cycle RUL cap -- the same
cap the priors use.  Channel placement mirrors training's eval mode:
sensors into the first process slots, op-settings 1-2 into the two load
slots, empty slots exactly zero.  FD001's settings are near-constant, so
after min-max scaling the load normalisation's constant guard fires and
the load slots become the clean constant-load case of the prior.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from rulbench.dataset_io import gen_grid
from rulbench.pretrain.model import ModelConfig, RULPretrainModel
from rulbench.pretrain.simulate import SimConfig, first_passage
from rulbench.pretrain.windows import WindowConfig, normalize_windows

# Raw C-MAPSS columns: 0-2 op settings, 3.. sensors 1-21.  Sensor list =
# rul-datasets' default 14 (Li et al.); settings 1-2 appended so they
# land at the end, i.e. in the load slots.
_SENSORS = [4, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 19, 22, 23]
_SETTINGS = [0, 1]
RUL_CAP = 125.0  # rul-datasets max_rul default == the priors' cap


def load_units(fd: int = 1, split: str = "test"):
    """Per-unit raw series (T, 16) float32 and the true capped RUL (one
    value per unit, at the truncation point)."""
    from rul_datasets.reader import CmapssReader  # slow import (lightning)

    reader = CmapssReader(fd=fd, window_size=1,
                          feature_select=_SENSORS + _SETTINGS)
    reader.prepare_data()
    # alias="dev" windows the FULL series (the default test alias crops
    # to the last window only); with window_size=1 the windows are the
    # raw per-step series.  Targets stay the official RUL-file values,
    # so the last label per unit is the benchmark truth.
    feats, targs = reader.load_split(split, alias="dev")
    series = [f[:, 0, :].astype(np.float32) for f in feats]
    targets = [t.astype(np.float32) for t in targs]
    return series, targets


def assemble_windows(series, cfg: WindowConfig | None = None):
    """Last <= 512 steps of each unit into the model's input contract:
    (B, W, 32) right-padded, mask, normalised exactly as in training."""
    cfg = cfg or WindowConfig()
    n_proc = cfg.max_channels - cfg.n_load_slots
    n_sens = len(_SENSORS)
    x = torch.zeros(len(series), cfg.window, cfg.max_channels)
    mask = torch.zeros(len(series), cfg.window, dtype=torch.bool)
    for j, s in enumerate(series):
        t = min(len(s), cfg.window)
        x[j, :t, :n_sens] = torch.from_numpy(s[-t:, :n_sens])
        x[j, :t, n_proc:] = torch.from_numpy(s[-t:, n_sens:])
        mask[j, :t] = True
    return normalize_windows(x, mask, n_proc, cfg), mask


def _slstm_cuda_to_vanilla(sd: dict, dim: int, num_heads: int = 4) -> dict:
    """A cuda-trained checkpoint stores the sLSTM cell parameters in the
    fused kernel's internal layout (recurrent kernel transposed, bias
    flattened in a different gate order).  Convert to the vanilla cell's
    layout via the library's own ext/int converters."""
    from xlstm.blocks.slstm.cell import (sLSTMCellConfig, sLSTMCell_cuda,
                                         sLSTMCell_vanilla)
    cfg = sLSTMCellConfig(hidden_size=dim, num_heads=num_heads)
    cu = sLSTMCell_cuda(cfg, skip_backend_init=True)
    va = sLSTMCell_vanilla(cfg, skip_backend_init=True)
    for k in list(sd):
        if k.endswith("slstm_cell._recurrent_kernel_"):
            sd[k] = va._recurrent_kernel_ext2int(
                cu._recurrent_kernel_int2ext(sd[k]))
        elif k.endswith("slstm_cell._bias_"):
            sd[k] = va._bias_ext2int(cu._bias_int2ext(sd[k]))
    return sd


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM08 asymmetric score (sum over units; late predictions cost more)."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13), np.exp(d / 10)) - 1))


def load_model(ckpt_path: str, device: str = "cpu") -> RULPretrainModel:
    """Load a checkpoint; cuda-trained sLSTM weights are converted to
    the vanilla layout when no GPU is available."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mcfg = ModelConfig(**ckpt["config"])
    sd = ckpt["state_dict"]
    if mcfg.slstm_backend == "cuda" and not torch.cuda.is_available():
        mcfg.slstm_backend = "vanilla"
        sd = _slstm_cuda_to_vanilla(sd, mcfg.embedding_dim)
    model = RULPretrainModel(mcfg)
    model.load_state_dict(sd)
    return model.to(device).eval()


def evaluate_checkpoint(ckpt_path: str, fd: int = 1, device: str = "cpu",
                        sim: SimConfig | None = None, batch: int = 64,
                        seed: int = 0) -> dict:
    """Run both readouts on the FD00x test set; returns per-unit arrays
    and summary metrics."""
    sim = sim or SimConfig()
    wcfg = WindowConfig()
    model = load_model(ckpt_path, device)

    series, targets = load_units(fd=fd)
    y_true = np.array([t[-1] for t in targets], dtype=np.float32)
    x, mask = assemble_windows(series, wcfg)
    grid_np = gen_grid().astype(np.float32)
    grid = torch.from_numpy(grid_np).to(device)

    parts = {"health": [], "dyn": [], "rul": []}
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out = model(x[i:i + batch].to(device),
                        mask[i:i + batch].to(device), grid)
            for k in parts:
                parts[k].append(out[k].cpu())
    health = torch.cat(parts["health"])                  # (B, W)
    dyn = torch.cat(parts["dyn"]).numpy()                # (B, G, 2)
    rul_head = torch.cat(parts["rul"]).numpy()           # (B,)

    last = mask.sum(dim=1) - 1
    hi_end = health[torch.arange(len(health)), last].numpy()
    direct = np.clip(rul_head, 0.0, 1.0) * RUL_CAP

    log_dyn = dyn * wcfg.dyn_log_std + wcfg.dyn_log_mean
    rng = np.random.default_rng(seed)
    B = len(series)
    sim_med = np.empty(B)
    sim_q10 = np.empty(B)
    sim_q90 = np.empty(B)
    frac_nc = np.empty(B)
    for j in range(B):
        r = first_passage(hi_end[j], np.exp(log_dyn[j, :, 0]),
                          np.exp(log_dyn[j, :, 1]), grid_np, sim, rng)
        sim_med[j] = r["rul"]
        sim_q10[j], sim_q90[j] = np.quantile(r["samples"], [0.1, 0.9])
        frac_nc[j] = r["frac_not_crossed"]

    def rmse(p):
        return float(np.sqrt(np.mean((p - y_true) ** 2)))

    metrics = {
        "n_units": B,
        "rmse_direct": rmse(direct),
        "rmse_simulated": rmse(sim_med),
        "nasa_direct": nasa_score(direct, y_true),
        "nasa_simulated": nasa_score(sim_med, y_true),
        "coverage_80": float(np.mean((y_true >= sim_q10) & (y_true <= sim_q90))),
        "frac_not_crossed_mean": float(frac_nc.mean()),
    }
    per_unit = {
        "y_true": y_true, "direct": direct, "simulated": sim_med,
        "sim_q10": sim_q10, "sim_q90": sim_q90, "hi_end": hi_end,
        "frac_not_crossed": frac_nc,
    }
    return {"metrics": metrics, "per_unit": per_unit,
            "health": health.numpy(), "mask": mask.numpy()}


def rul_trajectories(ckpt_path: str, fd: int = 1, units=None,
                     device: str = "cpu", start: int = 10,
                     sim_stride: int = 25,
                     sim_cfg: SimConfig | None = None, batch: int = 64,
                     seed: int = 0) -> dict:
    """RUL over time for selected test units: at every cycle the model
    sees the history up to that cycle (last <= 512 steps).  Returns per
    unit the true capped RUL curve, the direct readout per step, and
    the simulated readout every ``sim_stride`` steps.  Default units:
    five spread over the final-RUL range."""
    wcfg = WindowConfig()
    sim_cfg = sim_cfg or SimConfig()
    model = load_model(ckpt_path, device)
    series, targets = load_units(fd=fd)
    if units is None:
        order = np.argsort([t[-1] for t in targets])
        units = [int(order[k]) for k in
                 [0, len(order) // 4, len(order) // 2,
                  3 * len(order) // 4, len(order) - 1]]
    grid_np = gen_grid().astype(np.float32)
    grid = torch.from_numpy(grid_np).to(device)
    out = {}
    for u in units:
        s = series[u]
        steps = np.arange(start, len(s))
        history = [s[:t + 1] for t in steps]
        x, mask = assemble_windows(history, wcfg)
        direct = np.empty(len(steps))
        hi = np.empty(len(steps))
        dyn = np.empty((len(steps), len(grid_np), 2), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(x), batch):
                o = model(x[i:i + batch].to(device),
                          mask[i:i + batch].to(device), grid)
                last = (mask[i:i + batch].sum(dim=1) - 1).numpy()
                n = len(last)
                direct[i:i + n] = np.clip(o["rul"].cpu().numpy(),
                                          0, 1) * RUL_CAP
                hi[i:i + n] = o["health"].cpu().numpy()[np.arange(n), last]
                dyn[i:i + n] = o["dyn"].cpu().numpy()
        sim_idx = np.arange(0, len(steps), sim_stride)
        rng = np.random.default_rng(seed)
        log_dyn = dyn * wcfg.dyn_log_std + wcfg.dyn_log_mean
        sim = np.array([first_passage(hi[i], np.exp(log_dyn[i, :, 0]),
                                      np.exp(log_dyn[i, :, 1]), grid_np,
                                      sim_cfg, rng)["rul"]
                        for i in sim_idx])
        out[u] = dict(t=steps, true=targets[u][start:], direct=direct,
                      sim_t=steps[sim_idx], sim=sim)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fd", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    res = evaluate_checkpoint(a.ckpt, fd=a.fd, device=a.device)
    for k, v in res["metrics"].items():
        print(f"{k:24s} {v:.3f}" if isinstance(v, float) else f"{k:24s} {v}")


if __name__ == "__main__":
    main()
