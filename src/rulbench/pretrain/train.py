"""
Pretraining CLI -- plain PyTorch loop over sampled windows.

    uv run --group train python -m rulbench.pretrain.train \\
        --train data/hybrid_train.h5 data/sde_train.h5 \\
        --val   data/hybrid_val.h5   data/sde_val.h5 \\
        --out runs/v1 --steps 20000

The training arms of the ablation are selected with --variant:
A = RUL head only (direct baseline), B = health + dynamics (simulation
route), C = all three heads.  Everything else -- sampler, normalisation,
encoder, budget, validation draws -- is identical across arms by
construction (the validation draw seed is fixed, independent of --seed,
so seed replicates score on the same windows).

The first step logs the per-head loss magnitudes; the targets are all
standardised to O(1), and that line is where a regression of this
invariant shows up first.  Validation is reported PER FILE (a pooled
number would be a function of the val-file mixture); ``best.pt`` is
selected on the per-file criterion averaged with equal file weight --
the RUL part when the arm has one (A, C), the summed standardised
parts otherwise (B).  ``last.pt`` is always written, so arms can also
be compared at the fixed final step.  ``--train-weights`` declares the
arm/width sampling mixture explicitly instead of inheriting it from
the file sizes.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from rulbench.pretrain.model import (ModelConfig, RULPretrainModel,
                                     model_summary, pretrain_loss,
                                     save_checkpoint)
from rulbench.pretrain.windows import WindowConfig, WindowSampler

VARIANTS = {"A": dict(use_health=False, use_dynamics=False, use_rul=True),
            "B": dict(use_health=True, use_dynamics=True, use_rul=False),
            "C": dict(use_health=True, use_dynamics=True, use_rul=True)}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, sampler, draws, grid, batch_size, device) -> dict:
    """Per-head validation losses over fixed draws.  Each part is weighted
    by what it averages over (real steps for health, windows otherwise),
    so the numbers are independent of the eval batch size."""
    model.eval()
    sums, weights = {}, {}
    for batch in sampler.eval_batches(draws, batch_size):
        batch = to_device(batch, device)
        out = model(batch["x"], batch["mask"], grid)
        _, parts = pretrain_loss(out, batch, model.cfg)
        for k, v in parts.items():
            w = (float(batch["mask"].sum()) if k == "health"
                 else float(len(batch["y_rul"])))
            sums[k] = sums.get(k, 0.0) + float(v) * w
            weights[k] = weights.get(k, 0.0) + w
    model.train()
    return {k: sums[k] / weights[k] for k in sums}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--train-weights", type=float, nargs="+", default=None,
                    help="relative sampling mass per --train file "
                         "(default: uniform over the pooled units)")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="C", choices=sorted(VARIANTS))
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--no-slstm", action="store_true",
                    help="mLSTM-only stack (fast CPU dev config)")
    ap.add_argument("--slstm-backend", default="vanilla",
                    choices=["vanilla", "cuda"])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--p-terminal", type=float, default=0.15)
    ap.add_argument("--p-anchor-free", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--wandb", default="xlstm-rul-pretrain", metavar="PROJECT",
                    help="Weights & Biases project (default logging; needs "
                         "WANDB_API_KEY or `wandb login`)")
    ap.add_argument("--no-wandb", action="store_true",
                    help="disable W&B, keep only stdout + log.jsonl")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)

    wcfg = WindowConfig(window=a.window, p_terminal=a.p_terminal,
                        p_anchor_free=a.p_anchor_free)
    sampler = WindowSampler(a.train, wcfg, seed=a.seed,
                            path_weights=a.train_weights)
    val_sets = []
    for p in a.val:
        vs = WindowSampler([p], wcfg, seed=a.seed + 1)
        name = os.path.splitext(os.path.basename(p))[0]
        val_sets.append((name, vs, vs.fixed_eval_draws(per_unit=2, seed=0)))

    mcfg = ModelConfig(in_channels=sampler.n_channels, embedding_dim=a.dim,
                       num_blocks=a.blocks, context_length=a.window,
                       slstm_last=not a.no_slstm,
                       slstm_backend=a.slstm_backend, **VARIANTS[a.variant])
    model = RULPretrainModel(mcfg).to(device)
    if mcfg.slstm_last and device.type != "cuda":
        print("WARNING: the vanilla sLSTM recurrence is a Python loop over "
              "every time step; off CUDA it dominates the run (measured "
              "~556 s/step on MPS at the default size vs ~1.8 s/step "
              "mLSTM-only). Use --no-slstm locally, sLSTM on the cluster.")
    grid = torch.from_numpy(sampler.grid).to(device)
    print(f"{model_summary(model)}  device={device}  "
          f"units={sampler.n_units}  variant={a.variant}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": rest, "weight_decay": 0.0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=a.warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(
             opt, T_max=max(1, a.steps - a.warmup))],
        milestones=[a.warmup])

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "run.json"), "w") as fh:
        json.dump({"args": vars(a), "model": model_summary(model)}, fh,
                  indent=2)

    wb = None
    if not a.no_wandb:
        try:
            import wandb
            if not (os.environ.get("WANDB_API_KEY")
                    or os.environ.get("WANDB_MODE")
                    or os.path.exists(os.path.expanduser("~/.netrc"))):
                raise RuntimeError("no credentials -- set WANDB_API_KEY or "
                                   "run `wandb login` (or pass --no-wandb)")
            wb = wandb.init(project=a.wandb, dir=a.out,
                            config={**vars(a), "model": model_summary(model)})
        except Exception as e:
            print(f"WARNING: W&B logging off ({e}); using log.jsonl only")

    best_val, t0 = float("inf"), time.time()
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    for step in range(1, a.steps + 1):
        batch = to_device(sampler.sample_batch(a.batch), device)
        out = model(batch["x"], batch["mask"], grid)
        loss, parts = pretrain_loss(out, batch, mcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1:
            parts_f = {k: float(v.detach()) for k, v in parts.items()}
            balance = "  ".join(f"{k}={v:.3f}" for k, v in parts_f.items())
            print(f"step 1 head-loss balance: {balance}")
            log.write(json.dumps({"step": 1, **parts_f}) + "\n")
        if step % a.log_every == 0:
            rec = {"step": step, "loss": float(loss.detach()),
                   **{k: float(v.detach()) for k, v in parts.items()},
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if wb:
                wb.log(rec, step=step)
            print(f"step {step}  loss {rec['loss']:.4f}  "
                  + "  ".join(f"{k} {rec[k]:.4f}" for k in parts))
        if step % a.val_every == 0 or step == a.steps:
            val = {name: evaluate(model, vs, draws, grid, a.batch, device)
                   for name, vs, draws in val_sets}
            crit = [v.get("rul", sum(v.values())) for v in val.values()]
            val_crit = sum(crit) / len(crit)
            log.write(json.dumps({"step": step, "val": val}) + "\n")
            log.flush()
            if wb:
                wb.log({f"val/{n}/{k}": v for n, parts in val.items()
                        for k, v in parts.items()}, step=step)
            for n, parts in val.items():
                print(f"step {step}  VAL {n}  "
                      + "  ".join(f"{k} {v:.4f}" for k, v in parts.items()))
            save_checkpoint(os.path.join(a.out, "last.pt"), model,
                            {"step": step, "val": val})
            if val_crit < best_val:
                best_val = val_crit
                save_checkpoint(os.path.join(a.out, "best.pt"), model,
                                {"step": step, "val": val})
    log.close()
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
