"""Molab smoke test for the pretraining stack (marimo notebook).

Upload to molab.marimo.io.  FIRST switch the runtime to lazy (footer:
Runtime settings -> autorun off) so opening the notebook does not start
the whole pipeline, then run the cells top-down.  Assumes the
pretrain code is on GitHub (github.com/FelixSaretzky/xLSTM-RUL); the
12h GPU budget is spent as: env probe -> data generation -> 100-step
smoke of all three arms -> one real variant-C run.

The RTX PRO 6000 is Blackwell (sm_120): needs a recent torch cu128
wheel (the probe cell checks), and the fused sLSTM CUDA kernel is
compiled by xlstm at first use (needs nvcc; the probe cell tries it and
the training commands fall back to --no-slstm if it fails).
"""

import marimo

app = marimo.App()


@app.cell
def _():
    import subprocess

    def sh(cmd, timeout=None):
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        print(out[-4000:])
        return r.returncode, out

    return (sh,)


@app.cell
def _():
    # W&B key as a RUNTIME input -- never stored in this file.  Paste the
    # key from wandb.ai/authorize, or leave empty (falls back to log.jsonl).
    import marimo as mo
    wandb_key = mo.ui.text(kind="password",
                           label="WANDB_API_KEY (optional, runtime only)")
    wandb_key
    return (wandb_key,)


@app.cell
def _(wandb_key):
    import os
    if wandb_key.value:
        os.environ["WANDB_API_KEY"] = wandb_key.value
    wandb_env = bool(os.environ.get("WANDB_API_KEY"))
    print("W&B:", "key set" if wandb_env else "no key -> log.jsonl fallback")
    return (wandb_env,)


@app.cell
def _(sh):
    # --- clone + install ------------------------------------------------
    sh("git clone --depth 1 https://github.com/FelixSaretzky/xLSTM-RUL.git "
       "repo || (cd repo && git pull)")
    sh("pip install -q -e ./repo 'xlstm==2.0.5' wandb", timeout=900)
    return


@app.cell
def _(sh):
    # --- environment probe ---------------------------------------------
    sh("nvidia-smi | head -12")
    sh("python -c \"import torch; print('torch', torch.__version__, "
       "'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), "
       "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"")
    sh("nvcc --version | tail -1 || echo 'NO NVCC - sLSTM cuda kernel "
       "cannot compile, use --no-slstm or backend vanilla'")
    return


@app.cell
def _(sh):
    # --- sLSTM CUDA kernel probe (compiles at first use, takes minutes) --
    # xlstm crashes at IMPORT TIME (not compile time) when a GPU is
    # visible but CUDA_HOME is unset (its slstm cuda_init queries the
    # toolkit include paths on import).  Point CUDA_HOME at the real
    # toolkit if the image has one, else at a dummy: that unblocks the
    # import for every training cell, and the probe below then decides
    # whether the fused kernel actually compiles.
    import os
    if not os.environ.get("CUDA_HOME"):
        real = next((p for p in ("/usr/local/cuda", "/opt/cuda")
                     if os.path.isdir(p)), None)
        os.environ["CUDA_HOME"] = real or "/opt/cuda-not-installed"
        print(f"CUDA_HOME={os.environ['CUDA_HOME']}"
              + ("" if real else "  (dummy -> import unblocked, fused "
                                 "sLSTM kernel unavailable)"))
    code = (
        "import torch;"
        "from rulbench.pretrain.model import ModelConfig, RULPretrainModel;"
        "m = RULPretrainModel(ModelConfig(embedding_dim=64, num_blocks=2, "
        "context_length=128, slstm_backend='cuda')).cuda();"
        "x = torch.randn(2, 100, 10).cuda();"
        "mask = torch.ones(2, 100, dtype=torch.bool).cuda();"
        "import numpy as np;"
        "from rulbench.dataset_io import gen_grid;"
        "g = torch.from_numpy(gen_grid().astype('float32')).cuda();"
        "out = m(x, mask, g);"
        "print('sLSTM CUDA kernel OK', {k: tuple(v.shape) for k, v in out.items()})"
    )
    rc, _ = sh(f'cd repo && python -c "{code}"', timeout=1200)
    slstm_flag = "--slstm-backend cuda" if rc == 0 else "--no-slstm"
    print(f"\n==> training flag: {slstm_flag}")
    return (slstm_flag,)


@app.cell
def _(sh):
    # --- data: VARIABLE-WIDTH prior mix (PFN-style) ----------------------
    # sde arm: 8 widths x 1000 (incl. 24/30 so C-MAPSS-21 / N-CMAPSS-30
    # are IN-support, not extrapolation); hybrid arm: 3 widths x 500 (its
    # dotime calibration is the slow part -- shrink counts if the budget
    # runs short).  Val covers train widths AND held-out widths (10
    # interpolation; every width has a val file for width-conditional
    # curves).
    SDE_TRAIN_W = (3, 5, 8, 12, 16, 20, 24, 30)
    HYB_TRAIN_W = (3, 8, 16)
    for k in SDE_TRAIN_W:
        sh(f"cd repo && python -m rulbench.synthetic.rul_sde "
           f"--out data/sde_train_w{k}.h5 --n 1000 --seed {20 + k} "
           f"--n-sensors {k}", timeout=3600)
    for k, n, seed in ((8, 200, 3), (3, 100, 5), (10, 100, 7),
                       (16, 100, 6), (24, 100, 8), (30, 100, 9)):
        sh(f"cd repo && python -m rulbench.synthetic.rul_sde "
           f"--out data/sde_val_w{k}.h5 --n {n} --seed {seed} "
           f"--n-sensors {k}", timeout=600)
    for k in HYB_TRAIN_W:
        sh(f"cd repo && python -m rulbench.synthetic.rul_hybrid "
           f"--out data/hybrid_train_w{k}.h5 --n 500 --seed {100 + k} "
           f"--n-sensors {k}", timeout=14400)
    sh("cd repo && python -m rulbench.synthetic.rul_hybrid "
       "--out data/hybrid_val.h5 --n 200 --seed 1", timeout=7200)
    return (SDE_TRAIN_W, HYB_TRAIN_W)


@app.cell
def _(SDE_TRAIN_W, HYB_TRAIN_W):
    # Explicit file lists + weights: hybrid files carry weight 1, sde
    # files 3/8 -- equal TOTAL mass per arm (the mixture is a declared
    # part of the prior, not an accident of file sizes).
    TRAIN = ([f"data/hybrid_train_w{k}.h5" for k in HYB_TRAIN_W]
             + [f"data/sde_train_w{k}.h5" for k in SDE_TRAIN_W])
    WEIGHTS = " ".join(["1"] * len(HYB_TRAIN_W)
                       + [f"{3 / 8:.4f}"] * len(SDE_TRAIN_W))
    VAL = ("data/hybrid_val.h5 "
           + " ".join(f"data/sde_val_w{k}.h5" for k in (3, 8, 10, 16, 24, 30)))
    TRAIN_ARGS = (f"--train {' '.join(TRAIN)} --train-weights {WEIGHTS} "
                  f"--val {VAL}")
    return (TRAIN_ARGS,)


@app.cell
def _(sh, slstm_flag, TRAIN_ARGS, wandb_env):
    # --- 100-step smoke, all three arms ----------------------------------
    print(f"W&B enabled: {wandb_env}")
    for variant in "ABC":
        sh(f"cd repo && python -m rulbench.pretrain.train {TRAIN_ARGS} "
           f"--out runs/smoke_{variant} --variant {variant} "
           f"--steps 100 --batch 256 {slstm_flag} "
           f"--val-every 100 --log-every 25", timeout=3600)
    return


@app.cell
def _(sh, slstm_flag, TRAIN_ARGS, wandb_env):
    # --- the real run (variant C, ~20k steps).  W&B logging is on by
    # default (project xlstm-rul-pretrain) when a key is set above. ------
    print(f"W&B enabled: {wandb_env}")
    sh(f"cd repo && python -m rulbench.pretrain.train {TRAIN_ARGS} "
       f"--out runs/c_20k --variant C --steps 20000 --batch 256 "
       f"{slstm_flag}", timeout=None)
    return


if __name__ == "__main__":
    app.run()
