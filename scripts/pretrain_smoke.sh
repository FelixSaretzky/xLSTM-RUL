#!/usr/bin/env bash
# Pretraining smoke (+ optional first real run) on any CUDA box --
# vast.ai instance, university cluster, workstation.
#
# Setup on a fresh box (PyTorch devel image, python >= 3.11):
#   git clone https://github.com/FelixSaretzky/xLSTM-RUL.git && cd xLSTM-RUL
#   pip install -e . "xlstm==2.0.5" wandb
#   export WANDB_API_KEY=...          # optional; falls back to log.jsonl
#   bash scripts/pretrain_smoke.sh            # probe + data + 3-arm smoke
#   bash scripts/pretrain_smoke.sh full       # ... + the 20k variant-C run
#
# Long runs: start inside tmux/screen so an SSH drop does not kill them.
#
# The sLSTM decision is probed, not assumed: the fused kernel needs BOTH
# nvcc (a -devel image; xlstm compiles the kernel at first use) AND a
# GPU with compute capability >= 8.0.  Without either, the run falls
# back to the mLSTM-only stack (--no-slstm).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== environment probe =="
# `|| true`: under pipefail, head's early exit can SIGPIPE nvidia-smi
# and would abort the whole script.
nvidia-smi | head -5 || true
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device visible to torch"
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0),
      "| compute capability", ".".join(map(str, torch.cuda.get_device_capability())))
EOF

# xlstm crashes at IMPORT when a GPU is visible but CUDA_HOME is unset
# (its slstm cuda_init queries toolkit paths on import) -- set it to the
# real toolkit if present, else to a dummy that unblocks the import.
if [ -z "${CUDA_HOME:-}" ]; then
    for p in /usr/local/cuda /opt/cuda; do
        [ -d "$p" ] && export CUDA_HOME="$p" && break
    done
    export CUDA_HOME="${CUDA_HOME:-/opt/cuda-not-installed}"
fi
echo "CUDA_HOME=$CUDA_HOME"

SLSTM="--no-slstm"
if command -v nvcc >/dev/null 2>&1 \
   && [ "$(python -c 'import torch; print(int(torch.cuda.get_device_capability()[0] >= 8))')" = "1" ]; then
    echo "== sLSTM fused-kernel probe (first compile takes minutes) =="
    if python - <<'EOF'
import numpy as np, torch
from rulbench.dataset_io import gen_grid
from rulbench.pretrain.model import ModelConfig, RULPretrainModel
m = RULPretrainModel(ModelConfig(embedding_dim=64, num_blocks=2,
                                 context_length=128,
                                 slstm_backend="cuda")).cuda()
x = torch.randn(2, 100, 32).cuda()
mask = torch.ones(2, 100, dtype=torch.bool).cuda()
g = torch.from_numpy(gen_grid().astype("float32")).cuda()
out = m(x, mask, g)
print("sLSTM CUDA kernel OK", {k: tuple(v.shape) for k, v in out.items()})
EOF
    then
        SLSTM="--slstm-backend cuda"
    else
        echo "fused sLSTM kernel did not build -> mLSTM-only"
    fi
else
    echo "no nvcc or compute capability < 8.0 -> mLSTM-only"
fi
echo "training flag: $SLSTM"

echo "== data (skips files that already exist) =="
SDE_TRAIN_W=(3 5 8 12 16 20 24 30)
HYB_TRAIN_W=(3 8 16)
for k in "${SDE_TRAIN_W[@]}"; do
    [ -f "data/sde_train_w$k.h5" ] || python -m rulbench.synthetic.rul_sde \
        --out "data/sde_train_w$k.h5" --n 1000 --seed $((20 + k)) --n-sensors "$k"
done
while read -r k n seed; do
    [ -f "data/sde_val_w$k.h5" ] || python -m rulbench.synthetic.rul_sde \
        --out "data/sde_val_w$k.h5" --n "$n" --seed "$seed" --n-sensors "$k"
done <<'EOF'
8 200 3
3 100 5
10 100 7
16 100 6
24 100 8
30 100 9
EOF
for k in "${HYB_TRAIN_W[@]}"; do
    [ -f "data/hybrid_train_w$k.h5" ] || python -m rulbench.synthetic.rul_hybrid \
        --out "data/hybrid_train_w$k.h5" --n 500 --seed $((100 + k)) --n-sensors "$k"
done
[ -f data/hybrid_val.h5 ] || python -m rulbench.synthetic.rul_hybrid \
    --out data/hybrid_val.h5 --n 200 --seed 1

# Explicit train list + weights: hybrid files weight 1, sde files 3/8 --
# equal TOTAL sampling mass per prior arm (declared, not accidental).
TRAIN=()
WEIGHTS=()
for k in "${HYB_TRAIN_W[@]}"; do TRAIN+=("data/hybrid_train_w$k.h5"); WEIGHTS+=("1"); done
for k in "${SDE_TRAIN_W[@]}"; do TRAIN+=("data/sde_train_w$k.h5"); WEIGHTS+=("0.375"); done
VAL=(data/hybrid_val.h5)
for k in 3 8 10 16 24 30; do VAL+=("data/sde_val_w$k.h5"); done

echo "== 100-step smoke, all three arms =="
for V in A B C; do
    python -m rulbench.pretrain.train \
        --train "${TRAIN[@]}" --train-weights "${WEIGHTS[@]}" --val "${VAL[@]}" \
        --out "runs/smoke_$V" --variant "$V" \
        --steps 100 --batch 256 $SLSTM --val-every 100 --log-every 25
done

if [ "${1:-}" = "full" ]; then
    echo "== variant-C 20k run =="
    python -m rulbench.pretrain.train \
        --train "${TRAIN[@]}" --train-weights "${WEIGHTS[@]}" --val "${VAL[@]}" \
        --out runs/c_20k --variant C --steps 20000 --batch 256 $SLSTM
fi
echo "done. checkpoints under runs/ -- download them or rely on W&B."
