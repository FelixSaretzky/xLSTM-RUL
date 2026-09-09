#!/usr/bin/env bash
# Generate the rul_sde prior variants for the mixture / ablation study.
#
# Each variant isolates ONE axis so the training mixture is a declared
# quantity, not an accident.  Run the whole thing, then either train on a
# single file (isolation) or pass several to --train with --train-weights
# (mixture).  The cross-evaluation -- train on A, evaluate on B -- is what
# tells you whether an axis actually needs coverage; a wider prior costs
# capacity, so only axes that break under transfer earn their place.
#
#   bash scripts/gen_priors.sh              # full set
#   bash scripts/gen_priors.sh --smoke      # 200 units each, quick check
#
set -euo pipefail

GEN="uv run python -m rulbench.synthetic.rul_sde"
OUT="${OUT_DIR:-data}"
N_TRAIN="${N_TRAIN:-20000}"
N_VAL="${N_VAL:-2000}"

if [[ "${1:-}" == "--smoke" ]]; then
    N_TRAIN=200; N_VAL=50
    echo "smoke mode: ${N_TRAIN}/${N_VAL} units per variant"
fi

mkdir -p "$OUT"

# gen <name> <seed-offset> <extra args...>
gen () {
    local name="$1"; shift
    local off="$1"; shift
    for split in train val; do
        local n seed
        if [[ "$split" == train ]]; then n=$N_TRAIN; seed=$((100 + off));
        else                             n=$N_VAL;   seed=$((900 + off)); fi
        local f="$OUT/${name}_${split}.h5"
        if [[ -f "$f" ]]; then echo "skip $f (exists)"; continue; fi
        echo "=== $name / $split -> $f"
        $GEN --out "$f" --n "$n" --seed "$seed" "$@"
    done
}

# ---------------------------------------------------------------- baseline
# The reference configuration every other variant is compared against.
gen base 0 --n-sensors 8 --n-load 2

# ------------------------------------------------------------ sensor width
# The slot layout is built for variable width, and the benchmarks differ:
# C-MAPSS feeds 14 or 21 process sensors, N-CMAPSS 30.  Training on 8 only
# makes every benchmark width an extrapolation -- this is the axis where a
# transfer drop is most likely.
gen w04 10 --n-sensors 4  --n-load 2
gen w14 11 --n-sensors 14 --n-load 2
gen w21 12 --n-sensors 21 --n-load 2
gen w30 13 --n-sensors 30 --n-load 2

# --------------------------------------------------------- sequence length
# 60-100 steps (N-CMAPSS cycle level) vs several hundred are different
# aggregation regimes: drift is only estimable by averaging over many
# steps, so short units are a genuinely harder problem, not just a shorter
# one.  Length is set through the latent block, hence the config flags.
gen short 20 --n-sensors 8 --n-load 2 --min-length 40  --max-length 200
gen long  21 --n-sensors 8 --n-load 2 --min-length 300 --max-length 2000

# ------------------------------------------------------------ graph nuisance
# Measured cross-sensor correlation barely moved between topologies
# (0.161/0.170/0.178), so this axis is expected to be cheap -- worth one
# run to confirm rather than assume.
gen gnone 30 --n-sensors 8 --n-load 2 --graph-type none
gen ghub  31 --n-sensors 8 --n-load 2 --graph-type hub

# ---------------------------------------------------------------- load info
# Does explicit load observability help?  Without the dedicated channels
# the model sees the load only through the ~30% of process sensors that
# happen to carry a w_str edge -- in ~6% of units, not at all.
gen noload 40 --n-sensors 8 --n-load 0

# ------------------------------------------------------ negative control
# No degradation -> sensor edges at all.  A model trained on this MUST
# fail; if it does not, something leaks (window position, sequence length).
# Not part of any mixture -- it is a trust test.
gen nocouple 50 --n-sensors 8 --n-load 2 --no-couple

# --------------------------------------------- signature-scale ablation
# "free" restores the old behaviour where X' is only weakly identifiable
# through the prior (measured: HI correlation 0.59 vs 0.91, end-of-life
# norm 1.78 vs 1.00).
gen sigfree 60 --n-sensors 8 --n-load 2 --signature-norm free

echo
echo "done -- files in $OUT:"
ls -1sh "$OUT"/*.h5 2>/dev/null || true
cat <<'EOF'

Next:
  isolation   --train data/base_train.h5 --val data/base_val.h5
  mixture     --train data/base_train.h5 data/w21_train.h5 data/short_train.h5 \
              --train-weights 1 1 1 --val data/base_val.h5 data/w21_val.h5
  transfer    train on one, validate on another -> which axis breaks?
EOF