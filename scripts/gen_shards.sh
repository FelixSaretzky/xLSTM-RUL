#!/usr/bin/env bash
# Shard-parallel prior generation: S processes, one HDF5 file each.
# The training CLI pools shards natively (--train data/hybrid_train_*.h5).
# Resumable: finished shards are skipped; a crashed shard leaves no file
# (tmp+rename in write_hybrid_dataset), so re-running fills the gaps.
#
# usage (from the repo root):
#   S=8 N=200000 SEED0=0 scripts/gen_shards.sh rul_hybrid hybrid_train
#   S      shard count (~ physical cores)
#   N      total units across all shards
#   SEED0  seed of shard 0 -- give train and val DISJOINT ranges
#          (e.g. train SEED0=0, val SEED0=1000), same seed = same units.
#   OUT    output directory (default data)
set -euo pipefail
mod=$1; stem=$2; S=${S:-8}; N=${N:-200000}; SEED0=${SEED0:-0}; OUT=${OUT:-data}
for i in $(seq 0 $((S - 1))); do
    f=$OUT/${stem}_${i}.h5
    [ -e "$f" ] || uv run python -m rulbench.synthetic."$mod" \
        --out "$f" --n $((N / S)) --seed $((SEED0 + i)) &
done
wait
