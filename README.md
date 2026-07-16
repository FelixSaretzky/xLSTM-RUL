# xLSTM-RUL

Training-free remaining-useful-life (RUL) prognostics with prior-data fitted
networks: a model is pre-trained purely on **synthetic run-to-failure data
generated from structural causal models of degradation**, then predicts RUL
for real machinery zero-shot via in-context learning — labeled run-to-failure
trajectories in the context, a query sensor history in, RUL out, no training
on real data.

## Repository layout

| Path | Contents |
|------|----------|
| `src/rulbench/datasets.py` | Unified loaders for the real RUL benchmarks: C-MAPSS, N-CMAPSS, FEMTO (PRONOSTIA), XJTU-SY, Milling, PHME20, IMS. One canonical interface `load(name, fd, split) -> (units, meta)`; each unit carries `features (T, L, C)`, `rul (T,)` and a per-unit censoring flag. |
| `src/rulbench/synthetic/` | The synthetic RUL generator: `rul_mechanism.py` (degradation dynamics, first-passage failure against a family-calibrated threshold, right-censored suspensions, response-surface sensor signatures, capped piecewise-linear RUL labels) and `rul_prior.py` (`RULPrior`, an adapter over the CausalTimePrior SCM generator). |
| `src/notebooks/00_datasets.ipynb` | Executed walkthrough of the benchmark data API. |
| `src/notebooks/01_synthetic_rul_generator.ipynb` | Executed walkthrough of the synthetic generator, from a single unit to model-ready batches. |
| `third_party/` | Vendored dependencies as pinned git submodules (see below). |
| `paper/` | LaTeX sources of the companion paper. |

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/)
(`brew install uv`, or see the uv docs for other platforms).

```bash
git clone https://github.com/FelixSaretzky/xLSTM-RUL.git
cd xLSTM-RUL
git submodule update --init   # non-recursive on purpose, see below
uv sync
```

Use `git submodule update --init` (not `--recursive`, and not
`git clone --recurse-submodules`): the CausalTimePrior submodule declares
further nested submodules over SSH-only URLs that this project never uses --
recursing into them fails on machines without GitHub SSH keys, while the
non-recursive init fetches exactly the two pinned dependencies this project
needs, over HTTPS.

`uv sync` creates `.venv` with all locked dependencies (Python >= 3.11,
including the notebook stack) and installs `rulbench` itself in editable
mode, so `import rulbench` just works.

Open the walkthrough notebooks with:

```bash
uv run jupyter lab
```

The synthetic generator builds on two submodules under `third_party/`:

- [CausalTimePrior](https://github.com/thummd/CausalTimePrior) (Apache-2.0) —
  the SCM-based time-series generator, pinned at `ad9f0a4`.
- [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) — leaf utilities
  CausalTimePrior depends on, pinned at `287b0e0`. **No license declared**
  upstream, i.e. all rights reserved by its author: running this project
  requires viewing/forking rights GitHub grants plus a usage grant from the
  author; this repository only references the code as a pointer.

Both pins are version-critical (newer Do-PFN-prior commits removed symbols
CausalTimePrior imports) — do not `git submodule update --remote` without
re-testing.

Real-benchmark downloads are handled by
[rul-datasets](https://github.com/tilman151/rul-datasets) (cache in
`~/.rul-datasets`); Milling and PHME20 are fetched directly (cache in
`~/.rulbench`).

## Quickstart

Run inside the project environment (`uv run python`, or a notebook started
via `uv run jupyter lab`):

```python
# real benchmarks
from rulbench.datasets import load
units, meta = load("cmapss", fd=1, split="dev")

# synthetic run-to-failure data
from rulbench.synthetic import generate_rul_dataset
long_df, summary_df, rows, prior = generate_rul_dataset(n=5, seed=3, T_max=160)
batch = prior.generate_batch(rows)   # padded arrays, target_key='RUL'
```
