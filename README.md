# xLSTM-RUL

Synthetic run-to-failure data for pretraining zero-shot RUL (remaining
useful life) models. Both generators share the same latent block — a
health index degrading as an SDE with load-modulated drift, failure at
first passage, right-censored units, capped piecewise-linear RUL labels —
and differ only in how the sensors are generated from health and load.
That difference is the comparison this repo exists for:

- **SDE prior** (`rul_sde.py`) — sensors come from a hand-built dynamic
  Bayesian network: a small fixed graph with self-feedback, where health
  shifts the responsive sensors with a unit-norm signature.
- **Hybrid prior** (`rul_hybrid.py`) — same latent block, but the sensor
  network is a temporal SCM sampled from
  [dotime](https://github.com/thummd/dotime) (lagged cross-sensor edges,
  hidden nodes, diverse mechanisms). Health and load are injected as
  driver nodes, and the signature is calibrated to the same unit norm and
  noise range as the SDE prior.

Both write the same HDF5 layout, so their datasets are directly
comparable and mixable.

## Layout

| Path | Contents |
|---|---|
| `src/rulbench/dataset_io.py` | shared `Unit` record + streaming HDF5 store |
| `src/rulbench/synthetic/rul_sde.py` | SDE prior (DBN emission) |
| `src/rulbench/synthetic/rul_hybrid.py` | hybrid prior (dotime SCM emission) |
| `paper/` | LaTeX sources of the companion paper |

## Usage

```bash
uv sync
uv run python -m rulbench.synthetic.rul_sde    --out data/sde_train.h5    --n 1000 --seed 0
uv run python -m rulbench.synthetic.rul_hybrid --out data/hybrid_train.h5 --n 1000 --seed 0
```
