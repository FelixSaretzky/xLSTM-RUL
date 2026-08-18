"""Zero-shot C-MAPSS FD001 evaluation notebook (issue #8).

    uv run --group eval --group train marimo run notebooks/eval_cmapss_fd001.py

All logic lives in ``rulbench.eval``; this notebook only calls it and
plots.  Both readouts side by side: the direct head (point estimate,
comparable against published FD001 numbers) and the simulated first
passage (median + 10-90% band, the readout that carries uncertainty).
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    ckpt = mo.ui.text(value="runs/vast_full_20k/best.pt",
                      label="checkpoint", full_width=True)
    fd = mo.ui.number(value=1, start=1, stop=4, label="FD")
    mo.hstack([ckpt, fd])
    return ckpt, fd, mo


@app.cell
def _(ckpt, fd):
    from rulbench.eval import evaluate_checkpoint

    res = evaluate_checkpoint(ckpt.value, fd=int(fd.value))
    return (res,)


@app.cell
def _(mo, res):
    m = res["metrics"]
    mo.md(f"""
| metric | direct | simulated |
|---|---|---|
| RMSE | {m["rmse_direct"]:.2f} | {m["rmse_simulated"]:.2f} |
| NASA score | {m["nasa_direct"]:.1f} | {m["nasa_simulated"]:.1f} |
| 80% coverage | -- | {m["coverage_80"]:.2f} |

{m["n_units"]} test units, mean non-crossing fraction
{m["frac_not_crossed_mean"]:.3f}.
""")
    return


@app.cell
def _(res):
    import matplotlib.pyplot as plt
    import numpy as np

    p = res["per_unit"]
    order = np.argsort(p["y_true"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in [(axes[0], "direct", "direct head"),
                           (axes[1], "simulated", "simulated first passage")]:
        ax.scatter(p["y_true"], p[key], s=12)
        ax.plot([0, 125], [0, 125], lw=1, color="grey")
        ax.set_xlabel("true RUL")
        ax.set_title(title)
    axes[0].set_ylabel("predicted RUL")
    axes[1].errorbar(p["y_true"][order], p["simulated"][order],
                     yerr=np.vstack([p["simulated"][order] - p["sim_q10"][order],
                                     p["sim_q90"][order] - p["simulated"][order]]),
                     fmt="none", alpha=0.25, lw=1)
    fig.tight_layout()
    fig
    return


@app.cell
def _(ckpt, fd):
    from rulbench.eval import rul_trajectories

    traj = rul_trajectories(ckpt.value, fd=int(fd.value))
    return (traj,)


@app.cell
def _(traj):
    import matplotlib.pyplot as __plt

    figt, axst = __plt.subplots(1, len(traj), figsize=(3.2 * len(traj), 3.2),
                                sharey=True)
    for axt, (u, d) in zip(axst, sorted(traj.items(),
                                        key=lambda kv: kv[1]["true"][-1])):
        axt.plot(d["t"], d["true"], color="black", lw=1.5, label="true RUL")
        axt.plot(d["t"], d["direct"], lw=1, label="direct")
        axt.plot(d["sim_t"], d["sim"], "o", ms=3, label="simulated")
        axt.set_title(f"unit {u}  (final RUL {d['true'][-1]:.0f})", fontsize=9)
        axt.set_xlabel("cycle")
    axst[0].set_ylabel("RUL (cycles)")
    axst[0].legend(fontsize=7)
    figt.tight_layout()
    figt
    return


@app.cell
def _(res):
    import matplotlib.pyplot as _plt
    import numpy as _np

    h, msk = res["health"], res["mask"]
    worst = _np.argsort(res["per_unit"]["y_true"])[:6]
    fig2, ax2 = _plt.subplots(figsize=(8, 3.5))
    for j in worst:
        t = msk[j].sum()
        ax2.plot(range(t), h[j, :t], lw=1,
                 label=f"unit {j} (RUL {res['per_unit']['y_true'][j]:.0f})")
    ax2.axhline(1.0, color="grey", lw=1, ls="--")
    ax2.set_xlabel("window step")
    ax2.set_ylabel("predicted health X'")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2
    return


if __name__ == "__main__":
    app.run()
