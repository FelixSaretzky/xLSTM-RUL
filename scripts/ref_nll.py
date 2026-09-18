"""Uninformed reference: the Gaussian NLL of predicting the marginal.

This is the zero point every validation number is read against. Compute it
with EXACTLY the configuration under test -- window length, p_terminal, the
same fixed_eval_draws -- since each of those changes the target distribution
and therefore the achievable NLL.
"""
import torch
from rulbench.pretrain.windows import WindowSampler, WindowConfig

cfg = WindowConfig(window=256, p_terminal=0.0)
s = WindowSampler(["data3/base_val.h5"], cfg, seed=1)
draws = s.fixed_eval_draws(per_unit=2, seed=0)
b = s._assemble(draws[:4000])
print(f"{len(draws)} eval draws over {s.n_units} units\n")

def ref(name, y):
    sd = y.std()
    print(f"{name:16s}: sd {float(sd):.4f} -> uninformed NLL "
          f"{float(torch.log(sd) + 0.5):+.4f}")


def ref_per_point(name, y):
    """Reference for a model that knows the target STRUCTURE but nothing
    about the individual unit: one marginal per grid point and channel.

    The flat version below pools all grid points into one distribution and
    is therefore too easy to beat -- measured, a zero-input run reached
    0.059 against a flat reference of 0.187 without seeing any sensor.
    """
    sd = y.std(dim=0)                      # (G, 2)
    print(f"{name:16s}: sd {float(sd.mean()):.4f} (mean over points) -> "
          f"uninformed NLL {float((torch.log(sd) + 0.5).mean()):+.4f}")


ref("dyn (flat)",   b["y_dyn"].flatten())
ref_per_point("dyn (per-point)", b["y_dyn"])
ref("health",       b["y_health"][b["mask"]])
ref("health_end",   b["hi_end"])