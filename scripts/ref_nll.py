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
    print(f"{name:11s}: sd {float(sd):.4f} -> uninformed NLL "
          f"{float(torch.log(sd) + 0.5):+.4f}")

ref("dyn",        b["y_dyn"].flatten())          # over grid points and channels
ref("health",     b["y_health"][b["mask"]])      # over real steps, like the loss
ref("health_end", b["hi_end"])                   # one value per window