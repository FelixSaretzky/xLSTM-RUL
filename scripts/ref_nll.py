import numpy as np 
import torch
from rulbench.pretrain.windows import WindowSampler, WindowConfig
s = WindowSampler(["data/base_val.h5"], WindowConfig(window=256))
b = s._assemble(s.fixed_eval_draws(per_unit=2, seed=0)[:2000])
print("dyn   :", float(torch.log(b["y_dyn"].flatten().std()) + 0.5))
print("health:", float(torch.log(b["y_health"][b["mask"]].std()) + 0.5))


anchored, free = [], []
rng = np.random.default_rng(0)
for i in range(min(s.n_units, 500)):
    L, on = int(s.lengths[i]), int(s.onset[i])
    for _ in range(3):
        e = int(rng.integers(min(127, L - 1), L))
        start = max(0, e - 128 + 1)
        base = s.sensors[i][:on, :s.n_process[i]].mean(0)   # true healthy baseline
        dev = np.linalg.norm(s.sensors[i][e, :s.n_process[i]] - base)
        (anchored if start < on else free).append((dev, float(s.hi[i][e])))

for name, rows in (("window contains onset", anchored),
                   ("window after onset", free)):
    a = np.array(rows)
    if len(a) > 10:
        print(f"{name:24s} n={len(a):5d}  corr(||x-baseline||, X') = "
              f"{np.corrcoef(a[:,0], a[:,1])[0,1]:+.3f}")