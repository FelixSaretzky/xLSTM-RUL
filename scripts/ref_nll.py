import torch
from rulbench.pretrain.windows import WindowSampler, WindowConfig
s = WindowSampler(["data/base_val.h5"], WindowConfig(window=256))
b = s._assemble(s.fixed_eval_draws(per_unit=2, seed=0)[:2000])
print("dyn   :", float(torch.log(b["y_dyn"].flatten().std()) + 0.5))
print("health:", float(torch.log(b["y_health"][b["mask"]].std()) + 0.5))