import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset
from tirex import load_model, ForecastModel
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = 128
RUL_MAX = 125

class EmbeddingExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self._captured: torch.Tensor | None = None
        block = self._find_last_block(model)
        block.register_forward_hook(self._hook)
 
    @staticmethod
    def _find_last_block(model: nn.Module) -> nn.Module:
        candidates = [
            (name, mod)
            for name, mod in model.named_modules()
            if any(k in name.lower() for k in ("xlstm", "slstm", "mlstm", "block"))
        ]
        if not candidates:
            raise RuntimeError("No candidat found")
        return candidates[-1][1] 
 
    def _hook(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self._captured = out.detach()
 
    @torch.no_grad()
    def embed_univariate(self, context: torch.Tensor) -> torch.Tensor:
        self._captured = None
        _ = self.model.forecast(context=context.to(DEVICE), prediction_length=1)
        if self._captured is None:
            raise RuntimeError("Nothing found")
        h = self._captured 
        if h.dim() == 3:
            h = h.mean(dim=1) 
        return h.float().cpu()
 
    def embed_multivariate(self, window: np.ndarray) -> torch.Tensor:
        x = torch.tensor(window.T, dtype=torch.float32)  
        emb = self.embed_univariate(x)                   
        return emb.flatten()                             
 

class RULWindowDataset(Dataset):
    def __init__(self, trajectories: list[np.ndarray], extractor: EmbeddingExtractor,
                 context_len: int = CONTEXT_LEN, stride: int = 1):
        self.X, self.y = [], []
        for i, traj in enumerate(trajectories):
            T = len(traj)
            for end in tqdm(range(context_len, T + 1, stride), desc=f"Trajectory {i+1}/ {len(trajectories)}"):
                window = traj[end - context_len:end]        # (L, N)
                rul = min(T - end, RUL_MAX)                  # Label
                self.X.append(extractor.embed_multivariate(window))
                self.y.append(float(rul))
        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, i):
        return self.X[i], self.y[i]

tirex: ForecastModel = load_model('NX-AI/TiRex')
tirex_model = tirex 

print(type(tirex_model))

for p in tirex_model.parameters():
    p.requires_grad = False
tirex_model.eval().to(DEVICE)

extractor = EmbeddingExtractor(tirex_model)
rng = np.random.default_rng(0)
trajs = [rng.standard_normal((rng.integers(200, 400), 14)).astype("f")
             for _ in range(1)]

train_ds = RULWindowDataset(trajs, extractor)

print(train_ds[0])

#print(extractor(torch.tensor(trajs[0])))