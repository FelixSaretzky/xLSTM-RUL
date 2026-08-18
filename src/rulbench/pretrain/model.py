"""
The pretraining model: causal xLSTM encoder with three heads.

    window (B, T, C) --Linear--> xLSTM block stack --> hidden states (B, T, D)
        |-- health head    Linear(D, 1) per step          -> X'(t) estimate
        |-- dynamics head  cross-attention over the grid  -> log mu, log sigma
        `-- RUL head       AttentionPooling -> Linear     -> RUL / rul_cap

The encoder is causal by construction (mLSTM parallel form is lower-
triangular, sLSTM is recurrent), so with left-aligned content and
trailing pads no real step ever sees a pad.  The stack has no mask API;
masking happens exclusively in the heads (masked health loss, key
padding in the cross-attention, masked pooling).

The dynamics head reads the operator curves off the encoded window with
grid-query cross-attention.  Architecturally this follows the
AttentionOperator of OpenFIM (github.com/FIM4Science/OpenFIM @ cee2bb5,
MIT) in its GNOT-style repeated-query branch, reimplemented directly on
``nn.MultiheadAttention``; supervision is in log space, so the 5-orders-
of-magnitude spread of the operator targets becomes a relative-error
objective and positivity needs no softplus (exponentiate at simulation
time).  ``AttentionPooling`` mirrors ``src/model/encoding.py``.

Loss targets are all O(1) by construction: hi in threshold units, RUL
divided by rul_cap, and the log operators standardised by the sampler
(raw log values are ~N(-5, 2^2); unstandardised they would dominate the
shared gradient ~30x and silently unbalance the heads).  The
training CLI logs the per-head parts at step 1, so a regression of this
invariant is visible, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
from torch import nn

from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig,
)


@dataclass
class ModelConfig:
    in_channels: int = 32       # = WindowConfig.max_channels
    embedding_dim: int = 512
    num_blocks: int = 8
    context_length: int = 512
    slstm_last: bool = True         # n-1 mLSTM blocks + 1 sLSTM block
    slstm_backend: str = "vanilla"  # "cuda" on the cluster (CC >= 8.0)
    pool_heads: int = 4
    dyn_layers: int = 2
    dyn_heads: int = 4
    w_health: float = 1.0
    w_dyn: float = 1.0
    w_rul: float = 1.0


class AttentionPooling(nn.Module):
    """Masked single-query attention pooling over time (per
    ``src/model/encoding.py``); mask True = valid step."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Parameter(
            torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, z, mask=None):
        B, S, D = z.shape
        H, hd = self.num_heads, self.head_dim
        k = self.k_proj(z).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(z).view(B, S, H, hd).transpose(1, 2)
        scores = (self.query * self.scale) @ k.transpose(-2, -1)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        alpha = scores.softmax(dim=-1)
        h = (alpha @ v).squeeze(2)
        return self.out(h.reshape(B, D))


class GridCrossAttention(nn.Module):
    """Grid queries attend over the encoded window; one output vector per
    grid point (see module doc for provenance)."""

    def __init__(self, dim: int, n_layers: int = 2, n_heads: int = 4,
                 out_features: int = 2):
        super().__init__()
        self.query_embed = nn.Sequential(
            nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(dim, n_heads, batch_first=True)
            for _ in range(n_layers))
        self.norm1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                          nn.Linear(4 * dim, dim))
            for _ in range(n_layers))
        self.norm2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.out = nn.Linear(dim, out_features)

    def forward(self, h, mask, grid):
        # h (B, T, D), mask (B, T) True = valid, grid (G,)
        q = self.query_embed(grid[:, None])[None].expand(h.shape[0], -1, -1)
        for attn, n1, ff, n2 in zip(self.attn, self.norm1, self.ff, self.norm2):
            a, _ = attn(q, h, h, key_padding_mask=~mask, need_weights=False)
            q = n1(q + a)
            q = n2(q + ff(q))
        return self.out(q)                               # (B, G, out_features)


def _stack_config(cfg: ModelConfig) -> xLSTMBlockStackConfig:
    mlstm = mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
        conv1d_kernel_size=3, num_heads=4, qkv_proj_blocksize=4))
    if cfg.slstm_last:
        slstm = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=cfg.slstm_backend, num_heads=4, conv1d_kernel_size=3,
                bias_init="powerlaw_blockdependent"),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"))
        slstm_at = [cfg.num_blocks - 1]
    else:
        slstm, slstm_at = None, []
    return xLSTMBlockStackConfig(
        mlstm_block=mlstm, slstm_block=slstm, slstm_at=slstm_at,
        context_length=cfg.context_length, embedding_dim=cfg.embedding_dim,
        num_blocks=cfg.num_blocks)


class RULPretrainModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        D = cfg.embedding_dim
        self.in_proj = nn.Linear(cfg.in_channels, D)
        self.encoder = xLSTMBlockStack(_stack_config(cfg))
        self.health_head = nn.Linear(D, 1)
        self.dyn_head = GridCrossAttention(
            D, n_layers=cfg.dyn_layers, n_heads=cfg.dyn_heads)
        self.rul_pool = AttentionPooling(D, num_heads=cfg.pool_heads)
        self.rul_head = nn.Linear(D, 1)

    def forward(self, x, mask, grid) -> dict:
        """x (B, T, C) normalised windows, mask (B, T) True = real step,
        grid (G,) query locations for the dynamics head."""
        h = self.encoder(self.in_proj(x))
        return {
            "health": self.health_head(h).squeeze(-1),           # (B, T)
            "dyn": self.dyn_head(h, mask, grid),                 # (B, G, 2)
            "rul": self.rul_head(
                self.rul_pool(h, mask=mask)).squeeze(-1),        # (B,)
        }


def pretrain_loss(out: dict, batch: dict, cfg: ModelConfig
                  ) -> tuple[torch.Tensor, dict]:
    """Weighted sum of the three heads' losses; returns (total, parts)."""
    m = batch["mask"].float()
    se = (out["health"] - batch["y_health"]) ** 2 * m
    parts = {"health": se.sum() / m.sum().clamp(min=1.0),
             "dyn": ((out["dyn"] - batch["y_dyn"]) ** 2).mean(),
             "rul": ((out["rul"] - batch["y_rul"]) ** 2).mean()}
    total = (cfg.w_health * parts["health"] + cfg.w_dyn * parts["dyn"]
             + cfg.w_rul * parts["rul"])
    return total, parts


def model_summary(model: RULPretrainModel) -> str:
    n = sum(p.numel() for p in model.parameters())
    return (f"RULPretrainModel "
            f"D={model.cfg.embedding_dim} blocks={model.cfg.num_blocks} "
            f"params={n / 1e6:.2f}M")


def save_checkpoint(path, model: RULPretrainModel, extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(model.cfg), **(extra or {})}, path)


def load_checkpoint(path, map_location="cpu") -> RULPretrainModel:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = RULPretrainModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model
