import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig,
    sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig,
)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.query   = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.02)
        self.k_proj  = nn.Linear(dim, dim)
        self.v_proj  = nn.Linear(dim, dim)
        self.out     = nn.Linear(dim, dim)

    def forward(self, z, mask=None, return_weights=False):
        B, S, D = z.shape
        H, hd   = self.num_heads, self.head_dim

        k = self.k_proj(z).view(B, S, H, hd).transpose(1, 2)   
        v = self.v_proj(z).view(B, S, H, hd).transpose(1, 2)

        scores = (self.query * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))

        alpha = scores.softmax(dim=-1)                          
        h     = (alpha @ v).squeeze(2)                          
        h     = self.out(h.reshape(B, D))                       

        if return_weights:
            return h, alpha.squeeze(2)  
        return h


class ContextEncoder(nn.Module):
    def __init__(self, in_features: int = 2, embedding_dim: int = 256,
                 num_blocks: int = 6, context_length: int = 512):
        super().__init__()

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=3,
                    num_heads=4,
                    qkv_proj_blocksize=4,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=3,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            slstm_at=[num_blocks - 1],
        )

        self.in_proj = nn.Linear(in_features, embedding_dim)
        self.xlstm   = xLSTMBlockStack(cfg)
        self.pool    = AttentionPooling(embedding_dim, num_heads=4)

    def forward(self, ctx, mask=None, return_weights=False):
        z = self.in_proj(ctx)       
        z = self.xlstm(z)           
        return self.pool(z, mask=mask, return_weights=return_weights)
