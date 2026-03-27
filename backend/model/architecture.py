"""
backend/model/architecture.py
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

log = logging.getLogger("deeprisk.arch")


class GATBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int,
                 dropout: float, concat: bool = True, residual: bool = True):
        super().__init__()
        self.conv = GATConv(in_channels=in_dim, out_channels=out_dim, heads=heads,
                            concat=concat, dropout=dropout, add_self_loops=False)
        true_out     = out_dim * heads if concat else out_dim
        self.out_dim = true_out
        self.residual = residual
        self.proj: Optional[nn.Linear] = (
            nn.Linear(in_dim, true_out, bias=False)
            if residual and in_dim != true_out else None
        )
        self.norm = nn.LayerNorm(true_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        x_fp32 = x.float()
        if return_attn:
            h, attn_info = self.conv(x_fp32, edge_index, return_attention_weights=True)
        else:
            h = self.conv(x_fp32, edge_index)
            attn_info = None
        if not torch.isfinite(h).all():
            h = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.residual:
            res = self.proj(x_fp32) if self.proj is not None else x_fp32
            h   = h + res
        return self.norm(self.drop(F.elu(h))), attn_info


class LSTMEncoder(nn.Module):
    def __init__(self, hidden: int, n_layers: int, dropout: float,
                 chunk_size: int = 1024, use_grad_ckpt: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.out_dim       = hidden * 2
        self.chunk_size    = chunk_size
        self.use_grad_ckpt = use_grad_ckpt
        self.drop          = nn.Dropout(dropout)

    def forward(self, temporal: torch.Tensor,
                coverage_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = temporal.float().unsqueeze(-1)   # (N, T, 1)
        _, (h_n, _) = self.lstm(t)
        out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        if coverage_mask is not None:
            out = out * coverage_mask.unsqueeze(1).to(dtype=out.dtype)
        return self.drop(out)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, use_bn: bool):
        super().__init__()
        def block(i, o):
            l: List[nn.Module] = [nn.Linear(i, o)]
            if use_bn: l.append(nn.BatchNorm1d(o))
            return l + [nn.ELU(), nn.Dropout(dropout)]
        mid = max(hidden // 2, 16)
        self.net = nn.Sequential(*block(in_dim, hidden), *block(hidden, mid), nn.Linear(mid, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GATLSTM(nn.Module):
    """
    Inference-only GATLSTM.  Supports return_attention=True for explanations.
    **_extra absorbs unknown config keys from older checkpoints.
    """
    def __init__(self, n_node_feat: int, hidden: int = 96, gat_heads: int = 4,
                 gat_layers: int = 2, lstm_hidden: int = 96, lstm_layers: int = 2,
                 dropout: float = 0.40, use_bn: bool = True, use_skip: bool = True,
                 lstm_chunk_size: int = 1024, use_grad_ckpt: bool = False,
                 drop_edge_p: float = 0.0, **_extra):
        super().__init__()
        blocks: List[nn.Module] = []
        in_d = n_node_feat
        for i in range(gat_layers):
            b = GATBlock(in_dim=in_d, out_dim=hidden, heads=gat_heads,
                         dropout=dropout, concat=(i < gat_layers - 1), residual=use_skip)
            blocks.append(b); in_d = b.out_dim
        self.gat_blocks   = nn.ModuleList(blocks)
        self.lstm_encoder = LSTMEncoder(lstm_hidden, lstm_layers, dropout, lstm_chunk_size)
        self.fusion       = FusionHead(in_d + self.lstm_encoder.out_dim, hidden, dropout, use_bn)

    def forward(self, x, edge_index, temporal, coverage_mask=None, return_attention=False):
        attentions = []
        h = x
        for block in self.gat_blocks:
            h, attn = block(h, edge_index, return_attn=return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
        t = self.lstm_encoder(temporal, coverage_mask)
        logits = self.fusion(torch.cat([h, t], dim=-1))
        return (logits, attentions) if return_attention else logits


def load_from_checkpoint(path: str, device: str = "cpu") -> Tuple[GATLSTM, Dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    model = GATLSTM(n_node_feat=41, **{k: cfg[k] for k in cfg
                    if k in GATLSTM.__init__.__code__.co_varnames})
    miss, unexp = model.load_state_dict(ckpt["model_state"], strict=False)
    if miss:  log.warning(f"Missing keys: {miss}")
    if unexp: log.warning(f"Unexpected keys: {unexp}")
    model.to(device).eval()
    log.info(f"Loaded checkpoint: epoch={ckpt.get('epoch',0)}  val_auc={ckpt.get('val_auc',0):.4f}")
    return model, ckpt
