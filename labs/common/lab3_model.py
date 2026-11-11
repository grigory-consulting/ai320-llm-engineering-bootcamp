"""Mini-GPT aus Teil IV: Decoder-only, Pre-Norm mit RMSNorm, GELU-MLP, gelernte Position, Weight Tying.
Mit `ffn="swiglu"` ersetzt SwiGLU (Folie "SwiGLU als Variante") das GELU-MLP, innere Breite round(2/3 · 4d).

Standardkonfiguration V = 8.000, d = 256, L = 4, 4 Heads, T = 256 ergibt 5.261.568 Parameter.
Wie auf der Modellfolie in Teil IV initialisiert `_init_weights` lineare Schichten und Embeddings mit einer
Normalverteilung (std 0,02). Die Ausgabeprojektionen der Residual-Pfade nutzen std 0,02 / sqrt(2 L).
Ohne diese Initialisierung behalten Token- und Positions-Embedding die PyTorch-Voreinstellung std 1.
Über die gebundene Ausgabeschicht entstehen dann sehr große Logits. Der Start-Loss steigt von 9,04
auf rund 180.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)    # Q, K, V in einer Projektion
        self.proj = nn.Linear(d, d, bias=False)       # Ausgabeprojektion

    def forward(self, x):
        B, T, C = x.shape
        h = self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, h, C // h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, n_head, ffn="gelu"):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.norm2 = nn.RMSNorm(d)
        self.mlp = MLP(d) if ffn == "gelu" else SwiGLU(d, round(2 / 3 * 4 * d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))     # Pre-Norm, Residual
        return x + self.mlp(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab=8000, d=256, n_layer=4, n_head=4, ctx=256, ffn="gelu"):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.Sequential(*[Block(d, n_head, ffn) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # Weight Tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():     # Ausgabeprojektionen der Residual-Pfade kleiner (GPT-2)
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        return self.head(self.norm(self.blocks(x)))
