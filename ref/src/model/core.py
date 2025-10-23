# src/model/core.py
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import math, torch
import torch.nn as nn
import torch.nn.functional as F

AttnType = Literal["standard", "gqa", "mla"]

@dataclass
class ModelConfig:
    vocab_size: int = 128_000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    d_ff: int = 14336
    attn_type: AttnType = "mla"
    mla_rq: int = 1024
    mla_rkv: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    mask_fraction_min: float = 0.15
    mask_fraction_max: float = 0.90

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.weight * x / (norm + self.eps)

class SwiGLU(nn.Module):
    """
    y = (x W_upA) ⊙ swish(x W_upB) -> y W_down
    W_upA and W_upB are random but fixed (frozen).
    """
    def __init__(self, d_model, d_ff, dropout=0.0, freeze_up=True):
        super().__init__()
        self.W_upA = nn.Linear(d_model, d_ff, bias=False)
        self.W_upB = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        if freeze_up:
            for p in self.W_upA.parameters(): p.requires_grad_(False)
            for p in self.W_upB.parameters(): p.requires_grad_(False)
    def forward(self, x):
        a = self.W_upA(x)
        b = self.W_upB(x)
        y = a * torch.sigmoid(b)
        y = self.W_down(self.dropout(y))
        return y

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model; self.h = n_heads; self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q = self.Wq(x).view(B,T,self.h,self.dh).transpose(1,2)
        k = self.Wk(x).view(B,T,self.h,self.dh).transpose(1,2)
        v = self.Wv(x).view(B,T,self.h,self.dh).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dh)
        if attn_mask is not None: att = att + attn_mask
        p = self.drop(att.softmax(dim=-1))
        out = (p @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.Wo(out)

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert (n_heads % n_kv_heads) == 0
        self.d = d_model; self.hq = n_heads; self.hkv = n_kv_heads
        self.group = self.hq // self.hkv
        self.dhq = d_model // n_heads
        self.dhkv = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.hkv*self.dhkv, bias=False)
        self.Wv = nn.Linear(d_model, self.hkv*self.dhkv, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        B,T,D = x.shape
        q = self.Wq(x).view(B,T,self.hq,self.dhq).transpose(1,2)
        k = self.Wk(x).view(B,T,self.hkv,self.dhkv).transpose(1,2)
        v = self.Wv(x).view(B,T,self.hkv,self.dhkv).transpose(1,2)
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dhq)
        if attn_mask is not None: att = att + attn_mask
        p = self.drop(att.softmax(dim=-1))
        out = (p @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.Wo(out)

class MLA(nn.Module):
    """
    Low-rank Q/K/V with latent ranks rq, rkv, then group-expand KV heads (as in GQA).
    """
    def __init__(self, d_model, n_heads, n_kv_heads, rq, rkv, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model; self.hq = n_heads; self.hkv = n_kv_heads
        self.group = self.hq // self.hkv
        self.dh = d_model // n_heads
        self.Wq1 = nn.Linear(d_model, rq, bias=False)
        self.Wq2 = nn.Linear(rq, self.hq*self.dh, bias=False)
        self.Wk1 = nn.Linear(d_model, rkv, bias=False)
        self.Wk2 = nn.Linear(rkv, self.hkv*self.dh, bias=False)
        self.Wv1 = nn.Linear(d_model, rkv, bias=False)
        self.Wv2 = nn.Linear(rkv, self.hkv*self.dh, bias=False)
        self.Wo  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        B,T,D = x.shape
        q = self.Wq2(self.Wq1(x)).view(B,T,self.hq,self.dh).transpose(1,2)
        k = self.Wk2(self.Wk1(x)).view(B,T,self.hkv,self.dh).transpose(1,2)
        v = self.Wv2(self.Wv1(x)).view(B,T,self.hkv,self.dh).transpose(1,2)
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.dh)
        if attn_mask is not None: att = att + attn_mask
        p = self.drop(att.softmax(dim=-1))
        out = (p @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.Wo(out)

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        if cfg.attn_type == "standard":
            self.attn = MHA(cfg.d_model, cfg.n_heads, cfg.dropout)
        elif cfg.attn_type == "gqa":
            self.attn = GQA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout)
        else:
            self.attn = MLA(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.mla_rq, cfg.mla_rkv, cfg.dropout)
        self.mlp  = SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, freeze_up=True)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TiedEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.tie = cfg.tie_embeddings
        if not self.tie:
            self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.out = None
    def forward(self, input_ids):
        return self.embed(input_ids)
    def output_logits(self, h):
        if self.out is not None:
            return self.out(h)
        return F.linear(h, self.embed.weight)

class LayeredDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = TiedEmbedding(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
    @torch.no_grad()
    def forward_upto(self, input_ids, L, attn_mask=None):
        x = self.tok(input_ids)
        for i in range(L):
            x = self.blocks[i](x, attn_mask=attn_mask)
        return x
    def forward_layer(self, prev_act, layer_idx, attn_mask=None):
        return self.blocks[layer_idx](prev_act, attn_mask=attn_mask)
    def output_logits(self, h):
        h = self.final_norm(h)
        return self.tok.output_logits(h)
