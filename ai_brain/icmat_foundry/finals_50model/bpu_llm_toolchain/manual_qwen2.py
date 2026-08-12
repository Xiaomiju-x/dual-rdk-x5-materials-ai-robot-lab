"""Static-shape Qwen2/Qwen2.5 decoder segments for Bayes-e conversion.

This is an isolated, auditable successor of tools/bpu_transformer/qwen2_manual.py
and qwen2_bpu_split.py. It deliberately exports decoder blocks only. Token
embedding, final RMSNorm, and LM head remain CPU tensors in the sidecar runtime.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Qwen2StaticConfig:
    hidden_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_q_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @classmethod
    def from_hf_config(cls, path: Path, seq_len: int) -> "Qwen2StaticConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rope = raw.get("rope_parameters") or {}
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            num_q_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw["num_key_value_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            max_seq_len=seq_len,
            rope_theta=float(rope.get("rope_theta", raw.get("rope_theta", 10000.0))),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        )


class ManualRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = (value * value).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + self.eps) * self.weight


class ManualRoPE(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        frequencies = torch.outer(torch.arange(seq_len).float(), inv_freq)
        self.register_buffer("cos", torch.cat([frequencies.cos()] * 2, dim=-1))
        self.register_buffer("sin", torch.cat([frequencies.sin()] * 2, dim=-1))

    @staticmethod
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat([-value[..., half:], value[..., :half]], dim=-1)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos.unsqueeze(0).unsqueeze(0)
        sin = self.sin.unsqueeze(0).unsqueeze(0)
        return (
            query * cos + self.rotate_half(query) * sin,
            key * cos + self.rotate_half(key) * sin,
        )


class ManualAttention(nn.Module):
    def __init__(self, cfg: Qwen2StaticConfig, rope: ManualRoPE) -> None:
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        dim = cfg.hidden_size
        self.q_proj = nn.Linear(dim, cfg.num_q_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, cfg.num_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_q_heads * cfg.head_dim, dim, bias=False)
        mask = torch.triu(torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        length = cfg.max_seq_len
        query = self.q_proj(value).reshape(1, length, cfg.num_q_heads, cfg.head_dim).transpose(1, 2)
        key = self.k_proj(value).reshape(1, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        val = self.v_proj(value).reshape(1, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        query, key = self.rope(query, key)
        key = key.unsqueeze(2).expand(1, cfg.num_kv_heads, cfg.num_kv_groups, length, cfg.head_dim)
        val = val.unsqueeze(2).expand(1, cfg.num_kv_heads, cfg.num_kv_groups, length, cfg.head_dim)
        key = key.reshape(1, cfg.num_q_heads, length, cfg.head_dim)
        val = val.reshape(1, cfg.num_q_heads, length, cfg.head_dim)
        attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        attention = F.softmax(attention + self.causal_mask, dim=-1)
        output = torch.matmul(attention, val)
        output = output.transpose(1, 2).reshape(1, length, cfg.hidden_size)
        return self.o_proj(output)


class ManualMLP(nn.Module):
    def __init__(self, cfg: Qwen2StaticConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class ManualBlock(nn.Module):
    def __init__(self, cfg: Qwen2StaticConfig, rope: ManualRoPE) -> None:
        super().__init__()
        self.ln1 = ManualRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = ManualAttention(cfg, rope)
        self.ln2 = ManualRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = ManualMLP(cfg)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.ln1(value))
        return value + self.mlp(self.ln2(value))


class ManualSegment(nn.Module):
    def __init__(self, cfg: Qwen2StaticConfig, layer_start: int, layer_end: int) -> None:
        super().__init__()
        self.layer_start = layer_start
        self.layer_end = layer_end
        rope = ManualRoPE(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.layers = nn.ModuleList([ManualBlock(cfg, rope) for _ in range(layer_end - layer_start)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


WEIGHT_SUFFIXES = (
    ("self_attn.q_proj.weight", "attn.q_proj.weight"),
    ("self_attn.q_proj.bias", "attn.q_proj.bias"),
    ("self_attn.k_proj.weight", "attn.k_proj.weight"),
    ("self_attn.k_proj.bias", "attn.k_proj.bias"),
    ("self_attn.v_proj.weight", "attn.v_proj.weight"),
    ("self_attn.v_proj.bias", "attn.v_proj.bias"),
    ("self_attn.o_proj.weight", "attn.o_proj.weight"),
    ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
    ("mlp.up_proj.weight", "mlp.up_proj.weight"),
    ("mlp.down_proj.weight", "mlp.down_proj.weight"),
    ("input_layernorm.weight", "ln1.weight"),
    ("post_attention_layernorm.weight", "ln2.weight"),
)


def load_segment_weights(
    segment: ManualSegment,
    hf_state: dict[str, torch.Tensor],
    layer_start: int,
) -> dict[str, object]:
    state = segment.state_dict()
    missing: list[str] = []
    mismatched: list[str] = []
    loaded = 0
    for local_index in range(len(segment.layers)):
        hf_index = layer_start + local_index
        for hf_suffix, local_suffix in WEIGHT_SUFFIXES:
            source = f"model.layers.{hf_index}.{hf_suffix}"
            target = f"layers.{local_index}.{local_suffix}"
            if source not in hf_state or target not in state:
                missing.append(f"{source}->{target}")
                continue
            if tuple(hf_state[source].shape) != tuple(state[target].shape):
                mismatched.append(f"{source}:{tuple(hf_state[source].shape)}!={tuple(state[target].shape)}")
                continue
            state[target].copy_(hf_state[source].float())
            loaded += 1
    expected = len(segment.layers) * len(WEIGHT_SUFFIXES)
    if missing or mismatched or loaded != expected:
        raise ValueError(
            f"strict weight mapping failed: loaded={loaded}/{expected}, "
            f"missing={missing[:3]}, mismatched={mismatched[:3]}"
        )
    return {"loaded_tensors": loaded, "expected_tensors": expected}
