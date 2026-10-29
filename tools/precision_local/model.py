"""Small independent decoder reference; no Megatron or CUDA imports."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DeterministicByteEmbedding(nn.Embedding):
    """Use a dense FP32 gather equivalent for a small vocabulary on MPS.

    MPS embedding_dense_backward has no deterministic implementation. One-hot
    GEMM avoids that scatter reduction while preserving the lookup and gradient
    mathematics. This is a reference-only choice, not a large-vocabulary kernel.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.autocast(tokens.device.type, enabled=False):
            one_hot = F.one_hot(tokens, num_classes=self.num_embeddings).to(self.weight.dtype)
            return one_hot @ self.weight


class RMSNorm(nn.Module):
    """Accumulate normalization in FP32."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        return (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.epsilon) * self.weight
        )


class DecoderBlock(nn.Module):
    """Pre-norm causal attention and SwiGLU with explicit precision boundaries."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        width, hidden = config["width"], config["ffn_width"]
        self.heads = config["heads"]
        self.head_dim = width // self.heads
        if width % self.heads or self.head_dim % 2:
            raise ValueError("Head dimension must be even and divide model width")
        self.dropout = config["dropout"]
        self.attention_norm = RMSNorm(width, config["norm_epsilon"])
        self.ffn_norm = RMSNorm(width, config["norm_epsilon"])
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        frequencies = config["rope_theta"] ** (
            -torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )
        angles = torch.outer(torch.arange(config["sequence_length"]), frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.ones(config["sequence_length"], config["sequence_length"], dtype=torch.bool).triu(
                1
            ),
            persistent=False,
        )

    def rotate(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[-2]
        even, odd = value.float()[..., 0::2], value.float()[..., 1::2]
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(value.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        query, key = self.rotate(query), self.rotate(key)
        scores = (query @ key.transpose(-1, -2)).float() / math.sqrt(self.head_dim)
        probabilities = scores.masked_fill(self.causal_mask[:length, :length], -torch.inf).softmax(
            -1
        )
        context = (
            (probabilities.to(value.dtype) @ value).transpose(1, 2).reshape(batch, length, width)
        )
        x = x + F.dropout(self.projection(context).float(), self.dropout, self.training)
        normed = self.ffn_norm(x)
        hidden = F.silu(self.gate(normed).float()) * self.up(normed).float()
        return x + F.dropout(self.down(hidden).float(), self.dropout, self.training)


class TinyDecoder(nn.Module):
    """Untied byte embeddings/head with FP32 persistent parameters."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.embedding = DeterministicByteEmbedding(config["vocab_size"], config["width"])
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config["layers"])])
        self.norm = RMSNorm(config["width"], config["norm_epsilon"])
        self.head = nn.Linear(config["width"], config["vocab_size"], bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)).float()


def token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """All byte targets are valid; average CE over the exact B*T target tokens."""
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
