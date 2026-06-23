"""Engram hashed N-gram memory module (GPU-compatible).

Contains the embedding tables that store learned N-gram representations.
These tables are trained on a source model and then frozen for transfer.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .hashing import HashConfig, NgramHasher


@dataclass
class MemoryConfig:
    max_ngram: int = 3
    heads_per_order: int = 4
    table_size: int = 65536
    d_head: int = 64
    hash_seed: int = 0

    @property
    def total_heads(self) -> int:
        return (self.max_ngram - 1) * self.heads_per_order

    @property
    def d_mem(self) -> int:
        return self.total_heads * self.d_head

    @property
    def hash_config(self) -> HashConfig:
        return HashConfig(
            max_ngram=self.max_ngram,
            heads_per_order=self.heads_per_order,
            table_size=self.table_size,
            seed=self.hash_seed,
        )

    @property
    def total_params(self) -> int:
        return self.total_heads * self.table_size * self.d_head


class EngramMemory(nn.Module):
    """Hashed N-gram embedding tables.

    Architecture:
        - (max_ngram - 1) * heads_per_order embedding tables
        - Each table: (table_size, d_head)
        - Output: concatenation of all head lookups → (B, T, d_mem)

    For default config: 8 tables * 65536 * 64 = 33.6M params (~67MB FP16)
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.hasher = NgramHasher(cfg.hash_config)

        self.tables = nn.ModuleList([
            nn.Embedding(cfg.table_size, cfg.d_head)
            for _ in range(cfg.total_heads)
        ])

        # Small init for stability
        for emb in self.tables:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    @property
    def d_mem(self) -> int:
        return self.cfg.d_mem

    def forward(self, canon_ids: torch.LongTensor) -> torch.Tensor:
        """Look up memory vectors for each token position.

        Args:
            canon_ids: (B, T) canonical token IDs

        Returns:
            (B, T, d_mem) memory vectors
        """
        idx = self.hasher(canon_ids)  # (B, T, H)
        chunks = []
        for h, table in enumerate(self.tables):
            chunks.append(table(idx[:, :, h]))  # (B, T, d_head)
        return torch.cat(chunks, dim=-1)  # (B, T, d_mem)

    def forward_from_indices(self, indices: torch.LongTensor) -> torch.Tensor:
        """Look up memory vectors from pre-computed hash indices.

        Args:
            indices: (B, T, H) pre-computed hash indices

        Returns:
            (B, T, d_mem) memory vectors
        """
        chunks = []
        for h, table in enumerate(self.tables):
            chunks.append(table(indices[:, :, h]))
        return torch.cat(chunks, dim=-1)

    def permute_keys(self, seed: int = 0) -> None:
        """Apply per-head random row permutation (for permuted_keys ablation).

        Preserves the marginal distribution of embedding values but breaks
        the key→value mapping learned during source training.
        """
        rng = torch.Generator().manual_seed(seed)
        for table in self.tables:
            perm = torch.randperm(self.cfg.table_size, generator=rng)
            table.weight.data = table.weight.data[perm].clone()
