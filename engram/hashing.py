"""GPU-compatible deterministic multi-head N-gram hashing.

Produces per-token hash indices into the memory embedding tables.
All operations use torch tensors for GPU compatibility.
"""

import random
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class HashConfig:
    max_ngram: int = 3
    heads_per_order: int = 4
    table_size: int = 65536
    seed: int = 0

    @property
    def total_heads(self) -> int:
        return (self.max_ngram - 1) * self.heads_per_order


class NgramHasher:
    """Deterministic multi-head hashed N-gram indexing (GPU-compatible).

    Input: canonical_ids (B, T)
    Output: indices (B, T, H) where H = (max_ngram - 1) * heads_per_order

    All internal operations are tensor-based for GPU compatibility.
    """

    def __init__(self, cfg: HashConfig):
        self.cfg = cfg
        rng = random.Random(cfg.seed)

        # Position multipliers (odd) for XOR mixing
        self.pos_mult = [rng.randrange(1, 2**31, 2) for _ in range(cfg.max_ngram)]
        # Per-head salts
        self.head_salt = [rng.randrange(0, 2**31) for _ in range(cfg.total_heads)]

    def _shift_left_pad(self, x: torch.LongTensor, k: int) -> torch.LongTensor:
        """Shift x right by k positions, padding left with zeros.

        Returns a tensor of the same (B, T) shape as `x`. When T < k the
        entire output is zero (caller's input was shorter than the shift
        amount, so no original tokens survive). Without this guard the
        original implementation produced shape (B, T+k-...) for T<k due to
        torch's permissive negative-stride slicing, which then broke
        `torch.stack` downstream.
        """
        if k == 0:
            return x
        B, T = x.shape
        if k >= T:
            return torch.zeros(B, T, dtype=x.dtype, device=x.device)
        pad = torch.zeros(B, k, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x[:, :T - k]], dim=1)

    def __call__(self, canon_ids: torch.LongTensor) -> torch.LongTensor:
        """Compute hash indices for all N-gram orders and heads.

        Args:
            canon_ids: (B, T) canonical token IDs

        Returns:
            (B, T, H) hash indices into embedding tables
        """
        B, T = canon_ids.shape

        # Precompute shifted views for N-gram context
        shifts = [self._shift_left_pad(canon_ids, k) for k in range(self.cfg.max_ngram)]

        all_heads: List[torch.LongTensor] = []
        head_idx = 0

        for n in range(2, self.cfg.max_ngram + 1):
            # Gather tokens for this N-gram order
            tokens = shifts[n - 1 :: -1]  # [shift[n-1], shift[n-2], ..., shift[0]]

            # XOR-mix across positions
            mix = tokens[0] * self.pos_mult[0]
            for pos in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[pos] * self.pos_mult[pos])

            # Per-head salt and modulo
            for _ in range(self.cfg.heads_per_order):
                salt = self.head_salt[head_idx]
                head_hash = torch.remainder(
                    torch.abs(torch.bitwise_xor(mix, salt)), self.cfg.table_size
                )
                all_heads.append(head_hash)
                head_idx += 1

        return torch.stack(all_heads, dim=-1)  # (B, T, H)


class WordNgramHasher:
    """Hash word-level N-grams for cross-tokenizer canonicalization.

    Uses CRC32-based hashing on word strings, compatible with
    WordBoundaryCanonicalizer output.
    """

    def __init__(self, cfg: HashConfig):
        self.cfg = cfg
        rng = random.Random(cfg.seed)
        self.head_salt = [rng.randrange(0, 2**31) for _ in range(cfg.total_heads)]

    def hash_word_ngrams(
        self,
        word_ngrams: "List[List[List[tuple]]]",
        device: torch.device,
    ) -> torch.LongTensor:
        """Hash pre-computed word N-gram tuples.

        Args:
            word_ngrams: [B][T][n_orders] of word tuples
            device: target device

        Returns:
            (B, T, H) hash indices
        """
        import zlib

        B = len(word_ngrams)
        T = len(word_ngrams[0]) if B > 0 else 0
        H = self.cfg.total_heads

        # Build flat list in pure Python, then create tensor once at the end
        # to avoid B×T×H individual tensor element assignments
        flat = []
        for b in range(B):
            for t in range(len(word_ngrams[b])):
                head_idx = 0
                for order_idx, ngram_tuple in enumerate(word_ngrams[b][t]):
                    key_str = "|".join(ngram_tuple)
                    base_hash = zlib.crc32(key_str.encode("utf-8")) & 0xFFFFFFFF

                    for _ in range(self.cfg.heads_per_order):
                        salt = self.head_salt[head_idx]
                        flat.append((base_hash ^ salt) % self.cfg.table_size)
                        head_idx += 1

        indices = torch.tensor(flat, dtype=torch.long, device=device).view(B, T, H)
        return indices
