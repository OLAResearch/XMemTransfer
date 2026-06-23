"""LSH-based semantic key for Tier C C-1 prototype.

Motivated by the causal evidence that n-gram-context hash addressability is
the +15pp BoolQ ceiling cause.

This module provides a semantic alternative to the existing token-id-based
n-gram hash: instead of hashing 3-grams of canonical token IDs, we hash
sentence-pool embeddings of windowed text via random-hyperplane LSH. Q and A
that talk about the same content should produce nearby embeddings → similar
LSH bucket indices → routing to the same memory slot.

Designed as a drop-in alternative at the diagnostic level — the Phase 0.2
hit@k diagnostic can swap canonicalizers via `--canonicalizer lsh` and
re-measure on the existing 200-probe set without retraining anything.

Stage (i) acceptance per design doc: hit@10 ≥ 30% under LSH key would justify
escalating to Stage (ii) source-memory retrain.
"""

from __future__ import annotations

from typing import List, Optional

import torch


class HyperplaneLSH:
    """Random-hyperplane LSH producing per-head slot indices in [0, table_size).

    Generates `n_heads` independent hash families, each consisting of
    `log2(table_size)` random hyperplanes in the embedding space. For each
    input vector v and head h, project v onto each hyperplane, take the sign,
    pack the sign bits into an integer slot index ∈ [0, table_size).

    Reproducibility: hyperplanes are sampled from a torch.Generator seeded
    with `seed`. Same seed + same embedding space dim ⇒ same hyperplanes.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        table_size: int = 65536,
        seed: int = 42,
    ):
        if (table_size & (table_size - 1)) != 0 or table_size <= 1:
            raise ValueError(
                f"table_size must be a power of 2 ≥ 2; got {table_size}"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.table_size = table_size
        self.log2_table_size = int(table_size).bit_length() - 1  # 65536 → 16

        gen = torch.Generator().manual_seed(int(seed))
        # (n_heads, log2_table_size, embed_dim) random Gaussian hyperplanes
        self.hyperplanes = torch.randn(
            n_heads,
            self.log2_table_size,
            embed_dim,
            generator=gen,
        )
        # Bit weights for packing the sign bits to an integer
        self.bit_weights = (
            2 ** torch.arange(self.log2_table_size, dtype=torch.long)
        )

    def hash(self, embeddings: torch.Tensor) -> torch.LongTensor:
        """Hash a batch of embeddings to per-head slot indices.

        Args:
            embeddings: (B, T, D) tensor of mean-pooled or contextualized
                embeddings, where D == self.embed_dim. Caller is responsible
                for embedding production.

        Returns:
            (B, T, n_heads) LongTensor of slot indices, each in
            [0, table_size).
        """
        if embeddings.dim() != 3 or embeddings.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected (B, T, {self.embed_dim}); got {tuple(embeddings.shape)}"
            )
        device = embeddings.device
        # Project: (B, T, n_heads, log2_table_size)
        # embeddings: (B, T, 1, D) × hyperplanes: (n_heads, log2_table_size, D)
        # → einsum: BTD, HBD → BTHB (renamed: BTPD, HKD → BTHK)
        proj = torch.einsum(
            "btd,hkd->bthk", embeddings, self.hyperplanes.to(device)
        )
        bits = (proj > 0).long()
        # Pack bits → slot index in [0, table_size)
        bw = self.bit_weights.to(device)
        slot_indices = (bits * bw).sum(dim=-1)  # (B, T, n_heads)
        return slot_indices


class SentencePoolLSH:
    """Sentence-pool encoder + LSH for Tier C C-1 Stage (i).

    Pipeline per text:
        1. Tokenize text into word tokens (whitespace + punctuation strip)
        2. Slide a window of `window_size` words across the token stream,
           producing T_windows = max(1, n_words - window_size + 1) windows
        3. Encode each window with a sentence-transformer model
           (default: all-MiniLM-L6-v2, 22 MB, CPU-fast)
        4. Apply HyperplaneLSH to produce (T_windows, n_heads) slot indices

    This is a Stage (i) diagnostic only — no retraining involved. The
    existing source memory was trained against n-gram hash addresses, not
    LSH; the slots probed here are NOT the slots populated at source
    training. Stage (i) measures whether Q and A's LSH addresses overlap
    (a necessary condition for LSH-based retrieval to ever work); Stage (ii)
    would retrain source memory under LSH addressing.

    Lazy SBERT import: only triggered when first text is encoded, so the
    module is importable in environments without sentence-transformers.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        n_heads: int = 8,
        table_size: int = 65536,
        window_size: int = 3,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.n_heads = n_heads
        self.table_size = table_size
        self.window_size = window_size
        self.seed = seed
        self.device = device

        # Lazy-load SBERT on first encode
        self._sbert = None
        self._lsh: Optional[HyperplaneLSH] = None

    @property
    def sbert(self):
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "engram.lsh_key requires sentence-transformers. "
                    "Install via `uv add sentence-transformers`."
                ) from exc
            self._sbert = SentenceTransformer(self.model_name, device=self.device)
        return self._sbert

    @property
    def lsh(self) -> HyperplaneLSH:
        if self._lsh is None:
            embed_dim = self.sbert.get_sentence_embedding_dimension()
            self._lsh = HyperplaneLSH(
                embed_dim=embed_dim,
                n_heads=self.n_heads,
                table_size=self.table_size,
                seed=self.seed,
            )
        return self._lsh

    def _split_windows(self, text: str) -> List[str]:
        """Split text into overlapping word windows of `window_size`.

        Returns at least one window even if the text is shorter than
        `window_size`. Words are crude whitespace-split with punctuation
        stripped at boundaries.
        """
        import re

        # Crude word tokenization mirroring AliasCanonicalizer's pattern,
        # keeping numerics this time (don't drop "1997", "2/3", etc.)
        words = re.findall(r"\S+", text)
        words = [w.strip(".,;:!?\"'()[]{}") for w in words if w.strip(".,;:!?\"'()[]{}")]
        if not words:
            return [""]
        if len(words) < self.window_size:
            return [" ".join(words)]
        return [
            " ".join(words[i:i + self.window_size])
            for i in range(len(words) - self.window_size + 1)
        ]

    def slot_indices_for_text(self, text: str) -> torch.LongTensor:
        """Return (T_windows, n_heads) slot indices for `text`.

        Empty / whitespace-only input returns shape (0, n_heads).
        """
        windows = self._split_windows(text)
        if not windows or (len(windows) == 1 and not windows[0]):
            return torch.empty((0, self.n_heads), dtype=torch.long)

        # SBERT encode → (T_windows, D)
        embeddings = self.sbert.encode(
            windows,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Add batch dim for LSH
        embeddings_b = embeddings.unsqueeze(0)  # (1, T_windows, D)
        slot_indices = self.lsh.hash(embeddings_b)[0]  # (T_windows, n_heads)
        return slot_indices.cpu()

    @staticmethod
    def _word_split_with_offsets(text: str):
        """Word-split text mirroring `_split_windows`, plus char offsets.

        Returns: list of (start_char, end_char, stripped_word). Empty stripped
        words (pure-punctuation tokens) are dropped, which is what
        `_split_windows` does too.
        """
        import re
        out = []
        for m in re.finditer(r"\S+", text):
            raw = m.group(0)
            stripped = raw.strip(".,;:!?\"'()[]{}")
            if not stripped:
                continue
            offset = raw.find(stripped)
            if offset < 0:
                offset = 0
            s = m.start() + offset
            e = s + len(stripped)
            out.append((s, e, stripped))
        return out

    def per_token_slots(
        self,
        text: str,
        tokenizer,
    ):
        """Per-token slot indices for `text` via this LSH key + tokenizer.

        Used by both the offline preprocess (`scripts/precompute_lsh_indices.py`)
        and the online eval canonicalizer (`scripts/eval_downstream.py` lsh
        path). Keeping ONE implementation here avoids drift between train-time
        and eval-time slot semantics.

        Returns:
            (token_ids: list[int], slots: numpy.ndarray[n_tokens, n_heads])

        Edge cases:
          - Empty / whitespace text → ([], zeros(0, H))
          - Punctuation-only words / unalignable docs → zeros(n_tokens, H)
          - Punctuation tokens between two words STAY BOUND TO THE PREVIOUS
            WORD (strict `<` cursor advance), preventing causal context leak
            (codex 2026-05-15 finding).

        Args:
            text: raw text string
            tokenizer: HuggingFace fast tokenizer (offset_mapping required)
        """
        import numpy as np

        if not tokenizer.is_fast:
            raise ValueError(
                "per_token_slots requires a fast tokenizer (offset_mapping)"
            )

        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not token_ids:
            return [], np.zeros((0, self.n_heads), dtype=np.int32)

        words = self._word_split_with_offsets(text)
        if not words:
            return token_ids, np.zeros((len(token_ids), self.n_heads), dtype=np.int32)

        word_slots_t = self.slot_indices_for_text(text)
        word_slots = word_slots_t.numpy().astype(np.int32)
        n_windows = word_slots.shape[0]
        if n_windows == 0:
            return token_ids, np.zeros((len(token_ids), self.n_heads), dtype=np.int32)

        # Token → word alignment with strict `<` (codex P2 fix 2026-05-15):
        # punctuation/whitespace-only tokens stay bound to the previous word
        # rather than leaking the next word's content via slot routing.
        #
        # Codex P2 fix 2026-05-16: initial tokens (in words 0..window_size-2)
        # cannot legitimately use any full `window_size` window because that
        # window contains FUTURE words. Original `max(0, ...)` collapsed all
        # such positions to window 0 = [w0, w1, w2], which encodes future w1
        # and w2 content into the slot — causal leakage at every doc start.
        # FIX: assign zero slots to initial tokens (memory-blind at doc start,
        # but causal-clean). Matches the n-gram hasher's left-padded-zeros
        # initial-position behavior.
        token_slots = np.zeros((len(token_ids), self.n_heads), dtype=np.int32)
        w_idx = 0
        n_words = len(words)
        for t, (t_start, _t_end) in enumerate(offsets):
            while w_idx + 1 < n_words and words[w_idx][1] < t_start:
                w_idx += 1
            target_win = w_idx - (self.window_size - 1)
            if target_win < 0:
                # Initial position: not enough past context for a full window;
                # leave slot as zero (no future-context leak).
                continue
            win_idx = min(target_win, n_windows - 1)
            token_slots[t] = word_slots[win_idx]

        return token_ids, token_slots
