"""Tokenizer-agnostic canonicalization for cross-model N-gram hashing.

Two modes:
  1. Vocab-derived: Map token IDs to canonical IDs via normalized strings
     (works when tokenizers share the same surface forms).
  2. Word-boundary: Detect word boundaries from BPE/SentencePiece tokens and
     build word-level N-grams, so the same text produces the same hash
     indices regardless of subword segmentation.
"""

import unicodedata
from typing import Dict, List, Optional

import torch


def normalize_token(text: str) -> str:
    """NFKC + lowercase + strip accents + collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text


# ── Vocab-derived canonicalization (same-tokenizer transfer) ──────────────


class VocabCanonicalizer:
    """Maps token IDs → canonical IDs via normalized token strings.

    Used for same-tokenizer transfer (e.g. Pythia-160M → Pythia-410M)
    where both models share the GPT-NeoX tokenizer.
    """

    def __init__(self, token_strings: List[str]):
        norm_to_canon: Dict[str, int] = {}
        canon_tokens: List[str] = []
        for tok in token_strings:
            # Strip BPE/SentencePiece prefixes before normalization,
            # matching the logic in build_id_map()
            surface = _strip_prefix(tok)
            key = normalize_token(surface)
            if key not in norm_to_canon:
                norm_to_canon[key] = len(canon_tokens)
                canon_tokens.append(key)
        self.norm_to_canon = norm_to_canon
        self.canon_tokens = canon_tokens

    @property
    def canon_vocab_size(self) -> int:
        return len(self.canon_tokens)

    def build_id_map(self, tokenizer) -> torch.LongTensor:
        """Return tensor map[token_id] = canonical_id for a HF tokenizer."""
        vocab = tokenizer.get_vocab()
        mapping = torch.zeros(len(vocab), dtype=torch.long)
        for token_str, token_id in vocab.items():
            # Strip BPE/SentencePiece prefixes for normalization
            surface = _strip_prefix(token_str)
            key = normalize_token(surface)
            if key in self.norm_to_canon:
                mapping[token_id] = self.norm_to_canon[key]
            # Unknown tokens map to 0 (pad canonical)
        return mapping


def _strip_prefix(token: str) -> str:
    """Strip BPE/SentencePiece leading-space markers."""
    if token.startswith("\u0120"):  # GPT-NeoX/GPT-2 Ġ
        return token[1:]
    if token.startswith("\u2581"):  # SentencePiece ▁
        return token[1:]
    return token


# ── Word-boundary canonicalization (cross-tokenizer transfer) ─────────────


class WordBoundaryCanonicalizer:
    """Produces word-level N-gram contexts from BPE/SentencePiece token IDs.

    Detects word boundaries from token surface forms and maintains a running
    word buffer, so the same input text yields the same word-level N-gram
    sequences regardless of subword segmentation.

    Usage:
        canon = WordBoundaryCanonicalizer(tokenizer)
        word_ngram_ids = canon(input_ids)  # (B, T, max_ngram-1) indices
    """

    BOS_WORD = "<bos>"

    def __init__(self, tokenizer, max_ngram: int = 3):
        self.tokenizer = tokenizer
        self.max_ngram = max_ngram

        # Build token_id → surface string lookup
        vocab = tokenizer.get_vocab()
        self.id_to_surface: Dict[int, str] = {}
        for tok_str, tok_id in vocab.items():
            self.id_to_surface[tok_id] = tok_str

    def is_word_boundary(self, token_str: str) -> bool:
        """Detect word boundary from token surface form."""
        if token_str.startswith("\u0120"):  # GPT-NeoX Ġ
            return True
        if token_str.startswith("\u2581"):  # SentencePiece ▁
            return True
        # Fallback: decoded form starts with whitespace
        stripped = _strip_prefix(token_str)
        if stripped != token_str:
            return True
        return False

    def get_canonical_word(self, token_str: str) -> str:
        """Extract and normalize the word content from a token."""
        surface = _strip_prefix(token_str)
        return normalize_token(surface)

    def compute_word_ngrams(
        self, input_ids: torch.LongTensor
    ) -> List[List[List[str]]]:
        """Compute word-level N-gram contexts for each position.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            List[B] of List[T] of List[n_orders] word tuples for hashing
        """
        B, T = input_ids.shape
        # Single batched device→host transfer instead of B×T .item() calls
        ids_list = input_ids.cpu().tolist()
        results = []

        for b in range(B):
            seq_results = []
            current_word_parts: List[str] = []
            completed_words: List[str] = [self.BOS_WORD]

            for t in range(T):
                tid = ids_list[b][t]
                tok_str = self.id_to_surface.get(tid, "")

                if self.is_word_boundary(tok_str) and current_word_parts:
                    # Flush current word
                    word = normalize_token("".join(current_word_parts))
                    if word:
                        completed_words.append(word)
                    current_word_parts = []

                # Add current token content to word buffer
                surface = _strip_prefix(tok_str)
                if surface:
                    current_word_parts.append(surface)

                # Build N-gram contexts from completed words
                pos_ngrams = []
                for n in range(2, self.max_ngram + 1):
                    if len(completed_words) >= n:
                        ngram = tuple(completed_words[-n:])
                    else:
                        # Pad with BOS
                        pad_count = n - len(completed_words)
                        ngram = tuple(
                            [self.BOS_WORD] * pad_count + completed_words
                        )
                    pos_ngrams.append(ngram)
                seq_results.append(pos_ngrams)

            results.append(seq_results)
        return results


def build_canonicalizer(
    tokenizer,
    mode: str = "vocab",
    max_ngram: int = 3,
) -> "VocabCanonicalizer | WordBoundaryCanonicalizer":
    """Factory for canonicalizer based on mode."""
    if mode == "vocab":
        # Extract all token strings from HF tokenizer
        vocab = tokenizer.get_vocab()
        token_strings = sorted(vocab.keys(), key=lambda k: vocab[k])
        return VocabCanonicalizer(token_strings)
    elif mode == "word_boundary":
        return WordBoundaryCanonicalizer(tokenizer, max_ngram=max_ngram)
    else:
        raise ValueError(f"Unknown canonicalization mode: {mode}")
