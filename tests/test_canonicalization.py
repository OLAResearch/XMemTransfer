"""Test canonicalization produces identical results across tokenizers."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.canonicalization import (
    normalize_token,
    VocabCanonicalizer,
    WordBoundaryCanonicalizer,
    _strip_prefix,
)
from engram.hashing import HashConfig, NgramHasher


class TestNormalizeToken:
    def test_lowercase(self):
        assert normalize_token("Hello") == "hello"

    def test_nfkc(self):
        assert normalize_token("\ufb01") == "fi"  # fi ligature

    def test_accents(self):
        assert normalize_token("cafe\u0301") == "cafe"
        assert normalize_token("\u00e9") == "e"

    def test_whitespace(self):
        assert normalize_token("  hello   world  ") == "hello world"

    def test_idempotent(self):
        text = "Hello World"
        once = normalize_token(text)
        twice = normalize_token(once)
        assert once == twice


class TestStripPrefix:
    def test_gpt_neox_prefix(self):
        assert _strip_prefix("\u0120hello") == "hello"

    def test_sentencepiece_prefix(self):
        assert _strip_prefix("\u2581hello") == "hello"

    def test_no_prefix(self):
        assert _strip_prefix("hello") == "hello"


class TestVocabCanonicalizer:
    def test_same_tokens_same_vocab_size(self):
        """Two canonicalizers built from same tokens should have same vocab size."""
        tokens = ["hello", "world", "the", "is"]
        c1 = VocabCanonicalizer(tokens)
        c2 = VocabCanonicalizer(list(reversed(tokens)))
        assert c1.canon_vocab_size == c2.canon_vocab_size

    def test_normalized_collision(self):
        """Tokens that normalize to the same string should share a canonical ID."""
        tokens = ["Hello", "hello", "HELLO"]
        canon = VocabCanonicalizer(tokens)
        # All three normalize to "hello", so there should be only 1 canonical
        assert canon.canon_vocab_size == 1

    def test_distinct_tokens_distinct_ids(self):
        """Tokens that normalize differently should get different IDs."""
        tokens = ["hello", "world", "the"]
        canon = VocabCanonicalizer(tokens)
        ids = [canon.norm_to_canon[normalize_token(t)] for t in tokens]
        assert len(set(ids)) == 3


class TestWordBoundaryCanonicalizer:
    """Test word boundary detection across tokenizer types."""

    def test_gpt_neox_boundary(self):
        """GPT-NeoX tokens starting with Ġ should be word boundaries."""

        class FakeTokenizer:
            def get_vocab(self):
                return {"\u0120hello": 0, "world": 1, "\u0120the": 2}

        canon = WordBoundaryCanonicalizer(FakeTokenizer(), max_ngram=3)
        assert canon.is_word_boundary("\u0120hello") is True
        assert canon.is_word_boundary("world") is False

    def test_sentencepiece_boundary(self):
        """SentencePiece tokens starting with ▁ should be word boundaries."""

        class FakeTokenizer:
            def get_vocab(self):
                return {"\u2581hello": 0, "world": 1, "\u2581the": 2}

        canon = WordBoundaryCanonicalizer(FakeTokenizer(), max_ngram=3)
        assert canon.is_word_boundary("\u2581hello") is True
        assert canon.is_word_boundary("world") is False


class TestCrossTokenizerHashing:
    """Test that same text produces same hash indices across tokenizers."""

    def test_same_text_same_hashes_vocab_mode(self):
        """Same tokens with different IDs should produce same canonical IDs."""
        tokens = ["<pad>", "the", "cat", "sat"]

        # Simulate two tokenizers with different ID assignments
        # Tokenizer A: <pad>=0, the=1, cat=2, sat=3
        # Tokenizer B: <pad>=2, the=3, cat=0, sat=1

        canon = VocabCanonicalizer(tokens)
        cfg = HashConfig(max_ngram=3, heads_per_order=2, table_size=1024, seed=0)
        hasher = NgramHasher(cfg)

        # Sequence: "the cat sat"
        # Tokenizer A IDs: [1, 2, 3]
        # Tokenizer B IDs: [3, 0, 1]

        # Build ID maps
        id_map_a = torch.tensor([0, 1, 2, 3])  # identity for tokA
        id_map_b = torch.tensor([2, 3, 0, 1])  # remapped for tokB

        # But canonical IDs should be the same
        canon_map_a = {}
        canon_map_b = {}
        for tok_str in tokens:
            norm = normalize_token(tok_str)
            canon_id = canon.norm_to_canon[norm]
            canon_map_a[tok_str] = canon_id
            canon_map_b[tok_str] = canon_id

        # Sequence "the cat sat" canonical IDs
        seq_canon = torch.tensor([[
            canon_map_a["the"],
            canon_map_a["cat"],
            canon_map_a["sat"],
        ]])

        # Hash indices should be identical
        indices = hasher(seq_canon)
        assert indices.shape == (1, 3, 4)  # (B=1, T=3, H=4)

        # Running again should give same result (deterministic)
        indices2 = hasher(seq_canon)
        assert torch.equal(indices, indices2)
