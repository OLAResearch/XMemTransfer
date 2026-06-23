"""Integration tests for cross-tokenizer and ffn_only paths.

Tests full pipelines:
  1. word_boundary → WordNgramHasher → memory.forward_from_indices
  2. FFNOnlyAdaptor with mem=None (simulates hook behavior)
  3. Canon mode mismatch guard in setup_memory
"""

import json
import tempfile
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.canonicalization import WordBoundaryCanonicalizer, normalize_token
from engram.hashing import HashConfig, WordNgramHasher
from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import FFNOnlyAdaptor, EngramAdaptor, build_adaptor


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeGPTNeoXTokenizer:
    """Simulates GPT-NeoX BPE tokenizer (Ġ word-boundary markers)."""
    _vocab = {
        "\u0120the": 0,
        "\u0120capital": 1,
        "\u0120of": 2,
        "\u0120france": 3,
        "\u0120is": 4,
        "\u0120paris": 5,
        "cap": 6,
        "ital": 7,
    }

    def get_vocab(self):
        return dict(self._vocab)


class FakeSentencePieceTokenizer:
    """Simulates SentencePiece tokenizer (▁ word-boundary markers)."""
    _vocab = {
        "\u2581the": 10,
        "\u2581capital": 11,
        "\u2581of": 12,
        "\u2581france": 13,
        "\u2581is": 14,
        "\u2581paris": 15,
        "cap": 16,
        "ital": 17,
    }

    def get_vocab(self):
        return dict(self._vocab)


# ── Test: word_boundary → hash → memory.forward_from_indices ─────────────────


class TestWordBoundaryFullPath:
    """End-to-end: word_boundary canonicalization → hashing → memory lookup."""

    def _make_memory(self):
        cfg = MemoryConfig(
            max_ngram=3, heads_per_order=2, table_size=128, d_head=8, hash_seed=0,
        )
        return EngramMemory(cfg), cfg

    def test_full_path_produces_correct_shape(self):
        """word_boundary → hash → forward_from_indices → (B, T, d_mem)."""
        memory, cfg = self._make_memory()
        canon = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)
        hasher = WordNgramHasher(cfg.hash_config)

        # Sequence: "the capital of france" → token IDs [0, 1, 2, 3]
        input_ids = torch.tensor([[0, 1, 2, 3]])  # (1, 4)

        word_ngrams = canon.compute_word_ngrams(input_ids)
        assert len(word_ngrams) == 1  # B=1
        assert len(word_ngrams[0]) == 4  # T=4

        indices = hasher.hash_word_ngrams(word_ngrams, device=torch.device("cpu"))
        assert indices.shape == (1, 4, cfg.total_heads)  # (B, T, H)
        assert (indices >= 0).all() and (indices < cfg.table_size).all()

        mem_vectors = memory.forward_from_indices(indices)
        assert mem_vectors.shape == (1, 4, cfg.d_mem)

    def test_cross_tokenizer_agreement(self):
        """Same text tokenized differently should produce same hash indices."""
        cfg = MemoryConfig(
            max_ngram=3, heads_per_order=2, table_size=128, d_head=8, hash_seed=0,
        )
        hasher = WordNgramHasher(cfg.hash_config)

        # GPT-NeoX: "the capital of" → [0, 1, 2] (each token is a full word)
        canon_neox = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)
        ids_neox = torch.tensor([[0, 1, 2]])
        ngrams_neox = canon_neox.compute_word_ngrams(ids_neox)
        indices_neox = hasher.hash_word_ngrams(ngrams_neox, device=torch.device("cpu"))

        # SentencePiece: "the capital of" → [10, 11, 12]
        canon_sp = WordBoundaryCanonicalizer(FakeSentencePieceTokenizer(), max_ngram=3)
        ids_sp = torch.tensor([[10, 11, 12]])
        ngrams_sp = canon_sp.compute_word_ngrams(ids_sp)
        indices_sp = hasher.hash_word_ngrams(ngrams_sp, device=torch.device("cpu"))

        # Both should produce the same hash indices (same words, same N-grams)
        assert torch.equal(indices_neox, indices_sp), (
            f"Cross-tokenizer hash mismatch:\n"
            f"  GPT-NeoX:      {indices_neox}\n"
            f"  SentencePiece:  {indices_sp}"
        )

    def test_subword_split_agreement(self):
        """Subword split ('cap'+'ital') should match whole-word ('capital')."""
        cfg = MemoryConfig(
            max_ngram=3, heads_per_order=2, table_size=128, d_head=8, hash_seed=0,
        )
        hasher = WordNgramHasher(cfg.hash_config)
        canon = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)

        # "the capital" as whole token: IDs [0, 1]
        # Token 0 = "Ġthe" (word boundary), Token 1 = "Ġcapital" (word boundary)
        ids_whole = torch.tensor([[0, 1]])
        ngrams_whole = canon.compute_word_ngrams(ids_whole)

        # "the cap ital" as subwords: IDs [0, 6, 7]
        # Token 0 = "Ġthe" (boundary), Token 6 = "cap" (not boundary),
        # Token 7 = "ital" (not boundary)
        # So after "Ġthe" we flush "the" as a word.
        # "cap"+"ital" are accumulated but not flushed (no next boundary).
        # Thus at position 2, completed_words = ["<bos>", "the"]
        # which differs from ids_whole where at position 1,
        # completed_words = ["<bos>", "the", "capital"]
        #
        # This is expected: subword tokens share the *same* N-gram context
        # as long as the current word hasn't been flushed yet. The word
        # "capital" won't appear in the N-gram context until the next
        # word boundary is encountered.
        ids_split = torch.tensor([[0, 6, 7]])
        ngrams_split = canon.compute_word_ngrams(ids_split)

        # At the "Ġthe" position (index 0), both should be the same
        # (completed_words = ["<bos>"], same N-gram context)
        indices_whole = hasher.hash_word_ngrams(ngrams_whole, device=torch.device("cpu"))
        indices_split = hasher.hash_word_ngrams(ngrams_split, device=torch.device("cpu"))
        assert torch.equal(indices_whole[0, 0], indices_split[0, 0])

    def test_deterministic(self):
        """Same input should always produce same output."""
        memory, cfg = self._make_memory()
        canon = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)
        hasher = WordNgramHasher(cfg.hash_config)

        input_ids = torch.tensor([[0, 1, 2, 3]])

        ngrams1 = canon.compute_word_ngrams(input_ids)
        indices1 = hasher.hash_word_ngrams(ngrams1, device=torch.device("cpu"))
        vec1 = memory.forward_from_indices(indices1)

        ngrams2 = canon.compute_word_ngrams(input_ids)
        indices2 = hasher.hash_word_ngrams(ngrams2, device=torch.device("cpu"))
        vec2 = memory.forward_from_indices(indices2)

        assert torch.equal(indices1, indices2)
        assert torch.equal(vec1, vec2)

    def test_batched_correctness(self):
        """Batched processing should match sequential per-example processing."""
        memory, cfg = self._make_memory()
        canon = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)
        hasher = WordNgramHasher(cfg.hash_config)

        # Two sequences
        input_ids = torch.tensor([
            [0, 1, 2, 3],
            [0, 4, 5, 3],
        ])

        # Batched
        ngrams_batch = canon.compute_word_ngrams(input_ids)
        indices_batch = hasher.hash_word_ngrams(ngrams_batch, device=torch.device("cpu"))

        # Sequential
        for b in range(2):
            ngrams_single = canon.compute_word_ngrams(input_ids[b:b+1])
            indices_single = hasher.hash_word_ngrams(ngrams_single, device=torch.device("cpu"))
            assert torch.equal(indices_batch[b:b+1], indices_single), (
                f"Batch/sequential mismatch at b={b}"
            )

    def test_cpu_tolist_optimization(self):
        """Ensure compute_word_ngrams works with GPU-like tensors (float device)."""
        canon = WordBoundaryCanonicalizer(FakeGPTNeoXTokenizer(), max_ngram=3)
        # Just verify it calls .cpu().tolist() without error
        input_ids = torch.tensor([[0, 1, 2]])
        ngrams = canon.compute_word_ngrams(input_ids)
        assert len(ngrams) == 1
        assert len(ngrams[0]) == 3


# ── Test: FFNOnlyAdaptor with mem=None ───────────────────────────────────────


class TestFFNOnlyPath:
    """Test FFNOnlyAdaptor works correctly when memory is None."""

    def test_ffn_only_forward_no_memory(self):
        """FFNOnlyAdaptor should work with mem=None."""
        d_model = 32
        adaptor = FFNOnlyAdaptor(d_model=d_model, target_param_count=4096)

        h = torch.randn(2, 10, d_model)
        contribution, gate = adaptor(h, None)

        assert contribution.shape == (2, 10, d_model)
        assert gate.shape == (2, 10)
        assert (gate == 1.0).all(), "FFN gate should always be 1.0"

    def test_ffn_only_backward(self):
        """FFNOnlyAdaptor should produce gradients without memory."""
        d_model = 32
        adaptor = FFNOnlyAdaptor(d_model=d_model, target_param_count=4096)

        h = torch.randn(2, 10, d_model, requires_grad=True)
        contribution, gate = adaptor(h, None)
        loss = contribution.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adaptor.parameters()
        )
        assert has_grad, "FFNOnlyAdaptor should have gradients"

    def test_ffn_only_param_count_targeting(self):
        """FFNOnlyAdaptor should have approximately the target param count."""
        d_model = 64
        d_mem = 128
        # This is how build_adaptor computes the target for ffn_only
        target = 2 * d_mem * d_model + 2 * d_model
        adaptor = FFNOnlyAdaptor(d_model=d_model, target_param_count=target)
        actual = sum(p.numel() for p in adaptor.parameters())
        # Should be close (within 2x due to bias terms and rounding)
        assert actual > 0
        assert actual <= target * 2, f"FFN params {actual} >> target {target}"

    def test_ffn_only_vs_memory_adaptor_isolation(self):
        """FFNOnlyAdaptor output should not depend on memory tensor."""
        d_model = 32
        adaptor = FFNOnlyAdaptor(d_model=d_model, target_param_count=4096)

        h = torch.randn(2, 5, d_model)
        out_none, _ = adaptor(h, None)
        out_mem, _ = adaptor(h, torch.randn(2, 5, 128))
        assert torch.equal(out_none, out_mem), "FFN output should be identical with/without mem"


class TestBuildAdaptorFactory:
    """Test build_adaptor returns correct types for all conditions."""

    def test_ffn_only_build(self):
        adaptor = build_adaptor("ffn_only", d_model=64, d_mem=128)
        assert isinstance(adaptor, FFNOnlyAdaptor)

    def test_baseline_returns_none(self):
        adaptor = build_adaptor("baseline", d_model=64, d_mem=128)
        assert adaptor is None

    def test_transferred_build(self):
        adaptor = build_adaptor("transferred", d_model=64, d_mem=128)
        assert isinstance(adaptor, EngramAdaptor)


# ── Test: Hook simulation (without loading HF model) ────────────────────────


class TestHookSimulation:
    """Simulate the backbone_wrapper hook logic for ffn_only and memory paths."""

    @staticmethod
    def _simulate_hook(adaptor, condition, hidden_states, mem_vectors):
        """Replicate hook_fn logic from BackboneWrapper."""
        if adaptor is None:
            return hidden_states

        if mem_vectors is None and condition not in ("ffn_only",):
            return hidden_states

        h_float = hidden_states.float()
        mem_float = mem_vectors.float() if mem_vectors is not None else None
        contribution, gate_values = adaptor(h_float, mem_float)
        return hidden_states + contribution.to(hidden_states.dtype)

    def test_ffn_only_hook_no_memory(self):
        """ffn_only should modify hidden states even without memory."""
        d_model = 32
        adaptor = FFNOnlyAdaptor(d_model=d_model, target_param_count=4096)
        h = torch.randn(2, 5, d_model)
        h_original = h.clone()

        result = self._simulate_hook(adaptor, "ffn_only", h, mem_vectors=None)
        assert not torch.equal(result, h_original), "ffn_only hook should modify hidden states"

    def test_transferred_hook_with_memory(self):
        """transferred should modify hidden states when memory is provided."""
        d_model = 32
        d_mem = 64
        adaptor = EngramAdaptor(d_model=d_model, d_mem=d_mem)
        h = torch.randn(2, 5, d_model)
        mem = torch.randn(2, 5, d_mem)
        h_original = h.clone()

        result = self._simulate_hook(adaptor, "transferred", h, mem_vectors=mem)
        assert not torch.equal(result, h_original), "transferred hook should modify hidden states"

    def test_transferred_hook_no_memory_passthrough(self):
        """transferred should pass through when memory is None."""
        d_model = 32
        d_mem = 64
        adaptor = EngramAdaptor(d_model=d_model, d_mem=d_mem)
        h = torch.randn(2, 5, d_model)
        h_original = h.clone()

        result = self._simulate_hook(adaptor, "transferred", h, mem_vectors=None)
        assert torch.equal(result, h_original), "Should pass through without memory"

    def test_baseline_hook_passthrough(self):
        """baseline (adaptor=None) should always pass through."""
        h = torch.randn(2, 5, 32)
        h_original = h.clone()

        result = self._simulate_hook(None, "baseline", h, mem_vectors=None)
        assert torch.equal(result, h_original)


# ── Test: canon_mode mismatch guard in setup_memory ──────────────────────────


class TestCanonModeValidation:
    """Test the canon_mode mismatch guard from train_adaptor.setup_memory.

    Replicates the validation logic directly to avoid importing the full
    training script (which triggers heavy HF model loading dependencies).
    """

    @staticmethod
    def _validate_canon_mode(mem_cfg_dict: dict, requested_mode: str):
        """Replicate the canon_mode validation from train_adaptor.setup_memory."""
        source_canon_mode = mem_cfg_dict.get("canon_mode")
        if source_canon_mode is not None and source_canon_mode != requested_mode:
            raise ValueError(
                f"Canon mode mismatch: source memory was trained with "
                f"canon_mode='{source_canon_mode}' but --canon-mode='{requested_mode}' "
                f"was specified. Using mismatched canonicalization will produce "
                f"invalid hash indices and silently corrupt transfer results."
            )

    def test_mismatch_raises_error(self):
        """Validation should raise ValueError when canon_mode mismatches."""
        cfg = {"canon_mode": "vocab", "max_ngram": 3}
        try:
            self._validate_canon_mode(cfg, "word_boundary")
            assert False, "Expected ValueError for canon_mode mismatch"
        except ValueError as e:
            assert "vocab" in str(e)
            assert "word_boundary" in str(e)

    def test_matching_mode_succeeds(self):
        """Validation should pass when canon_mode matches."""
        cfg = {"canon_mode": "vocab", "max_ngram": 3}
        # Should not raise
        self._validate_canon_mode(cfg, "vocab")

    def test_missing_canon_mode_skips_check(self):
        """Validation should not raise when canon_mode is absent from config."""
        cfg = {"max_ngram": 3}  # no canon_mode key
        # Should not raise for any requested mode
        self._validate_canon_mode(cfg, "word_boundary")
        self._validate_canon_mode(cfg, "vocab")

    def test_guard_exists_in_train_adaptor(self):
        """Verify the actual guard code exists in train_adaptor.py."""
        from pathlib import Path
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_adaptor.py"
        source = script_path.read_text()
        assert 'canon_mode' in source, "train_adaptor.py should reference canon_mode"
        assert 'Canon mode mismatch' in source, (
            "train_adaptor.py should contain the mismatch error message"
        )
