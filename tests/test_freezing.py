"""Test that frozen parameters have zero gradient norm."""

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.memory import EngramMemory, MemoryConfig
from engram.adaptor import EngramAdaptor, build_adaptor


class TestFreezing:
    """Test freeze/unfreeze behavior."""

    def test_frozen_memory_no_grad(self):
        """Frozen memory parameters should produce no-grad output."""
        cfg = MemoryConfig(table_size=64, d_head=8, max_ngram=2, heads_per_order=2)
        memory = EngramMemory(cfg)

        # Freeze
        for p in memory.parameters():
            p.requires_grad = False

        # Forward should produce output with no grad_fn
        canon_ids = torch.randint(0, 64, (2, 10))
        out = memory(canon_ids)
        assert not out.requires_grad, "Output should not require grad when all params frozen"

    def test_unfrozen_memory_has_grad(self):
        """Unfrozen memory parameters should accumulate gradients."""
        cfg = MemoryConfig(table_size=64, d_head=8, max_ngram=2, heads_per_order=2)
        memory = EngramMemory(cfg)

        canon_ids = torch.randint(0, 64, (2, 10))
        out = memory(canon_ids)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in memory.parameters()
        )
        assert has_grad, "No gradients found for unfrozen memory"

    def test_adaptor_grad_isolation(self):
        """Only adaptor should have gradients when memory is frozen."""
        cfg = MemoryConfig(table_size=64, d_head=8, max_ngram=2, heads_per_order=2)
        memory = EngramMemory(cfg)

        # Freeze memory
        for p in memory.parameters():
            p.requires_grad = False

        d_model = 32
        adaptor = EngramAdaptor(d_model=d_model, d_mem=cfg.d_mem)

        # Forward (detach memory output since it has no grad_fn)
        canon_ids = torch.randint(0, 64, (2, 10))
        mem_out = memory(canon_ids).detach()
        h = torch.randn(2, 10, d_model, requires_grad=True)
        contribution, gate = adaptor(h, mem_out)
        loss = contribution.sum()
        loss.backward()

        # Memory should have no grads
        for name, p in memory.named_parameters():
            assert p.grad is None, f"Frozen memory param {name} has grad"

        # Adaptor should have grads
        adaptor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in adaptor.parameters()
        )
        assert adaptor_has_grad, "Adaptor has no gradients"

    def test_all_conditions_build(self):
        """All condition adaptors should build without error."""
        d_model = 64
        d_mem = 32

        conditions = [
            "baseline", "transferred", "random_memory", "permuted_keys",
            "no_gate", "train_from_scratch", "ffn_only", "affine_stitch",
        ]

        for cond in conditions:
            adaptor = build_adaptor(cond, d_model, d_mem)
            if cond == "baseline":
                assert adaptor is None
            else:
                assert adaptor is not None, f"Adaptor is None for {cond}"
                # Forward pass should work
                h = torch.randn(1, 5, d_model)
                mem = torch.randn(1, 5, d_mem)
                out, gate = adaptor(h, mem)
                assert out.shape == (1, 5, d_model), f"Wrong output shape for {cond}"

    def test_permute_keys_changes_weights(self):
        """Permuting keys should change table weights."""
        cfg = MemoryConfig(table_size=64, d_head=8, max_ngram=2, heads_per_order=2)
        memory = EngramMemory(cfg)

        original_weight = memory.tables[0].weight.data.clone()
        memory.permute_keys(seed=42)
        permuted_weight = memory.tables[0].weight.data

        # Should be different ordering
        assert not torch.equal(original_weight, permuted_weight)
        # But same values (just reordered)
        orig_sorted = original_weight.sort(dim=0).values
        perm_sorted = permuted_weight.sort(dim=0).values
        assert torch.allclose(orig_sorted, perm_sorted)
