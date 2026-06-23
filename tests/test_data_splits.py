"""Test that WikiText-103 train/val/test splits are disjoint.

Note: This test requires downloading WikiText-103, so it's marked as slow.
Run with: pytest tests/test_data_splits.py -m "not slow" to skip.
"""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.mark.slow
def test_split_disjointness():
    """Verify WikiText-103 splits have minimal overlapping content.

    Note: WikiText-103 splits articles, not individual lines, so some
    common sentences (e.g. "= = References = =") appear in multiple splits.
    We check that overlap is below a reasonable threshold rather than zero.
    """
    from engram.data import verify_split_disjointness

    results = verify_split_disjointness(tokenizer=None, seq_len=512)

    # WikiText-103 has known line-level overlaps across splits (common
    # boilerplate like section headers). Allow up to 2% overlap per pair.
    train_size = results.get("train_size", 1)

    for pair, key in [
        ("train/val", "train_val_overlap"),
        ("train/test", "train_test_overlap"),
        ("val/test", "val_test_overlap"),
    ]:
        overlap = results[key]
        frac = overlap / max(train_size, 1)
        assert frac < 0.02, (
            f"{pair} overlap too high: {overlap} texts ({frac:.1%} of train)"
        )


def test_dataset_shapes():
    """Test that WikiTextDataset produces correct shapes (uses tiny subset)."""
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")

    from transformers import AutoTokenizer
    from engram.data import WikiTextDataset

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = WikiTextDataset(
        split="test",
        tokenizer=tokenizer,
        seq_len=128,
        max_tokens=10000,
    )

    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    assert item["labels"].shape == (128,)
