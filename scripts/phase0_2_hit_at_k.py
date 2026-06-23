"""Phase 0.2 — hit@k retrieval addressability diagnostic.

Given an Engram source-memory checkpoint trained on HQ-DQA and a probe set
of (Q, A) pairs harvested from HQ-DQA, the diagnostic measures whether the
slot indices touched by Q overlap with those touched by A.

Interpretation thresholds for the hit@10 result:
    hit@10 <20%  → addressability bottleneck
    hit@10 25-60% → dual-axis (retrieval + downstream USE)
    hit@10 >60%  → retrieval is fine; bottleneck is downstream gate/integration

Combined with Phase 0.1 q-only result (+3.49pp preserved out of +15.47pp), this
narrows the bottleneck further.

Hash mechanism (matches source training exactly):
    1. Tokenize text with the target-model tokenizer (canon_mode='vocab' required —
       diagnostic asserts this matches memory_config.json).
    2. Apply VocabCanonicalizer: token IDs → canonical IDs
    3. NgramHasher(canon_ids) → (1, T, total_heads) slot indices in [0, table_size)
    4. Per-head identity is preserved by encoding addresses as
       `addr = head_idx * table_size + slot_idx`. EngramMemory uses one embedding
       table per head, so (head=0, slot=123) and (head=7, slot=123) are different
       memory locations — flattening without offset would inflate hit@k via false
       cross-head collisions (codex P1 caught this).

Definitions (over per-head addresses, not bare slot indices):
    top_k(Q)     = the k most-frequent addresses when hashing Q's tokens
    gold(A)      = the unique set of addresses when hashing A's tokens
    hit_at_k(p)  = 1 if any element of gold(A_p) appears in top_k(Q_p), else 0

Note on "populated-only" filtering: EngramMemory initializes every embedding
row from a nonzero Normal distribution, so weight-norm cannot distinguish
training-touched slots from never-touched ones. This script therefore does
NOT support a populated-only mode. A real occupancy signal would require
explicit usage tracking added to source training (codex P1).

Usage:
    uv run python scripts/phase0_2_hit_at_k.py \\
        --memory-config data/phase0/memory_config.json \\
        --target-model Qwen/Qwen3.5-2B-Base \\
        --probe-set data/phase0/probe_set_seed42.jsonl \\
        --canonicalizer vocab \\
        --output data/phase0/hit_at_k_seed42.json \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

# Repo root is the parent of scripts/. Add it for engram.* imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.canonicalization import VocabCanonicalizer, build_canonicalizer  # noqa: E402
from engram.hashing import HashConfig, NgramHasher  # noqa: E402
from engram.alias_canonicalization import AliasCanonicalizer  # noqa: E402
from engram.lsh_key import SentencePoolLSH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0.2 hit@k diagnostic")
    parser.add_argument("--memory-config", type=str, required=True,
                        help="Path to memory_config.json produced by source training")
    parser.add_argument("--target-model", type=str, required=True,
                        help="HF model id whose tokenizer was used at source-training "
                             "time (must match canon_mode='vocab' alignment)")
    parser.add_argument("--probe-set", type=str, required=True,
                        help="Path to probe_set JSONL with {question, answer} entries")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file with summary + per-probe records")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k threshold for hit@k (default 10)")
    parser.add_argument("--canonicalizer", type=str, default="vocab",
                        choices=["vocab", "alias", "lsh"],
                        help="Canonicalization to apply before hashing. "
                             "'vocab' (default) = VocabCanonicalizer matching "
                             "source-training. 'alias' = AliasCanonicalizer "
                             "(stopword + lemma + compound merge); used to "
                             "test Tier A alias-aware hypothesis in Stage (i). "
                             "'lsh' = SentencePoolLSH (Tier C C-1 semantic key); "
                             "uses SBERT embeddings + random-hyperplane LSH "
                             "instead of n-gram token hashing. Source-memory is "
                             "NOT retrained — measures whether Q and A reach "
                             "overlapping LSH slots under a semantic key.")
    parser.add_argument("--lsh-model", type=str, default=None,
                        help="SBERT model for --canonicalizer lsh (default: "
                             "sentence-transformers/all-MiniLM-L6-v2, 22MB)")
    parser.add_argument("--lsh-window-size", type=int, default=3,
                        help="Sliding window size in words for --canonicalizer lsh")
    parser.add_argument("--max-probes", type=int, default=None,
                        help="Truncate probe set for fast iteration")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device; CPU is sufficient for hashing")
    return parser.parse_args()


def load_hash_config(
    memory_config_path: Path, canonicalizer_mode: str
) -> tuple[HashConfig, str]:
    """Read memory_config.json and return (HashConfig, source_canon_mode).

    source_canon_mode is the canonicalization scheme used at source training
    time. The diagnostic supports two scenarios:

    - `canonicalizer_mode='vocab'`: must match the source's canon_mode='vocab'.
      We assert this to avoid measuring hash overlap in the wrong address space.

    - `canonicalizer_mode='alias'`: intentionally uses a DIFFERENT
      canonicalizer than the source memory was trained with. This is Stage (i)
      of the alias-aware validation protocol: we measure whether Q and A
      reach overlapping hash slots under an alias-aware key function, BEFORE
      committing to a full source-memory retrain. The canon_mode assertion
      is intentionally bypassed here — source-memory faithfulness is not the
      goal of Stage (i).
    """
    with memory_config_path.open() as f:
        cfg = json.load(f)
    source_canon_mode = str(cfg.get("canon_mode", "vocab"))
    if canonicalizer_mode == "vocab" and source_canon_mode != "vocab":
        raise SystemExit(
            f"memory_config.canon_mode={source_canon_mode!r} is not supported "
            f"by the 'vocab' canonicalizer path; only 'vocab' source training "
            f"is implemented. Running hit@k under a different scheme would "
            f"measure hash overlap in the wrong address space (codex P1)."
        )
    return (
        HashConfig(
            max_ngram=int(cfg["max_ngram"]),
            heads_per_order=int(cfg["heads_per_order"]),
            table_size=int(cfg["table_size"]),
            seed=int(cfg["hash_seed"]),
        ),
        source_canon_mode,
    )


def _canon_ids_to_addresses(
    canon_ids: torch.LongTensor, hasher: NgramHasher, device: torch.device
) -> torch.LongTensor:
    """Shared tail: pad-to-max_ngram (so hasher runs) → hash → drop padded
    positions → encode per-head address → flatten.

    The pad-to-max_ngram is required because `NgramHasher._shift_left_pad`
    produces a mismatched shape when T < max_ngram (engram/hashing.py bug,
    deferred). We pad enough to satisfy the hasher, run it, then SLICE OFF
    the padded-suffix contributions before counting addresses (codex P1:
    counting padded positions as real tokens inflates the multiset for
    short Q/A and can create false overlaps from the synthetic zero pad).
    """
    if canon_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    original_T = canon_ids.shape[1]
    if canon_ids.shape[1] < hasher.cfg.max_ngram:
        pad_amount = hasher.cfg.max_ngram - canon_ids.shape[1]
        canon_ids = torch.cat([
            canon_ids,
            torch.zeros(1, pad_amount, dtype=canon_ids.dtype, device=device),
        ], dim=1)
    hashed = hasher(canon_ids)  # (1, T_padded, H)
    # Drop padded-position contributions: the hashes at positions ≥ original_T
    # come from the zero-pad suffix and would inflate hit@k via synthetic
    # zero-zero-zero n-grams that aren't part of the real text.
    hashed = hashed[:, :original_T, :]
    _, _, H = hashed.shape
    head_offsets = (
        torch.arange(H, dtype=torch.long, device=device) * hasher.cfg.table_size
    )
    addresses = hashed.long() + head_offsets
    return addresses.reshape(-1)


def slot_addresses_for_text_lsh(
    text: str,
    lsh_canon: SentencePoolLSH,
    table_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """Tier C C-1 path: sentence-pool LSH (no NgramHasher involved).

    Bypasses the n-gram token hashing pipeline entirely. The
    SentencePoolLSH instance is responsible for tokenization (whitespace
    splitting), windowed encoding via SBERT, and producing (T_windows,
    n_heads) slot indices directly. We then add per-head offsets so the
    address space matches the vocab/alias paths (head_idx * table_size +
    slot_idx ∈ [0, n_heads * table_size)).
    """
    slot_indices = lsh_canon.slot_indices_for_text(text)  # (T_w, H) on CPU
    if slot_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    slot_indices = slot_indices.to(device)
    T_w, H = slot_indices.shape
    head_offsets = (
        torch.arange(H, dtype=torch.long, device=device) * table_size
    )
    addresses = slot_indices.long() + head_offsets
    return addresses.reshape(-1)


def slot_addresses_for_text_alias(
    text: str,
    tokenizer,
    alias_canon: AliasCanonicalizer,
    hasher: NgramHasher,
    device: torch.device,
) -> torch.LongTensor:
    """Alias-aware path: tokenize → alias-canonicalize → hash → addresses.

    Note: alias canonicalization may produce an EMPTY stream when the input
    is entirely stopwords (e.g., A = "Yes"/"No" which NLTK marks as
    stopwords). Returns empty tensor in that case; caller skips the probe.
    """
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    ids = ids.to(device)
    canon_ids = alias_canon.canonicalize(ids, device)
    return _canon_ids_to_addresses(canon_ids, hasher, device)


def slot_addresses_for_text(
    text: str,
    tokenizer,
    canon: VocabCanonicalizer,
    id_map: torch.LongTensor,
    hasher: NgramHasher,
    device: torch.device,
) -> torch.LongTensor:
    """Vocab-canonicalizer path: tokenize → vocab-canonicalize → hash → addresses.

    Each address = head_idx * table_size + slot_idx, so collisions are only
    counted within the same head (which is how EngramMemory's per-head
    embedding tables work).
    """
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    ids = ids.to(device)
    canon_ids = id_map.to(device)[ids]  # (1, T)
    # _canon_ids_to_addresses handles pad-to-max_ngram + hash + offset.
    return _canon_ids_to_addresses(canon_ids, hasher, device)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    hash_cfg, source_canon_mode = load_hash_config(
        Path(args.memory_config), args.canonicalizer
    )
    print(f"HashConfig: {hash_cfg}, total_heads={hash_cfg.total_heads}, "
          f"source_canon_mode={source_canon_mode}, "
          f"canonicalizer={args.canonicalizer}")

    # Tokenizer + canonicalizer setup
    from transformers import AutoTokenizer
    print(f"Loading tokenizer {args.target_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True
    )

    hasher = NgramHasher(hash_cfg)

    # Dispatch on canonicalizer mode. `vocab` matches source-training; `alias`
    # is the Tier A experimental key function (stopword + lemma + compound).
    if args.canonicalizer == "vocab":
        canon = build_canonicalizer(
            tokenizer, mode=source_canon_mode, max_ngram=hash_cfg.max_ngram
        )
        id_map = canon.build_id_map(tokenizer)
        print(f"Canonical vocab size: {canon.canon_vocab_size}")

        def addresses_for_text(text: str) -> torch.LongTensor:
            return slot_addresses_for_text(
                text, tokenizer, canon, id_map, hasher, device
            )
    elif args.canonicalizer == "alias":
        # Alias canonicalizer — stateful, shared between Q and A across all probes.
        alias_canon = AliasCanonicalizer(tokenizer)
        print(
            f"AliasCanonicalizer: {len(alias_canon.stopwords)} stopwords, "
            f"{len(alias_canon.compounds)} compounds"
        )

        def addresses_for_text(text: str) -> torch.LongTensor:
            return slot_addresses_for_text_alias(
                text, tokenizer, alias_canon, hasher, device
            )
    else:
        # Tier C C-1: SBERT sentence-pool + random-hyperplane LSH
        lsh_canon = SentencePoolLSH(
            model_name=args.lsh_model,
            n_heads=hash_cfg.total_heads,
            table_size=hash_cfg.table_size,
            window_size=args.lsh_window_size,
            seed=hash_cfg.seed,
            device=("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"),
        )
        # Eager-load the SBERT model so the first probe doesn't pay an outlier latency
        _ = lsh_canon.sbert
        print(
            f"SentencePoolLSH: model={lsh_canon.model_name}, "
            f"n_heads={hash_cfg.total_heads}, table_size={hash_cfg.table_size}, "
            f"window_size={args.lsh_window_size}"
        )

        def addresses_for_text(text: str) -> torch.LongTensor:
            return slot_addresses_for_text_lsh(
                text, lsh_canon, hash_cfg.table_size, device
            )

    probes: list[dict] = []
    with Path(args.probe_set).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            probes.append(json.loads(line))
    if args.max_probes is not None:
        probes = probes[: args.max_probes]
    print(f"Loaded {len(probes)} probes")

    records: list[dict] = []
    hits = 0
    empty_q = 0
    empty_a = 0
    gold_sizes: list[int] = []
    q_slot_counts: list[int] = []

    for idx, probe in enumerate(probes):
        q_slots = addresses_for_text(probe["question"])
        a_slots = addresses_for_text(probe["answer"])
        if q_slots.numel() == 0 or a_slots.numel() == 0:
            # Empty stream (alias canon may produce this on stopword-only A
            # like "Yes"/"No"). Per codex P1: record as hit=0 with full
            # denominator rather than skipping, so denominator stays
            # consistent with the vocab path (where empty is essentially
            # impossible). This keeps vocab vs alias comparisons apples-to-
            # apples for the Stage (i) decision.
            if q_slots.numel() == 0:
                empty_q += 1
            if a_slots.numel() == 0:
                empty_a += 1
            records.append({
                "idx": idx,
                "skipped": False,
                "empty_stream_q": bool(q_slots.numel() == 0),
                "empty_stream_a": bool(a_slots.numel() == 0),
                "q_slot_total": int(q_slots.numel()),
                "q_unique_slots": 0,
                "gold_unique_slots": int(a_slots.numel()),
                "top_k_size": 0,
                "intersection_size": 0,
                "hit": False,
            })
            gold_sizes.append(0)
            q_slot_counts.append(int(q_slots.numel()))
            continue

        q_counter = Counter(q_slots.cpu().tolist())
        a_set = set(a_slots.cpu().tolist())

        top_k_slots = {slot for slot, _ in q_counter.most_common(args.top_k)}
        intersection = top_k_slots & a_set
        hit = bool(intersection)

        records.append({
            "idx": idx,
            "skipped": False,
            "q_slot_total": int(q_slots.numel()),
            "q_unique_slots": int(len(q_counter)),
            "gold_unique_slots": int(len(a_set)),
            "top_k_size": int(len(top_k_slots)),
            "intersection_size": int(len(intersection)),
            "hit": hit,
        })
        if hit:
            hits += 1
        gold_sizes.append(len(a_set))
        q_slot_counts.append(q_slots.numel())

    # Per codex P1: empty streams (alias canon stopword-only inputs) are
    # recorded as hit=0 in `records`, NOT removed from the denominator. This
    # keeps vocab vs alias comparisons apples-to-apples.
    n_eval = len(probes)
    summary = {
        "n_probes": len(probes),
        "n_evaluated": n_eval,
        "n_empty_q": empty_q,
        "n_empty_a": empty_a,
        "top_k": args.top_k,
        "hits": hits,
        "hit_at_k_pct": (100.0 * hits / n_eval) if n_eval else 0.0,
        "gold_size_mean": (sum(gold_sizes) / len(gold_sizes)) if gold_sizes else 0.0,
        "q_slot_total_mean":
            (sum(q_slot_counts) / len(q_slot_counts)) if q_slot_counts else 0.0,
        "address_space_size": int(
            hash_cfg.table_size * hash_cfg.total_heads
        ),
        "canonicalizer": args.canonicalizer,
        "hash_config": {
            "max_ngram": hash_cfg.max_ngram,
            "heads_per_order": hash_cfg.heads_per_order,
            "table_size": hash_cfg.table_size,
            "seed": hash_cfg.seed,
            "total_heads": hash_cfg.total_heads,
            "source_canon_mode": source_canon_mode,
        },
        "target_model": args.target_model,
        "probe_set": args.probe_set,
        "memory_config": args.memory_config,
    }

    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with output_path.open("w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
