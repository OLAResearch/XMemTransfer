# Reproducing the Hash Hit Rate (hit@k addressability diagnostic)

This guide reproduces the **retrieval-addressability** diagnostic reported as
the "hash hit rate". It answers one question: *when the Engram memory is keyed
by an n-gram hash, do the slots a question (Q) hashes to overlap the slots its
answer (A) hashes to?* The headline result is **hit@10 = 3.0% (6/200)** under
the vocabulary canonicalizer, about 20x the random baseline.

This is a pure hashing diagnostic. It does **not** load any memory weights or
run a model forward pass. It only tokenizes text, canonicalizes it, hashes it
to per-head slot addresses, and measures Q/A overlap. CPU is sufficient.

## What is measured

For each `(Q, A)` probe (200 probes harvested from Nemotron-CC HQ-DQA):

- Hash Q's tokens and A's tokens separately into per-head slot **addresses**.
- `gold(A)` = the set of **unique** addresses produced by A.
- `topk(Q)` = the `k` **most frequent** addresses produced by Q.
- `hit = 1` if `topk(Q) ∩ gold(A)` is non-empty, else `0`.
- `hit@k` = mean of `hit` over all probes.

A per-head address is `addr = head_idx * table_size + slot_idx`, so a slot in
head 0 and the same slot index in head 7 are **different** addresses.

## Files

| Path | Role |
|---|---|
| `scripts/phase0_2_hit_at_k.py` | The diagnostic (entry point). |
| `engram/canonicalization.py` | Token-id to canonical-id mapping. |
| `engram/hashing.py` | `NgramHasher`: canonical n-grams to per-head slots. |
| `engram/alias_canonicalization.py` | Optional `--canonicalizer alias` path. |
| `engram/lsh_key.py` | Optional `--canonicalizer lsh` semantic-key path. |
| `data/phase0/memory_config.json` | Hash configuration (read for hash params only). |
| `data/phase0/probe_set_seed42.jsonl` | The exact 200 `(Q, A)` probes (see note below). |

## Run it

```bash
uv sync   # or: bash run/00_install.sh

uv run python scripts/phase0_2_hit_at_k.py \
    --memory-config data/phase0/memory_config.json \
    --target-model Qwen/Qwen3.5-2B-Base \
    --probe-set data/phase0/probe_set_seed42.jsonl \
    --canonicalizer vocab \
    --top-k 10 \
    --output data/phase0/hit_at_k_seed42.json
```

Expected summary: `hit_at_k_pct ≈ 3.0`, `hits = 6`, `n_evaluated = 200`,
`address_space_size = 524288` (8 heads x 65536). The hash parameters that must
match are `max_ngram=3, heads_per_order=4, table_size=65536, hash_seed=42` and
`canon_mode=vocab`; they are read from `memory_config.json`.

The semantic-key variant (Stage-ii probe, hit@10 rises to about 9%) uses
`--canonicalizer lsh` and additionally requires `sentence-transformers`.

## Reproduction notes (read before re-implementing)

If you re-implement the metric from scratch instead of running the script
above, these five details decide whether you recover `3.0%`. They are the
common causes of a mismatch.

1. **Canonical-id assignment order.** Canonical ids are assigned by iterating
   the tokenizer vocabulary in **token-id order** and assigning a new id on the
   **first occurrence** of each normalized surface string
   (`engram/canonicalization.py`, `build_canonicalizer` + `VocabCanonicalizer`).
   They are **not** assigned by sorting the normalized strings
   lexicographically. A different assignment changes the integer fed to the
   hash, which changes every downstream slot. (The paper prose describing a
   lexicographic sort is imprecise here; the code is authoritative.)

2. **Per-head address offset.** Encode each address as
   `head_idx * table_size + slot_idx` before counting overlaps. Flattening raw
   slot indices across heads inflates hit@k with false cross-head collisions.

3. **top-k is by frequency, not by uniqueness.** `topk(Q)` is the `k` most
   frequent addresses (`Counter.most_common(k)`), then reduced to a set. It is
   not "the first `k` unique addresses". Because most addresses occur once,
   the flatten order of the `(T, H)` address grid decides which addresses
   survive ties; reshape addresses in the same row-major `(token, head)` order.

4. **Short-text padding.** Sequences shorter than `max_ngram` are zero-padded
   so the hasher runs, then the padded-suffix positions are sliced off before
   counting. Counting the synthetic zero n-grams creates false overlaps.

5. **Tokenizer and canonicalizer must match.** Use the same target-model
   tokenizer and `canon_mode=vocab`. A different tokenizer or canonicalizer
   measures overlap in a different address space.

## Probe set provenance

`hit@10 = 3.0%` is `6/200` on one specific probe file. `data/phase0/probe_set_seed42.jsonl`
is that exact file; use it for exact reproduction. The probes were sampled
(seed 42) from a streamed, shuffled subset of the public source corpus, so
re-sampling from scratch depends on the `datasets` library version and the
corpus revision and need not return the same 200 items. A fresh sample keeps
hit@10 near 3% but need not match the same 6 probes.
