#!/usr/bin/env bash
# Phase 1: Train all source memories used in the paper.
#
# This script trains ALL source memories needed to reproduce main experiments.
# Each block is guarded by a checkpoint-existence check so you can resume
# after an interrupt.
#
# Total compute budget (approximate, on a single H100):
#   - Pythia-160M (50M tok, WikiText)     ~4 h
#   - Pythia-160M (50M tok, word-boundary) ~4 h  (for Tier 2)
#   - QWen3.5-0.8B (50M tok, WikiText)     ~6 h
#   - QWen3.5-4B (50M tok, FineWeb-Edu)   ~24 h
#   - QWen3.5-9B (200M tok, FineWeb-Edu)  ~96 h  (batch_size=1, 8-bit optim)
#   - QWen3.5-4B (50M tok, Nemotron-CC HQ-DQA)  ~24 h
#   - QWen3.5-9B (50M tok, Nemotron-CC HQ-DQA)  ~96 h
#   - QWen3.5-4B (50M tok, Nemotron-CC HQ 26B)  ~24 h
#
# On a smaller GPU (e.g., RTX 4090 24 GB), cut --max-tokens by 10x for a
# qualitative reproduction; the 9B runs are not feasible without A100/H100.
#
# Comment-out any block you do not need.

set -euo pipefail
cd "$(dirname "$0")/.."

run_if_absent() {
    local out_dir="$1"; shift
    if [ -f "$out_dir/memory.pt" ]; then
        echo "==== SKIP: $out_dir/memory.pt exists ===="
        return
    fi
    echo "==== TRAIN: $out_dir ===="
    "$@"
}

# ----- Pythia-160M (vocab canonicalizer) -----
# Used by: Tier 1, Tier 3, Tier 4, CKA, LoRA/KD baselines
run_if_absent results/source_memory \
    uv run python scripts/train_source_memory.py \
        --model EleutherAI/pythia-160m \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 16 \
        --lr 3e-5 \
        --warmup-steps 1000 \
        --output-dir results/source_memory

# ----- Pythia-160M (word-boundary canonicalizer) -----
# Used by: Tier 2 cross-tokenizer (Pythia -> TinyLlama)
run_if_absent results/source_memory_wb \
    uv run python scripts/train_source_memory.py \
        --config configs/tier2_cross_tokenizer.yaml \
        --canon-mode word_boundary \
        --output-dir results/source_memory_wb

# ----- QWen3.5-0.8B (WikiText) -----
# Used by: cross-arch matrix (0.8B -> Pythia/TinyLlama/QWen-*), scaling analysis
run_if_absent results/qwen35_source_memory \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 16 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 1000 \
        --output-dir results/qwen35_source_memory

# ----- QWen3.5-4B (FineWeb-Edu) -----
# Used by: Table downstream (FW matched), Table domain (vs Nemo)
run_if_absent results/slimpajama_source_4b \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-4B-Base \
        --corpus fineweb-edu \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 2 \
        --grad-accum-steps 8 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 1000 \
        --gradient-checkpointing \
        --output-dir results/slimpajama_source_4b

# ----- QWen3.5-9B (FineWeb-Edu, 200M tok) -----
# Used by: Table downstream (FW-9B source)
# Requires H100 80GB or A100 80GB; uses 8-bit AdamW
run_if_absent results/slimpajama_source_9b \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-9B-Base \
        --corpus fineweb-edu \
        --max-tokens 200000000 \
        --seq-len 512 \
        --batch-size 1 \
        --grad-accum-steps 16 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 500 \
        --gradient-checkpointing \
        --optimizer-8bit \
        --output-dir results/slimpajama_source_9b

# ----- QWen3.5-4B (Nemotron-CC HQ-DQA, 8B STEM Q&A) -----
# Used by: Table domain (Nemo-4B source)
run_if_absent results/nemotron_source_4b \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-4B-Base \
        --corpus nemotron-cc \
        --corpus-subset hq-dqa \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 4 \
        --grad-accum-steps 4 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 1000 \
        --gradient-checkpointing \
        --output-dir results/nemotron_source_4b

# ----- QWen3.5-9B (Nemotron-CC HQ-DQA) -----
# Used by: Table domain (Nemo-9B source)
run_if_absent results/nemotron_source_9b \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-9B-Base \
        --corpus nemotron-cc \
        --corpus-subset hq-dqa \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 1 \
        --grad-accum-steps 16 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 500 \
        --gradient-checkpointing \
        --optimizer-8bit \
        --output-dir results/nemotron_source_9b

# ----- QWen3.5-4B (Nemotron-CC HQ, 26B organic web) -----
# Used by: Table corpus_control (HQ 26B vs HQ-DQA 8B)
run_if_absent results/nemotron_hq26b_source_4b \
    uv run python scripts/train_source_memory.py \
        --model Qwen/Qwen3.5-4B-Base \
        --corpus nemotron-cc \
        --corpus-subset hq \
        --max-tokens 50000000 \
        --seq-len 512 \
        --batch-size 2 \
        --grad-accum-steps 8 \
        --lr 3e-5 \
        --memory-lr 1e-3 \
        --warmup-steps 1000 \
        --gradient-checkpointing \
        --output-dir results/nemotron_hq26b_source_4b

echo ""
echo "==== All source memories DONE ===="
ls -la results/*source*/memory.pt 2>/dev/null || true
