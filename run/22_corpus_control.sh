#!/usr/bin/env bash
# §5.2 Corpus control: HQ 26B organic web vs HQ-DQA 8B STEM Q&A.
# Tests whether STEM structure (HQ-DQA) or corpus size (HQ 26B) drives the gain.
# QWen-2B target, 3 seeds.
# Prerequisite: results/nemotron_hq26b_source_4b/memory.pt (run/02_source_memories.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

KNOW_TASKS="boolq rte openbookqa sciq truthfulqa race"

SRC_DIR="results/nemotron_hq26b_source_4b"
SOURCE_MEM="${SRC_DIR}/memory.pt"
MEM_CONFIG="${SRC_DIR}/memory_config.json"
TARGET="Qwen/Qwen3.5-2B-Base"
OUT_PREFIX="results/hq26b_matched_to_qwen2b"

if [ ! -f "$SOURCE_MEM" ]; then
    echo "ERROR: $SOURCE_MEM not found (train via run/02_source_memories.sh)"
    exit 1
fi

for SEED in 42 137 2024; do
    echo ""
    echo "==== HQ-26B -> QWen-2B seed=${SEED} (HQ matched adaptor) ===="

    ADAPT_DIR="${OUT_PREFIX}/transferred_seed${SEED}"
    if [ -f "${ADAPT_DIR}/adaptor.pt" ] || [ -f "${ADAPT_DIR}/adaptor_best.pt" ]; then
        echo "  [skip] adaptor exists"
    else
        uv run python scripts/train_adaptor.py \
            --condition transferred --target-model "$TARGET" \
            --source-memory "$SOURCE_MEM" --memory-config "$MEM_CONFIG" \
            --canon-mode vocab --max-tokens 20000000 \
            --batch-size 4 --grad-accum-steps 4 --gradient-checkpointing \
            --early-stopping-patience 5 --seed $SEED \
            --corpus nemotron-cc --corpus-subset hq \
            --output-dir "$ADAPT_DIR"
    fi

    RAND_DIR="${OUT_PREFIX}/random_memory_seed${SEED}"
    if [ -f "${RAND_DIR}/adaptor.pt" ] || [ -f "${RAND_DIR}/adaptor_best.pt" ]; then
        echo "  [skip] random adaptor exists"
    else
        uv run python scripts/train_adaptor.py \
            --condition random_memory --target-model "$TARGET" \
            --source-memory "$SOURCE_MEM" --memory-config "$MEM_CONFIG" \
            --canon-mode vocab --max-tokens 20000000 \
            --batch-size 4 --grad-accum-steps 4 --gradient-checkpointing \
            --early-stopping-patience 5 --seed $SEED \
            --corpus nemotron-cc --corpus-subset hq \
            --output-dir "$RAND_DIR"
    fi

    EVAL_OUT="results/downstream/qwen2b_hq26b_matched_seed${SEED}"
    if [ -d "$EVAL_OUT" ] && [ "$(ls "$EVAL_OUT"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [skip] downstream eval exists"
    elif [ -f "${ADAPT_DIR}/adaptor.pt" ] || [ -f "${ADAPT_DIR}/adaptor_best.pt" ]; then
        uv run python scripts/eval_downstream.py \
            --target-model "$TARGET" \
            --adaptor-dir "$ADAPT_DIR" \
            --source-memory "$SOURCE_MEM" --memory-config "$MEM_CONFIG" \
            --canon-mode vocab --tasks $KNOW_TASKS --seed $SEED \
            --output-dir "$EVAL_OUT"
    else
        echo "  [skip] downstream eval: no adaptor checkpoint"
    fi
done

echo ""
echo "==== Corpus control DONE ===="
