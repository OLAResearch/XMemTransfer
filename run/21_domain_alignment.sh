#!/usr/bin/env bash
# §5.2 Domain-aligned source memory: Nemotron-CC HQ-DQA (8B STEM Q&A).
# Reproduces Table: domain (2 sources × 4 targets × 3 seeds = 24 runs + 24 evals).
# Prerequisites: results/nemotron_source_{4b,9b}/memory.pt (run/02_source_memories.sh).

set -euo pipefail
cd "$(dirname "$0")/.."

KNOW_TASKS="boolq rte openbookqa sciq truthfulqa race"

declare -A SRC_DIRS
SRC_DIRS[nemo4b]="results/nemotron_source_4b"
SRC_DIRS[nemo9b]="results/nemotron_source_9b"

TARGETS=(
    "Qwen/Qwen3.5-0.8B-Base"
    "Qwen/Qwen3.5-2B-Base"
    "Qwen/Qwen3.5-4B-Base"
    "Qwen/Qwen3.5-9B-Base"
)
TARGET_LABELS=("qwen08b" "qwen2b" "qwen4b" "qwen9b")

for SRC_KEY in nemo4b nemo9b; do
    SRC_DIR="${SRC_DIRS[$SRC_KEY]}"
    SOURCE_MEM="${SRC_DIR}/memory.pt"
    MEM_CONFIG="${SRC_DIR}/memory_config.json"

    if [ ! -f "$SOURCE_MEM" ]; then
        echo "SKIP $SRC_KEY: $SOURCE_MEM not found (train via run/02_source_memories.sh)"
        continue
    fi

    for i in "${!TARGETS[@]}"; do
        TARGET="${TARGETS[$i]}"
        TLABEL="${TARGET_LABELS[$i]}"
        OUT_PREFIX="results/${SRC_KEY}_matched_to_${TLABEL}"

        for SEED in 42 137 2024; do
            echo ""
            echo "==== ${SRC_KEY} -> ${TLABEL} seed=${SEED} (HQ-DQA matched adaptor) ===="

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
                    --corpus nemotron-cc --corpus-subset hq-dqa \
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
                    --corpus nemotron-cc --corpus-subset hq-dqa \
                    --output-dir "$RAND_DIR"
            fi

            EVAL_OUT="results/downstream/${TLABEL}_${SRC_KEY}_matched_seed${SEED}"
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
    done
done

echo ""
echo "==== Domain alignment (Nemo matched) DONE ===="
