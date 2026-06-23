#!/usr/bin/env bash
# Iso-parameter baselines: LoRA, Cross-LoRA (projection), KNN-LM.
# All target-size ranks are chosen to match the Engram adaptor parameter count
# (≈1.03 M for Pythia-410M, ≈2.10 M for TinyLlama-1.1B).

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results/baselines

echo "==== Installing optional deps (peft, faiss, safetensors) ===="
uv add peft faiss-cpu safetensors 2>/dev/null || echo "Already installed"

# ---------- 1. LoRA (iso-parameter) ----------
echo ""
echo "================================================"
echo "  LoRA (iso-parameter)"
echo "================================================"

for SEED in 42 137 2024; do
    OUT="results/baselines/lora_pythia410m_seed${SEED}"
    if [ -f "$OUT/results.json" ]; then continue; fi
    uv run python scripts/train_lora_baseline.py \
        --target-model EleutherAI/pythia-410m \
        --lora-rank 7 \
        --max-tokens 20000000 \
        --seed $SEED \
        --output-dir "$OUT"
done

for SEED in 42 137 2024; do
    OUT="results/baselines/lora_tinyllama_seed${SEED}"
    if [ -f "$OUT/results.json" ]; then continue; fi
    uv run python scripts/train_lora_baseline.py \
        --target-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --lora-rank 12 \
        --max-tokens 20000000 \
        --seed $SEED \
        --output-dir "$OUT"
done

# ---------- 2. Cross-LoRA (projection baseline) ----------
echo ""
echo "================================================"
echo "  Cross-LoRA (projection baseline)"
echo "================================================"

for SEED in 42 137 2024; do
    OUT="results/baselines/crosslora_pythia410m_seed${SEED}"
    if [ -f "$OUT/results.json" ]; then continue; fi
    uv run python scripts/train_crosslora.py \
        --source-model EleutherAI/pythia-160m \
        --target-model EleutherAI/pythia-410m \
        --lora-rank 7 \
        --max-tokens 20000000 \
        --seed $SEED \
        --output-dir "$OUT"
done

# ---------- 3. KNN-LM ----------
echo ""
echo "================================================"
echo "  KNN-LM (nearest-neighbor datastore)"
echo "================================================"

for TGT_LABEL in pythia410m tinyllama; do
    case "$TGT_LABEL" in
        pythia410m) TARGET="EleutherAI/pythia-410m" ;;
        tinyllama)  TARGET="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
    esac
    OUT="results/baselines/knnlm_${TGT_LABEL}"
    if [ -f "$OUT/results.json" ]; then continue; fi
    uv run python scripts/eval_knnlm.py \
        --target-model "$TARGET" \
        --datastore-tokens 20000000 \
        --output-dir "$OUT"
done

echo ""
echo "==== Baselines DONE ===="
