#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
LLAMA_MODEL=${LLAMA_MODEL:-NousResearch/Llama-2-7b-hf}
MISTRAL_MODEL=${MISTRAL_MODEL:-mistralai/Mistral-7B-v0.3}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
SOURCE_TOKENS=${SOURCE_TOKENS:-10000000}
SOURCE_PATIENCE=${SOURCE_PATIENCE:-3}
INJECTION_LAYERS_SPEC=${INJECTION_LAYERS_SPEC:-2:10}
OPENQA_TASKS=${OPENQA_TASKS:-"nq webqa triviaqa truthfulqa hotpotqa"}

# Strict Engram-only setup: d_mem = d_model = 4096 with matched total memory params.
TABLE_SIZE=${TABLE_SIZE:-8192}
D_HEAD=${D_HEAD:-512}
HEADS_PER_ORDER=${HEADS_PER_ORDER:-4}
MAX_NGRAM=${MAX_NGRAM:-3}

submit_job() {
    local dependency=$1
    shift
    if [ -n "$dependency" ]; then
        sbatch --parsable --dependency=afterok:"$dependency" "$@"
    else
        sbatch --parsable "$@"
    fi
}

submit_memory_only_chain() {
    local label=$1
    local source_model=$2

    local source_out="$REPO_ROOT/results/${label}_source_true_duallayer_memory_only_10m_seed${SEED}"
    local eval_out="$REPO_ROOT/results/openqa/${label}_true_duallayer_memory_only_10m_seed${SEED}"

    local source_job
    local eval_job

    source_job=$(submit_job "" \
        --export=ALL,MODEL="$source_model",CONDITION=memory_only,SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$SOURCE_TOKENS",TABLE_SIZE="$TABLE_SIZE",D_HEAD="$D_HEAD",MAX_NGRAM="$MAX_NGRAM",HEADS_PER_ORDER="$HEADS_PER_ORDER",EARLY_STOPPING_PATIENCE="$SOURCE_PATIENCE",OUTPUT_DIR="$source_out",INJECTION_LAYERS="$INJECTION_LAYERS_SPEC" \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_source.slurm")

    eval_job=$(submit_job "$source_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$source_out",ADAPTOR_OUT="$source_out",OUTPUT_DIR="$eval_out",OVERWRITE=1,CONDITIONS="memory_only",OPENQA_TASKS="$OPENQA_TASKS" \
        "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

    echo "$label source_job=$source_job eval_job=$eval_job"
    echo "  source_model=$source_model"
    echo "  source_out=$source_out"
    echo "  eval_out=$eval_out"
    echo "  table_size=$TABLE_SIZE d_head=$D_HEAD"
}

submit_memory_only_chain "mistral_from_mistral_7b_wiki2021" "$MISTRAL_MODEL"
submit_memory_only_chain "mistral_from_llama2_7b_wiki2021" "$LLAMA_MODEL"
