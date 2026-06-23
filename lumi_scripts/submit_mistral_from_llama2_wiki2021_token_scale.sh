#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
MODEL=${MODEL:-NousResearch/Llama-2-7b-hf}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
TABLE_SIZE=${TABLE_SIZE:-65536}
SOURCE_PATIENCE=${SOURCE_PATIENCE:-3}
ADAPTOR_PATIENCE=${ADAPTOR_PATIENCE:-3}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
OPENQA_TASKS=${OPENQA_TASKS:-"nq webqa triviaqa truthfulqa hotpotqa"}

submit_job() {
    local dependency=$1
    shift
    if [ -n "$dependency" ]; then
        sbatch --parsable --dependency=afterok:"$dependency" "$@"
    else
        sbatch --parsable "$@"
    fi
}

submit_chain() {
    local source_tokens=$1
    local adaptor_tokens=$2
    local tag="s$((source_tokens/1000000))m_a$((adaptor_tokens/1000000))m"

    local source_out="$REPO_ROOT/results/llama2_7b_wiki2021_source_${tag}_es${SOURCE_PATIENCE}_seed${SEED}"
    local adaptor_out="$REPO_ROOT/results/mistral_from_llama2_7b_wiki2021_transferred_${tag}_es${ADAPTOR_PATIENCE}_seed${SEED}"
    local eval_out="$REPO_ROOT/results/openqa/mistral_from_llama2_7b_wiki2021_${tag}_transfer_only_seed${SEED}"

    local source_job
    local adaptor_job
    local eval_job

    source_job=$(submit_job "" \
        --export=ALL,MODEL="$MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$source_tokens",TABLE_SIZE="$TABLE_SIZE",EARLY_STOPPING_PATIENCE="$SOURCE_PATIENCE",OUTPUT_DIR="$source_out" \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_source.slurm")

    adaptor_job=$(submit_job "$source_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$adaptor_tokens",EARLY_STOPPING_PATIENCE="$ADAPTOR_PATIENCE",SOURCE_OUT="$source_out",OUTPUT_DIR="$adaptor_out",SKIP_FINAL_TEST_EVAL=1 \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_transfer.slurm")

    eval_job=$(submit_job "$adaptor_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$source_out",ADAPTOR_OUT="$adaptor_out",OUTPUT_DIR="$eval_out",OVERWRITE=1,CONDITIONS="transferred",OPENQA_TASKS="$OPENQA_TASKS" \
        "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

    echo "tag=$tag source_tokens=$source_tokens adaptor_tokens=$adaptor_tokens source_job=$source_job adaptor_job=$adaptor_job eval_job=$eval_job"
    echo "  source_out=$source_out"
    echo "  adaptor_out=$adaptor_out"
    echo "  eval_out=$eval_out"
}

submit_chain 15000000 25000000
submit_chain 20000000 30000000
