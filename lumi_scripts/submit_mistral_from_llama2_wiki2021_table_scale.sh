#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
MODEL=${MODEL:-NousResearch/Llama-2-7b-hf}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
SOURCE_TOKENS=${SOURCE_TOKENS:-10000000}
ADAPTOR_TOKENS=${ADAPTOR_TOKENS:-20000000}
SOURCE_PATIENCE=${SOURCE_PATIENCE:-3}
ADAPTOR_PATIENCE=${ADAPTOR_PATIENCE:-3}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
BASE_TABLE_SIZE=${BASE_TABLE_SIZE:-65536}
TABLE_MULTIPLIERS=${TABLE_MULTIPLIERS:-"2 4 6"}

submit_job() {
    local dependency=$1
    shift
    if [ -n "$dependency" ]; then
        sbatch --parsable --dependency=afterok:"$dependency" "$@"
    else
        sbatch --parsable "$@"
    fi
}

for factor in $TABLE_MULTIPLIERS; do
    table_size=$((BASE_TABLE_SIZE * factor))
    source_out="$REPO_ROOT/results/llama2_7b_wiki2021_source_10m_es${SOURCE_PATIENCE}_table${factor}x_seed${SEED}"
    adaptor_out="$REPO_ROOT/results/mistral_from_llama2_7b_wiki2021_transferred_20m_es${ADAPTOR_PATIENCE}_table${factor}x_seed${SEED}"
    eval_out="$REPO_ROOT/results/openqa/mistral_from_llama2_7b_wiki2021_table${factor}x_seed${SEED}"

    source_job=$(submit_job "" \
        --export=ALL,MODEL="$MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$SOURCE_TOKENS",TABLE_SIZE="$table_size",EARLY_STOPPING_PATIENCE="$SOURCE_PATIENCE",OUTPUT_DIR="$source_out" \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_source.slurm")

    adaptor_job=$(submit_job "$source_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$ADAPTOR_TOKENS",EARLY_STOPPING_PATIENCE="$ADAPTOR_PATIENCE",SOURCE_OUT="$source_out",OUTPUT_DIR="$adaptor_out",SKIP_FINAL_TEST_EVAL=1 \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_transfer.slurm")

    eval_job=$(submit_job "$adaptor_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$source_out",ADAPTOR_OUT="$adaptor_out",OUTPUT_DIR="$eval_out",OVERWRITE=1 \
        "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

    echo "factor=${factor}x table_size=${table_size} source_job=${source_job} adaptor_job=${adaptor_job} eval_job=${eval_job}"
    echo "  source_out=$source_out"
    echo "  adaptor_out=$adaptor_out"
    echo "  eval_out=$eval_out"
done
