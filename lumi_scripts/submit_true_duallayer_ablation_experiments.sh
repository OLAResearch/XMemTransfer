#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
ADAPTOR_TOKENS=${ADAPTOR_TOKENS:-20000000}
SOURCE_PATIENCE=${SOURCE_PATIENCE:-3}
ADAPTOR_PATIENCE=${ADAPTOR_PATIENCE:-3}
INJECTION_LAYERS_SPEC=${INJECTION_LAYERS_SPEC:-2:10}
ANCHOR_BRANCHES=${ANCHOR_BRANCHES:-4}
OPENQA_TASKS=${OPENQA_TASKS:-"nq webqa triviaqa truthfulqa hotpotqa"}
SOURCE_OUT=${SOURCE_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_source_true_duallayer_multibranch_10m_seed${SEED}}

submit_job() {
    local dependency=$1
    shift
    if [ -n "$dependency" ]; then
        sbatch --parsable --dependency=afterok:"$dependency" "$@"
    else
        sbatch --parsable "$@"
    fi
}

submit_ablation() {
    local label=$1
    local condition=$2
    local branches=$3

    local adaptor_out="$REPO_ROOT/results/mistral_from_llama2_7b_wiki2021_true_duallayer_${label}_20m_seed${SEED}"
    local eval_out="$REPO_ROOT/results/openqa/mistral_from_llama2_7b_wiki2021_true_duallayer_${label}_20m_seed${SEED}"

    local transfer_job
    transfer_job=$(submit_job "" \
        --export=ALL,CONDITION="$condition",TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$ADAPTOR_TOKENS",EARLY_STOPPING_PATIENCE="$ADAPTOR_PATIENCE",SOURCE_OUT="$SOURCE_OUT",OUTPUT_DIR="$adaptor_out",SKIP_FINAL_TEST_EVAL=1,INJECTION_LAYERS="$INJECTION_LAYERS_SPEC",ADAPTOR_BRANCHES="$branches" \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_transfer.slurm")

    local eval_job
    eval_job=$(submit_job "$transfer_job" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$SOURCE_OUT",ADAPTOR_OUT="$adaptor_out",OUTPUT_DIR="$eval_out",OVERWRITE=1,CONDITIONS="$condition",OPENQA_TASKS="$OPENQA_TASKS" \
        "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

    echo "$label transfer_job=$transfer_job eval_job=$eval_job"
    echo "  condition=$condition"
    echo "  injection_layers=$INJECTION_LAYERS_SPEC"
    echo "  adaptor_branches=$branches"
    echo "  adaptor_out=$adaptor_out"
    echo "  eval_out=$eval_out"
}

if [ ! -f "$SOURCE_OUT/memory.pt" ] || [ ! -f "$SOURCE_OUT/memory_config.json" ]; then
    echo "Source artifacts not found at $SOURCE_OUT"
    exit 1
fi

submit_ablation "branch1_transferred" "transferred" "1"
submit_ablation "ffn_only" "ffn_only" "$ANCHOR_BRANCHES"
submit_ablation "random_memory" "random_memory" "$ANCHOR_BRANCHES"
submit_ablation "permuted_keys" "permuted_keys" "$ANCHOR_BRANCHES"
submit_ablation "train_from_scratch" "train_from_scratch" "$ANCHOR_BRANCHES"
