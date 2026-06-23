#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
SOURCE_MODEL_LLAMA=${SOURCE_MODEL_LLAMA:-NousResearch/Llama-2-7b-hf}
SOURCE_MODEL_MISTRAL=${SOURCE_MODEL_MISTRAL:-mistralai/Mistral-7B-v0.3}
TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
SOURCE_TOKENS=${SOURCE_TOKENS:-10000000}
ADAPTOR_TOKENS=${ADAPTOR_TOKENS:-20000000}
SOURCE_PATIENCE=${SOURCE_PATIENCE:-3}
ADAPTOR_PATIENCE=${ADAPTOR_PATIENCE:-3}
INJECTION_LAYERS_SPEC=${INJECTION_LAYERS_SPEC:-2:10}
ADAPTOR_BRANCHES=${ADAPTOR_BRANCHES:-4}
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

submit_eval_only() {
    local label=$1
    local source_out=$2
    local adaptor_out=$3
    local dependency=$4
    local eval_out="$REPO_ROOT/results/openqa/${label}"
    local eval_job

    eval_job=$(submit_job "$dependency" \
        --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$source_out",ADAPTOR_OUT="$adaptor_out",OUTPUT_DIR="$eval_out",OVERWRITE=1,CONDITIONS="transferred",OPENQA_TASKS="$OPENQA_TASKS" \
        "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

    echo "$label eval_job=$eval_job"
    echo "  source_out=$source_out"
    echo "  adaptor_out=$adaptor_out"
    echo "  eval_out=$eval_out"
}

# 1) Mistral -> Mistral native dual-layer multi-branch: train source + target adaptor + eval.
mistral_source_out="$REPO_ROOT/results/mistral_7b_wiki2021_source_true_duallayer_multibranch_10m_seed${SEED}"
mistral_adaptor_out="$REPO_ROOT/results/mistral_from_mistral_7b_wiki2021_transferred_true_duallayer_multibranch_20m_seed${SEED}"
mistral_eval_out="$REPO_ROOT/results/openqa/mistral_from_mistral_7b_wiki2021_true_duallayer_multibranch_20m_seed${SEED}"

mistral_source_job=$(submit_job "" \
    --export=ALL,MODEL="$SOURCE_MODEL_MISTRAL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$SOURCE_TOKENS",TABLE_SIZE=65536,EARLY_STOPPING_PATIENCE="$SOURCE_PATIENCE",OUTPUT_DIR="$mistral_source_out",INJECTION_LAYERS="$INJECTION_LAYERS_SPEC",ADAPTOR_BRANCHES="$ADAPTOR_BRANCHES" \
    "$REPO_ROOT/lumi_scripts/train_wiki2021_source.slurm")

mistral_adaptor_job=$(submit_job "$mistral_source_job" \
    --export=ALL,CONDITION="transferred",TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$ADAPTOR_TOKENS",EARLY_STOPPING_PATIENCE="$ADAPTOR_PATIENCE",SOURCE_OUT="$mistral_source_out",OUTPUT_DIR="$mistral_adaptor_out",SKIP_FINAL_TEST_EVAL=1,INJECTION_LAYERS="$INJECTION_LAYERS_SPEC",ADAPTOR_BRANCHES="$ADAPTOR_BRANCHES" \
    "$REPO_ROOT/lumi_scripts/train_wiki2021_transfer.slurm")

mistral_eval_job=$(submit_job "$mistral_adaptor_job" \
    --export=ALL,TARGET_MODEL="$TARGET_MODEL",SEED="$SEED",SOURCE_OUT="$mistral_source_out",ADAPTOR_OUT="$mistral_adaptor_out",OUTPUT_DIR="$mistral_eval_out",OVERWRITE=1,CONDITIONS="transferred",OPENQA_TASKS="$OPENQA_TASKS" \
    "$REPO_ROOT/lumi_scripts/eval_transfer_openqa.slurm")

echo "mistral_native_duallayer_multibranch source_job=$mistral_source_job adaptor_job=$mistral_adaptor_job eval_job=$mistral_eval_job"
echo "  source_out=$mistral_source_out"
echo "  adaptor_out=$mistral_adaptor_out"
echo "  eval_out=$mistral_eval_out"

# 2) Frozen Mistral source adaptor -> Mistral: no target adaptor training, direct eval.
submit_eval_only \
    "mistral_from_mistral_7b_wiki2021_true_duallayer_frozen_source_adaptor_10m_seed${SEED}" \
    "$mistral_source_out" \
    "$mistral_source_out" \
    "$mistral_source_job"

# 3) Frozen Llama source adaptor -> Mistral: no target adaptor training, direct eval.
llama_source_out="$REPO_ROOT/results/llama2_7b_wiki2021_source_true_duallayer_multibranch_10m_seed${SEED}"
if [ ! -f "$llama_source_out/memory.pt" ] || [ ! -f "$llama_source_out/source_adaptor.pt" ]; then
    llama_source_job=$(submit_job "" \
        --export=ALL,MODEL="$SOURCE_MODEL_LLAMA",SEED="$SEED",SEQ_LEN="$SEQ_LEN",BATCH_SIZE="$BATCH_SIZE",MAX_TOKENS="$SOURCE_TOKENS",TABLE_SIZE=65536,EARLY_STOPPING_PATIENCE="$SOURCE_PATIENCE",OUTPUT_DIR="$llama_source_out",INJECTION_LAYERS="$INJECTION_LAYERS_SPEC",ADAPTOR_BRANCHES="$ADAPTOR_BRANCHES" \
        "$REPO_ROOT/lumi_scripts/train_wiki2021_source.slurm")
else
    llama_source_job=""
fi

submit_eval_only \
    "mistral_from_llama2_7b_wiki2021_true_duallayer_frozen_source_adaptor_10m_seed${SEED}" \
    "$llama_source_out" \
    "$llama_source_out" \
    "$llama_source_job"
