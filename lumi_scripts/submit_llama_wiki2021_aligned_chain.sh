#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

SEED=${SEED:-42}
PREPROCESS_JOB_ID=${PREPROCESS_JOB_ID:-}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data/enwiki-dec2021-preprocessed-llama2-7b}
SOURCE_OUT=${SOURCE_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_source_seed${SEED}}
ADAPTOR_OUT=${ADAPTOR_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_transferred_seed${SEED}}
EVAL_OUT=${EVAL_OUT:-$REPO_ROOT/results/openqa/llama2_7b_wiki2021_aligned_seed${SEED}}

submit_job() {
    local dependency=$1
    shift
    if [ -n "$dependency" ]; then
        sbatch --parsable --dependency=afterok:"$dependency" "$@"
    else
        sbatch --parsable "$@"
    fi
}

if [ -n "$PREPROCESS_JOB_ID" ]; then
    preprocess_job=$PREPROCESS_JOB_ID
elif [ -f "$DATASET_DIR/preprocessing_config.json" ]; then
    preprocess_job=""
else
    preprocess_job=$(sbatch --parsable \
        --export=ALL,DATASET_NAME=brimmann2/enwiki-dec2021,TOKENIZER_PATH=NousResearch/Llama-2-7b-hf,OUTPUT_DIR="$DATASET_DIR",BLOCK_SIZE=2048,STRIDE=1024,NUM_PROC=16 \
        "$REPO_ROOT/lumi_scripts/preprocess_llama_wiki2021.slurm")
fi

source_job=$(submit_job "$preprocess_job" \
    --export=ALL,MODEL=NousResearch/Llama-2-7b-hf,SEED="$SEED",DATASET_DIR="$DATASET_DIR",OUTPUT_DIR="$SOURCE_OUT",MAX_TOKENS=5000000,SEQ_LEN=2048,BATCH_SIZE=1 \
    "$REPO_ROOT/lumi_scripts/train_llama_wiki2021_aligned_source.slurm")

adaptor_job=$(submit_job "$source_job" \
    --export=ALL,TARGET_MODEL=NousResearch/Llama-2-7b-hf,SEED="$SEED",DATASET_DIR="$DATASET_DIR",SOURCE_OUT="$SOURCE_OUT",OUTPUT_DIR="$ADAPTOR_OUT",MAX_TOKENS=2000000,SEQ_LEN=2048,BATCH_SIZE=1 \
    "$REPO_ROOT/lumi_scripts/train_llama_wiki2021_aligned_adaptor.slurm")

eval_job=$(submit_job "$adaptor_job" \
    --export=ALL,TARGET_MODEL=NousResearch/Llama-2-7b-hf,SEED="$SEED",SOURCE_OUT="$SOURCE_OUT",ADAPTOR_OUT="$ADAPTOR_OUT",OUTPUT_DIR="$EVAL_OUT",OVERWRITE=1 \
    "$REPO_ROOT/lumi_scripts/eval_llama_wiki2021_aligned_openqa.slurm")

echo "preprocess_job=${preprocess_job:-<reused-existing-dataset>}"
echo "source_job=$source_job"
echo "adaptor_job=$adaptor_job"
echo "eval_job=$eval_job"
