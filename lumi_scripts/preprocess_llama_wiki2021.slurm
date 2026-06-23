#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=llama_wiki_prep

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
OVERLAY=${OVERLAY:-$REPO_ROOT/.lumi-venv.sqsh}
OVERLAY_DIR=${OVERLAY_DIR:-$REPO_ROOT/.lumi-venv}

if [ -d "$OVERLAY_DIR" ]; then
    BIND_USER_SOFTWARE="-B $OVERLAY_DIR:/user-software"
elif [ -f "$OVERLAY" ]; then
    BIND_USER_SOFTWARE="-B $OVERLAY:/user-software:image-src=/"
else
    echo "Overlay $OVERLAY_DIR or $OVERLAY not found."
    echo "Create it first with: bash lumi_scripts/create_overlay_venv.sh"
    exit 1
fi

export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_HF_CACHE_BASE=${HF_CACHE_BASE:-/scratch/project_462001050/cache/huggingface}
export SINGULARITYENV_HF_HOME=${HF_HOME:-/scratch/project_462001050/cache/huggingface}
export SINGULARITYENV_HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/project_462001050/cache/huggingface/hub}
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/scratch/project_462001050/cache/huggingface/hub}
export SINGULARITYENV_TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/scratch/project_462001050/cache/huggingface/hub}
export SINGULARITYENV_HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/scratch/project_462001050/cache/huggingface/datasets}
export SINGULARITYENV_HF_MODULES_CACHE=${HF_MODULES_CACHE:-/scratch/project_462001050/cache/huggingface/modules}
export SINGULARITYENV_HF_ASSETS_CACHE=${HF_ASSETS_CACHE:-/scratch/project_462001050/cache/huggingface/assets}

if [ -f "$REPO_ROOT/hf.txt" ]; then
    HF_TOKEN_VALUE=$(tr -d '\n\r' < "$REPO_ROOT/hf.txt")
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN_VALUE"
    export SINGULARITYENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VALUE"
fi

DATASET_NAME=${DATASET_NAME:-brimmann2/enwiki-dec2021}
TOKENIZER_PATH=${TOKENIZER_PATH:-NousResearch/Llama-2-7b-hf}
TEXT_COLUMN=${TEXT_COLUMN:-text}
BLOCK_SIZE=${BLOCK_SIZE:-2048}
STRIDE=${STRIDE:-1024}
NUM_PROC=${NUM_PROC:-16}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/data/enwiki-dec2021-preprocessed-llama2-7b}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/preprocessing_config.json" ]; then
    echo "Output already exists: $OUTPUT_DIR"
    echo "Skipping preprocessing."
    exit 0
fi

singularity run $BIND_USER_SOFTWARE "$SIF" bash -lc "
set -euo pipefail
source /user-software/bin/activate
cd '$REPO_ROOT'

python scripts/preprocess_dataset.py \
  --dataset-name '$DATASET_NAME' \
  --text-column '$TEXT_COLUMN' \
  --tokenizer-path '$TOKENIZER_PATH' \
  --output-dir '$OUTPUT_DIR' \
  --block-size '$BLOCK_SIZE' \
  --stride '$STRIDE' \
  --num-proc '$NUM_PROC'
"
