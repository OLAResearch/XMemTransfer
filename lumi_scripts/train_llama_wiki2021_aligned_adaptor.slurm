#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --job-name=llama_wiki_ada

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

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

TARGET_MODEL=${TARGET_MODEL:-NousResearch/Llama-2-7b-hf}
SEED=${SEED:-42}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-2000000}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/data/enwiki-dec2021-preprocessed-llama2-7b}
SOURCE_OUT=${SOURCE_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_source_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_transferred_seed${SEED}}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ -f "$OUTPUT_DIR/adaptor_best.pt" ] || [ -f "$OUTPUT_DIR/adaptor.pt" ]; then
    echo "Reusing existing aligned adaptor artifacts from $OUTPUT_DIR"
    exit 0
fi

if [ ! -f "$DATASET_DIR/preprocessing_config.json" ]; then
    echo "Tokenizer-aligned dataset not found at $DATASET_DIR"
    exit 1
fi

if [ ! -f "$SOURCE_OUT/memory.pt" ] || [ ! -f "$SOURCE_OUT/memory_config.json" ]; then
    echo "Aligned source artifacts not found at $SOURCE_OUT"
    exit 1
fi

singularity run $BIND_USER_SOFTWARE "$SIF" bash -lc "
set -euo pipefail
source /user-software/bin/activate
cd '$REPO_ROOT'

python scripts/train_adaptor.py \
  --condition transferred \
  --target-model '$TARGET_MODEL' \
  --seed '$SEED' \
  --canon-mode word_boundary \
  --corpus wikipedia-2021 \
  --wikipedia2021-dataset '$DATASET_DIR' \
  --wikipedia2021-source-tokenizer '$TARGET_MODEL' \
  --wikipedia2021-require-tokenizer-match \
  --seq-len '$SEQ_LEN' \
  --batch-size '$BATCH_SIZE' \
  --max-tokens '$MAX_TOKENS' \
  --source-memory '$SOURCE_OUT/memory.pt' \
  --memory-config '$SOURCE_OUT/memory_config.json' \
  --output-dir '$OUTPUT_DIR'
"
