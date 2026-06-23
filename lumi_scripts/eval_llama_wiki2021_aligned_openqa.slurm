#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=36:00:00
#SBATCH --job-name=llama_wiki_qa

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
OPENQA_TASKS=${OPENQA_TASKS:-"nq webqa triviaqa truthfulqa hotpotqa"}
OPENQA_MAX_EXAMPLES=${OPENQA_MAX_EXAMPLES:-}
OVERWRITE=${OVERWRITE:-1}
SOURCE_OUT=${SOURCE_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_source_seed${SEED}}
ADAPTOR_OUT=${ADAPTOR_OUT:-$REPO_ROOT/results/llama2_7b_wiki2021_aligned_transferred_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/openqa/llama2_7b_wiki2021_aligned_seed${SEED}}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ ! -f "$SOURCE_OUT/memory.pt" ] || [ ! -f "$SOURCE_OUT/memory_config.json" ]; then
    echo "Aligned source artifacts not found at $SOURCE_OUT"
    exit 1
fi

if [ ! -f "$ADAPTOR_OUT/adaptor_best.pt" ] && [ ! -f "$ADAPTOR_OUT/adaptor.pt" ]; then
    echo "Aligned adaptor artifacts not found at $ADAPTOR_OUT"
    exit 1
fi

singularity run $BIND_USER_SOFTWARE "$SIF" bash -lc "
set -euo pipefail
source /user-software/bin/activate
cd '$REPO_ROOT'

eval_args=(
  --target-model '$TARGET_MODEL'
  --tasks $OPENQA_TASKS
  --conditions baseline transferred
  --canon-mode word_boundary
  --max-new-tokens 15
  --adaptor-dir '$ADAPTOR_OUT'
  --source-memory '$SOURCE_OUT/memory.pt'
  --memory-config '$SOURCE_OUT/memory_config.json'
  --output-dir '$OUTPUT_DIR'
)

if [ '$OVERWRITE' = '1' ]; then
  eval_args+=(--overwrite)
fi

if [ -n '$OPENQA_MAX_EXAMPLES' ]; then
  eval_args+=(--max-examples '$OPENQA_MAX_EXAMPLES')
fi

python scripts/eval_openqa.py \"\${eval_args[@]}\"
"
