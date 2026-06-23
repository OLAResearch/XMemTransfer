#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=wiki_src
#SBATCH --output=%x-%j.out

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

MODEL=${MODEL:-NousResearch/Llama-2-7b-hf}
CONDITION=${CONDITION:-transferred}
SEED=${SEED:-42}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TOKENS=${MAX_TOKENS:-10000000}
TABLE_SIZE=${TABLE_SIZE:-65536}
D_HEAD=${D_HEAD:-64}
MAX_NGRAM=${MAX_NGRAM:-3}
HEADS_PER_ORDER=${HEADS_PER_ORDER:-4}
INJECTION_LAYERS=${INJECTION_LAYERS:-}
ADAPTOR_BRANCHES=${ADAPTOR_BRANCHES:-1}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-3}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/wiki2021_source_seed${SEED}}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_DIR")"

if [ -f "$OUTPUT_DIR/memory.pt" ] && [ -f "$OUTPUT_DIR/memory_config.json" ]; then
    echo "Reusing existing source artifacts from $OUTPUT_DIR"
    exit 0
fi

singularity run $BIND_USER_SOFTWARE "$SIF" bash -lc "
set -euo pipefail
source /user-software/bin/activate
cd '$REPO_ROOT'

extra_args=()
if [ -n '$INJECTION_LAYERS' ]; then
  extra_args+=(--injection-layers '$INJECTION_LAYERS')
fi
extra_args+=(--adaptor-branches '$ADAPTOR_BRANCHES')

python scripts/train_source_memory.py \
  --model '$MODEL' \
  --condition '$CONDITION' \
  --seed '$SEED' \
  --freeze-backbone \
  --canon-mode word_boundary \
  --corpus wikipedia-2021 \
  --seq-len '$SEQ_LEN' \
  --batch-size '$BATCH_SIZE' \
  --max-tokens '$MAX_TOKENS' \
  --table-size '$TABLE_SIZE' \
  --d-head '$D_HEAD' \
  --max-ngram '$MAX_NGRAM' \
  --heads-per-order '$HEADS_PER_ORDER' \
  --early-stopping-patience '$EARLY_STOPPING_PATIENCE' \
  --output-dir '$OUTPUT_DIR' \
  \"\${extra_args[@]}\"
"
