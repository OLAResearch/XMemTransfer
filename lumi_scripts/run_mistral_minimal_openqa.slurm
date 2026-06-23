#!/bin/bash
#SBATCH --account=project_462001050
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00

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

TARGET_MODEL=${TARGET_MODEL:-mistralai/Mistral-7B-v0.3}
SEED=${SEED:-42}
SEQ_LEN=${SEQ_LEN:-2048}
BATCH_SIZE=${BATCH_SIZE:-1}
SOURCE_TOKENS=${SOURCE_TOKENS:-5000000}
ADAPTOR_TOKENS=${ADAPTOR_TOKENS:-2000000}
OPENQA_TASKS=${OPENQA_TASKS:-"nq webqa triviaqa truthfulqa hotpotqa"}
OPENQA_MAX_EXAMPLES=${OPENQA_MAX_EXAMPLES:-}
SKIP_ADAPTOR_FINAL_TEST_EVAL=${SKIP_ADAPTOR_FINAL_TEST_EVAL:-0}

SOURCE_OUT=${SOURCE_OUT:-results/mistral_wiki2021_source_seed${SEED}}
ADAPTOR_OUT=${ADAPTOR_OUT:-results/mistral_wiki2021_transferred_seed${SEED}}
EVAL_OUT=${EVAL_OUT:-results/openqa/mistral_wiki2021_seed${SEED}}

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/.hf_cache"

singularity run $BIND_USER_SOFTWARE "$SIF" bash -lc "
set -euo pipefail
source /user-software/bin/activate
cd '$REPO_ROOT'

if [ -f '$SOURCE_OUT/memory.pt' ] && [ -f '$SOURCE_OUT/memory_config.json' ]; then
  echo 'Reusing existing source memory artifacts from $SOURCE_OUT'
else
  python scripts/train_source_memory.py \
    --model '$TARGET_MODEL' \
    --seed '$SEED' \
    --freeze-backbone \
    --canon-mode word_boundary \
    --corpus wikipedia-2021 \
    --seq-len '$SEQ_LEN' \
    --batch-size '$BATCH_SIZE' \
    --max-tokens '$SOURCE_TOKENS' \
    --output-dir '$SOURCE_OUT'
fi

if [ -f '$ADAPTOR_OUT/adaptor_best.pt' ] || [ -f '$ADAPTOR_OUT/adaptor.pt' ]; then
  echo 'Reusing existing adaptor artifacts from $ADAPTOR_OUT'
else
  extra_adaptor_args=()
  if [ '$SKIP_ADAPTOR_FINAL_TEST_EVAL' = '1' ]; then
    extra_adaptor_args+=(--skip-final-test-eval)
  fi

  python scripts/train_adaptor.py \
    --condition transferred \
    --target-model '$TARGET_MODEL' \
    --seed '$SEED' \
    --canon-mode word_boundary \
    --corpus wikipedia-2021 \
    --seq-len '$SEQ_LEN' \
    --batch-size '$BATCH_SIZE' \
    --max-tokens '$ADAPTOR_TOKENS' \
    --source-memory '$SOURCE_OUT/memory.pt' \
    --memory-config '$SOURCE_OUT/memory_config.json' \
    --output-dir '$ADAPTOR_OUT' \
    \"\${extra_adaptor_args[@]}\"
fi

eval_args=(
  --target-model '$TARGET_MODEL'
  --tasks $OPENQA_TASKS
  --canon-mode word_boundary
  --max-new-tokens 15
  --overwrite
  --adaptor-dir '$ADAPTOR_OUT'
  --source-memory '$SOURCE_OUT/memory.pt'
  --memory-config '$SOURCE_OUT/memory_config.json'
  --output-dir '$EVAL_OUT'
)

if [ -n '$OPENQA_MAX_EXAMPLES' ]; then
  eval_args+=(--max-examples '$OPENQA_MAX_EXAMPLES')
fi

python scripts/eval_openqa.py \"\${eval_args[@]}\"
"
