#!/usr/bin/env bash
# Install dependencies and verify environment.
#
# Prerequisites:
#   - Python >= 3.11 (any recent 3.11/3.12/3.13 works)
#   - uv package manager (https://docs.astral.sh/uv/)
#       Install on Linux/macOS:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   - CUDA-capable GPU for training (≥24 GB VRAM recommended; H100/A100 for 9B source)
#   - HuggingFace account + token for gated models (TinyLlama/Pythia are open)
#
# Run from the repository root:
#   bash run/00_install.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==== uv version ===="
uv --version

echo ""
echo "==== uv sync (resolves pyproject.toml) ===="
uv sync

echo ""
echo "==== Import smoke test ===="
uv run python -c "
import torch, transformers, datasets
print(f'torch:        {torch.__version__}  (cuda={torch.cuda.is_available()})')
print(f'transformers: {transformers.__version__}')
print(f'datasets:     {datasets.__version__}')
from engram.memory import EngramMemory
from engram.adaptor import build_adaptor
from engram.backbone_wrapper import BackboneWrapper
print('engram.* imports OK')
"

echo ""
echo "==== Install complete ===="
echo "Next: run 'bash run/01_pilot.sh' to run a ~30 min smoke test."
