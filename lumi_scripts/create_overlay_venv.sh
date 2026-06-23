#!/bin/bash
set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV_DIR=${VENV_DIR:-$REPO_ROOT/.lumi-venv}
SQSH_PATH=${SQSH_PATH:-$REPO_ROOT/.lumi-venv.sqsh}

cd "$REPO_ROOT"

echo "Creating container-backed virtualenv at $VENV_DIR"
singularity run "$SIF" bash -lc "
set -euo pipefail
python -m venv '$VENV_DIR' --system-site-packages
source '$VENV_DIR/bin/activate'
python -m pip install --upgrade pip wheel
python -m pip install --no-cache-dir -e .
"

chmod -R a+rX "$VENV_DIR"

if command -v mksquashfs >/dev/null 2>&1; then
    echo "Packing $VENV_DIR into $SQSH_PATH"
    rm -f "$SQSH_PATH"
    (
        cd "$VENV_DIR"
        mksquashfs . "$SQSH_PATH" -all-root -noappend >/dev/null
    )
    echo "Created $SQSH_PATH"
else
    echo "mksquashfs not found; using unpacked virtualenv at $VENV_DIR"
fi
