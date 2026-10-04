#!/usr/bin/env bash
# One-time environment setup on Ubuntu: WSL2 on the laptop, or a rented Linux GPU box.
# Usage (from the repo root):
#   bash setup/install_linux.sh            # GPU machine
#   CPU_ONLY=1 bash setup/install_linux.sh # laptop / WSL2 (smaller CPU-only PyTorch)
set -euo pipefail

ENV_NAME="${ENV_NAME:-humanoid}"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/code/lerobot}"
CPU_ONLY="${CPU_ONLY:-0}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ">> System packages (build tools + offscreen OpenGL for MuJoCo)"
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
    libosmesa6-dev libgl1-mesa-dev libegl1 libglfw3 patchelf

echo ">> Conda env '$ENV_NAME' (Python 3.12)"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"
conda install -y ffmpeg -c conda-forge

if [ "$CPU_ONLY" = "1" ]; then
    echo ">> CPU-only PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ">> LeRobot + LIBERO (cloned outside this repo, at $LEROBOT_DIR)"
if [ ! -d "$LEROBOT_DIR" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
pip install -e "${LEROBOT_DIR}[libero]"

echo ">> Project requirements"
pip install -r "$REPO_ROOT/requirements.txt"

echo ">> Writing LIBERO's config (skips its interactive first-run prompt)"
echo N | python -c "import libero.libero" > /dev/null

echo ">> Sanity check"
python "$REPO_ROOT/src/view_scenes.py" --list-only --suites libero_object
echo ">> Done. Run: conda activate $ENV_NAME"
