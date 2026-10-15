#!/usr/bin/env bash
# Create the conda environment: Python 3.12, PyTorch, LeRobot (SmolVLA + LIBERO) and this
# repo's requirements. Works on Linux, Windows (inside WSL2) and macOS. Safe to rerun.
#
# Usage, from the repo root with conda installed:
#   bash setup/install.sh
# Options (environment variables):
#   ENV_NAME=real2sim        name of the conda env
#   LEROBOT_DIR=../lerobot   where LeRobot is cloned
#   SKIP_APT=1               don't apt-get the OpenGL libraries (e.g. no sudo; ask an admin)
set -eo pipefail

ENV_NAME="${ENV_NAME:-real2sim}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$REPO_ROOT/../lerobot}"
OS="$(uname -s)"

if [ "$OS" = "Linux" ] && [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null; then
  echo ">> OpenGL libraries for the simulator's offscreen rendering (needs sudo; SKIP_APT=1 to skip)"
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -y
  $SUDO apt-get install -y git libegl1 libgl1 libosmesa6 libglfw3
fi

echo ">> conda env '$ENV_NAME' (Python 3.12 + ffmpeg)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
conda install -y -c conda-forge ffmpeg

echo ">> PyTorch"
if python -c "import torch" 2>/dev/null; then
  echo "   already installed"
elif [ "$OS" = "Linux" ] && command -v nvidia-smi >/dev/null; then
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
elif [ "$OS" = "Linux" ]; then
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch==2.11.0 torchvision==0.26.0      # macOS: CPU / Apple GPU (MPS)
fi

echo ">> LeRobot, pinned to the commit the results were produced with"
[ -d "$LEROBOT_DIR" ] || git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
git -C "$LEROBOT_DIR" fetch -q origin
git -C "$LEROBOT_DIR" checkout -q "$(cat "$REPO_ROOT/setup/lerobot_commit.txt")"
# same package versions as the original runs (torch/CUDA lines left out: installed above)
CONSTRAINTS="$(mktemp)"
grep -vE '^(#|torch==|torchvision==|nvidia-|triton==|cuda-)' "$REPO_ROOT/setup/requirements-lock.txt" > "$CONSTRAINTS"
pip install -e "${LEROBOT_DIR}[smolvla,libero,training]" -c "$CONSTRAINTS"   # LIBERO is Linux-only
pip install -r "$REPO_ROOT/requirements.txt" -c "$CONSTRAINTS"

echo ">> LIBERO first-run config (answers its interactive question)"
echo N | python -c "import libero.libero" >/dev/null 2>&1 || true

echo ">> checks"
python -c "import torch; print('   torch', torch.__version__, '| CUDA GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if [ "$OS" != "Linux" ]; then
  echo "   macOS: video -> trajectory steps only; the simulator (LIBERO) needs Linux or WSL2"
elif python "$REPO_ROOT/setup/check_render.py"; then
  echo "   simulator renders"
else
  echo "   simulator could not render: see 'Troubleshooting' in the README"
fi
echo ">> done. In every new terminal: conda activate $ENV_NAME"
