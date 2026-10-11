#!/usr/bin/env bash
# One-time setup on a rented Linux GPU machine (e.g. a RunPod pod), run as root.
# Installs conda, a Python 3.12 env, LeRobot with SmolVLA + LIBERO, and this repo's
# requirements. Everything goes under /workspace so it survives a pod restart.
#
# Usage (from the repo root, on the GPU machine):
#   bash setup/install_gpu.sh
#   source /workspace/miniforge3/bin/activate humanoid
set -euo pipefail

WS="${WS:-/workspace}"
CONDA_DIR="$WS/miniforge3"
ENV_NAME="humanoid"
LEROBOT_DIR="$WS/lerobot"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo ">> System packages (offscreen OpenGL for the LIBERO simulator, ffmpeg for videos)"
$SUDO apt-get update -y
DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y git wget unzip build-essential \
    libegl1 libgl1 libglvnd0 libosmesa6 libglfw3 ffmpeg

echo ">> Conda (Miniforge) in $CONDA_DIR"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/bin/activate"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

echo ">> LeRobot with SmolVLA, LIBERO and training extras"
if [ ! -d "$LEROBOT_DIR" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
pip install -e "${LEROBOT_DIR}[smolvla,libero,training]"
pip install -r "$REPO_ROOT/requirements.txt"

echo ">> LIBERO config (skips its interactive first-run question)"
echo N | python -c "import libero.libero" > /dev/null || true

echo ">> Checks"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE FOUND')"
for gl in egl osmesa; do
    # open the actual LIBERO task with cameras, exactly as evaluation does
    if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python "$REPO_ROOT/setup/check_render.py" >/dev/null 2>&1; then
        echo "Simulator rendering works with MUJOCO_GL=$gl"
        echo "export MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl" > "$WS/render_env.sh"
        break
    fi
done
[ -f "$WS/render_env.sh" ] || echo "WARNING: no working MuJoCo renderer found (needed for evaluation only)"
echo ">> Done. In every new terminal run:"
echo "   source $CONDA_DIR/bin/activate $ENV_NAME && source $WS/render_env.sh"
