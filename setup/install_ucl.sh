#!/usr/bin/env bash
# Rebuild the training/evaluation environment on a UCL GPU workstation (no admin rights,
# everything under /scratch0/$USER). Safe to rerun: finished steps are skipped.
#
# Before running (fresh session):
#   cd /scratch0/$USER && git clone https://github.com/YOUR_GITHUB/Real2Sim-Task.git
#   cd Real2Sim-Task && bash setup/install_ucl.sh
# Afterwards, in every new terminal:
#   source ~/.bashrc && conda activate humanoid && cd /scratch0/$USER/Real2Sim-Task
#
# Uses the exact versions saved by setup/backup_session.sh when present:
#   setup/lerobot_commit.txt, setup/requirements_lock_ucl.txt
set -eo pipefail

SCRATCH="${SCRATCH:-/scratch0/$USER}"
CONDA_DIR="$SCRATCH/miniconda3"
ENV_NAME="humanoid"
LEROBOT_DIR="$SCRATCH/lerobot"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SCRATCH"/{hf,pip,conda/pkgs,tmp,libero_cache}

echo ">> 1/7 cache folders inside $SCRATCH (added to ~/.bashrc once)"
if ! grep -q "HF_LEROBOT_HOME=$SCRATCH" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<EOF
export HF_HOME=$SCRATCH/hf
export HF_LEROBOT_HOME=$SCRATCH/hf/lerobot
export PIP_CACHE_DIR=$SCRATCH/pip
export CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs
export TMPDIR=$SCRATCH/tmp
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
EOF
fi
export HF_HOME=$SCRATCH/hf HF_LEROBOT_HOME=$SCRATCH/hf/lerobot PIP_CACHE_DIR=$SCRATCH/pip \
       CONDA_PKGS_DIRS=$SCRATCH/conda/pkgs TMPDIR=$SCRATCH/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# LIBERO downloads its 3D assets to ~/.cache/libero; keep them in scratch instead
mkdir -p ~/.cache
if [ -d ~/.cache/libero ] && [ ! -L ~/.cache/libero ]; then
  cp -rn ~/.cache/libero/. "$SCRATCH/libero_cache/" && rm -rf ~/.cache/libero
fi
[ -L ~/.cache/libero ] || ln -s "$SCRATCH/libero_cache" ~/.cache/libero

echo ">> 2/7 Miniconda in $CONDA_DIR"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH/tmp/mc.sh"
  bash "$SCRATCH/tmp/mc.sh" -b -p "$CONDA_DIR"
  "$CONDA_DIR/bin/conda" init bash >/dev/null
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
# newer Miniconda asks to accept channel terms once; harmless if not needed
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo ">> 3/7 conda env '$ENV_NAME' (Python 3.12 + ffmpeg)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"
conda install -y -c conda-forge ffmpeg >/dev/null

echo ">> 4/7 PyTorch for CUDA 12.8 (works with driver 570+; check nvidia-smi if unsure)"
python -c "import torch" 2>/dev/null || \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo ">> 5/7 LeRobot (SmolVLA + LIBERO + training extras)"
[ -d "$LEROBOT_DIR" ] || git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
if [ -f "$REPO_ROOT/setup/lerobot_commit.txt" ]; then
  COMMIT=$(cat "$REPO_ROOT/setup/lerobot_commit.txt")
  echo "   pinning LeRobot to $COMMIT (the version used for the earlier runs)"
  git -C "$LEROBOT_DIR" fetch -q origin && git -C "$LEROBOT_DIR" checkout -q "$COMMIT"
fi
CONSTRAINT=()
if [ -f "$REPO_ROOT/setup/requirements_lock_ucl.txt" ]; then
  # same package versions as last session (torch lines skipped: already installed above)
  grep -viE '^(torch==|torchvision==|nvidia-|triton==|lerobot==)' "$REPO_ROOT/setup/requirements_lock_ucl.txt" \
    | grep -v ' @ ' > "$SCRATCH/tmp/constraints.txt" || true
  CONSTRAINT=(-c "$SCRATCH/tmp/constraints.txt")
fi
pip install -e "$LEROBOT_DIR[smolvla,libero,training]" "${CONSTRAINT[@]}"
pip install -r "$REPO_ROOT/requirements.txt" "${CONSTRAINT[@]}"

echo ">> 6/7 LIBERO first-run config"
echo N | python -c "import libero.libero" >/dev/null 2>&1 || true

echo ">> 7/7 checks"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('   GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
python "$REPO_ROOT/setup/check_render.py" 2>/dev/null | tail -1 | sed 's/^/   simulator: /'
hf auth whoami >/dev/null 2>&1 && echo "   Hugging Face: logged in" || echo "   Hugging Face: not logged in (run: hf auth login)"
grep -q api.wandb.ai ~/.netrc 2>/dev/null && echo "   W&B: logged in" || echo "   W&B: not logged in (run: wandb login)"
echo ">> environment ready. In new terminals: source ~/.bashrc && conda activate $ENV_NAME"
