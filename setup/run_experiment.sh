#!/usr/bin/env bash
# One command for a whole experiment: build the dataset, train SmolVLA, evaluate it.
#
# Usage (repo root, env active, inside tmux):
#   bash setup/run_experiment.sh HF_USERNAME NAME [make_dataset options...]
# Examples:
#   bash setup/run_experiment.sh yjoshi scripted --motion scripted --placement human      # control C
#   bash setup/run_experiment.sh yjoshi phone_spawn4 --placement spawn --augment 3       # A2
# Creates data/lerobot/ketchup_NAME, outputs/train/smolvla_NAME, outputs/eval/smolvla_NAME_*,
# logs/make_dataset_NAME.txt and logs/smolvla_NAME_train_log.txt. Stops at the first failure.
# Set WANDB=0 to skip W&B, STEPS/BATCH as in train_both.sh.
set -eo pipefail
HF_USER="${1:?usage: bash setup/run_experiment.sh HF_USERNAME NAME [make_dataset options]}"
NAME="${2:?give the experiment a short name, e.g. scripted}"
shift 2
DATA="data/lerobot/ketchup_$NAME"
mkdir -p logs

if [ -f "$DATA/meta/info.json" ]; then
  echo ">> dataset $DATA already exists, reusing it"
else
  echo ">> building dataset $DATA  (options: --speed 1.3 $*)"
  python src/make_dataset.py --speed 1.3 "$@" --out "$DATA" --repo-id "local/ketchup_$NAME" \
    2>&1 | tee "logs/make_dataset_$NAME.txt"
fi

echo ">> training smolvla_$NAME"
ONLY=A WANDB="${WANDB:-1}" A_NAME="smolvla_$NAME" A_REPO="local/ketchup_$NAME" A_ROOT="$DATA" \
  bash setup/train_both.sh "$HF_USER"

echo ">> evaluating smolvla_$NAME"
RUNS="smolvla_$NAME" CKPTS="010000 last" bash setup/eval_all.sh
echo ">> finished $NAME. Back up before the session ends: bash setup/backup_session.sh $HF_USER"
