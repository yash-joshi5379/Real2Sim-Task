#!/usr/bin/env bash
# Train and evaluate ACT (a small transformer policy, no language model, trained from scratch
# apart from an ImageNet ResNet-18) on an existing dataset, for comparison with SmolVLA.
#
# Usage (repo root, env active, inside tmux):
#   bash setup/run_act.sh phone_eval          # uses data/lerobot/ketchup_phone_eval (run D's data)
# Creates outputs/train/act_NAME, outputs/eval/act_NAME_*, logs/act_NAME_train_log.txt.
set -eo pipefail
NAME="${1:?give the dataset name, e.g. phone_eval}"
DATA="data/lerobot/ketchup_$NAME"
STEPS="${STEPS:-30000}"
[ -f "$DATA/meta/info.json" ] || { echo "no dataset at $DATA (build it with run_experiment.sh first)"; exit 1; }
mkdir -p logs

echo ">> training act_$NAME on $DATA ($STEPS steps)"
lerobot-train \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 \
  --dataset.repo_id="local/ketchup_$NAME" --dataset.root="$DATA" \
  --batch_size=32 --steps="$STEPS" --save_freq=10000 --log_freq=100 --seed=1000 \
  --policy.device=cuda --policy.push_to_hub=false \
  --wandb.enable="$([ "${WANDB:-1}" = 1 ] && echo true || echo false)" --wandb.project=real2sim-ketchup --wandb.disable_artifact=true \
  --output_dir="outputs/train/act_$NAME" --job_name="act_$NAME" \
  2>&1 | tee "logs/act_${NAME}_train_log.txt"

echo ">> evaluating act_$NAME"
RUNS="act_$NAME" CKPTS="010000 last" bash setup/eval_all.sh
