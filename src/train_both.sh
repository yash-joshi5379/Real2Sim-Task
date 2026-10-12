#!/usr/bin/env bash
# Train both SmolVLA policies, one after the other, with identical settings:
#   A  smolvla_phone       - the demos retargeted from phone video  (HF_USER/ketchup_from_phone)
#   B  smolvla_libero_ref  - LIBERO's own robot demos of the same task (lerobot/libero subset)
#
# Usage (from the repo root, inside tmux, env activated):
#   bash setup/train_both.sh YOUR_HF_USERNAME
#   STEPS=20000 BATCH=32 bash setup/train_both.sh YOUR_HF_USERNAME     # the defaults
#   ONLY=B bash setup/train_both.sh YOUR_HF_USERNAME                    # just one of them
#   WANDB=1 bash setup/train_both.sh YOUR_HF_USERNAME                   # live loss curves on wandb.ai
#   ONLY=A A_NAME=smolvla_phone_spawn4 A_REPO=local/ketchup_from_phone_spawn4 \
#     A_ROOT=data/lerobot/ketchup_from_phone_spawn4 bash setup/train_both.sh X  # A on a local dataset
#                                                                       # (run `wandb login` once first)
# Text logs (everything printed during training):
#   logs/smolvla_phone_train_log.txt, logs/smolvla_libero_ref_train_log.txt
set -uo pipefail

HF_USER="${1:?give your Hugging Face username, e.g. bash setup/train_both.sh yjoshi}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"
ONLY="${ONLY:-AB}"
WANDB="${WANDB:-0}"
A_REPO="${A_REPO:-$HF_USER/ketchup_from_phone}"   # dataset for run A
A_ROOT="${A_ROOT:-}"                              # local folder of that dataset (optional)
A_NAME="${A_NAME:-smolvla_phone}"                 # run name for A
A_ROOT_ARGS=(); [[ -n "$A_ROOT" ]] && A_ROOT_ARGS=(--dataset.root="$A_ROOT")
mkdir -p logs

if [[ "$WANDB" == "1" ]]; then
  # curves and system stats go to wandb.ai; checkpoints stay on this machine (no upload)
  WANDB_ARGS=(--wandb.enable=true --wandb.project=real2sim-ketchup --wandb.disable_artifact=true)
else
  WANDB_ARGS=(--wandb.enable=false)
fi

COMMON=(
  --policy.path=lerobot/smolvla_base
  '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
  --batch_size="$BATCH" --steps="$STEPS" --save_freq=5000 --log_freq=100 --seed=1000
  --policy.device=cuda --policy.push_to_hub=false "${WANDB_ARGS[@]}"
)

if [[ "$ONLY" == *A* ]]; then
  echo ">> A: training $A_NAME on the phone-video demos ($A_REPO)"
  lerobot-train "${COMMON[@]}" \
    --dataset.repo_id="$A_REPO" "${A_ROOT_ARGS[@]}" \
    --output_dir="outputs/train/$A_NAME" --job_name="$A_NAME" \
    2>&1 | tee "logs/${A_NAME}_train_log.txt"
fi

if [[ "$ONLY" == *B* ]]; then
  echo ">> B: finding LIBERO's own demos of the ketchup task"
  EPS_LINE=$(python src/reference_episodes.py ${REF_ARGS:-} | tee /dev/stderr | grep -- '--dataset.episodes=')
  EPS="${EPS_LINE#--dataset.episodes=}"
  if [[ -z "$EPS" ]]; then echo "could not get the episode list; stopping"; exit 1; fi
  echo ">> B: training on lerobot/libero episodes $EPS"
  lerobot-train "${COMMON[@]}" \
    --dataset.repo_id=lerobot/libero --dataset.episodes="$EPS" \
    --output_dir=outputs/train/smolvla_libero_ref --job_name=smolvla_libero_ref \
    2>&1 | tee logs/smolvla_libero_ref_train_log.txt
fi
echo ">> done"
