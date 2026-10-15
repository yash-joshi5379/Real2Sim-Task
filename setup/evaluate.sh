#!/usr/bin/env bash
# Evaluate a trained policy on LIBERO-Object task 4 over LIBERO's 50 standard layouts.
#
# Usage (repo root, env active, Linux):
#   bash setup/evaluate.sh smolvla_phone_blend10              # final checkpoint
#   bash setup/evaluate.sh smolvla_phone_blend10 010000 last  # several checkpoints
#   MODEL=path/or/hf_repo bash setup/evaluate.sh my_name      # a model from elsewhere
# Writes outputs/eval/<run>_<checkpoint>/eval_info.json and videos/ (first 10 episodes).
set -eo pipefail
RUN="${1:?usage: bash setup/evaluate.sh <run name> [checkpoints...]}"; shift
CKPTS="${*:-last}"

for ck in $CKPTS; do
  model="${MODEL:-outputs/train/$RUN/checkpoints/$ck/pretrained_model}"
  out="outputs/eval/${RUN}_${ck}"
  if [ -z "$MODEL" ] && [ ! -d "$model" ]; then echo "skip: no checkpoint at $model"; continue; fi
  if [ -f "$out/eval_info.json" ]; then echo "skip: $out already evaluated"; continue; fi
  # SmolVLA was fine-tuned with camera names camera1/camera2; ACT uses the dataset's own names
  rename=('--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}')
  [[ "$RUN" == act_* ]] && rename=()
  echo ">> evaluating $RUN @ $ck"
  lerobot-eval --policy.path="$model" --policy.device="${DEVICE:-cuda}" \
    --env.type=libero --env.task=libero_object --env.task_ids='[4]' \
    --env.observation_height=256 --env.observation_width=256 \
    --eval.n_episodes=50 --eval.batch_size=10 --seed=1000 \
    "${rename[@]}" --output_dir="$out"
done
