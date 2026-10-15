#!/usr/bin/env bash
# Reproduce one row of the results table: build its dataset, train, evaluate.
#
# Usage (repo root, env active; a CUDA GPU for training):
#   bash setup/run_experiment.sh G             # the best run
#   STEPS=2000 bash setup/run_experiment.sh G  # quick smoke test
#   WANDB=1 ...                                # log curves to Weights & Biases (run `wandb login` once)
#   DEVICE=mps ...                             # Apple GPU (training only; evaluation needs Linux)
#
# Runs:  A  A2  B  C  C2  D  D4  D40k  E  F  G  H   (what each one is: see the README)
# Output: data/lerobot/ketchup_<data>/  outputs/train/<run>/  outputs/eval/<run>_<checkpoint>/
set -eo pipefail
EXP="${1:?usage: bash setup/run_experiment.sh <A|A2|B|C|C2|D|D4|D40k|E|F|G|H>}"
EVAL_SPOT="--placement eval --init-state all"
POLICY=smolvla; STEPS="${STEPS:-}"; CKPTS="010000 last"

case "$EXP" in
  A)    DATA=phone;         ARGS="--placement human" ;;
  A2)   DATA=phone_spawn4;  ARGS="--placement spawn --augment 3" ;;
  C)    DATA=scripted;      ARGS="--motion scripted --placement human" ;;
  C2)   DATA=scripted_eval; ARGS="--motion scripted $EVAL_SPOT" ;;
  D)    DATA=phone_eval;    ARGS="$EVAL_SPOT" ;;
  D4)   DATA=phone_eval4;   ARGS="$EVAL_SPOT --augment 3" ;;
  D40k) DATA=phone_eval;    ARGS="$EVAL_SPOT"; RUN=smolvla_phone_eval_40k; STEPS="${STEPS:-40000}" ;;
  E)    DATA=phone_timed;   ARGS="--motion timed $EVAL_SPOT" ;;
  F)    DATA=phone_eval;    ARGS="$EVAL_SPOT"; POLICY=act; RUN=act_phone_eval; STEPS="${STEPS:-30000}" ;;
  G)    DATA=phone_blend10; ARGS="$EVAL_SPOT --blend 1.0" ;;
  H)    DATA=phone_smooth;  ARGS="$EVAL_SPOT --smooth 0.75" ;;
  B)    DATA=""; RUN=smolvla_libero_ref; CKPTS="010000 015000 last" ;;
  *)    echo "unknown run '$EXP'"; exit 1 ;;
esac
RUN="${RUN:-smolvla_$DATA}"
STEPS="${STEPS:-20000}"
mkdir -p logs

# 1. dataset (B uses LIBERO's own demos instead, downloaded by lerobot-train)
if [ -n "$DATA" ]; then
  if [ -f "data/lerobot/ketchup_$DATA/meta/info.json" ]; then
    echo ">> dataset data/lerobot/ketchup_$DATA exists, reusing it"
  else
    echo ">> building data/lerobot/ketchup_$DATA  ($ARGS)"
    python src/make_dataset.py $ARGS --out "data/lerobot/ketchup_$DATA" --repo-id "local/ketchup_$DATA" \
      2>&1 | tee "logs/make_dataset_$DATA.txt"
  fi
  DATASET=(--dataset.repo_id="local/ketchup_$DATA" --dataset.root="data/lerobot/ketchup_$DATA")
else
  EPS=$(python src/reference_episodes.py | grep -- '--dataset.episodes=' | cut -d= -f2)
  DATASET=(--dataset.repo_id=lerobot/libero --dataset.episodes="$EPS")
fi

# 2. training (same settings for every SmolVLA run)
if [ "$POLICY" = act ]; then
  POLICY_ARGS=(--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50)
  SAVE=10000
else
  POLICY_ARGS=(--policy.path=lerobot/smolvla_base
    '--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}')
  SAVE=5000
fi
if [ "${WANDB:-0}" = 1 ]; then
  WB=(--wandb.enable=true --wandb.project=real2sim-ketchup --wandb.disable_artifact=true)
else
  WB=(--wandb.enable=false)
fi
echo ">> training $RUN ($STEPS steps)"
lerobot-train "${POLICY_ARGS[@]}" "${DATASET[@]}" "${WB[@]}" \
  --batch_size=32 --steps="$STEPS" --save_freq="$SAVE" --log_freq=100 --seed=1000 \
  --policy.device="${DEVICE:-cuda}" --policy.push_to_hub=false \
  --output_dir="outputs/train/$RUN" --job_name="$RUN" 2>&1 | tee "logs/${RUN}_train.txt"

# 3. evaluation on LIBERO's 50 layouts, then the results table
bash setup/evaluate.sh "$RUN" $CKPTS
python src/results.py
