#!/usr/bin/env bash
# Evaluate the trained policies on LIBERO-Object task 4 ("pick up the ketchup and place
# it in the basket") over LIBERO's 50 standard starting layouts, then print a table.
#
# Usage (from the repo root, env activated; can run while nothing else uses the GPU):
#   bash setup/eval_all.sh                         # final checkpoint of both runs
#   CKPTS="010000 020000" bash setup/eval_all.sh   # several checkpoints
#   RUNS="smolvla_phone" bash setup/eval_all.sh    # just one run
# Results: outputs/eval/<run>_<checkpoint>/eval_info.json (+ videos/), and
#          outputs/eval/results.md (the table, ready for the README)
set -uo pipefail

RUNS="${RUNS:-smolvla_phone smolvla_phone_spawn4 smolvla_scripted smolvla_phone_eval smolvla_phone_eval4 smolvla_scripted_eval smolvla_libero_ref}"
CKPTS="${CKPTS:-last}"
EPISODES="${EPISODES:-50}"
mkdir -p outputs/eval
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

for run in $RUNS; do
  for ck in $CKPTS; do
    model="outputs/train/$run/checkpoints/$ck/pretrained_model"
    out="outputs/eval/${run}_${ck}"
    if [[ ! -d "$model" ]]; then echo "skip: $model not found"; continue; fi
    if [[ -f "$out/eval_info.json" ]]; then echo "skip: $out already evaluated"; continue; fi
    # SmolVLA expects camera1/camera2; ACT uses the dataset's own camera names
    rename=('--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}')
    [[ "$run" == act_* ]] && rename=()
    echo ">> evaluating $run @ $ck"
    lerobot-eval \
      --policy.path="$model" \
      --env.type=libero --env.task=libero_object --env.task_ids='[4]' \
      --env.observation_height=256 --env.observation_width=256 \
      --eval.n_episodes="$EPISODES" --eval.batch_size=10 --seed=1000 \
      "${rename[@]}" \
      --output_dir="$out" 2>&1 | tee "$out.log"
  done
done

python - <<'EOF'
import json
from pathlib import Path

names = {"smolvla_phone": "A: demos from phone video",
         "smolvla_phone_spawn4": "A2: phone video, ketchup in spawn region x4",
         "smolvla_scripted": "C: scripted motion, same pipeline (control)",
         "smolvla_phone_eval": "D: phone video, ketchup at the eval spot, all 50 layouts",
         "smolvla_phone_eval4": "D4: as D, each demo replayed in 4 layouts (216 episodes)",
         "smolvla_scripted_eval": "C2: scripted motion, ketchup at the eval spot, all 50 layouts",
         "smolvla_phone_eval_n10": "D, re-planning every 10 steps",
         "smolvla_phone_eval_n5": "D, re-planning every 5 steps",
         "smolvla_phone_eval_40k": "D, trained 40k steps",
         "smolvla_phone_timed": "E: your paths, fixed smooth timing, eval spot, all layouts",
         "smolvla_phone_blend10": "G: average of your path shapes, eval spot, all layouts",
         "smolvla_phone_blend05": "G50: paths 50% towards the average, eval spot, all layouts",
         "act_phone_eval": "F: ACT (no language model) on D's data",
         "smolvla_phone_smooth": "H: as D, hand-tracking jitter smoothed out",
         "smolvla_phone_dart": "I: as H, + 3 recovery copies per demo (pushed off course), 40k steps",
         "smolvla_libero_ref": "B: LIBERO robot demos",
         "smolvla_libero_ref_n10": "B, re-planning every 10 steps"}
rows = []
order = list(names)
for f in sorted(Path("outputs/eval").glob("*/eval_info.json"),
              key=lambda f: (order.index(f.parent.name.rpartition("_")[0]) if f.parent.name.rpartition("_")[0] in order else 99, f.parent.name)):
    o = json.loads(f.read_text())["overall"]
    run, _, ck = f.parent.name.rpartition("_")
    lo, hi = o["pc_success_ci95"]
    rows.append(f"| {names.get(run, run)} | {ck} | {o['n_success']}/{o['n_episodes']} | "
                f"{o['pc_success']:.0f}% | {lo:.0f}-{hi:.0f}% |")
table = ["| Policy | Checkpoint | Successes | Success rate | 95% interval |",
         "|---|---|---|---|---|"] + rows
Path("outputs/eval/results.md").write_text("\n".join(table) + "\n")
print("\n".join(table))
EOF
