#!/usr/bin/env bash
# Run at the END of a GPU session, before it expires. Saves everything the next session
# (possibly on a different machine, with /scratch0 wiped) needs:
#   1. exact software versions: setup/environment_ucl.yml, setup/requirements_lock_ucl.txt,
#      setup/lerobot_commit.txt  (used by setup/install_ucl.sh next time)
#   2. results: outputs/eval/*/eval_info.json, outputs/eval/results.md, logs/*.txt
#      -> committed to git (small text files)
#   3. trained models (final checkpoint of each run) and evaluation videos
#      -> uploaded to Hugging Face (needs `hf auth login` with a write token)
#
# Usage (repo root, env active):  bash setup/backup_session.sh YOUR_HF_USERNAME
set -uo pipefail
HF_USER="${1:?give your Hugging Face username}"
LEROBOT_DIR="${LEROBOT_DIR:-/scratch0/yjoshi/lerobot}"

echo ">> 1. recording software versions"
conda env export --no-builds > setup/environment_ucl.yml
pip list --format=freeze --exclude-editable > setup/requirements_lock_ucl.txt   # name==version only
git -C "$LEROBOT_DIR" rev-parse HEAD > setup/lerobot_commit.txt
echo "   lerobot commit: $(cat setup/lerobot_commit.txt)"

echo ">> 2. committing results and logs to git"
git add setup/environment_ucl.yml setup/requirements_lock_ucl.txt setup/lerobot_commit.txt
git add -f outputs/eval/*/eval_info.json outputs/eval/results.md logs/*.txt 2>/dev/null
git commit -m "Session backup: results, logs, environment" || echo "   (nothing new to commit)"
if ! git push; then
  echo "   git push failed (no GitHub login on this machine?) - the same files are uploaded to"
  echo "   Hugging Face below; or run: git push  after setting up a GitHub token"
fi

echo ">> 3. uploading models and eval videos to Hugging Face"
if ! hf auth whoami >/dev/null 2>&1; then
  echo "   not logged in - run: hf auth login   (write token), then rerun this script"; exit 1
fi
for dir in outputs/train/*/; do
  run=$(basename "$dir")
  ck=$(ls -d "$dir"checkpoints/0*/ 2>/dev/null | sort | tail -1)
  [[ -z "$ck" ]] && continue
  echo "   $run  <- $(basename "$ck")"
  hf upload "$HF_USER/$run" "${ck}pretrained_model" . --repo-type=model --private \
    --commit-message "$run $(basename "$ck")" || echo "   upload of $run failed"
done
hf upload "$HF_USER/real2sim-ketchup-results" . . --repo-type=dataset --private \
  --include "outputs/eval/**" --include "logs/**" --include "setup/*ucl*" --include "setup/lerobot_commit.txt" \
  --commit-message "session backup" || echo "   results upload failed"
echo ">> done. Models: huggingface.co/$HF_USER/<run name>; results: datasets/$HF_USER/real2sim-ketchup-results"
