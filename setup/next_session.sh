#!/usr/bin/env bash
# Everything for the next UCL GPU session, in one go (about 3.5 hours; run inside tmux):
#   D   phone-video motion, ketchup exactly where evaluation puts it, all 50 LIBERO layouts
#   C2  the same with scripted motion (control)
# then evaluates both and backs up. Usage (repo root, env active):
#   bash setup/next_session.sh YOUR_HF_USERNAME
set -eo pipefail
HF_USER="${1:?give your Hugging Face username}"
bash setup/run_experiment.sh "$HF_USER" phone_eval --placement eval --init-state all
bash setup/run_experiment.sh "$HF_USER" scripted_eval --motion scripted --placement eval --init-state all
cat outputs/eval/results.md   # the table with every run evaluated so far
bash setup/backup_session.sh "$HF_USER"
