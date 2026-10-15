#!/usr/bin/env bash
# Upload the phone clips, the training datasets of runs D and G and the trained models
# to the Hugging Face Hub (public), under the names the README links to.
#
# Usage (repo root, after `hf auth login` with a write token):
#   bash setup/publish_hf.sh HF_USERNAME
# Uploads whatever exists locally and skips the rest, so it can run on the machine with
# the clips (data/clips) and again on the GPU machine with the datasets and checkpoints.
set -eo pipefail
U="${1:?usage: bash setup/publish_hf.sh HF_USERNAME}"

up() {  # up <local dir> <repo> <type>
  if [ -d "$1" ]; then
    echo ">> $1 -> huggingface.co/$([ "$3" = dataset ] && echo datasets/)$U/$2"
    hf upload "$U/$2" "$1" . --repo-type="$3" --commit-message "upload from Real2Sim-Task"
  else
    echo "   (skipped $2: no $1 here)"
  fi
}

up data/clips                         real2sim-ketchup-phone-clips dataset   # 54 phone demos (mp4)
up data/lerobot/ketchup_phone_eval    real2sim-ketchup-D           dataset   # each demo's own path
up data/lerobot/ketchup_phone_blend10 real2sim-ketchup-G           dataset   # averaged path shape
up outputs/train/smolvla_phone_blend10/checkpoints/last/pretrained_model smolvla-real2sim-ketchup-G model
up outputs/train/smolvla_phone_eval/checkpoints/last/pretrained_model    smolvla-real2sim-ketchup-D model
