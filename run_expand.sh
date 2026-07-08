#!/bin/bash
source env.sh
COMMON="--with_MediaBG true --media_data all --source data/dataset/HumC.pkl.gz --n 100 --k 5 --type chunks --gpu 1"

for MODEL in "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3"; do
  for METHOD in Explain ConflictLocSoft ConflictLocEvidence; do
    echo "===== $MODEL / $METHOD ====="
    python -u main.py --method "$METHOD" --model "$MODEL" $COMMON
  done
done