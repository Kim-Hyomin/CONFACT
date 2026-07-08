#!/bin/bash
source env.sh
COMMON="--with_MediaBG true --media_data all --source data/dataset/HumC.pkl.gz --n 100 --k 5 --type chunks --model Qwen/Qwen2-7B-Instruct --gpu 1"

for M in ConflictLocExact ConflictLocClaim ConflictLocEvidence ConflictLocSoft; do
  echo "=== $M ==="
  python -u main.py --method $M $COMMON
done