#!/bin/bash
source env.sh
unset CUDA_VISIBLE_DEVICES   # env.sh의 고정값 해제 (아래서 모델별로 지정)
COMMON="--with_MediaBG true --media_data all --source data/dataset/ModC.pkl.gz --n 100 --k 5 --type chunks --gpu 1"

run_model () {  # $1=GPU번호, $2=모델
  for METHOD in Explain ConflictLocSoft ConflictLocEvidence; do
    echo "===== GPU$1 / $2 / $METHOD ====="
    CUDA_VISIBLE_DEVICES=$1 python -u main.py --method "$METHOD" --model "$2" $COMMON
  done
}

run_model 0 "Qwen/Qwen2-7B-Instruct" > log_qwen.txt 2>&1 &
run_model 1 "meta-llama/Llama-3.1-8B-Instruct" > log_llama.txt 2>&1 &
run_model 2 "mistralai/Mistral-7B-Instruct-v0.3" > log_mistral.txt 2>&1 &
wait
echo "=== 전부 완료 ==="