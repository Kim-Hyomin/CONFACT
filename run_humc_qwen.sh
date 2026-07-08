#!/bin/bash
# 공통 환경변수 (이번 세션에서 확정된 조합)
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_P2P_DISABLE=1
export VLLM_HOST_IP=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0

MODEL="Qwen/Qwen2-7B-Instruct"
SRC="data/dataset/HumC.pkl.gz"
COMMON="--source $SRC --n 100 --k 5 --type chunks --model $MODEL --gpu 1"

echo "=== [1/3] CoT (baseline) ==="
python -u main.py --method CoT --with_MediaBG false --media_data all $COMMON

echo "=== [2/3] SBACoT (CoT + MediaBG) ==="
python -u main.py --method CoT --with_MediaBG true --media_data all $COMMON

echo "=== [3/3] SBAexp (Explain + MediaBG) ==="
python -u main.py --method Explain --with_MediaBG true --media_data all $COMMON

echo "=== 평가 ==="
python eval.py --folder ./results/results_all_media