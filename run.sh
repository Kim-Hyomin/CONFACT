#!/bin/bash
# ============================================================
# CONFACT + ConflictLoc 통합 실행 스크립트
#
# 사용법:
#   ./run.sh                          # 전체: 2 split × 3 model × 주요 method (GPU 병렬)
#   ./run.sh HumC                     # HumC만, 3 model 병렬
#   ./run.sh ModC Qwen2               # ModC · Qwen2만 (단일 GPU)
#   ./run.sh HumC Qwen2 ConflictLocSoft   # 특정 조합 하나만
#
# 인자:
#   $1 = split  : HumC | ModC        (기본: 둘 다)
#   $2 = model  : Qwen2 | Llama | Mistral   (기본: 셋 다, 병렬)
#   $3 = method : Explain | ConflictLocSoft | ConflictLocEvidence | ... (기본: 주요 3종)
# ============================================================
source env.sh

# ---- 기본 설정 ----
N=100; K=5; TYPE=chunks; MEDIA=all; MEDIABG=true

# 모델 별칭 -> 실제 HF 경로
declare -A MODELS=(
  [Qwen2]="Qwen/Qwen2-7B-Instruct"
  [Llama]="meta-llama/Llama-3.1-8B-Instruct"
  [Mistral]="mistralai/Mistral-7B-Instruct-v0.3"
)
# 모델 -> GPU 번호 (병렬 실행용)
declare -A GPUS=( [Qwen2]=0 [Llama]=1 [Mistral]=2 )

# ---- 인자 파싱 ----
SPLITS=(${1:-HumC ModC})
MODEL_KEYS=(${2:-Qwen2 Llama Mistral})
METHODS=(${3:-Explain ConflictLocSoft ConflictLocEvidence})

# 한 (split, model)에 대해 method들을 순차 실행하는 함수
run_one () {
  local split=$1 mkey=$2 gpu=$3
  local src="data/dataset/${split}.pkl.gz"
  local model="${MODELS[$mkey]}"
  for method in "${METHODS[@]}"; do
    echo "===== [$split / $mkey / $method] (GPU $gpu) ====="
    CUDA_VISIBLE_DEVICES=$gpu python -u main.py \
      --method "$method" --with_MediaBG $MEDIABG --media_data $MEDIA \
      --source "$src" --n $N --k $K --type $TYPE --model "$model" --gpu 1
  done
}

# ---- 실행 ----
for split in "${SPLITS[@]}"; do
  # 전처리: split별 retrieved pkl이 없으면 생성
  if [ ! -f "./results/top${N}_retrieved_${TYPE}_${split}.pkl" ]; then
    echo ">>> preprocess $split"
    python preprocess.py --source "data/dataset/${split}.pkl.gz" --n $N --type $TYPE --chunk_size 256
  fi

  if [ ${#MODEL_KEYS[@]} -gt 1 ]; then
    # 여러 모델 -> GPU 병렬
    for mkey in "${MODEL_KEYS[@]}"; do
      run_one "$split" "$mkey" "${GPUS[$mkey]}" &
    done
    wait
  else
    # 단일 모델 -> 그냥 실행 (GPU 0)
    run_one "$split" "${MODEL_KEYS[0]}" 0
  fi
done

echo "=== 전부 완료 ==="
echo ">>> 평가:"
python eval.py --folder ./results/results_${MEDIA}_media