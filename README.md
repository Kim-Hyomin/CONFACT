# CONFACT + ConflictLoc

Extending **CONFACT** with lightweight, prompt-based **conflict localization** for credibility-aware RAG fact-checking.

This repository is a fork of [zoeyyes/CONFACT](https://github.com/zoeyyes/CONFACT) (Ge et al., IJCAI 2025). It keeps the original benchmark and pipeline, and adds a prompt-based method that checks *claim–evidence exactness* and *evidence–evidence conflicts* before the credibility-weighted verdict — without any agentic structure or extra training.

---

## What's new in this fork

| Addition | Where | Description |
|---|---|---|
| **ConflictLoc prompt methods** | `method/prompt_processor.py`, `config.py`, `main.py` | New single-prompt methods (`ConflictLocSoft`, `ConflictLocEvidence`, `ConflictLocExact`, `ConflictLocClaim`, and a full 4-step `ConflictLoc`) that insert lightweight conflict-localization checks into the SBAexp prompt. Same number of LLM calls as SBAexp (one per claim). |
| **Robust answer parsing** | `inference.py` | Fixed the `Final Answer` regex so it captures the model's actual verdict (not the prompt text) and handles model-specific phrasings such as `final answer is no.` |
| **Blackwell-GPU support** | `inference.py`, `env.sh` | Runs on NVIDIA RTX PRO 6000 Blackwell (sm_120) with vLLM 0.17 + torch 2.10+cu128, `enforce_eager`, greedy decoding, and the required NCCL / FlashInfer environment variables. |
| **Split-safe outputs** | `main.py`, `preprocess.py` | Result and retrieval files now include the split name (`HumC` / `ModC`), so running one split never overwrites the other. |
| **Unified runner** | `run.sh` | One script to preprocess, run (with optional multi-GPU parallelism across models), and evaluate. |
| **Evaluation upsert** | `eval.py` | `evaluation_results.csv` now updates existing rows and appends new ones instead of blindly accumulating duplicates. |
| **Analysis scripts** | `compare.py`, `compare_by_model.py` | Per-method / per-model comparison of Accuracy, Macro-F1, empty-prediction counts, and confusion (no→yes) for quick inspection. |

The original CONFACT README content (data format, media-background generation, reranking) is preserved below.

---

## Setup

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. (Blackwell GPUs) load the required environment variables each session
source env.sh
```

`env.sh` sets the environment variables needed to run vLLM on Blackwell (sm_120) hardware:

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_P2P_DISABLE=1
export VLLM_HOST_IP=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0
```

Gated models (Llama-3.1, Mistral) require a Hugging Face token with access granted:

```bash
export HF_TOKEN=your_token_here   # do not commit this
```

---

## Quick start

The unified runner handles preprocessing, inference, and evaluation.

```bash
chmod +x run.sh

# full sweep: 2 splits × 3 models × main methods (models run in parallel across GPUs)
./run.sh

# a single split, 3 models in parallel
./run.sh HumC

# a single (split, model)
./run.sh ModC Qwen2

# one specific combination
./run.sh HumC Qwen2 ConflictLocSoft
```

Arguments:

- `$1` split: `HumC` | `ModC` (default: both)
- `$2` model: `Qwen2` | `Llama` | `Mistral` (default: all three, run in parallel)
- `$3` method: `Explain` (= SBAexp baseline) | `ConflictLocSoft` | `ConflictLocEvidence` | `ConflictLocExact` | `ConflictLocClaim` | `CoT` | `DirectAnswer` | ... (default: `Explain ConflictLocSoft ConflictLocEvidence`)

Results are written to `./results/results_all_media/` with split-prefixed filenames, e.g.
`HumC_Top_5_chunks_ConflictLocEvidence_MediaBD_True_model_Qwen2-7B-Instruct.json`.

### Manual run (equivalent to what run.sh does internally)

```bash
# preprocess a split (creates ./results/top100_retrieved_chunks_<split>.pkl)
python preprocess.py --source data/dataset/HumC.pkl.gz --n 100 --type chunks --chunk_size 256

# run one method
python main.py --method ConflictLocEvidence --with_MediaBG true --media_data all \
  --source data/dataset/HumC.pkl.gz --n 100 --k 5 --type chunks \
  --model Qwen/Qwen2-7B-Instruct --gpu 1

# evaluate a results folder
python eval.py --folder ./results/results_all_media
```

---

## Methods

Baselines (from the original CONFACT):

- `DirectAnswer`, `CoT` — no credibility information
- `Explain` with `--with_MediaBG true` — **SBAexp**, the strongest baseline and the direct comparison target

ConflictLoc (this fork):

- `ConflictLocEvidence` — evidence–evidence conflict check only *(most robust; highest Macro-F1 in our experiments)*
- `ConflictLocSoft` — exact-match + evidence-conflict check, softened to avoid over-correction *(best Accuracy)*
- `ConflictLocExact` / `ConflictLocClaim` — single-element ablations
- `ConflictLoc` — full 4-step version *(kept for ablation; underperforms due to reasoning overload)*

All ConflictLoc methods keep the SBAexp prompt and add only short checking instructions, so they use the same one-call-per-claim budget as SBAexp.

---

## Analysis

```bash
# per-method summary (Acc, Macro-F1, empty count, no->yes bias)
python compare.py

# grouped by model, comparing SBAexp vs ConflictLoc variants
python compare_by_model.py
```

---

## Notes on reproduction

- Decoding is greedy (`temperature=0`) for reproducibility and fair comparison.
- Credibility-predictor checkpoints are not included upstream, so credibility-weighted reranking (CW) methods are out of scope here; this fork focuses on the generation-stage methods (Baseline / SBA / ConflictLoc).
- Retrieval and result files are split-tagged, so `HumC` and `ModC` can be run in any order without overwriting each other.

---

## Attribution

This work builds directly on **CONFACT**:

> Ge, Z., Wu, Y., Chin, D. W. K., Lee, R. K.-W., & Cao, R. (2025). *Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs.* IJCAI 2025. arXiv:2505.17762.

Original repository: https://github.com/zoeyyes/CONFACT

Please cite the original CONFACT paper when using this benchmark. The additions in this fork (ConflictLoc prompts, parsing fixes, Blackwell setup, unified runner) were developed as part of an undergraduate research project (UROP).

---

<details>
<summary>Original CONFACT README (preserved)</summary>

*See the upstream repository for the original data-format description, media-background generation, and reranking instructions:* https://github.com/zoeyyes/CONFACT

</details>
