# CONFACT
This repository maintains the dataset and implementation for the experiments described in our paper: Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs

## Data Format
The dataset consists of a list of JSON objects.
```json
[
  {
    "id": 1,
    "claim": "Nigeria has an estimated physician-patient ratio of one doctor to every 4,000 to 5,000 patients.",
    "label": "Supported",
    "claim_date": "2019-10-10",
    "review_date": null,
    "country": "NG",
    "question": "Does Nigeria have an estimated physician-patient ratio of one doctor to every 4,000 to 5,000 patients?",
    "original_claim_url": "https://web.archive.org/web/20201217182533/https://www.aljazeera.com/economy/2019/10/2/nigeria-has-a-mental-health-problem",
    "fact_checking_article": "https://web.archive.org/web/20210127230319/https://africacheck.org/fact-checks/reports/fact-checked-al-jazeeras-claims-about-nigerias-mental-health-problem",
    "evidence_url": [
      {
        "evidence_id": "1_1",
        "original_link": "https://www.theguardian.com/global-development/2023/aug/14/africa-health-worker-brain-drain-acc#:~:text=In%20Nigeria%2C%20there%20is%20one,for%20about%20every%20254%20people.",
        "content": "..."
      }
    ]
  }
]
```
## Requirements


## Generate Media Description and Predict Credibility Label
To get prepared for media background check, collect information using the following information:

```bash
cd mediaBG_check & python article_collection.py & python wiki_collection.py & python google_search.py
```

To generate the media description and predict the credibility label, run the following command:

```bash
python main.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 2
```

## RAG
### Preprocess Questions with Evidence

Use the following command to preprocess questions and split the corresponding evidence into sentences or chunks. The processed and retrieved evidence will be stored in the `results` folder:

```bash
python preprocess.py --source data/dataset/ModC.pkl.gz --k 100 --type chunk --chunk_size 256
```

If using reranking method, continue processing the dataset using the following command:
```bash
cd method & python rerank.py --n 100 --type chunks --media_data all
```

### Prediction
To run experiments using different methods, use the following command:

```bash
python main.py --method "${method}" --k "${k}" --with_MediaBG "${media}" --model "${model}" --gpu 2 --media_data all
```
- **method**: Choose from ["DirectAnswer", "DiscernAndAnswer", "ExplainAndAnswer", "MajorityVoting", "AgentBased", "Filter", "RerankSoft", "RerankHard"].
- **with_MediaBG**: Specify whether to use media background knowledge when evaluating the evidence.
- **media_data**: Choose from ["mbfc", "all"]. This determines whether to use only MBFC information (mbfc) or all available information (both MBFC data and generated media background data).

Results will be saved in the results folder (e.g., ``./results/results_mbfc_media`` or ``./results/results_all_media``).

### Evaluation of Results
To evaluate the outcomes (precision, recall, F1 score, etc.), run the following command to get the evaluation metrics within the specific folder:

```bash
python eval.py --folder './results/results_all_media'
```
