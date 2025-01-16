# UncertainQA

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

## Dataset Construction
### Searching for Evidence on Google

To search for evidence corresponding to a set of claims, use the following command:

```bash
python data/google_search.py --input dataset/averitec_data.csv --type Averitec --output dataset/claimWevidence.json
```

### Scraping Web Page Content
After collecting the URLs, scrape the content of the web pages using this command:

```bash
python data/web_scraping.py --input dataset/claimWevidence.json --output dataset/claimWevidence_txt.json
```

### GPT Annotation for Evidence Analysis
#### Labeling Evidence with GPT-4o
To identify conflicting evidence, GPT-4o is used for the first round of filtering. The agent labels each piece of evidence as supporting or refuting the claim.

To reduce costs, the batch API is used for sending requests:

```bash
python data/batch_submit.py --input claimWevidence.json --type majority_vote
```
A folder named in the format ``batch_submissionYYYY-MM-DD_HH_MM`` is created to record the submission ID and store related data.

#### Retrieving Results from GPT-4
To retrieve the sent results of GPT-4o processing, use the following command:

```bash
python data/batch_retrieve_results.py --source claimWevidence.json --folder batch_submission2024-12-01_04_12 --submission_time "2024-12-01-04-12"
```

The retrieved responses will be saved in the specified folder, organized under the provided submission details.

## Generate Media Description and Predict Credibility Label
To generate the media description and predict the credibility label, run the following command:

```bash
python mediaBG_check/main.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 2
```

## RAG
### Preprocess Questions with Evidence

Use the following command to preprocess questions and split the corresponding evidence into sentences or chunks. The processed and retrieved evidence will be stored in the `results` folder:

```bash
python preprocess.py --source ../data/dataset/MUQA.pkl --k 100 --type chunk --chunk_size 256
```

If using reranking method, continue processing the dataset using the following command:
```bash
python method/rerank.py --n 100 --type chunks --media_data all
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
To evaluate the outcomes (precision, recall, F1 score, etc.), run the following command:

```bash
python eval.py --prediction './results/results_mbfc_media/Top_5_AgentBased_MediaBD_true_model_Llama-3.1-8B-Instruct.json'
```
