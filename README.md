# fact-checking  

## Files in `./data`

### Source Datasets
1. AVeriTec dataset
2. FactCheckQA dataset

filtered dataset in `./data/dataset`

### Codes

#### `retriever.py`
processes a JSON file of claims and searches for related information, outputting URLs in JSON format.
```bash
python retriever.py --input input_json_file --output output_file --n number_of_evidence_retrieved_per_claim  
```

#### `web_scraping.py`
Scrapes content from the specified URLs.

#### `reranking.py`
Reranks the retrieved content using the BM25 algorithm.

#### `LLM_checker.py`
Uses a language model to determine if the evidence supports or refutes the corresponding claim.



