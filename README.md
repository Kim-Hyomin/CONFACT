# fact-checking  

## Files in `./RAG`

### Codes

#### `split_sentences.py`
split sentences from strings in the claim file into a text file. Resulted txt files would be saved in a newly created folder, the default path of which is `./data/dataset/evidence`.

#### `sentence_retriver.py`
Given a claim and its corresponding evidence (split into sentences), the code retrieves the most relevant sentences related to the claim. Claims can be reformulated as binary questions if needed. (see codes: question = claim['question'])

## Files in `./data`

### Source Datasets
1. AVeriTec dataset
2. FactCheckQA dataset

### Datasets ready-to-use for testing framework
a small sample dataset (50 claims) `./data/dataset/intermediate_datasets/sample.json`

### Codes

#### `google_search.py`
processes a JSON file of claims and searches for related information, outputting URLs in JSON by directly scraping google search results.

#### `APIretriever.py`
processes a JSON file of claims and searches for related information, outputting URLs in JSON format using API requests.

#### `web_scraping.py`
Scrapes content from the specified URLs.

#### `web_archive.py`
Backup web pages using the Wayback Machine API.

#### `batch_submit.py`
Uses a language model to determine if the evidence supports or refutes the corresponding claim. Sending data by Batch API.

#### `batch_retrieve_results.py`
Retrieve results from the Batch API and parse the results.




