from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from web_scraping import parse_html
import json
from tqdm import tqdm

def bm25_rerank(query, documents):

    # tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
    tokenized_documents = [word_tokenize(doc['content'].lower()) for doc in documents]
   
    tokenized_query = word_tokenize(query.lower())

    bm25 = BM25Okapi(tokenized_documents)

    scores = bm25.get_scores(tokenized_query)

    ranked_docs = sorted(zip(scores, documents), key=lambda x: x[0],reverse=True)

    return ranked_docs

# def save_to_json(ranked_documents, filename):
#     data = [{'score': score, 'document': doc} for score, doc in ranked_documents]
#     with open(filename, 'w') as f:
#         json.dump(data, f, indent=4)

def save_to_json(ranked_documents, filename):
    data = [
        {'score': score, 'document': doc['content'], 'original_rank': doc['original_rank']}
        for score, doc in ranked_documents
    ]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

        
with open('./outputs/test.json', 'r') as f:
    data = json.load(f) 


filename = f'./outputs/test_reranked.json'
with open(filename, 'w') as outfile:
    json.dump([], outfile)

for item in tqdm(data, desc="Processing claims"):
    claim = item['claim']
    retrieved_results = item['retrieved results']

    documents = []
    for evidence in retrieved_results:
        original_rank = evidence['rank']
        url = evidence['url']
        content = parse_html(url)
        documents.append({'original_rank': original_rank, 'url': url, 'content': content})

    ranked_docs = bm25_rerank(claim, documents)


    formatted_ranked_docs = []
    for bm25_rank, (score, doc) in enumerate(ranked_docs):
        formatted_ranked_docs.append({
            'score': score,
            'url': doc['url'],
            'original_rank': doc['original_rank'],
            'bm25_rank': bm25_rank + 1,
            'content': doc['content']
        })

    ranked_data = {
        'claim': claim,
        'evidence': formatted_ranked_docs
    }


    with open(filename, 'r+') as outfile:
        existing_data = json.load(outfile)
        existing_data.append(ranked_data)
        outfile.seek(0)
        json.dump(existing_data, outfile, indent=4)




