import json
import os
from rank_bm25 import BM25Okapi
import nltk
from tqdm import tqdm
import argparse
import numpy as np

# python sentence_retriever.py 

parser = argparse.ArgumentParser(description='BM25 Retrieval')
parser.add_argument('--claims_file', default="../data/dataset/intermediate_datasets/sample.json", help='File with claims')
parser.add_argument('--evidence_folder', default="../data/dataset/evidence", help='Folder with evidence sentences')
parser.add_argument('--k', default=5, type=int, help='Number of top relevant sentences to retrieve')
parser.add_argument('--output_file', default="../data/dataset/retrieved_sentences.json", help='Output file to save results')


args = parser.parse_args()


with open(args.claims_file, 'r') as f:
    claims = json.load(f)



results = []


for claim in tqdm(claims):
    claim_text = claim['claim']
    claim_id = claim['id']

    all_sentences = []
    sentence_corpus = []

    for filename in os.listdir(args.evidence_folder):
        if filename.startswith(str(claim_id)+'_'):
            with open(os.path.join(args.evidence_folder, filename), 'r') as f:
                url = f.readline().strip()  # extract the URL

                f.readline()  # skip newline
                f.readline() 

                # read the remaining sentences
                sentences = f.readlines()
                sentences = [sentence.strip() for sentence in sentences if sentence.strip()] 
                all_sentences.extend(sentences)

                for sentence in sentences:
                    sentence_corpus.append({
                        'evidence_id': str(filename),
                        'original_link': url,
                        'sentence': sentence
                    })

  
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in all_sentences]
    bm25 = BM25Okapi(tokenized_sentences)
   
    claim_tokens = nltk.word_tokenize(claim_text)
    scores = bm25.get_scores(claim_tokens)
    top_idx = np.argsort(scores)[::-1][:args.k]
    # top_K = [sentence_corpus[i] for i in top_idx]
    top_K = []
    
    for i in top_idx:
        if scores[i] > 0:  # Only include sentences with a non-zero score
            top_K.append({
                'sentence': sentence_corpus[i]['sentence'],
                'original_link': sentence_corpus[i]['original_link'],
                'score': scores[i]
            })

    results.append({
        'claim_id': claim_id,
        'claim_text': claim_text,
        'top_k': top_K,
    })


with open(args.output_file, 'w') as f:
    json.dump(results, f)
