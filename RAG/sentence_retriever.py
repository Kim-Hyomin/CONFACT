import argparse
import os
import json
import re
import pickle
from tqdm import tqdm
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
import pickle as pkl
import logging


def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data


class SentenceProcessor:
    def __init__(self):
        logging.info(f"initializing sentence splitter..")

    @staticmethod
    def split_into_sentences(text):
        abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'e.g', 'i.e']
        for abbr in abbreviations:
            text = re.sub(r'\b' + abbr + r'\.', abbr + '<ABBR>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [sentence.replace('<ABBR>', '.') for sentence in sentences]
        return sentences

    def store_sentences(self, source, store_path):
        data = load_json(source)
        
        all_evidence = {}
        
        for qn in tqdm(data, desc="Storing sentences"):
            for evidence in qn['evidence_url']:
                evidence_id = evidence['evidence_id']
                evidence_url = evidence['original_link']
                sentences = self.split_into_sentences(evidence['content'])
                
                all_evidence[evidence_id] = {
                    'url': evidence_url,
                    'sentences': sentences
                }
        
        if not os.path.exists(os.path.dirname(store_path)):
            os.mkdir(os.path.dirname(store_path))
        logging.info(f"Saving evidence to {store_path}...")

        with open(store_path, 'wb') as f:
            pickle.dump(all_evidence, f)
        logging.info("Finished storing sentences in .pkl file.")

class BM25Retriever:
    def __init__(self, k=5):
        self.k = k
        logging.info(f"Initialized BM25Retriever with k={k}")

    def retrieve_for_qn(self, item, evidence_data):
        question_id = item['id']
        question = item['question']
        logging.info(f"Retrieving sentences for question: {question}")
        
        all_sentences = []
        sentence_corpus = []
        
        for evidence_id, content in evidence_data.items():
            if evidence_id.startswith(str(question_id) + '_'):
                url = content['url']
                sentences = content['sentences']
                all_sentences.extend(sentences)

                for sentence in sentences:
                    sentence_corpus.append({
                        'evidence_id': evidence_id,
                        'original_link': url,
                        'sentence': sentence
                    })

        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in all_sentences]
        bm25 = BM25Okapi(tokenized_sentences)

        qn_tokens = nltk.word_tokenize(question)
        scores = bm25.get_scores(qn_tokens)
        top_idx = np.argsort(scores)[::-1][:self.k]
        
        top_K = [
            {
                'sentence': sentence_corpus[i]['sentence'],
                'original_link': sentence_corpus[i]['original_link'],
                'score': scores[i]
            }
            for i in top_idx if scores[i] > 0
        ]

        return {
            'question_id': question_id,
            'question': question,
            'top_k': top_K,
        }

    def retrieve_for_file(self, source_file, evidence_file, output_file):
        logging.info(f"Loading file for BM25 retrieval from {source_file}...")
        qns = load_json(source_file)

        logging.info(f"Loading evidence data from {evidence_file}...")
        evidence_data = load_pkl(evidence_file)


        results = []
        for qn in tqdm(qns, desc="Processing each question"):
            result = self.retrieve_for_qn(qn, evidence_data)
            results.append(result)

        logging.info(f"Saving retrieval results to {output_file}.")

        with open(output_file, 'w') as f:
            json.dump(results, f)

        logging.info("BM25 retrieval completed and results saved.")

def main():
    parser = argparse.ArgumentParser(description='BM25 Retrieval and Sentence Splitting')
    parser.add_argument('--source', default="../data/dataset/intermediate_datasets/sample.json", help='Path to qns')
    parser.add_argument('--store_path', default="../data/dataset/evidence/evidence.pkl", help='Path to store evidence sentences in a .pkl file')
    parser.add_argument('--k', default=5, type=int, help='Number of top relevant sentences to retrieve')
    parser.add_argument('--output_file', default="../data/dataset/retrieved_sentences.json", help='Output file to save results')
    
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)

    # Split sentences and store in .pkl file
    sentence_processor = SentenceProcessor()
    sentence_processor.store_sentences(args.source, args.store_path)

    # Perform BM25 retrieval for each qn
    retriever = BM25Retriever(k=args.k)
    retriever.retrieve_for_file(args.source, args.store_path, args.output_file)

if __name__ == "__main__":
    main()

