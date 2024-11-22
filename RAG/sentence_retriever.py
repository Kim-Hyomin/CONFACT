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

sentence_logger = logging.getLogger('sentence_retriever')

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data


class SentenceProcessor:
    def __init__(self):
        sentence_logger.info(f"initializing sentence splitter..")

    @staticmethod
    def split_into_sentences(text):
        abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'e.g', 'i.e']
        for abbr in abbreviations:
            text = re.sub(r'\b' + abbr + r'\.', abbr + '<ABBR>', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [sentence.replace('<ABBR>', '.') for sentence in sentences]
        return sentences

    def store_sentences(self, source_path, store_path):
        data = load_pkl(source_path)
        
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
        sentence_logger.info(f"Saving evidence to {store_path}...")

        with open(store_path, 'wb') as f:
            pickle.dump(all_evidence, f)
        sentence_logger.info("Finished storing sentences in .pkl file.")

class BM25Retriever:
    def __init__(self, k=100):
        self.k = k
        sentence_logger.info(f"Initialized BM25Retriever with k={k}")

    def retrieve_for_qn(self, item, evidence_data):
        question_id = item['id']
        question = item['question']
        
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

    def retrieve_for_file(self, qns_entities, evidence_file, output_file = "./results/retrieved_sentences.pkl"):
        
 
        sentence_logger.info(f"Loading evidence data from {evidence_file}...")
        sentence_logger.info(f"Retrieving evidence data...")

        evidence_data = load_pkl(evidence_file)


        results = []
        for qn in tqdm(qns_entities, desc="Retrieving evidence for each question"):
            result = self.retrieve_for_qn(qn, evidence_data)
            results.append(result)

        # [{
        #     'question_id': question_id,
        #     'question': question,
        #     'top_k': top_K,
        # }, {}, ...
        # ]

        with open(output_file, 'wb') as f:
            pickle.dump(results, f)

        return results

# def main():


#     parser = argparse.ArgumentParser(description='BM25 Retrieval and Sentence Splitting')
#     parser.add_argument('--source', default="../data/dataset/intermediate_datasets/sample.json", help='Path to qns')
#     parser.add_argument('--store_path', default="../data/dataset/evidence/evidence.pkl", help='Path to store evidence sentences in a .pkl file')
#     parser.add_argument('--k', default=100, type=int, help='Number of top relevant sentences to retrieve')
#     parser.add_argument('--output_file', default="../data/dataset/retrieved_sentences.pkl", help='Output file to save results')
    
#     args = parser.parse_args()

#     nltk.download('punkt_tab')

#     # Split sentences and store in .pkl file
#     sentence_processor = SentenceProcessor()
#     sentence_processor.store_sentences(args.source, args.store_path)

#     # Perform BM25 retrieval for each qn
#     retriever = BM25Retriever(k=args.k)

#     qns_entities = load_json(source_file)

#     retriever.retrieve_for_file(qns_entities, args.store_path)

# if __name__ == "__main__":
#     main()

