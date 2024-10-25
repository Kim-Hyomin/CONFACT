import argparse
import json
import re
import os
from tqdm import tqdm

#python split_sentences.py --n_compute 50

parser = argparse.ArgumentParser(description='Sentence Splitting')
parser.add_argument('--source', default="../data/dataset/intermediate_datasets/sample.json", help='the path to claims')
parser.add_argument('--store_folder', default="../data/dataset/evidence", help='the folder where all evidenc sentences to be stored.')
parser.add_argument('--start_idx', default=0, type=int, help='the position where the processing to start at')
parser.add_argument('--n_compute', default=0, type=int, help='the amount of claims to process in this round')

args = parser.parse_args()

with open(args.source, 'r') as f:
    data = json.load(f)

if not os.path.exists(args.store_folder):
    os.makedirs(args.store_folder)


def split_into_sentences(text):
    # Define abbreviations that should not split the sentence
    abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'e.g', 'i.e']
    for abbr in abbreviations:
        text = re.sub(r'\b' + abbr + r'\.', abbr + '<ABBR>', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Replace the placeholder back with the period
    sentences = [sentence.replace('<ABBR>', '.') for sentence in sentences]
    
    return sentences


end_idx = -1
if args.n_compute != -1:
    end_idx = args.start_idx+args.n_compute


# #create/extend a mapping for all evidence
# if os.path.exists(os.path.join(args.store_folder, "evidence_mapping.json")):
#     with open(os.path.join(args.store_folder, "evidence_mapping.json"), 'r') as f:
#         evidence_mapping = json.load(f)
# else:
#     evidence_mapping = []

for claim in tqdm(data[args.start_idx:end_idx]):
    for evidence in claim['evidence_url']:
        id=evidence['evidence_id']
        evidence_url = evidence['original_link'] 
        sentences = split_into_sentences(evidence['content'])


        with open(os.path.join(args.store_folder, f"{id}.txt"), 'w') as f:
            f.write(evidence_url + '\n\n\n')
            for sentence in sentences:
                f.write(sentence + '\n')  # Write each sentence followed by a newline

        # evidence_mapping.append({
        #     'evidence_id': id,
        #     'original_link': evidence_url,
        #     'archive_url': evidence['archive_url'],
        #     'html_file': evidence['html_file']
        # })


    
# with open(os.path.join(args.store_folder, "evidence_mapping.json"), 'w') as f:
#     json.dump(evidence_mapping, f)

     