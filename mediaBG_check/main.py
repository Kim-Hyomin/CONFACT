import argparse
import logging
from urllib.parse import urlparse
import random
import vllm
from vllm import SamplingParams
import os
import pickle as pkl
import re
import torch
import numpy as np
from dotenv import load_dotenv
import argparse
from UncertainQA.utils import initialize_pkl_file, load_pkl, load_gzip, extract_domain, set_seed
from media_check import MediaCheck

load_dotenv()

logger = logging.getLogger('media_check')

def expend_credibility_dataset(existing_data, new_data):
    existing_data.extend(new_data)
    return existing_data

# python main.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 2
def main():
    parser = argparse.ArgumentParser(description="Run Media Background Check")
    parser.add_argument("--seed", type=int, default = 2024, help="seed")
    parser.add_argument("--model", type=int, default = 'meta-llama/Llama-3.1-8B-Instruct', help="model used for media background generation and label prediction")
    parser.add_argument('--gpu', type=int, default=2, help="GPU device number to use.")

    args = parser.parse_args()

    set_seed(args.seed)
    
    sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=args.seed, stop = '*** END')

    LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.95, trust_remote_code=True)

    query_file = "../media_bg_collected/google_queries.json"
    label_demo_file = "./media_bg_collected/label_demos.pkl.gz"
    description_demo_file = "./media_bg_collected/description_demos.pkl.gz"
    credibility_data_file = "../data/dataset/all_media_data.pkl"  # this was initialzed with mbfc dataset

    all_evidence_url_file = "./media_bg_collected/google_evidence_urls.pkl"
    all_scraped_text_file = "./media_bg_collected/google_evidence_text.pkl"
    article_info_file = "./media_bg_collected/article_info.pkl"
    wiki_info_file = "./media_bg_collected/wiki_info.pkl"

    files = [all_evidence_url_file, all_scraped_text_file, article_info_file, wiki_info_file]

    for file in files:
        initialize_pkl_file(file)

    media_checker = MediaCheck(
        LLM = LLM,
        sampling_params = sampling_params,
        wiki_flag = True,
        article_flag = True, 
        google_flag = True, 
        credibility_data_file = credibility_data_file, 
        label_demo_file = label_demo_file, 
        description_demo_file = description_demo_file, 
        query_file = query_file, 
        all_evidence_url_file = all_evidence_url_file, 
        all_evidence_text_file = all_scraped_text_file,
        article_info_file = article_info_file,
        wiki_info_file= wiki_info_file)


    credibility_data = load_pkl(credibility_data_file)
    media_saved = set(credibility_data.keys())
            
    data_path = "../data/dataset/MUQA.pkl.gz"
    data = load_gzip(data_path)

    domains_to_describe = set()
    for item in data:
        for evidence in item['evidence_url']:
            url = evidence['original_link']
            domain = extract_domain(url)
            if domain not in media_saved:
                domains_to_describe.add(domain)


    domains_to_describe = list(domains_to_describe)
    domains_to_describe = sorted(domains_to_describe) 

    print("#######number of domains to describe: ", len(domains_to_describe))

    descriptions = media_checker.generate_description(domains_to_describe, 1)

    labels = media_checker.predict_credibility_level(domains_to_describe, 1)

    with open(credibility_data_file, "wb") as f:
        pkl.dump(media_checker.credibility_data, f) # save updated credibility data

if __name__ == "__main__":
    main()
