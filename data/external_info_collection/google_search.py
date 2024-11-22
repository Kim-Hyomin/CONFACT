import json
import os
import pickle as pkl
import random
from collections import defaultdict
import argparse 
import re
from urllib.parse import urlparse
import time
from others.secure_info import google_key,search_engine_id
from googleapiclient.discovery import build
from html2lines import url2text
from goose3 import Goose
import trafilatura

g = Goose()
blacklist = [
    "jstor.org", # Blacklisted because their pdfs are not labelled as such, and clog up the download
    "facebook.com", # Blacklisted because only post titles can be scraped, but the scraper doesn't know this,
    "ftp.cs.princeton.edu", # Blacklisted because it hosts many large NLP corpora that keep showing up
    "nlp.cs.princeton.edu",
    "mediabiasfactcheck.com", # We don't want to retrieve the test data
    "ground.news" # Cites mediabiasfactcheck too often
]

blacklist_files = [ # Blacklisted some additional NLP files that show up in search results and cause OOM errors
    "/glove.", 
    "ftp://ftp.cs.princeton.edu/pub/cs226/autocomplete/words-333333.txt",
    "https://web.mit.edu/adamrose/Public/googlelist"
]

def load_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

service = build("customsearch", "v1", developerKey=google_key)
import sys

def search_pages(search_term, **kwargs):
    res = service.cse().list(q=search_term, cx=search_engine_id, **kwargs).execute()
    if "items" in res:
        return res['items']
    else:
        return []

def get_domain_name(url):
    if '://' not in url:
        url = 'http://' + url

    domain = urlparse(url).netloc

    if domain.startswith("www."):
        return domain[4:]
    else:
        return domain

def process_search_results(results, search_string):
    #print (results)
    all_results=[]
    for i,result in enumerate(results):
        link = str(result["link"])
        all_results.append(link)
    return all_results


def get_google_search_results(search_string, sort_date=None, page=0):
    search_results = []
    sort = None if sort_date is None else ("date:r:19000101:"+sort_date)
    print ('\tSearch string:',search_string)
    for _ in range(3):
        try:
            search_results += search_pages(
                search_string,
                num=10,
                start=0,
                sort=sort,
                dateRestrict=None,
                gl="US"
            )
            break
        except:
            print("I encountered an error trying to search +\""+search_string+"\". Maybe the connection dropped. Trying again in 3 seconds...", file=sys.stderr)
            time.sleep(1)
    return process_search_results(search_results, search_string)

def google_search(queries, source_name, saved_entities):
    entity={} 
    #print(source)
    for item in queries:
        localized_question = item['question'].strip().replace("X", "\"" + source_name + "\"")
        localized_search_query = item['statement'].strip().replace("X", "\"" + source_name + "\"")
        entity[localized_question]=[]

        if localized_question in saved_entities and len(saved_entities[localized_question]):
            entity[localized_question]=saved_entities[localized_question]
            print ('\tAlready exists:',localized_question)
        else:
            search_results  = get_google_search_results(localized_search_query)
            entity[localized_question]=search_results
    return entity

def save_results(names, output_path, debug = False):
    print("saving url results to {output_path}")
    if not os.exists(output_path):
        total={}
    else:
        total=load_pkl(output_path)
    

    for i,name in enumerate(names):
        if i%100==0:
            print ("dumping!!!")
            pkl.dump(total,open(output_path,'wb'))

        if name in total:
            saved_entities=total[name]
        else:
            saved_entities={}
        print(i, name)

        google_search_result=google_search(queries, name,saved_entities)

        total[name]=google_search_result
        if debug:
            for qn in google_search_result:
                print(qn,len(google_search_result[qn]))
    pkl.dump(total,open(output_path,'wb'))

def scapper(web_pages):
    all_texts=[]
    for url in web_pages:
        try:
            downloaded = trafilatura.fetch_url(url)
            texts=trafilatura.extract(downloaded)
            all_texts.append(texts)
        except:
            print ("Invalid evidence url:",url)
    return all_texts

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    """
    Whether generating detailed background information
        If false, only categorize media 
    """
    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)
    parser.add_argument('--media_file',
                        type=str,
                        default="../dataset/mfbc_updated.pkl")
    parser.add_argument('--query_file',
                        type=str,
                        default="../dataset/google_queries.json")                   
    args=parser.parse_args()

    queries=load_json(args.QUERY_FILE)
    media_file=load_pkl(args.media_file)

    names=[]
    for domain in media_file:
        media=media_file[domain]
        media_name=media["media"].split('(')[0][:-1]
        if len(media_name):
            names.append(media_name)
        else:
            names.append("")


    print ("Length of websites included:",len(media_file))
    
    if args.DEBUG:
        names=names[:5]

    save_results(names, "../dataset/google_search_urls.pkl", args.DEBUG)


        