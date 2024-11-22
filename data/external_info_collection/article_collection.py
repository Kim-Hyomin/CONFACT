import json
import os
import pickle as pkl
import random
from collections import defaultdict
import argparse 
import re
from feedsearch import search
import feedparser

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

def extract_article_url(websites, input_type, article_urls_file):

    if input_type == "mfbc":
        os.chdir("./mfbc_external_info")

    idx=0
    if os.path.exists(article_urls_file):
        saved_file=load_jsonl()
        save_rows=len(saved_file)
    else:
        save_rows=0
    print ('Already finished:',save_rows)

    if not os.path.exists("valid_" + article_urls_file + ".jsonl"):
        with open("valid_" + article_urls_file + ".jsonl", 'w') as file:
            pass

    if not os.path.exists("invalid_" + article_urls_file + ".jsonl"):
        with open("invalid_" + article_urls_file + ".jsonl", 'w') as file:
            pass

    article_urls=open("valid_"+article_urls_file+".jsonl",'a')
    invalid_websites=open("invalid_"+article_urls_file+".jsonl",'a')

    for i,website in enumerate(websites):
        if idx <save_rows:
            print ('Already exists!',idx)
            idx+=1
            continue
        try:
            feeds=search(website)
            urls=[f.url for f in feeds]
        except:
            print ('Invalid!!:',i, website)
            invalid_websites.write(json.dumps((i,website)) + '\n')
            invalid_websites.flush()
            #cur_article_urls=[]
            urls=[]
        
        cur_article_urls=[]
        if len(urls) > 0:
            try:
                news_feed = feedparser.parse(urls[0])
                entries=news_feed["entries"]
            except:
                entries = []
            for entry in entries:
                link=""
                pub=""
                title=""
                if "title" in entry:
                    title=entry["title"]
                if "link" in entry:
                    link=entry["link"]
                if "published" in entry:
                    pub=entry["published"]
                cur_article_urls.append({
                    "title":title,
                    "published":pub,
                    "link":link
                })
        article_urls.write(json.dumps(cur_article_urls) + '\n')
        article_urls.flush()
        idx+=1
        print (i, website)
        print ('\tNumber of articles',len(cur_article_urls))
    return 


if __name__=='__main__':

    print("changes saved!")

    parser=argparse.ArgumentParser()
    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)
    
    parser.add_argument('--input_type',
                        type=str,
                        default='mfbc')
    
    parser.add_argument('--input_file',
                        type=str,
                        default="../mfbc/mfbc_updated.pkl")
    parser.add_argument('--output_article_urls_file_name',
                        type=str,
                        default="mfbc_article_urls")
    args=parser.parse_args()


    media_file=load_pkl(args.input_file)
    websites=list(media_file.keys())

    print ("Length of websites included:",len(websites))
    if args.DEBUG:
        websites=websites[:10]
        
    extract_article_url(websites, args.input_type, args.output_article_urls_file_name)
    
    