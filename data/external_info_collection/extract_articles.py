import json
import os
import pickle as pkl
import random
from collections import defaultdict
import argparse 
import re
import unicodedata
from goose3 import Goose

def fix(text):
    try:
        text = text.decode("ascii", "ignore")
    except:
        t=[unicodedata.normalize('NFKD', str(q)).encode('ascii','ignore') for q in text]
        text=''.join(t).strip()
    print (text)
    return text

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

def extract_articles(websites, input_type, article_urls):
    if input_type == "mfbc":
        os.chdir("./mfbc_external_info")
        if not os.path.exists("extracted_articles.jsonl"):
            ext_articles=open("extracted_articles.jsonl",'w')
            ext_articles.close()
        if not os.path.exists("invalid_ext_articles.jsonl"):
            invalid=open("invalid_ext_articles.jsonl",'w')
            invalid.close()

        ext_articles=open("extracted_articles.jsonl",'a')
        invalid=open("invalid_ext_articles.jsonl",'a')

    g = Goose()

    for i, website in enumerate(websites):
        print ('Extracting for the ',i)
        articles=article_urls[i]
        cur_ext_info=[]

        for article in articles:
            if "link" not in article:
                continue
            link=article["link"]
            #print (link)
            #ext_info= g.extract(url=link)
            #print (ext_info.title,ext_info.meta_description,ext_info.cleaned_text)
            try:
                ext_info= g.extract(url=link)
                #print (ext_info.title)
                #print (ext_info.cleaned_text)
                cur_ext_info.append({
                    "title":ext_info.title,
                    #"meta_data":ext_info.meta_description,
                    "cleaned_text":ext_info.cleaned_text,
                })
            except:
                print ('Inavlid!',link)
                invalid.write(json.dumps((i,link)) + '\n')
                invalid.flush()
        ext_articles.write(json.dumps(cur_ext_info) + '\n')
        ext_articles.flush()

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)
    parser.add_argument('--input_type',
                        type=str,
                        default="mfbc")
    parser.add_argument('--media_file',
                        type=str,
                        default="../mfbc/mfbc_updated.pkl")
    args=parser.parse_args()

    if args.input_type=="mfbc":
        print ('extract mfbc data')
        media_file=load_pkl(args.media_file)
        websites=list(media_file.keys())
        article_urls=load_jsonl("./mfbc_external_info/valid_mfbc_article_urls.jsonl")
        invalid_rows=load_jsonl("./mfbc_external_info/invalid_mfbc_article_urls.jsonl")
        invalid_dict=defaultdict(int)
        for row in invalid_rows:
            invalid_dict[row[0]]+=1
    else:
        pass

    print ("Length of websites included:",len(websites),len(article_urls))

    if args.DEBUG:
        websites=websites[:4]

    extract_articles(websites, args.input_type, article_urls)

