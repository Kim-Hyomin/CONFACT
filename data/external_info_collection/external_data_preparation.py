import json
import pickle as pkl
import os
from article_collection import extract_article_url
from extract_articles import extract_articles
from wiki_collection import extract_wiki
import argparse

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


def mfbc_preprocess(args):
    media_file=load_pkl(args.input_file)

    # extracting article urls
    print(f"extracting article urls, saving to {args.output_article_urls_file_name}")

    websites=list(media_file.keys())
    extract_article_url(websites, args.input_type, args.output_article_urls_file_name)

    # extracting articles
    print(f"extracting articles")
    article_urls=load_jsonl("./mfbc_external_info/valid_mfbc_article_urls.jsonl")

    print ("Length of websites included:",len(websites),len(article_urls))

    extract_articles(websites, args.input_type, article_urls)
    
    # extracting wiki info
    names = []
    for domain in media_file:
        media=media_file[domain]
        media_name=media["media"].split('(')[0][:-1]
        if len(media_name):
            names.append(media_name)
        else:
            names.append("")

    print(f"extracting wiki info, saving to {args.output_wiki_file_name}")
    extract_wiki(names, args.input_type, args.output_wiki_file_name)

    # consolidating data
    collected_articles=load_jsonl("./mfbc_external_info/extracted_articles.jsonl")
    print ('Length of extracted articles:',len(collected_articles))

    wiki_info=load_jsonl("./mfbc_external_info/wiki_info.jsonl")
    print ('Length of wiki info:',len(wiki_info))

    valid_articles=0
    valid_wiki=0

    for i,name in enumerate(websites):
        articles=collected_articles[i]
        if len(articles):
            valid_articles+=1
            
        cur_wiki=wiki_info[i]
        if "title" in cur_wiki:
            valid_wiki+=1

        media_file[name]['articles']=articles
        media_file[name]["wiki"]=cur_wiki

    with open("../dataset/mfbc_W_external_info.pkl", "wb") as f:
        pkl.dump(media_file, f)

    

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

    parser.add_argument('--output_wiki_file_name',
                        type=str,
                        default="wiki_info")

    args=parser.parse_args()

    mfbc_preprocess(args)

    
    