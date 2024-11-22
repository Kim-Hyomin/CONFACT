import json
import os
import pickle as pkl
import random
from collections import defaultdict
import argparse 
import re
import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')


def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def extract_wiki(websites, input_type, wiki_file):

    if input_type == "mfbc":
        os.chdir("./mfbc_external_info")

    if not os.path.exists(wiki_file +".jsonl"):
        with open(wiki_file +".jsonl", 'w') as file:
            pass

    wiki=open(wiki_file +".jsonl",'a')
    for i,website in enumerate(websites):
        page_py = wiki_wiki.page(website)
        if page_py.exists():
            wiki_info={
                "title":page_py.title,
                "summary":page_py.summary,
                "text":page_py.text
                }
        else:
            wiki_info={}
        wiki.write(json.dumps(wiki_info) + '\n')
        wiki.flush()
        print (i, website)
        print ('\tExit',page_py.exists())
    return 

if __name__=='__main__':
    print("new changes saved!")

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
    parser.add_argument('--output_wiki_file_name',
                        type=str,
                        default="wiki_info")
    args=parser.parse_args()


    media_file=load_pkl(args.input_file)
    websites=[]

    for domain in media_file:
        media=media_file[domain]
        media_name=media["media"].split('(')[0][:-1]
        if len(media_name):
            websites.append(media_name)
        else:
            websites.append("")

    print ("Length of websites included:",len(websites))

    if args.DEBUG:
        print("DEBUG MODE")
        # websites=websites[:30]
        websites=["us"]
    # print(websites)
    extract_wiki(websites, args.input_type, args.output_wiki_file_name)
    
    