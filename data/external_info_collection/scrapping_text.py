import numpy as np
import argparse
import json
import os
import pickle as pkl
import random
import trafilatura

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

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
    Testing whether the system works properly
        For a given website, use the pipeline, rather than taking the code directly
    """
    parser.add_argument('--input_file',
                        type=str,
                        default="../dataset/google_search_urls.pkl")
    parser.add_argument('--DEBUG',
                        type=bool,
                        default=False)
    args=parser.parse_args()
    
    file=load_pkl(args.input_file)
    print ('Scrapping evidence from web pages related to pre-defined queries')
    print ('\tLength of instances:',len(file))
    total={}
    names=list(file.keys())
    
    for i,source_name in enumerate(names):
        if args.DEBUG:
            if i>=1:
                break
        print ('Scapping evidence for the ',i,'-th instance:',source_name)

        all_evidence=file[source_name]
        all_evid_texts=[]
        for query_ques in all_evidence:
            collected_pages=all_evidence[query_ques]
            collected_texts=scapper(collected_pages)
            all_evid_texts.append(collected_texts)
            print ('\t\t',query_ques,len(collected_texts))
            #print ('\t\t',query_ques,len(collected_pages),collected_pages[0])
        total[source_name]=all_evid_texts
    pkl.dump(total,open('../dataset/text.pkl','wb'))