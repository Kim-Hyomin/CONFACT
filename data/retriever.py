import requests
import json
from openai import OpenAI 
import argparse
from datetime import datetime
from nltk import pos_tag, word_tokenize

API_KEY = ""
CX = ""
client = OpenAI(api_key ="") 


def get_search_results(api_key, cx, query, num_results=10):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': cx,
        'q': query,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()

    items = results.get('items', [])
    num = len(items)

    while num < num_results and results['queries'].get('nextPage',[]):
        start = results['queries']['nextPage'][0]['startIndex']
        params = {
        'key': api_key,
        'cx': cx,
        'q': query,
        'start': start
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        items = items + results.get('items', [])
        num = len(items)

    return items[:num_results]

def claim2qn(claim):
    system_prompt = """
To convert a claim into a question, keep the core elements of the statement while rephrasing it in an interrogative format.
"""
    user_prompt = """
Sample 1:  
Claim: Secret Service agents smiled while moving former President Donald Trump to safety after he was wounded in an assassination attempt at a Pennsylvania rally.   
Question: Did Secret Service agents smile while moving former President Donald Trump to safety after he was wounded in an assassination attempt at a Pennsylvania rally?  
  
Sample 2:  
Claim: Secret Service officials stopped an agent named "Jonathan Willis" from shooting an attempted assassin of former President Donald Trump at a rally.  
Question: Did Secret Service officials stop an agent named "Jonathan Willis" from shooting an attempted assassin of former President Donald Trump at a rally?  
  
Sample 3:  
Claim: A photo circulating on social media shows Donald and Melania Trump with Stormy Daniels, providing evidence of Trump's affair with Daniels.  
Question: A photo circulating on social media shows Donald and Melania Trump with Stormy Daniels, does it provide evidence of Trump's affair with Daniels?"""

    user_prompt += f"\Claim: {claim}\nQuestion:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            # model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        question = response.choices[0].message.content.strip()
        return question
    except Exception as e:  
        print(f"An error occurred in question generation: {e}")
        return None
    

def to_search_query(claim, date):
    token_words = word_tokenize(claim.strip())
    tags = pos_tag(token_words)

    target_tags = ["CD", "JJ", "NN", "VB"]
    search_string = []


    for token, tag in zip(token_words, tags):
        for keep_tag in target_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)

    # if date:
    #     try: 
    #         parsed_date = datetime.strptime(date, "%d/%m/%y").strftime("%Y-%m-%d")
    #     except ValueError:
    #         parsed_date = date
    #     search_string.append(parsed_date)


    search_string = " ".join(search_string)

    return search_string

    
parser = argparse.ArgumentParser(description='Process a JSON file of claims and search for related information.')
parser.add_argument('--input', type=str, help='The input JSON file containing claims')
parser.add_argument('--output', type=str, help='The outpt JSON file containing retrieved results')
parser.add_argument('--n', type=int, help='The number of evidence retrieved for each claim')
args = parser.parse_args()



with open(args.input, 'r') as file:
    articles = json.load(file)


output_filename = args.output

with open(output_filename, 'w') as outfile:
    json.dump([], outfile)

for article in articles:
    claim = article["claim"]
    date = article["claim_date"]
    label = article["label"]
    # QUERY = claim
    QUERY = to_search_query(claim2qn(claim), date)

    items = get_search_results(API_KEY, CX, QUERY, args.n)

    retrieved_results = []

    for i, item in enumerate(items):
        title = item.get('title')
        link = item.get('link')
        retrieved_results.append({
            'rank': i + 1,
            'title': title,
            'url': link
        })

    # claim_result = {
    #     'claim': claim,
    #     'retrieved results': retrieved_results
    # }
    claim_result = {
        'label': label,
        'claim': claim,
        'question': QUERY,
        'retrieved results': retrieved_results
    }

    with open(output_filename, 'r+') as outfile:
        existing_data = json.load(outfile)
        existing_data.append(claim_result)
        outfile.seek(0)
        json.dump(existing_data, outfile, indent=4)


