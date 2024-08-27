import requests
import json
from openai import OpenAI 


client = OpenAI(api_key ="sk-proj-txyILNTruRXHhU9X8d5HT3BlbkFJk8n6KZvrIJGuSXAXY2i6") 

def LLM_factchecker(claim, evidence):
    system_prompt = """You are an expert in fact-checking."""

    user_prompt = f"\Claim: {claim}\nURL: {evidence}\nDetermine if the evidence provided in the URL supports or rejects the claim. Respond with either 'Support' or 'Reject.'"

    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:  
        print(f"An error occurred in question generation: {e}")
        return None
    

claim = 'Jammu and Kashmir (J&K) in Pakistan/India was removed from the United Nations (UN) list of ‚Äúunresolved disputes‚Äù.'
evidence = 'https://www.thehindu.com/news/national/Jammu-and-Kashmir-out-of-U.N.-list-of-disputes/article15687886.ece'


result = LLM_factchecker(claim, evidence)
print(result)