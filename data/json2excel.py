import json
import pandas as pd

with open("./test_files/retrieved_NoDate.json",'r') as f:
    data = json.load(f)

records = []
for entry in data:
        record = {
            'label': entry['label'],
            'claim': entry['claim'],
            'question': entry['question']
        }
        
        for result in entry['retrieved results']:
            rank = result['rank']
            url_key = f'rank{rank}_url'
            record[url_key] = result['url']
        

        records.append(record)

df = pd.DataFrame(records)

df.to_excel("./test_files/retrieved_NoDate.xlsx", index=False)


