import waybackpy
from waybackpy.exceptions import WaybackError
import time
import os
import json
from tqdm import tqdm

user_agent = os.getenv("WAYBACK_USER_AGENT")


def get_archive_url(url):
    wayback = waybackpy.Url(url, user_agent)
    
    # Check if the domain is blocked
    try:
        n_archived = wayback.total_archives()
    except Exception as e:
        print(f"The domain seems to block archival. Skipping. Error: {e}")
        return
    
    if n_archived > 0:
        try:
            archive = wayback.newest()
        except Exception as e:
            print(f"Failed to retrieve a saved archival page. Building a new page. Error: {e}")
            for _ in range(5):
                try:
                    archive = wayback.save()
                    break
                except Exception as e:
                    print(f"Couldn't reach the archive to build a page. Trying again in 3 seconds. Error: {e}")
                    time.sleep(3)
                    archive = None

    else:
        for _ in range(5):
            try:
                archive = wayback.save()
                break
            except Exception as e:
                print(f"Couldn't reach the archive to build a page. Trying again in 3 seconds. Error: {e}")
                time.sleep(3)
                archive = None

    if archive is not None:
        archive_url = archive.archive_url
        return archive_url
    else:
        return None


file_path='dataset/claimWevidence.json'
# file_path='claimWevidence.json'

with open(file_path, 'r') as f:
    data = json.load(f)

        # "claim": "For the year under review all County Governments allocated out of their budgets at least 20% to health.",
        # "label": "Refuted",
        # "original_claim_url": "https://web.archive.org/web/20210418085957/https://africacheck.org/sites/default/files/STATE-OF-DEVOLUTION-ADDRESS-2019-.pdf",
        # "fact_checking_article": "https://web.archive.org/web/20210720042508/https://africacheck.org/fact-checks/reports/claims-kenyas-counties-fact-checking-2019-state-devolution-address",
        # "country": "KE",
        # "query": "Did County Governments allocate least 20 % budgets health year review",
        # "evidence_url": [
        #     {
        #         "original_link": "http://www.healthpolicyplus.com/ns/pubs/18441-18879_KenyaNCBABrief.pdf"
        #     }]

for item in tqdm(data, total=len(data), desc="archieving web urls"):

    for element in item['evidence_url']:
        if element:
            if 'archive_url' not in element or element['archive_url'] is None:
                url = element["original_link"]
                result = get_archive_url(url)
                element['archive_url'] = result

        # with open(file_path, 'w') as f:
        #     json.dump(data, f, indent=4)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)