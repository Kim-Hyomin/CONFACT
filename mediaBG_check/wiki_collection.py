import wikipediaapi
import logging
from UncertainQA.utils import load_pkl
import pickle as pkl

logger = logging.getLogger(__name__)

def extract_wiki_info(domain_name, wiki_info_file):

    logger.info(f"Extracting Wikipedia info for {domain_name}")

    all_wiki_info = load_pkl(wiki_info_file)
    
    if domain_name in all_wiki_info:
        wiki_info = all_wiki_info[domain_name]
        logger.info(f"existing Wikipedia info found for {domain_name}")
    else:
        wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        page_py = wiki_wiki.page(domain_name)

        if page_py.exists():
            wiki_info = {
                "title": page_py.title,
                "summary": page_py.summary,
                "text": page_py.text
            }
            logger.info(f"Wikipedia page found for {domain_name}")
        else:
            wiki_info = {}
            logger.warning(f"No Wikipedia page found for {domain_name}")
        
        all_wiki_info[domain_name] = wiki_info

        with open(wiki_info_file, 'wb') as f:
            pkl.dump(all_wiki_info, f) #update the file with newly extracted info
            
    return wiki_info