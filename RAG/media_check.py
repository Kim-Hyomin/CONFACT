import argparse
import logging
import pickle as pkl
import json
import re
import wikipediaapi
from feedsearch import search
import feedparser
from goose3 import Goose
from urllib.parse import urlparse
import random
import time
from googleapiclient.discovery import build
import trafilatura
import vllm
from vllm import SamplingParams
import os
from tqdm import tqdm
import re
import torch
import numpy as np
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

load_dotenv()


logger = logging.getLogger('media_check')

CX = os.getenv('GOOGLE_CX')
google_api = os.getenv("GOOGLE_API_KEY")

service = build("customsearch", "v1", developerKey=google_api)

init_guess_sys_prompt = "You are InfoHuntGPT, a world-class AI assistant used by journalists to quickly build knowledge of new sources."

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def write_to_file(file_path, data, append=True):
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(data + "\n")


def load_pkl(path):
    logger.info(f"Loading pickle file from {path}")
    return pkl.load(open(path, 'rb'))

def load_json(path):
    logger.info(f"Loading JSON file from {path}")
    return json.load(open(path, 'r'))


def clean_text(text):

    text = re.sub(r'(]]>\s*){2,}', ']]>', text)
    text = re.sub(r'(\]</a>\s*){2,}', ']</a> ', text)
    text = re.sub(r'(<\!\[CDATA\[.*?\]\]>\s*){2,}', '<![CDATA[DATA]]>', text)
    text = re.sub(r'</?\w+>\s*', '', text)  # Strip simple HTML/XML tags
    text = re.sub(r'[\[\]]+', '', text)  # Remove stray brackets that have no semantic purpose
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Strip leading and trailing whitespace
    text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Remove non-alphanumeric and punctuation except those needed

    return text


def extract_wiki_info(domain_name):
    logger.info(f"Extracting Wikipedia info for {domain_name}")
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
        
    return wiki_info 

def extract_articles(article_urls):
    logger.info("Extracting articles from URLs")
    g = Goose()
    article_info = []
    invalid_article_urls = []

    for article in article_urls:
        if "link" not in article:
            continue
        link = article["link"]

        try:
            ext_info = g.extract(url=link)
            article_info.append({
                "title": ext_info.title,
                "cleaned_text": ext_info.cleaned_text,
            })
            logger.info(f"Article extracted from {link}")
        except Exception as e:
            logger.error(f"Invalid article URL: {link} - {e}")
            invalid_article_urls.append(link)

    return article_info, invalid_article_urls

def collect_article_urls(domain_name):
    logger.info(f"Collecting article URLs for {domain_name}")
    invalid_domains = []
    try:
        feeds = search(domain_name)
        urls = [f.url for f in feeds]
    except Exception as e:
        logger.error(f"Error searching feeds for {domain_name} - {e}")
        invalid_domains.append(domain_name)
        urls = []

    valid_article_urls = []
    if urls:
        try:
            news_feed = feedparser.parse(urls[0])
            entries = news_feed["entries"]
        except Exception as e:
            logger.error(f"Error parsing feed for {domain_name} - {e}")
            entries = []

        for entry in entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            pub = entry.get("published", "")
            valid_article_urls.append({
                "title": title,
                "published": pub,
                "link": link
            })
            logger.info(f"URL collected: {link}")

    return valid_article_urls, invalid_domains

class GoogleSearch:
    def __init__(self, query_file, all_evidence_url_file, all_scraped_text_file):
        logger.info("Initializing GoogleSearch")
        self.queries = load_json(query_file)
        self.all_evidence_url_path = all_evidence_url_file
        self.all_evidence_url = load_pkl(all_evidence_url_file)
        # {"media_name_1": {
        #     "qn1": [link1, link2, link3], 
        #     "qn2": [link1, link2, link3]
        #     },
        # }

        self.all_scraped_text = load_pkl(all_scraped_text_file)
        self.all_scraped_path = all_scraped_text_file
        # {"media_name_1": [
        #     [text content of qn1 evidence1, text content of qn1 evidence2, text content of qn1 evidence3...],
        #     [text content of qn2 evidence1, text content of qn2 evidence2, text content of qn2 evidence3...]
        # ]}

    def scraper(self, web_pages, timeout=60, max_retries=3):
        logger.info("Scraping text content from web pages")
        all_texts = []
        for url in web_pages:
            retries = 0
            while retries < max_retries:
                try:
                    logger.info(f"Attempting to fetch: {url} (Retry {retries + 1}/{max_retries})")
                    
                    downloaded = trafilatura.fetch_url(url, timeout=timeout)
                    if not downloaded:
                        raise ValueError("Failed to fetch content or invalid URL")
                    
                    texts = trafilatura.extract(downloaded)
                    if not texts:
                        raise ValueError("Failed to extract content or empty HTML tree")

                    all_texts.append(texts)
                    logger.info(f"Content scraped successfully from {url}")
                    break  # Exit retry loop on success

                except RequestException as e:
                    retries += 1
                    logger.warning(f"Request exception for {url}: {e}. Retrying...")
                    time.sleep(3) 

                except ValueError as e:
                    logger.error(f"Content error for {url}: {e}")
                    break

                except Exception as e:
                    logger.error(f"Unexpected error for {url}: {e}")
                    break  # Exit loop on unexpected errors

            if retries == max_retries:
                logger.error(f"Max retries reached for {url}. Skipping...")
                all_texts.append("") 

        return all_texts


    def search_pages(self, search_term, **kwargs):
        try:
            res = service.cse().list(q=search_term, cx=CX, **kwargs).execute()

            logger.info(f"Search completed for term: {search_term}")
            return res.get('items', [])
        except Exception as e:
            logger.error(f"Error during search for term {search_term} - {e}")
            return []

    def process_search_results(self, results):
        logger.info("Processing search results")
        return [str(result["link"]) for result in results]

    def get_google_search_results(self, search_string, sort_date=None, page=0):
        logger.info(f"Getting Google search results for: {search_string}")
        search_results = []
        sort = None if sort_date is None else ("date:r:19000101:" + sort_date)
        for _ in range(3):
            try:
                search_results += self.search_pages(
                    search_string,
                    num=10,
                    start=0,
                    sort=sort,
                    dateRestrict=None,
                    gl="US"
                )
                break
            except Exception as e:
                logger.warning(f"Retrying search for {search_string} due to error: {e}")
                time.sleep(1)
        return self.process_search_results(search_results)


    def google_search(self, source_name):
        logger.info(f"Performing Google search for source: {source_name}")
        saved_urls = self.all_evidence_url.get(source_name, {})

        if source_name not in self.all_scraped_text:
            self.all_scraped_text[source_name] = []

        entity = {}
        for idx, item in enumerate(tqdm(self.queries, desc="Processing queries")):  # loop over the saved entities again as google connection is not stable, some query results might be lost
            localized_question = item['question'].replace("X", f"\"{source_name}\"").strip()
            localized_search_query = item['statement'].replace("X", f"\"{source_name}\"").strip()

            entity[localized_question] =  saved_urls.get(localized_question, [])
            if entity[localized_question]:
                logger.info(f"Using cached results for: {localized_question}")
            else:
                search_results = self.get_google_search_results(localized_search_query)
                entity[localized_question] = search_results

                # Ensure that the list is long enough to handle the current index
                while len(self.all_scraped_text[source_name]) <= idx:
                    self.all_scraped_text[source_name].append([])  # Append empty list for missing indices

                self.all_scraped_text[source_name][idx] = self.scraper(search_results)
                logger.info(f"Scraped text content for: {localized_question}")

        self.all_evidence_url[source_name] = entity
        pkl.dump(self.all_evidence_url, open(self.all_evidence_url_path, "wb"))
        logger.info(f"Saved all evidence urls for source: {source_name}")

        pkl.dump(self.all_scraped_text, open(self.all_scraped_path, "wb"))
        logger.info(f"Saved all scraped text for source: {source_name}")

        return self.all_scraped_text[source_name]


class MediaCheck:
    
    def __init__(self, LLM, sampling_params, wiki_flag = True, article_flag = True, google_flag = False, credibility_data_file = None, label_demo_file = None, description_demo_file = None, query_file = None, all_evidence_url_file = None, all_evidence_text_file = None):

        logger.info("Initializing MediaCheck class.")

        self.LLM = LLM
        self.sampling_params = sampling_params

        self.credibility_data_file = credibility_data_file
        self.credibility_data = load_pkl(credibility_data_file) # files where details and labels are saved, initialized using mfbc_updated.pkl [keeps updating this]

        self.predict_label_demo = load_pkl(label_demo_file) # files used for demo, mfbc_W_external_info.pkl [this should not be modified]

        self.description_demo_file = description_demo_file
        self.query_file = query_file
        self.all_evidence_url_file = all_evidence_url_file
        self.all_evidence_text_file = all_evidence_text_file

        self.wiki_flag = wiki_flag
        self.article_flag = article_flag
        self.google_flag = google_flag

    def extract_domain(self, url):
        if url:
            domain_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            match = re.search(domain_pattern, url)
            if match:
                domain = match.group(1)
                logger.info("Extracted domain: %s", domain)
                return domain
            
        logger.warning("No domain found for URL: %s", url)
        return url  
            

    def get_answers_from_evidence(self, domain, queries, google_evidence, output_file="./media_bg_generated/answers.txt"):
        answer_sys_prompt = "Please provide an answer in a sentence to the question: %s, based on provided evidence.\nEvidence:%s\n\nBegin your response with Answer:"

        prompts = []
        for k, item in enumerate(queries):
            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
            cur_evid = google_evidence[k]     
            if len(cur_evid):
                evid = str(cur_evid)[0][:3000]
            else:
                evid = "No external evidence available."
            prompts.append(answer_sys_prompt % (cur_query, evid))

        logger.info("Generating answers from evidence for domain %s.", domain)

        outputs = self.LLM.generate(prompts, self.sampling_params)
        answers = [clean_text(output.outputs[0].text)[:1500] for output in outputs]

        # Write the results to the file instead of printing
        results = ""
        for k, item in enumerate(queries):
            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
            answer = answers[k]
            results += f"Question: {cur_query}\n\tAnswer: {answer}\n"
            results += "-------------------------\n"
        write_to_file(output_file, results, append = True)  # Save to file
        
        return answers


    def generate_initial_guess(self, source_names, wiki_collection, articles_collection, num_demo, output_file="./media_bg_generated/initial_guess.txt"):
        prompts = []
        for idx, source_name in enumerate(source_names):
            logger.info("Generating initial guess for %s", source_name)

            wiki = wiki_collection[idx]
            articles = articles_collection[idx]
        
            prompt = f"Build a background check for the news source \"{source_name}\". Write down everything you know about them in no more than 500 words."

            if 'title' not in wiki:
                wiki = 'It does not have a wiki page'
            else:
                wiki = wiki['summary']

            prompt += f'Here is its wiki information: {wiki}'

            if len(articles):
                titles = [article['title'] for article in articles][:10]
                titles = '\t'.join(titles)
            else:
                titles = 'No articles from the website available'
            prompt += f'Here are some article titles from the website: {titles}'

            
            if num_demo > 0:
                prompt += "\n\n Below are some examples of background checks, you may follow the format."
                prompt += "\n" + self.get_icl_string(num_demo)
            
            prompt += "\nIf you are aware that they have failed any fact-checks, mention which. Begin your response with \"**Background check**\"."



            input_prompt = init_guess_sys_prompt + prompt
            logger.info("Initial prompt generated for initial guess.")
            
            prompts.append(input_prompt)

        outputs = self.LLM.generate(prompts, self.sampling_params)
        # extracted_outputs = [clean_text(output.outputs[0].text[len(input_prompt):]) for output in outputs]
        extracted_outputs = [clean_text(output.outputs[0].text)[:3000] for output in outputs]

        # Write initial guesses and prompts to the file
        results = ""
        for idx, initial_prompt in enumerate(prompts):
            # guess = extracted_outputs[idx].split("**")[1:]
            # guess = "**" + "**".join(guess)
            # result = f"Prompt: {initial_prompt}\n\nInitial Guess: {guess}"
            results += f"Prompt: {initial_prompt}\n\nInitial Guess: {extracted_outputs[idx]}"
            results += "----------------------------------------------------"

        write_to_file(output_file, results)  # Save to file

        return extracted_outputs, prompts


    def get_icl_string(self, num_demo):
        #./media_bg_collected/description_demos.pkl
        all_texts=["The following are examples of background checks:"]

        demos = load_pkl(self.description_demo_file)

        for cat_name in demos:
            demonstrations=demos[cat_name][:num_demo]
            for demo in demonstrations:
                cur_text=[]
                media=re.findall(r'\(.*?\)',demo['media'])
                if len(media):
                    removed=media[0]
                    media=demo['media'].split(removed)[0].strip()
                else:
                    media=demo['media']

                head="**"+media+"**"
                details=demo['details']
                details="".join(details)
                cur_text.append(head)

               
                wiki=demo['wiki']
                if 'title' not in wiki:
                    wiki='It does not have a wiki page'
                else:
                    wiki=wiki['summary']
                    cur_text.append('\nHere is its wiki information: '+wiki)

            
                articles=demo['articles']
                if len(articles):
                    titles=[article['title'] for article in articles][:10]
                    titles='\t'.join(titles)
                else:
                    titles='No articles from the website available'

                cur_text.append('\nHere are some article titles from the website: '+titles)
            cur_text.append("Credibility:\t"+cat_name)
            cur_text.append("\n**Background Check**:\t"+details)
            cur_text='\n'.join(cur_text)
            all_texts.append(cur_text)
        all_texts='\n\n'.join(all_texts)
        return all_texts

    def generate_description(self, evidence_links, num_demo, output_file="./media_bg_generated/output.txt"):
        domains = []
        wiki_collection = []
        article_collection = []
        external_info_collection = []
        descriptions = {}

        for evidence_link in tqdm(evidence_links, desc = "generating media BG descriptions."):
            domain = self.extract_domain(evidence_link)

            if domain in self.credibility_data.keys(): 
                description = self.credibility_data[domain]["details"]
                descriptions[domain] = description
                logger.info(f"found existing description for {domain}")

            elif domain in descriptions:
                logger.info(f"found generated description for {domain}")
                continue
            
            else:
                logger.info(f"generating description for {domain}")
                domains.append(domain)

                if self.google_flag:
                    google_search = GoogleSearch(self.query_file, self.all_evidence_url_file, self.all_evidence_text_file)
                    google_evidence = google_search.google_search(domain)

                    queries = load_json(self.query_file)

                    answers = self.get_answers_from_evidence(domain, queries, google_evidence)  # list of answers
                    ext_prompt = "Google search has revealed some new information:\n\n%s\n\n Update your background check for \"%s\" using the new information..."
                    extra_answers = ""
                    for k, item in enumerate(queries):
                        cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
                        extra_answers += cur_query.replace("\"", "") + " " + answers[k]
                    external_info = ext_prompt % (domain, extra_answers)

                else:
                    external_info = ""

                external_info_collection.append(external_info)

                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain)
                else: 
                    wiki_info = {}

                wiki_collection.append(wiki_info)

                if self.article_flag:
                    valid_article_urls, invalid_domains = collect_article_urls(domain)
                    article_info, invalid_article_urls = extract_articles(valid_article_urls)
                else:
                    article_info = []

                article_collection.append(article_info)

        initial_guesses, init_prompts = self.generate_initial_guess(domains, wiki_collection, article_collection, num_demo)

        prompts = []
        for idx, guess in enumerate(initial_guesses):
            # guess = guess.split("**")[1:]
            # guess = "**" + "**".join(guess)

            final_instruction = f"Now generate description for the target domain {domains[idx]} in no more than 500 words."

            prompts.append(init_guess_sys_prompt + init_prompts[idx] + guess + external_info_collection[idx] + final_instruction)             

        outputs = self.LLM.generate(prompts, self.sampling_params)

        # Write final descriptions to the file
        
        for idx, output in enumerate(outputs):
            # description = "**" + "**".join(clean_text(output.outputs[0].text).split("**")[1:])
            # descriptions[domains[idx]] = description

            description = clean_text(output.outputs[0].text)[:3000]
            descriptions[domains[idx]] = description

        
        results = ""
        for domain, description in descriptions.items():
            results += f"Domain: {domain}\nDescription: {description}\n"
            results += "\n-------------------------------------------\n"

        write_to_file(output_file, results)  # Save to file

        return descriptions

    def create_input_prompt(self, domains_to_label, descriptions, article_collection, wiki_collection, num_demo, output_file = "./media_bg_generated/prompts.txt"):
        logger.info("creating input prompt for media classification")

        input_prompts = []

        for domain in domains_to_label: #loop over to output prompts in a known sequence
            wiki_info = wiki_collection[domain]
            article_info = article_collection[domain]
            details = descriptions[domain]

            if "title" in wiki_info:
                wiki_summary = wiki_info["summary"][:512]
            else:
                wiki_summary = "None"
            
            if len(article_info) > 0:
                article = article_info[0]["cleaned_text"][:512]
            else:
                article = "None"

       
            examples=self.get_valid_examples(num_demo)
            # use mfbc data for few shot examples

            prompt_temp="Media Description:%s\nArticle:%s\nWiki info:%s\nCredibility:"

            domain_prompt = []
            for exp in examples:
                domain_prompt.append((prompt_temp % (exp["details"], exp["article_info"], exp["wiki_summary"]))+exp["credibility"])

            domain_prompt.append((prompt_temp % (details, article, wiki_summary)))

            domain_prompt="\n\n".join(domain_prompt)

            input_prompts.append(domain_prompt)

        
        for idx, domain in enumerate(domains_to_label):
            result = f"domain: {domain}\n"
            result += f"prompt: {input_prompts[idx]}\n"
            result += "\n------------------------------------------------\n"
            write_to_file(output_file, result)
            
        return input_prompts


    def get_valid_examples(self, num_demo):
        logger.info("Getting valid examples for media classification")

        
        selection = {k: self.predict_label_demo[k] for k in random.sample(list(self.predict_label_demo.keys()), num_demo)}

        examples = []
        for key, item in selection.items():
            article = item["articles"][0]["cleaned_text"][:512]
            wiki_summary = item["wiki"]["summary"][:512]

            examples.append({
                "media": item["media"],
                "details": item["details"],
                "url" : item["url"],
                "article_info":article,
                "wiki_summary":wiki_summary,
                "credibility": item["credibility"]
            })

        return examples
    
    def predict_credibility_level(self, evidence_links, num_demo, output_file = "./media_bg_generated/labels.txt"):

        results = {}

        evidence_links_to_label = []
        domains_to_label = []
        wiki_collection = {}
        article_collection = {}

        for evidence_link in evidence_links:
            logger.info("Predicting credibility level for evidence link: %s", evidence_link)

            domain = self.extract_domain(evidence_link)
            if domain == evidence_link:
                print("we cannot extract domain for link:", evidence_link)
                results[domain] = "low"
                # return low credibility if domain is not found as it is highly possible to be a suspicous link

            if domain in self.credibility_data.keys(): 
                # if yes, use existing credibility label, which saves time and resources
                logger.info("Using existing credibility level for domain {domain}")
                
                credibility_level = self.credibility_data[domain]["credibility"].lower()
                results[domain] = credibility_level

            else:
                # if no, generate label using collected information
                logger.info("No existing credibility level for domain {domain}, now generating...")

                evidence_links_to_label.append(evidence_link)
                domains_to_label.append(domain)

                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain)
                else:
                    wiki_info = {}

                wiki_collection[domain] = wiki_info

                if self.article_flag:
                    valid_article_urls, invalid_domains = collect_article_urls(domain)
                    article_info, invalid_article_urls = extract_articles(valid_article_urls)
                else:
                    article_info = []

                article_collection[domain] = article_info
    
                
        descriptions = self.generate_description(evidence_links_to_label, num_demo)

        sys_msg="You are InfoHuntGPT, a world-class AI assistant used by journalists to quickly predict the credibility of new sources. "

        sys_msg+="You need to categorize the credibility of a source into: high, medium or low, based on provided information (e.g., articles, wiki information, description of the media source)."

        msgs = self.create_input_prompt(domains_to_label, descriptions, article_collection, wiki_collection, num_demo)


        prompts = [sys_msg+msg for msg in msgs]

        outputs = self.LLM.generate(prompts, self.sampling_params)

        valid_preds=['low','medium','high']

        has_valid_pred = False
        for idx, output in enumerate(outputs):
            output = output.outputs[0].text.lower()
            predict_result = f"domain: {domains_to_label[idx]}\n outputs: {output}\n"
            predict_result += "\n-----------------------------------\n"
            write_to_file(output_file, predict_result)
            
            for valid_pred in valid_preds:
                if valid_pred in output:
                    results[domains_to_label[idx]] = valid_pred 
                    has_valid_pred = True
                    break
            if not has_valid_pred:
                results[domains_to_label[idx]] = 'unknown' 
        return results


def main():
    # args = argparse.ArgumentParser(description='')
    set_seed(412)

    query_file = "../data/dataset/google_queries.json"
    all_evidence_url_file = "./media_bg_collected/google_evidence_urls.pkl"
    all_scraped_text_file = "./media_bg_collected/google_evidence_text.pkl"
    credibility_data_file = "./media_bg_collected/media_credibility_data.pkl"
    label_demo_file = "./media_bg_collected/label_demos.pkl"
    description_demo_file = "./media_bg_collected/description_demos.pkl"



    # google = GoogleSearch(query_file, all_evidence_url_file, all_scraped_text_file)
    # results = google.google_search("4Chan.org")
    # print("-----------------------------------")
    # print("final_results:" ,results)

    def extract_domain(url):
        if url:
            domain_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            match = re.search(domain_pattern, url)
            if match:
                domain = match.group(1)
                return domain
    
        return url  


    credibility_data = load_pkl(credibility_data_file)
    media_saved = set(credibility_data.keys())
            

    filtered_results_path = "../data/dataset/filtered_results.json"
    filtered_results = load_json(filtered_results_path)


    domains_to_google = set()
    for item in filtered_results:
        for evidence in item['evidence_url']:
            url = evidence['original_link']
            domain = extract_domain(url)
            if domain not in media_saved:
                domains_to_google.add(domain)

    google_search = GoogleSearch(query_file, all_evidence_url_file, all_scraped_text_file)

    domains_to_google = list(domains_to_google)
    domains_to_google = sorted(domains_to_google)
    for domain in tqdm(domains_to_google):
        google_evidence = google_search.google_search(domain)
    



    # sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=412, stop = '*** END')

    # LLM = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.95)


    # media_checker = MediaCheck(
    #     LLM = LLM,
    #     sampling_params = sampling_params,
    #     wiki_flag = True,
    #     article_flag = True, 
    #     google_flag = False, 
    #     credibility_data_file = credibility_data_file, 
    #     label_demo_file = label_demo_file, 
    #     description_demo_file = description_demo_file, 
    #     query_file = query_file, 
    #     all_evidence_url_file = all_evidence_url_file, 
    #     all_evidence_text_file = all_scraped_text_file
    # )

    # debug = ["https://4Chan.org/", 
    # "https://www.24jours.com/lukraine-va-ac", 
    # "https://www.republicworld.com/india/politics/"]

    # # pred = media_checker.generate_description(debug, 0)
    # # print("---------------------result-----------------------------")
    # # print(pred)

    # label = media_checker.predict_credibility_level(
    #     evidence_links = debug,
    #     num_demo = 1
    # )
    # print("---------------------result-----------------------------")
    # print(label)

    # pass



if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print("CUDA:", os.environ["CUDA_VISIBLE_DEVICES"])
    main()



    
    
