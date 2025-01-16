import logging
import re
import random
from tqdm import tqdm
from UncertainQA.utils import load_pkl, load_json
from UncertainQA.utils import clean_text, extract_domain, load_gzip
from UncertainQA.utils import sys_msg, init_guess_sys_prompt, answer_sys_prompt
from google_search import GoogleSearch
from wiki_collection import extract_wiki_info
from article_collection import collect_article_info

logger = logging.getLogger(__name__)

class MediaCheck:
    
    def __init__(self, 
                 LLM, 
                 sampling_params, 
                 wiki_flag = True, 
                 article_flag = True, 
                 google_flag = False, 
                 credibility_data_file = None, 
                 label_demo_file = None, 
                 description_demo_file = None, 
                 query_file = None, 
                 all_evidence_url_file = None, 
                 all_evidence_text_file = None, 
                 article_info_file = None,
                 wiki_info_file = None):

        logger.info("Initializing MediaCheck class.")

        self.LLM = LLM
        self.sampling_params = sampling_params

        self.credibility_data_file = credibility_data_file
        self.predict_label_demo_file = label_demo_file
        self.description_demo_file = description_demo_file
        self.query_file = query_file
        self.all_evidence_url_file = all_evidence_url_file
        self.all_evidence_text_file = all_evidence_text_file
        self.article_info_file = article_info_file
        self.wiki_info_file = wiki_info_file

        self.wiki_flag = wiki_flag
        self.article_flag = article_flag
        self.google_flag = google_flag

        self.credibility_data = load_pkl(credibility_data_file)

    def get_answers_from_evidence(self, domain, queries, google_evidence):

        prompts = []
        no_evidence_query_idx = []
        for k, item in enumerate(queries):
            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
            cur_evid = google_evidence[k]     
            if len(cur_evid):
                evid = str(cur_evid)[0][:3000]
                prompts.append(answer_sys_prompt % (cur_query, evid))
            else:
                evid = "No external evidence available."
                no_evidence_query_idx.append(k)

        logger.info("Generating answers from evidence for domain %s.", domain)

        outputs = self.LLM.generate(prompts, self.sampling_params)
        answers = [clean_text(output.outputs[0].text)[:512] for output in outputs]

        return answers, no_evidence_query_idx


    def generate_initial_guess(self, source_names, wiki_collection, articles_collection, num_demo):
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
        extracted_outputs = [clean_text(output.outputs[0].text)[:3000] for output in outputs]

        return extracted_outputs, prompts


    def get_icl_string(self, num_demo):
        all_texts=["The following are examples of background checks:"]

        demos = load_gzip(self.description_demo_file)

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

    def generate_description(self, evidence_links, num_demo):
        domains = []
        wiki_collection = []
        article_collection = []
        external_info_collection = []
        descriptions = {}

        for evidence_link in tqdm(evidence_links, desc = "generating media BG descriptions."):
            domain = extract_domain(evidence_link)

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

                    answers, no_evidence_query_idx = self.get_answers_from_evidence(domain, queries, google_evidence)  # list of answers
                    ext_prompt = "Google search has revealed some new information:\n\n%s\n\n Update your background check for \"%s\" using the new information..."
                    extra_answers = ""

                    idx = 0
                    for k, item in enumerate(queries):
                        if k not in no_evidence_query_idx:
                            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
                            extra_answers += cur_query.replace("\"", "") + " " + answers[idx]
                            idx+=1
                    external_info = ext_prompt % (domain, extra_answers)

                else:
                    external_info = ""

                external_info_collection.append(external_info)

                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain, self.wiki_info_file)
                else: 
                    wiki_info = {}

                wiki_collection.append(wiki_info)

                if self.article_flag:
                    article_info = collect_article_info(domain, self.article_info_file)
                else:
                    article_info = []

                article_collection.append(article_info)

        initial_guesses, init_prompts = self.generate_initial_guess(domains, wiki_collection, article_collection, num_demo)

        prompts = []
        for idx, guess in enumerate(initial_guesses):

            final_instruction = f"Now generate description for the target domain {domains[idx]} in no more than 500 words."

            prompts.append(init_guess_sys_prompt + init_prompts[idx] + guess + external_info_collection[idx] + final_instruction)             

        outputs = self.LLM.generate(prompts, self.sampling_params)
        
        for idx, output in enumerate(outputs):
            description = clean_text(output.outputs[0].text)[:3000]
            descriptions[domains[idx]] = description
            self.credibility_data[domains[idx]]['details'] = description

        return descriptions

    def create_input_prompt(self, domains_to_label, descriptions, article_collection, wiki_collection, num_demo):
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

       
            examples=self.get_valid_examples(num_demo) # use mbfc data for few shot examples

            prompt_temp="Media Description:%s\n\nArticle:%s\n\nWiki info:%s\n\nCredibility:"

            domain_prompt = []
            for exp in examples:
                if isinstance(exp["details"], list):
                    details = " ".join(exp["details"])
                domain_prompt.append((prompt_temp % (details, exp["article_info"], exp["wiki_summary"]))+exp["credibility"])

            domain_prompt.append((prompt_temp % (details, article, wiki_summary)))

            domain_prompt="\n\n".join(domain_prompt)

            input_prompts.append(domain_prompt)

        return input_prompts


    def get_valid_examples(self, num_demo):
        logger.info("Getting valid examples for media classification")

        predict_label_demo = load_gzip(self.predict_label_demo_file) 
        
        selection = {k: predict_label_demo[k] for k in random.sample(list(predict_label_demo.keys()), num_demo)}

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
    
    def predict_credibility_level(self, evidence_links, num_demo):

        results = {}
        evidence_links_to_label = []
        domains_to_label = []
        wiki_collection = {}
        article_collection = {}

        for evidence_link in evidence_links:
            logger.info("Predicting credibility level for evidence link: %s", evidence_link)

            domain = extract_domain(evidence_link)

            if domain in self.credibility_data.keys(): 
                # if yes, use existing credibility label
                logger.info("Using existing credibility level for domain {domain}")
                
                credibility_level = self.credibility_data[domain]["credibility"].lower()
                results[domain] = credibility_level

            else:
                # if no, generate label using collected information
                logger.info("No existing credibility level for domain {domain}, now generating...")

                evidence_links_to_label.append(evidence_link)
                domains_to_label.append(domain)

                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain, self.wiki_info_file)
                else:
                    wiki_info = {}

                wiki_collection[domain] = wiki_info

                if self.article_flag:
                    article_info = collect_article_info(domain, self.article_info_file)
                else:
                    article_info = []

                article_collection[domain] = article_info
    
        descriptions = self.generate_description(evidence_links_to_label, num_demo)

        msgs = self.create_input_prompt(domains_to_label, descriptions, article_collection, wiki_collection, num_demo)


        prompts = [sys_msg+msg for msg in msgs]

        outputs = self.LLM.generate(prompts, self.sampling_params)

        valid_preds=['low','medium','high']

        for idx, output in enumerate(outputs):
            output = output.outputs[0].text.lower()
            pattern = re.compile('|'.join(valid_preds))
            match = pattern.search(output)

            if match:
                result = match.group(0)
            else:
                result = 'unknown'

            results[domains_to_label[idx]] = result
            self.credibility_data[domains_to_label[idx]]['credibility'] = result

        return results
