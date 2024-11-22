import argparse
import logging
from typing import List, Dict
import pickle as pkl
import json
import vllm
from vllm import SamplingParams
import re
from sentence_retriever import SentenceProcessor
from sentence_retriever import BM25Retriever
from media_check import MediaCheck
from media_check import clean_text
from media_check import write_to_file
import nltk
import torch
import numpy as np
import random
import os
import time
from config import parse_args

nltk.download('punkt_tab')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logfile.log"),    
        logging.StreamHandler()               
    ]
)

main_logger = logging.getLogger('__name__')



def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

def load_json(path):
    data=json.load(open(path,'r'))
    return data

def log_hyperpara(opt):
    dic = vars(opt)
    for k,v in dic.items():
        main_logger.info(f'{k} : {v}')
        print(k + ' : ' + str(v))


        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def extract_ground_truth(source_data):
    label_mapping = {
            "Supported": "yes",
            "Refuted": "no",
        }

    return [label_mapping.get(entity['label'], "unsure") for entity in source_data]



class RetrievalProcessor:
    def __init__(self, method, sampling_params, LLM):
        self.method = method
        self.sampling_params = sampling_params
        self.LLM = LLM

        main_logger.info(f"Retrieval Processor initialized with method: {method}")

    def process(self, qns_entities, args, bm25_k = 100):
        
        if self.method == "BM25":
            main_logger.info("Using direct BM25 results")

            retriever = BM25Retriever(bm25_k)

            retrieved_results = retriever.retrieve_for_file(qns_entities, args.sentence_store_path)

            # {
            # 'question_id': question_id,
            # 'question': question,
            # 'top_k': [
            #     {
            #         'sentence': sentence text,
            #         'original_link': original_link,
            #         'score': scores
            #     },...]
            # }

            return retrieved_results
        
        elif self.method == "filtered":
            main_logger.info("Filtering retrieved results based on credibility")

            retriever = BM25Retriever(bm25_k)
            evidence_data = load_pkl(args.store_path)
            retrieved_results = retriever.retrieve_for_qn(item, evidence_data)
            # {
            # 'question_id': question_id,
            # 'question': question,
            # 'top_k': top_K,
            # }
            top_k = retrieved_results["top_k"]
            # top_K = [
            #     {
            #         'sentence': sentence_corpus[i]['sentence'],
            #         'original_link': sentence_corpus[i]['original_link'],
            #         'score': scores[i]
            #     },,[]
            #     ]

            media_checker = MediaCheck(
                LLM = self.LLM,
                sampling_params = self.sampling_params,
                wiki_flag = args.wiki_flag,
                article_flag = args.article_flag, 
                google_flag = args.google_flag, 
                credibility_data_file = args.credibility_data_file, 
                label_demo_file = args.label_demo_file, 
                description_demo_file = args.description_demo_file, 
                query_file = args.search_query_file, 
                all_evidence_url_file = args.all_evidence_url_file, 
                all_evidence_text_file = args.all_scraped_text_file
            )

            filtered_results = []
            for result in topk:
                evidence_link = result['original_link']
                label = media_checker.predict_credibility_level(
                    evidence_link = evidence_link,
                    num_demo = 1
                )
                if label != 'low':
                    
                    description = media_checker.generate_description(
                        evidence_link=evidence_link, 
                        num_demo=1)

                    result["details"] = description

                    filtered_results.append(result)


            #  最后选 args.k 个
            return filtered_results


        elif self.method == "reranked":
            main_logger.info("Re-ranking retrieved results based on credibility")
            # ######################### to implement
            pass

        else:
            raise ValueError(f"Invalid retrieval method: {self.method}")


class PromptProcessor:
    def __init__(self, prompt_type, with_MediaBG = False):
        self.prompt_type = prompt_type
        self.with_MediaBG = with_MediaBG
        main_logger.info(f"Prompt Processor initialized with prompt type: {prompt_type}")

    def generate_prompt(self, question, contexts):
            # contexts
            # top_K = [
            #     {
            #         'sentence': sentence_corpus[i]['sentence'],
            #         'original_link': sentence_corpus[i]['original_link'],
            #         'score': scores[i]
            #     },,[]
            #     ]

        if self.prompt_type == "DiscernAndAnswer":
            return self._generate_discern_and_answer_prompt(question, contexts)

        elif self.prompt_type == "ExplainAndAnswer":
            return self._generate_explain_and_answer_prompt(question, contexts)

        else:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    def _generate_discern_and_answer_prompt(self, question, contexts):
        prompt = "You are given a question and several pieces of evidence. Your task is to analyze the evidence and provide a concise answer.\n\n"

        prompt += f"Question: {question}\n\n"

        prompt += (
            "For each piece of evidence below, assess its relevance and credibility. "
            "Decide if the evidence is reliable and useful in answering the question. "
            "If the evidence is useful, include it in your final answer; if not, exclude it.\n\n"
        )

        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                )

        prompt += (
            "After reviewing all the evidence, provide your answer to the question at the end of the analysis. "
            "Start your final answer with 'Final Answer:' and ensure it is clearly separated from the evidence analysis.\n"
            "Your final answer should be either 'yes' or 'no'.\n"
            "Make sure to include only one final answer and do not include any additional text after it.\n\n"
            "Example:\n"
            "Final Answer: yes\n"
        )


        return prompt



    
    def _generate_explain_and_answer_prompt(self, question, contexts):

        prompt = "You are given a question and several pieces of evidence. Your task is to analyze the evidence and provide a concise answer.\n\n"

        prompt += f"Question: {question}\n\n"
        prompt += (
            "Given the following evidence, first explain your reasoning for any contradictions or conflicting information. "
            "Then provide a concise answer to the question. Make sure your reasoning is clear and addresses any inconsistencies.\n"
        )
 
        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:

                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                )


        prompt += (
            "\nAfter your reasoning, provide your final answer to the question. "
            "Start your answer with 'Final Answer:' and clearly separate it from the rest of your analysis.\n"
            "Your final answer should be either 'yes' or 'no'. "
            "Include only one final answer, and avoid adding any additional explanation after it.\n\n"
            "Example:\n"
            "Final Answer: yes\n"
        )

        return prompt

def create_prompt_BM25_wo_BG(retrieved_entities):
    # [{
    # 'question_id': question_id,
    # 'question': question,
    # 'top_k': [top_K],
    # }]
    # top_k = [
    #     {
    #         'sentence': sentence_corpus[i]['sentence'],
    #         'original_link': sentence_corpus[i]['original_link'],
    #         'score': scores[i]
    #     },,[]
    #     ]
    prompts = []

    for qn_entity in retrieved_entities:
        question_text = qn_entity['question']
        evidence_list = qn_entity['top_k']

        prompt = "Given the relevant information provided, answer the question concisely.\n\n"

        prompt += f"Question: {question_text}\n\n"

        # Add evidence
        prompt += "Evidence:\n"
        for evidence in evidence_list:
            prompt += f"- {evidence['sentence']}\n"

        # Final instruction
        prompt += "\nBased on the evidence, answer 'yes' or 'no'. Your answer:"

        prompts.append(prompt)

    return prompts


# Zero-Shot Processing
class ZeroShotProcessor:
    def __init__(self):
        main_logger.info("Zero-Shot Processor initialized")

    def zero_shot_prompt(self, query):
        main_logger.info(f"Creating Zero-Shot prompt for query: {query}")
        return ""


class InferenceEngine:
    def __init__(self, llm, sampling_params):
        main_logger.info("Inference Engine initialized")
        self.LLM = llm
        self.sampling_params = sampling_params
        self.predictions = []

    def infer(self, prompts, method, output_file = "./results/NoMedia_explain_inference.txt"):

        start_time = time.time()
        # Get the CUDA device being used
        if torch.cuda.is_available():
            cuda_device_id = torch.cuda.current_device()
            cuda_device_name = torch.cuda.get_device_name(cuda_device_id)
            cuda_info = f"CUDA Device ID: {cuda_device_id}, Device Name: {cuda_device_name}"
        else:
            cuda_info = "No CUDA device available"

        main_logger.info(f"Using {cuda_info}")

        
        main_logger.info(f"Running inference on {len(prompts)} prompts")
        
        outputs = self.LLM.generate(prompts, self.sampling_params)

        if method == "direct":
            for idx, output in enumerate(outputs):
                
                result = clean_text(output.outputs[0].text).strip()
                answer = result.split(" ")[0].lower()
                self.predictions.append(answer)

                results = f"prompt: {prompts[idx]}\n\n"
                results += f"Final answer: {answer}\n\n"
                results += f"output: {result}\n"
                results += "\n\n-----------------------------------------------------\n\n"
                write_to_file(output_file, results)
        
        elif method == "discern" or method == 'explain':
            for idx, output in enumerate(outputs):
                result = clean_text(output.outputs[0].text).strip().lower()
                pattern = r"final answer:\s*(yes|no)\b"
                match = re.search(pattern, result)
                if match:
                    answer =  match.group(1).strip()
                else:
                    answer = ""
                self.predictions.append(answer)
                results = f"prompt: {prompts[idx]}\n\n"
                results += f"Final answer: {answer}\n\n"
                results += f"output: {result}\n"
                results += "\n\n-----------------------------------------------------\n\n"
                write_to_file(output_file, results)

        end_time = time.time()
        duration = end_time - start_time
        duration_message = f"Inference completed in {duration:.2f} seconds.\n"
        duration_message += f"CUDA Device: {cuda_info}\n"
        with open("runtime_tracking.txt", "a") as f:
            f.write(duration_message)

        return self.predictions

    def evaluate(self, ground_truth):
        main_logger.info(f"Evaluating predictions")

        em = 0
        for idx, prediction in enumerate(self.predictions):
            if ground_truth[idx] in prediction:
                em += 1

        stat = em/len(self.predictions)

        return stat

def save_results(qns, predictions, output_file = "./results/NoMedia_explain.json"):
    results = []
    for idx, prediction in enumerate(predictions):
        qn = qns[idx]['question']
        qn_id = qns[idx]['id']
        ground_truth = qns[idx]['label']
        results.append({"id": qn_id, "question": qn, "ground_truth": ground_truth, "prediction": prediction})

    with open(output_file, 'w') as f:
        json.dump(results, f)
    
    return 



if __name__ == "__main__":

    args = parse_args()
    log_hyperpara(args)

    set_seed(args.seed)

    sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=412, stop = '*** END')
    
    LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.95)
    

# bm25 without bg

    # sentence_processor = SentenceProcessor()
    # sentence_processor.store_sentences(args.source, args.sentence_store_path)

    
    # processor = RetrievalProcessor(
    #     method = args.method, 
    #     sampling_params = sampling_params, 
    #     LLM = LLM)

    # qns_entities = load_pkl(args.source)

    # qns_entities = qns_entities[:5]

    # processed_results = processor.process(qns_entities, args, bm25_k = args.k)

    # processed_results = load_pkl("./results/retrieved_sentences.pkl")

    # prompts = create_prompt_BM25_wo_BG(processed_results)


    
    # engine = InferenceEngine(LLM, sampling_params)
    # predictions = engine.infer(prompts)

    # ground_truths = extract_ground_truth(qns_entities)

    # stat = engine.evaluate(ground_truths)
    # print("stat: ", stat)

    # save_results(qns_entities, predictions)


# discern

    # qns_entities = load_pkl(args.source)

    # qns_entities = qns_entities

    # processed_results = load_pkl("./results/retrieved_sentences.pkl")

    # processed_results = processed_results


    # prompt_generator = PromptProcessor("DiscernAndAnswer")

    # prompts = []
    # for entity in processed_results:
    #     qn = entity['question']
    #     contexts = entity['top_k']
    #     prompt = prompt_generator.generate_prompt(qn, contexts)
    #     prompts.append(prompt)

    
    # engine = InferenceEngine(LLM, sampling_params)
    # predictions = engine.infer(prompts, 'discern')

    # ground_truths = extract_ground_truth(qns_entities)

    # stat = engine.evaluate(ground_truths)
    # print("stat: ", stat)

    # save_results(qns_entities, predictions)


#explain
    # qns_entities = load_pkl(args.source)

    # qns_entities = qns_entities

    # processed_results = load_pkl("./results/retrieved_sentences.pkl")

    # processed_results = processed_results


    # prompt_generator = PromptProcessor("ExplainAndAnswer")

    # prompts = []
    # for entity in processed_results:
    #     qn = entity['question']
    #     contexts = entity['top_k']
    #     prompt = prompt_generator.generate_prompt(qn, contexts)
    #     prompts.append(prompt)

    
    # engine = InferenceEngine(LLM, sampling_params)
    # predictions = engine.infer(prompts, 'explain')

    # ground_truths = extract_ground_truth(qns_entities)

    # stat = engine.evaluate(ground_truths)
    # print("stat: ", stat)

    # save_results(qns_entities, predictions)

#explain with BG
    qns_entities = load_pkl(args.source)
    qns_entities = qns_entities

    processed_results = load_pkl("./results/top100_retrieved_sentences.pkl")


    search_query_file = "../data/dataset/google_queries.json"
    all_evidence_url_file = "./media_bg_collected/google_evidence_urls.pkl"
    all_scraped_text_file = "./media_bg_collected/google_evidence_text.pkl"
    credibility_data_file = "./media_bg_collected/media_credibility_data.pkl"
    label_demo_file = "./media_bg_collected/label_demos.pkl"
    description_demo_file = "./media_bg_collected/description_demos.pkl"


    media_checker = MediaCheck(
        LLM = LLM,
        sampling_params = sampling_params,
        wiki_flag = True,
        article_flag = True, 
        google_flag = False, 
        credibility_data_file = credibility_data_file, 
        label_demo_file = label_demo_file, 
        description_demo_file = description_demo_file, 
        query_file = search_query_file, 
        all_evidence_url_file = all_evidence_url_file, 
        all_evidence_text_file = all_scraped_text_file
    )

    evidence_links = set()
    for qn in processed_results:
        for evidence in qn['top_k']:
            evidence_links.add(evidence['original_link'])

    evidence_links = sorted(list(evidence_links))
    # evidence_links = evidence_links[:5]

    print("number of evidence links:", len(evidence_links))

    descriptions = media_checker.generate_description(evidence_links, num_demo = 1, output_file = "./media_bg_generated/output.txt")


    with open("./media_bg_generated/top100_generated_description.json", "wb") as f:
        json.dump(descriptions, f, indent = 4)
        
j

    # prompt_generator = PromptProcessor("ExplainAndAnswer", with_MediaBG = True)

    # prompts = []
    # for entity in processed_results:
    #     qn = entity['question']
    #     contexts = entity['top_k']
    #     prompt = prompt_generator.generate_prompt(qn, contexts)
    #     prompts.append(prompt)

    
    # engine = InferenceEngine(LLM, sampling_params)
    # predictions = engine.infer(prompts, 'explain')

    # ground_truths = extract_ground_truth(qns_entities)

    # stat = engine.evaluate(ground_truths)
    # print("stat: ", stat)

    # save_results(qns_entities, predictions)

        



















#---------------------------------------------------------------------------------------
    # prediction = engine.infer(prompt)

    # print("------------------------------------------------")
    # print("number of claims after filtering:", len(processed_results))

    # query = "is estimated physician-patient ratio Nigeria one doctor 4,000 5,000 patients"

    # # Step 3 & 4: Generate Prompt
    # if args.zero_shot:
    #     zero_shot_processor = ZeroShotProcessor()
    #     prompt = zero_shot_processor.zero_shot_prompt(args.query)
    # else:
    #     prompt_processor = PromptProcessor(args.prompt_type)
    #     # prompt = prompt_processor.generate_prompt(args.query, processed_results)
    #     prompt = prompt_processor.generate_prompt(query, processed_results)

    # # Step 5: Inference and Evaluation
    # engine = InferenceEngine()
    # prediction = engine.infer(prompt)
    # print("------------------------------------------")
    # print(prediction)


    # # # Placeholder: Add ground truth here for evaluation
    # # em_score = engine.evaluate("ground_truth_placeholder", prediction)

    # # logging.info(f"Final EM Score: {em_score}")

