import torch
import time
import re
import json
import vllm
from vllm import SamplingParams
from config import parse_args
from utils import write_to_file, clean_text

args = parse_args()

sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=args.seed, stop = '*** END')

LLM = vllm.LLM(model=args.model, tensor_parallel_size=args.gpu, gpu_memory_utilization=0.9,trust_remote_code=True)


def save_results(qns, predictions, output_file):
    results = []
    for idx, prediction in enumerate(predictions):
        qn = qns[idx]['question']
        qn_id = qns[idx]['id']
        ground_truth = qns[idx]['label']
        results.append({"id": qn_id, "question": qn, "ground_truth": ground_truth, "prediction": prediction})

    with open(output_file , 'w') as f:
        json.dump(results, f, indent=4)
    
    return 

class InferenceEngine:
    def __init__(self):
        # main_logger.info("Inference Engine initialized")
        self.LLM = LLM
        self.sampling_params = sampling_params
        self.predictions = []

    def infer(self, prompts, method, output_file):

        start_time = time.time()
        # Get the CUDA device being used
        if torch.cuda.is_available():
            cuda_device_id = torch.cuda.current_device()
            cuda_device_name = torch.cuda.get_device_name(cuda_device_id)
            cuda_info = f"CUDA Device ID: {cuda_device_id}, Device Name: {cuda_device_name}"
        else:
            cuda_info = "No CUDA device available"

        # main_logger.info(f"Using {cuda_info}")

        # main_logger.info(f"Running inference on {len(prompts)} prompts")
        
        outputs = self.LLM.generate(prompts, self.sampling_params)

        if method == "DirectAnswer":
            for idx, output in enumerate(outputs):
                
                answer = clean_text(output.outputs[0].text).strip().lower()
                self.predictions.append(answer)

                results = f"prompt: {prompts[idx]}\n\n"
                results += f"Answer: {answer}\n"
                results += "\n\n-----------------------------------------------------\n\n"
                write_to_file(output_file, results)
        
        elif method == "DiscernAndAnswer" or method == 'ExplainAndAnswer':
            for idx, output in enumerate(outputs):
                result = clean_text(output.outputs[0].text).strip().lower()
                pattern = r"final answer:\s*(.*)"
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
