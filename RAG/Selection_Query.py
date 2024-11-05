import vllm
from vllm import SamplingParams
import time
import re
import json
from datasets import load_dataset
import csv
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
template_prompt  = """
Please evaluate the following statement based on the given criteria. Answer each question concisely, using #*# to separate the question number from your final response (Only Yes/No). Provide a brief explanation.
Criteria:
1. Does the statement contain only English characters?
2. Does this task require in-depth thinking and analysis, indicating that it is a complex “writing” task?
3. Is the expected output likely to exceed 1000 words and consist of multiple paragraphs?
4. Does the statement involve sensitive topics (e.g., politics, explicit content, violence)?
5. Is the primary focus of the task on generating written content rather than requiring mathematical reasoning or coding?



### Example 1:
Input: Is Sanskrit the oldest 语言?
Answer:
1. The statement includes the non-English word "语言" (language in Chinese). #*# No
2. The question only requires factual information and lacks complexity. #*# No
3. A concise factual answer suffices for this query. #*# No
4. The subject of historical linguistics is not sensitive. #*# No
5. The task involves written content as it requires a written answer to a question.  #*# Yes
*** END

### Example 2:
Input: Please summary of this text: The coil designer is obliged to As the electromotive force in the secondary is proportional to the fall in the magnetic field the iron core from the experience of others , is made too , it is greater with a straight determine the length of as the mathe matics for calculating it is too complex although simple and useful in the case of closed circui t transformers too large while if made too is.
1. The statement is written entirely in English. #*# Yes
2. The statement does not require in-depth thinking and analysis, as it mainly asks for a summary of the given text. #*# No
3. A summary of this text would likely be concise and not exceed 1000 words. #*# No
4. The content does not touch on sensitive topics. #*# No
5. The task focuses on generating written content by summarizing a text, not mathematical reasoning or coding. #*# Yes
*** END

### Example 3:
Input: Give me a business plan about the cat litter.
Answer:
1. The statement is written entirely in English #*# Yes
2. Creating a business plan involves detailed thinking and organizing multiple components like market analysis, financial planning, and product strategy. #*# Yes
3. A business plan typically contains multiple sections and is usually longer than 500 word. #*# Yes
4. "Cat litter" does not involve sensitive topics. #*# No
5.  The task is centered on generating written content for a business plan, not involving mathematical or coding activities. #*# Yes
*** END

Input: <Statement to be analyzed here>
Answer:
1. """


with open('/mnt/yuhao/LongWriter/Position_paper/conversation_data_100K_300K.json', 'r', encoding='utf-8') as f:
    inputs = json.load(f)



# 提取每个 content 中的内容并过滤 word_count == 1 的项
prompts_for_inference = []
for conversation, messages in inputs.items():
    for message in messages:
        if message['word_count'] > 1:
            prompts_for_inference.append(message['content'])

# import json
# import pandas as pd

# # 读取 CSV 文件中的 'prompt' 列
# df = pd.read_csv('/mnt/yuhao/LongWriter/agentwrite/filtered_unique_results.csv')

# # 提取 'prompt' 列中的每个元素并加入到 prompts_for_inference 列表中
# prompts_for_inference = df['prompt'].tolist()

# prompts_for_inference = prompts_for_inference[:1000]
prepared_prompts = []
for input_data in prompts_for_inference:
    prepared_prompt = template_prompt.replace("<Statement to be analyzed here>", input_data)
    prepared_prompts.append(prepared_prompt)

# 设置推理参数
sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=6211027, stop = '*** END')

# 初始化vllm推理器
llm = vllm.LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=8, gpu_memory_utilization=0.95)

# 进行推理
start_time = time.time()
outputs = llm.generate(prepared_prompts, sampling_params)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

# 处理推理结果


# 处理推理结果
results = []
for output, input_data in zip(outputs, prompts_for_inference):
    output_text = output.outputs[0].text
    matches = re.findall(r'#\*#\s*(\w+)', output_text)
    extracted_words = {str(i + 1): matches[i] if i < len(matches) else None for i in range(5)}
    results.append({
        #"number": input_data['number'],
        "query": input_data,
        "output": output_text,
        "extracted_words": extracted_words
    })

# 将推理结果保存为json文件
with open('filtered_inference_results_llama_100K_300K.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("已将推理结果保存为filtered_inference_results_llama_100K_300K文件")


# def save_filtered_unique_results_to_csv(results, output_file, threshold=0.5):
#     filtered_results = []
#     prompts = []

#     # 筛选符合条件的行
#     for result in results:
#         if (result['extracted_words'].get('1') == 'Yes' and
#             result['extracted_words'].get('2') == 'Yes' and
#             result['extracted_words'].get('3') == 'Yes' and
#             result['extracted_words'].get('4') == 'No' and
#             result['extracted_words'].get('5') == 'Yes'):
#             filtered_results.append(result)
#             prompts.append(result['query'])

#     # 使用 TF-IDF 向量化并计算余弦相似度去重
#     if prompts:
#         vectorizer = TfidfVectorizer().fit_transform(prompts)
#         vectors = vectorizer.toarray()
#         unique_indices = []

#         for i in range(len(vectors)):
#             is_duplicate = False
#             for j in unique_indices:
#                 similarity = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
#                 if similarity > threshold:
#                     is_duplicate = True
#                     break
#             if not is_duplicate:
#                 unique_indices.append(i)

#         filtered_results = [filtered_results[i] for i in unique_indices]

#     # 将过滤后的结果保存为 CSV 文件
#     with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['prompt', 'output', '1', '2', '3', '4', '5']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         writer.writeheader()
#         for result in filtered_results:
#             row = {
#                 # 'number': result['number'],
#                 'prompt': result['query'],
#                 'output': result['output'],
#                 '1': result['extracted_words'].get('1'),
#                 '2': result['extracted_words'].get('2'),
#                 '3': result['extracted_words'].get('3'),
#                 '4': result['extracted_words'].get('4'),
#                 '5': result['extracted_words'].get('5')
#             }
#             writer.writerow(row)

# # 保存符合条件且去重后的推理结果为csv文件
# save_filtered_unique_results_to_csv(results, 'filtered_inference_results_llama_100K_300K.csv')
# print("已将符合条件且去重后的推理结果保存为filtered_inference_results.csv文件")