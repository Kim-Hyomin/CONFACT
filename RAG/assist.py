import pickle

with open("/home/yuhao/UncertainQA/RAG/results/retrieved_sentences.pkl", "rb") as f:
    data = pickle.load(f)

print("hi")
print(data)