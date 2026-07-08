import json, glob, re
from collections import Counter

def norm(p):
    p = p.strip().lower()
    if p.startswith('yes'): return 'yes'
    if p.startswith('no'): return 'no'
    m = re.search(r'\b(yes|no)\b', p)
    return m.group(1) if m else 'EMPTY'

def f1(tp, fp, fn):
    pr = tp/(tp+fp) if tp+fp else 0
    rc = tp/(tp+fn) if tp+fn else 0
    return 2*pr*rc/(pr+rc) if pr+rc else 0

def stats(f):
    d = json.load(open(f))
    correct=empty=tp=fp=fn=tn=0
    for x in d:
        g = 'yes' if x['ground_truth'].strip()=='Supported' else 'no'
        p = norm(x['prediction'])
        if p=='EMPTY': empty+=1
        if g==p: correct+=1
        if g=='yes' and p=='yes': tp+=1
        elif g=='no' and p=='yes': fp+=1
        elif g=='yes' and p=='no': fn+=1
        elif g=='no' and p=='no': tn+=1
    acc=correct/len(d)*100
    macro=(f1(tp,fp,fn)+f1(tn,fn,fp))/2*100
    return acc, macro, empty, fp  # fp = no->yes 오답

models = {'Qwen2':'Qwen2-7B-Instruct', 'Llama':'Llama-3.1-8B-Instruct', 'Mistral':'Mistral-7B-Instruct-v0.3'}
methods = ['Explain', 'ConflictLocSoft', 'ConflictLocEvidence']
label = {'Explain':'SBAexp (baseline)', 'ConflictLocSoft':'ConflictLocSoft', 'ConflictLocEvidence':'ConflictLocEvidence'}

for mkey, mstr in models.items():
    print(f"\n===== {mkey} =====")
    print(f"{'method':<24}{'Acc':>7}{'MacroF1':>9}{'empty':>7}{'no->yes':>9}")
    print('-'*56)
    for meth in methods:
        matches = glob.glob(f'results/results_all_media/Top_5_chunks_{meth}_MediaBD_True_model_{mstr}.json')
        if not matches:
            print(f"{label[meth]:<24}{'(없음)':>7}")
            continue
        acc, macro, empty, fp = stats(matches[0])
        print(f"{label[meth]:<24}{acc:>7.2f}{macro:>9.2f}{empty:>7}{fp:>9}")