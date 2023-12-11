import json
import itertools
from datasets import load_dataset
from tqdm import tqdm

from .en2vi_vinai_translate import translate_en2vi

dataset = load_dataset("meta-math/MetaMathQA")

# Select only simple and short questions
dataset = dataset.filter(lambda example: ('MATH' not in example['type']) and (len('query') + len(example['response']) < 256))

# Translate in batch to speed-up process
L, R = 0, -1
sub = dataset['train'][:]
BS = 4
trans_query = itertools.chain.from_iterable([translate_en2vi(sub['query'][i:i+BS]) for i in tqdm(range(0, len(sub['query']), BS))])
trans_ans = itertools.chain.from_iterable([translate_en2vi(sub['response'][i:i+BS]) for i in tqdm(range(0, len(sub['query']), BS))])

trans_query = list(trans_query)
trans_ans = list(trans_ans)

outs = []
for q, a in zip(trans_query, trans_ans):
    lst = a.split("Câu trả lời là:")
    explanation = "Câu trả lời là:".join(lst[:-1]).strip()
    ans = lst[-1].strip()
    # print(q, explanation, ans)
    out = {
        'question': q,
        'choices': None,
        'explanation': explanation,
        'answer': ans
    }
    outs.append(out)

# Save question in json format
with open("D8K-256w.json","w", encoding='utf8') as f:
    json.dump(outs, f, ensure_ascii=False)