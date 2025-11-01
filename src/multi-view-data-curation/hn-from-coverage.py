from tqdm import tqdm 
import argparse
import numpy as np
from datasets import DatasetDict, Dataset
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopics

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='Dataset split to analyze')
parser.add_argument('--tau', type=int, default=3, help='Threshold for answerable subtopics')
args = parser.parse_args()

# main
split = args.split
tau = args.tau

topic = load_topic(split)
subtopics = load_subtopics(split)
run = load_run_or_qrel(f'/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
judge = load_ratings('/exp/scale25/artifacts/crux/crux-researchy/judge/')

dataset_dict = {}
for qid in tqdm(run):

    # stat1: answerable 
    n = len(subtopics[qid])
    n_ans = (np.max([judge[qid][docid] for docid in judge[qid]], 0) >= tau).sum()
    n_unans = n - n_ans

    # stat2
    document_ids = {0.75: [], 0.5: [], 0.25: [], 0.0: [], -1: []}
    coverage_per_doc = []
    coverage_accumulated = []
    for i, docid in enumerate(run[qid]):
        rating = (judge[qid][docid] or [0])
        c = sum(np.array(rating) >= tau) / n
        coverage_per_doc.append(float(c))

        selected = [docid for j, docid in enumerate(run[qid]) if j <= i]
        rating = [judge[qid][docid] for docid in selected if judge[qid][docid]]
        try:
            c_acc = sum(np.max(rating, 0) >= tau) / n
        except:
            c_acc = 0.0
        coverage_accumulated.append(float(c_acc))

        for threshold in [0.75, 0.5, 0.25, 0.0]:
            if c >= threshold:
                document_ids[threshold].append(docid)
                if c == 0:
                    document_ids[-1].append(docid)
                break

    # create datasetlist
    for positive_category, negative_category in [
        ('high', 'zero'),
        ('high', 'low'),
        ('high', 'quarter'),
        ('high', 'half'),
        ('half', 'zero'),
    ]:
        tag = f"pos_{positive_category}.neg_{negative_category}"
        if tag not in dataset_dict:
            dataset_dict[tag] = []

        positive_docs = []
        if positive_category == 'high':
            positive_docs += document_ids[0.75]
        if positive_category == 'half':
            positive_docs += document_ids[0.75] + document_ids[0.5]

        negative_docs = []
        if negative_category == 'zero':
            negative_docs += document_ids[-1]
        if negative_category == 'low':
            negative_docs += document_ids[0.0] + document_ids[-1]
        if negative_category == 'quarter':
            negative_docs += document_ids[0.25] + document_ids[0.0] + document_ids[-1]
        if negative_category == 'half':
            negative_docs += document_ids[0.5] + document_ids[0.25] + document_ids[0.0] + document_ids[-1]

        if len(positive_docs) > 0 and len(negative_docs) > 0:
            dataset_dict[tag].append({
                'query_id': qid, 
                'query_text': topic[str(qid)],
                'positive_document_ids': positive_docs,
                'negative_document_ids': negative_docs,
                'answer': None,
                'source': f'clueweb22-B.tau:{tau}',
            })

## Transform to dataset
dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
print(dataset)
dataset.push_to_hub("DylanJHJ/crux-researchy")
