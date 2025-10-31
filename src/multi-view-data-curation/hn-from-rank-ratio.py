import math
import numpy as np
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopics

split = 'train'
run1 = load_run_or_qrel(f'/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-{split}-init-q.bm25+qwen3.clueweb22-b.txt')
run2 = load_run_or_qrel(f'/exp/scale25/artifacts/crux/crux-researchy/runs/run.researchy-{split}-init-q.bm25.clueweb22-b.txt')

def convert_score_to_reciprocal_rank(run):
    reciprocal_rank_run = {}
    for qid in run:
        reranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        reciprocal_rank_run[qid] = \
            {docid: rank+1 for rank, (docid, score) in enumerate(reranked)}
    return reciprocal_rank_run


run1_rr = convert_score_to_reciprocal_rank(run1)
run2_rr = convert_score_to_reciprocal_rank(run2)

rank_ratio = {}
for qid in run1_rr:
    rank_ratio[qid] = {docid: math.log( run1_rr[qid][docid] / run2_rr[qid][docid]) for docid in run1_rr[qid]}


# output the document and log-rank-ratio
values = []
with open('log-rank-ratio.txt', 'w') as f:
    for qid in rank_ratio:
        sorted_docs = sorted(rank_ratio[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (docid, log_rr) in enumerate(sorted_docs):
            f.write(f'{qid} Q0 {docid} {rank+1} {log_rr} {run1_rr[qid][docid]}/{run2_rr[qid][docid]}\n')

# report = {'min': [], 'max': [], 'mean': []}
# for qid in rank_ratio:
#     min_value = min(rank_ratio[qid].values())
#     max_value = max(rank_ratio[qid].values())
#     mean_value = np.mean(list(rank_ratio[qid].values()))
#     report['min'].append(math.log(min_value))
#     report['max'].append(math.log(max_value))
#     report['mean'].append(math.log(mean_value))
#
# # Average
# for key in report:
#     print('key', np.mean(report[key]))
