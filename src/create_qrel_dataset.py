import os
import ir_datasets
from datasets import DatasetDict, Dataset
from crux.tools import load_run_or_qrel

dataset_dict = {}
home_dir = os.path.expanduser("~/")

## QRELS from TREC
for split in ['msmarco_passage.trec_dl_2019', 'msmarco_passage.trec_dl_2020']:

    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)
    query = {}
    for qid, qtext in d.queries_iter():
        query[qid] = qtext

    qrel = load_run_or_qrel(
        f"{home_dir}/.ir_datasets/{tag}/qrels",
        threshold=-1
    )

    dataset_dict[split] = []
    for qid in qrel:
        # relevances = [relevance for docid, relevance in qrel[qid].items()]

        dataset_dict[split].append({
            'query_id': qid, 
            'query_text': query[str(qid)],
            'positive_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance > 0],
            'negative_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance == 0],
            'answer': None,
            'source': 'msmarco-passage'
        })

## QRELS from MSMARCO
for split in ['msmarco_passage.dev.small']:

    tag = split.replace("_", "-")
    tag = tag.replace(".", "/")
    d = ir_datasets.load(tag)
    query = {}
    for qid, qtext in d.queries_iter():
        query[qid] = qtext

    qrel = load_run_or_qrel(
        f"{home_dir}/.ir_datasets/{tag}/qrels",
        threshold=-1
    )


    with open(f"{home_dir}/.ir_datasets/{tag}/dev.top1000.txt", 'r') as f:
        for line in f:
            qid, docid = line.strip().split()[:2]
            if docid not in qrel[qid]:
                qrel[qid].update({docid: 0})

    dataset_dict[split] = []
    for qid in qrel:
        # relevances = [relevance for docid, relevance in qrel[qid].items()]

        dataset_dict[split].append({
            'query_id': qid, 
            'query_text': query[str(qid)],
            'positive_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance > 0],
            'negative_document_ids': [docid for docid, relevance in qrel[qid].items() if relevance == 0],
            'answer': None,
            'source': 'msmarco-passage'
        })

## Transform to dataset
qrel_dataset = DatasetDict( {key: Dataset.from_list(dataset_dict[key]) for key in dataset_dict})
print(qrel_dataset)
qrel_dataset.push_to_hub("DylanJHJ/Qrels")
