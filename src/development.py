import os
from crux.tools import load_run_or_qrel

home_dir = os.path.expanduser("~/")



dataset_dict = {}
for tag in ['msmarco-passage/trec-dl-2019', 'msmarco-passage/trec-dl-2020']:
    qrel = load_run_or_qrel(
        f"{home_dir}/datasets/ir_datasets/{dataset_name}/qrels",
        threshold=-1
    )

    dataset_dict[tag] = []
    for qid in qrel:
        for docid, relevance in qrel[qid].items():
            dataset_dict.append({
                'query_id': qid, 
                'query_text': None, 
                'positive_document_ids': 

            })

