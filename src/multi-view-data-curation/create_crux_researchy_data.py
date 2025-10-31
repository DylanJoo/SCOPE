import argparse
from crux.tools import load_ratings, load_run_or_qrel
from crux.tools.researchy.ir_utils import load_topic, load_subtopic

def create_dataset(split):
    dataset_dict = {split: []}

    for qid in topic:
        positive_document_ids = []
        negative_document_ids = []

        for docid, relevance in subtopics[qid].items():
            if relevance > 0:
                positive_document_ids.append(docid)
            elif relevance == 0:
                negative_document_ids.append(docid)

        dataset_dict[split].append({
            'query_id': qid,
            'query_text': topic[str(qid)],
            'positive_document_ids': positive_document_ids,
            'negative_document_ids': negative_document_ids,
            'answer': None,
            'source': 'researchy-clueweb22'
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True)
    args = parser.parse_args()


    topic = load_topic(split=args.split)
    subtopics = load_subtopics(split=args.split)
    run = load_run_or_qrel(args.run)
    corpus = load_corpus(args.corpus)
    ratings = load_ratings(args.ratings_path)
