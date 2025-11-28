"""
Sub-query retrieval for diversity ranking.

Implements retrieval using ground-truth sub-questions (subtopics)
followed by post-hoc fusion strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils import batch_cosine_similarity


def retrieve_for_subquery(
    subquery_emb: np.ndarray,
    doc_ids: List[str],
    doc_embs: np.ndarray,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Retrieve documents for a single sub-query.
    
    Args:
        subquery_emb: Sub-query embedding vector.
        doc_ids: List of document IDs.
        doc_embs: Document embeddings matrix (n_docs x dim).
        top_k: Number of documents to retrieve.
        
    Returns:
        List of (doc_id, score) tuples sorted by score.
    """
    # Compute similarities
    scores = batch_cosine_similarity(subquery_emb, doc_embs)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = [(doc_ids[idx], float(scores[idx])) for idx in top_indices]
    return results


def retrieve_for_all_subqueries(
    main_qid: str,
    subquery_embeddings: Dict[str, np.ndarray],
    doc_ids: List[str],
    doc_embs: np.ndarray,
    top_k_per_subquery: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Retrieve documents for all sub-queries of a main query.
    
    Args:
        main_qid: Main query ID.
        subquery_embeddings: Dictionary {subquery_id: embedding}.
        doc_ids: List of document IDs.
        doc_embs: Document embeddings matrix.
        top_k_per_subquery: Documents to retrieve per sub-query.
        
    Returns:
        Dictionary {subquery_id: [(docid, score), ...]}.
    """
    subquery_results = {}
    
    for sq_id, sq_emb in subquery_embeddings.items():
        subquery_results[sq_id] = retrieve_for_subquery(
            subquery_emb=sq_emb,
            doc_ids=doc_ids,
            doc_embs=doc_embs,
            top_k=top_k_per_subquery
        )
    
    return subquery_results


def subquery_retrieval_from_candidates(
    subquery_emb: np.ndarray,
    candidate_doc_ids: List[str],
    doc_embeddings: Dict[str, np.ndarray],
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Retrieve from a candidate set (e.g., from initial retrieval).
    
    Args:
        subquery_emb: Sub-query embedding.
        candidate_doc_ids: List of candidate document IDs.
        doc_embeddings: Dictionary {docid: embedding}.
        top_k: Number of documents to return.
        
    Returns:
        List of (doc_id, score) tuples.
    """
    # Filter to documents that have embeddings
    valid_docs = [docid for docid in candidate_doc_ids if docid in doc_embeddings]
    
    if not valid_docs:
        return []
    
    doc_embs = np.array([doc_embeddings[docid] for docid in valid_docs])
    
    scores = batch_cosine_similarity(subquery_emb, doc_embs)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    
    return [(valid_docs[idx], float(scores[idx])) for idx in sorted_indices]


def subquery_rerank_candidates(
    main_run: Dict[str, List[Tuple[str, float]]],
    subquery_embeddings: Dict[str, Dict[str, np.ndarray]],
    doc_embeddings: Dict[str, np.ndarray],
    top_k_per_subquery: int = 100
) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Re-rank candidate documents using sub-query embeddings.
    
    Args:
        main_run: Main query run {qid: [(docid, score), ...]}.
        subquery_embeddings: {main_qid: {sq_id: embedding}}.
        doc_embeddings: Dictionary {docid: embedding}.
        top_k_per_subquery: Documents to return per sub-query.
        
    Returns:
        Nested dictionary {qid: {sq_id: [(docid, score), ...]}}.
    """
    subquery_runs = {}
    
    for qid, doc_list in main_run.items():
        candidate_doc_ids = [docid for docid, _ in doc_list]
        
        if qid not in subquery_embeddings:
            continue
        
        subquery_runs[qid] = {}
        
        for sq_id, sq_emb in subquery_embeddings[qid].items():
            subquery_runs[qid][sq_id] = subquery_retrieval_from_candidates(
                subquery_emb=sq_emb,
                candidate_doc_ids=candidate_doc_ids,
                doc_embeddings=doc_embeddings,
                top_k=top_k_per_subquery
            )
    
    return subquery_runs
