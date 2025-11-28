"""
Maximal Marginal Relevance (MMR) re-ranking.

MMR balances relevance and diversity by iteratively selecting documents
that are both relevant to the query and dissimilar to already selected documents.

Formula: MMR = λ * sim(d, q) - (1-λ) * max(sim(d, d_i)) for d_i in selected set
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils import cosine_similarity, min_max_normalize


def mmr_rerank(
    query_emb: np.ndarray,
    doc_ids: List[str],
    doc_embs: np.ndarray,
    doc_scores: Optional[np.ndarray] = None,
    lambda_param: float = 0.5,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Apply MMR re-ranking to a set of candidate documents.
    
    Args:
        query_emb: Query embedding vector.
        doc_ids: List of document IDs.
        doc_embs: Document embeddings matrix (n_docs x dim).
        doc_scores: Original relevance scores (if None, computed from embeddings).
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
        top_k: Number of documents to return.
        
    Returns:
        List of (doc_id, mmr_score) tuples sorted by MMR score.
    """
    n_docs = len(doc_ids)
    
    if n_docs == 0:
        return []
    
    top_k = min(top_k, n_docs)
    
    # Compute relevance scores if not provided
    if doc_scores is None:
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_normalized = query_emb / query_norm
        else:
            query_normalized = query_emb
            
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        docs_normalized = doc_embs / doc_norms
        
        relevance_scores = docs_normalized @ query_normalized
    else:
        relevance_scores = doc_scores
    
    # Normalize relevance scores to [0, 1]
    relevance_scores = min_max_normalize(relevance_scores)
    
    # Precompute document similarity matrix
    doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)
    docs_normalized = doc_embs / doc_norms
    doc_sim_matrix = docs_normalized @ docs_normalized.T
    
    # MMR selection
    selected_indices = []
    selected_set = set()
    remaining_indices = set(range(n_docs))
    
    results = []
    
    for _ in range(top_k):
        best_idx = None
        best_mmr = float('-inf')
        
        for idx in remaining_indices:
            rel_score = relevance_scores[idx]
            
            # Compute max similarity to selected documents
            if len(selected_indices) == 0:
                max_sim = 0.0
            else:
                sims_to_selected = doc_sim_matrix[idx, selected_indices]
                max_sim = np.max(sims_to_selected)
            
            # MMR formula
            mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            remaining_indices.remove(best_idx)
            results.append((doc_ids[best_idx], float(best_mmr)))
    
    return results


def apply_mmr_to_run(
    run: Dict[str, List[Tuple[str, float]]],
    query_embeddings: Dict[str, np.ndarray],
    doc_embeddings: Dict[str, np.ndarray],
    lambda_param: float = 0.5,
    top_k: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Apply MMR re-ranking to an entire run file.
    
    Args:
        run: Original run dictionary {qid: [(docid, score), ...]}.
        query_embeddings: Dictionary of query embeddings.
        doc_embeddings: Dictionary of document embeddings.
        lambda_param: MMR lambda parameter.
        top_k: Number of documents to return per query.
        
    Returns:
        Re-ranked run dictionary.
    """
    reranked_run = {}
    
    for qid, doc_list in run.items():
        if qid not in query_embeddings:
            reranked_run[qid] = doc_list[:top_k]
            continue
        
        query_emb = query_embeddings[qid]
        
        # Gather document embeddings and scores
        doc_ids = []
        doc_embs = []
        doc_scores = []
        
        for docid, score in doc_list:
            if docid in doc_embeddings:
                doc_ids.append(docid)
                doc_embs.append(doc_embeddings[docid])
                doc_scores.append(score)
        
        if len(doc_ids) == 0:
            reranked_run[qid] = doc_list[:top_k]
            continue
        
        doc_embs = np.array(doc_embs)
        doc_scores = np.array(doc_scores)
        
        # Apply MMR
        reranked_run[qid] = mmr_rerank(
            query_emb=query_emb,
            doc_ids=doc_ids,
            doc_embs=doc_embs,
            doc_scores=doc_scores,
            lambda_param=lambda_param,
            top_k=top_k
        )
    
    return reranked_run
