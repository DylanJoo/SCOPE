"""
LANCER-style score summation with min-max normalization.

LANCER (Learning to Aggregate for Complex Queries) combines retrieval scores
from multiple sub-queries using weighted score aggregation.

This implementation provides:
1. Score aggregation with min-max normalization
2. Weighted combinations of sub-query scores
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def get_uniform_weight(n_subqueries: int) -> float:
    """Get uniform weight for a given number of sub-queries."""
    return 1.0 / n_subqueries if n_subqueries > 0 else 0.0


def lancer_score_aggregation(
    subquery_scores: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
    aggregation: str = "sum"
) -> Dict[str, float]:
    """
    Aggregate scores from multiple sub-queries using LANCER-style approach.
    
    Args:
        subquery_scores: Dictionary {subquery_id: {docid: score}}.
        weights: Optional weights for each sub-query (uniform if None).
        normalize: If True, apply min-max normalization per sub-query.
        aggregation: Aggregation method - "sum", "max", or "avg".
        
    Returns:
        Dictionary {docid: aggregated_score}.
    """
    if not subquery_scores:
        return {}
    
    n_subqueries = len(subquery_scores)
    uniform_weight = get_uniform_weight(n_subqueries)
    
    # Set uniform weights if not provided
    if weights is None:
        weights = {sq_id: uniform_weight for sq_id in subquery_scores}
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}
    
    # Apply min-max normalization per sub-query
    normalized_scores = {}
    for sq_id, doc_scores in subquery_scores.items():
        if not doc_scores:
            normalized_scores[sq_id] = {}
            continue
        
        scores = np.array(list(doc_scores.values()))
        
        if normalize:
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                norm_scores = (scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(scores)
            
            normalized_scores[sq_id] = dict(zip(doc_scores.keys(), norm_scores))
        else:
            normalized_scores[sq_id] = doc_scores
    
    # Collect all unique document IDs
    all_docs = set()
    for doc_scores in normalized_scores.values():
        all_docs.update(doc_scores.keys())
    
    # Aggregate scores
    aggregated = {}
    for docid in all_docs:
        doc_scores_list = []
        weighted_sum = 0.0
        
        for sq_id in normalized_scores:
            if docid in normalized_scores[sq_id]:
                score = normalized_scores[sq_id][docid] * weights.get(sq_id, uniform_weight)
                weighted_sum += score
                doc_scores_list.append(normalized_scores[sq_id][docid])
        
        if aggregation == "sum":
            aggregated[docid] = weighted_sum
        elif aggregation == "max":
            aggregated[docid] = max(doc_scores_list) if doc_scores_list else 0.0
        elif aggregation == "avg":
            aggregated[docid] = np.mean(doc_scores_list) if doc_scores_list else 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return aggregated


def lancer_rerank(
    main_query_scores: Dict[str, float],
    subquery_scores: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
    normalize: bool = True,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Re-rank documents combining main query scores with sub-query aggregation.
    
    Final_score = α * main_score + (1-α) * aggregated_subquery_score
    
    Args:
        main_query_scores: Scores from main query {docid: score}.
        subquery_scores: Scores from sub-queries {sq_id: {docid: score}}.
        alpha: Weight for main query (0 to 1).
        normalize: If True, apply min-max normalization.
        top_k: Number of documents to return.
        
    Returns:
        Re-ranked list of (docid, score) tuples.
    """
    # Normalize main query scores
    if normalize and main_query_scores:
        scores = np.array(list(main_query_scores.values()))
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            norm_main = {
                docid: (score - min_score) / (max_score - min_score)
                for docid, score in main_query_scores.items()
            }
        else:
            norm_main = {docid: 0.0 for docid in main_query_scores}
    else:
        norm_main = main_query_scores
    
    # Aggregate sub-query scores
    aggregated_sub = lancer_score_aggregation(
        subquery_scores,
        normalize=normalize,
        aggregation="sum"
    )
    
    # Combine scores
    all_docs = set(norm_main.keys()) | set(aggregated_sub.keys())
    final_scores = {}
    
    for docid in all_docs:
        main_score = norm_main.get(docid, 0.0)
        sub_score = aggregated_sub.get(docid, 0.0)
        final_scores[docid] = alpha * main_score + (1 - alpha) * sub_score
    
    # Sort and return top-k
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


def apply_lancer_to_runs(
    main_run: Dict[str, List[Tuple[str, float]]],
    subquery_runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    alpha: float = 0.5,
    normalize: bool = True,
    top_k: int = 100
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Apply LANCER-style aggregation to runs.
    
    Args:
        main_run: Main query run {qid: [(docid, score), ...]}.
        subquery_runs: Sub-query runs {qid: {sq_id: [(docid, score), ...]}}.
        alpha: Weight for main query scores.
        normalize: If True, apply min-max normalization.
        top_k: Number of documents per query.
        
    Returns:
        Re-ranked run dictionary.
    """
    reranked_run = {}
    
    for qid in main_run:
        # Convert main run to dict
        main_scores = {docid: score for docid, score in main_run[qid]}
        
        # Convert sub-query runs to dict format
        sq_scores = {}
        if qid in subquery_runs:
            for sq_id, sq_list in subquery_runs[qid].items():
                sq_scores[sq_id] = {docid: score for docid, score in sq_list}
        
        if sq_scores:
            reranked_run[qid] = lancer_rerank(
                main_query_scores=main_scores,
                subquery_scores=sq_scores,
                alpha=alpha,
                normalize=normalize,
                top_k=top_k
            )
        else:
            # No sub-queries, keep original ranking
            reranked_run[qid] = main_run[qid][:top_k]
    
    return reranked_run
