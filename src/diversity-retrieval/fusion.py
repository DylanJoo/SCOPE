"""
Ranklist fusion strategies for diversity retrieval.

Implements:
1. Round-robin fusion: Interleave documents from multiple ranklists
2. Reciprocal Rank Fusion (RRF): Combine scores using reciprocal rank formula
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def round_robin_fusion(
    ranklists: Dict[str, List[Tuple[str, float]]],
    top_k: int = 100,
    skip_duplicates: bool = True
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranklists using round-robin interleaving.
    Takes one document from each ranklist in turn.
    
    Args:
        ranklists: Dictionary mapping sub-query IDs to their ranklists.
                   Each ranklist is [(docid, score), ...].
        top_k: Number of documents to return.
        skip_duplicates: If True, skip documents already selected.
        
    Returns:
        Fused ranklist as [(docid, score), ...].
    """
    if not ranklists:
        return []
    
    # Initialize pointers for each ranklist
    pointers = {key: 0 for key in ranklists}
    subquery_ids = list(ranklists.keys())
    
    results = []
    seen_docs = set()
    current_rank = 0
    
    while len(results) < top_k:
        made_progress = False
        
        for sq_id in subquery_ids:
            if len(results) >= top_k:
                break
            
            ranklist = ranklists[sq_id]
            ptr = pointers[sq_id]
            
            while ptr < len(ranklist):
                docid, score = ranklist[ptr]
                pointers[sq_id] = ptr + 1
                
                if skip_duplicates:
                    if docid not in seen_docs:
                        seen_docs.add(docid)
                        # Score based on position in fused list
                        fused_score = 1.0 / (len(results) + 1)
                        results.append((docid, fused_score))
                        made_progress = True
                        break
                else:
                    fused_score = 1.0 / (len(results) + 1)
                    results.append((docid, fused_score))
                    made_progress = True
                    break
                
                ptr += 1
        
        # Check if all ranklists are exhausted
        if not made_progress:
            break
    
    return results


def reciprocal_rank_fusion(
    ranklists: Dict[str, List[Tuple[str, float]]],
    k: int = 60,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranklists using Reciprocal Rank Fusion (RRF).
    
    RRF Score = Σ 1 / (k + rank_i)
    
    Args:
        ranklists: Dictionary mapping sub-query IDs to their ranklists.
        k: RRF smoothing parameter (default: 60).
        top_k: Number of documents to return.
        
    Returns:
        Fused ranklist as [(docid, score), ...].
    """
    doc_scores = defaultdict(float)
    
    for sq_id, ranklist in ranklists.items():
        for rank, (docid, _) in enumerate(ranklist, 1):
            doc_scores[docid] += 1.0 / (k + rank)
    
    # Sort by RRF score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_docs[:top_k]


def score_summation_fusion(
    ranklists: Dict[str, List[Tuple[str, float]]],
    normalize: bool = True,
    top_k: int = 100
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranklists by summing scores.
    Optionally applies min-max normalization to each ranklist first.
    
    Args:
        ranklists: Dictionary mapping sub-query IDs to their ranklists.
        normalize: If True, apply min-max normalization before summing.
        top_k: Number of documents to return.
        
    Returns:
        Fused ranklist as [(docid, score), ...].
    """
    doc_scores = defaultdict(float)
    
    for sq_id, ranklist in ranklists.items():
        if not ranklist:
            continue
        
        if normalize:
            # Apply min-max normalization
            scores = np.array([score for _, score in ranklist])
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score > min_score:
                norm_scores = (scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(scores)
            
            for i, (docid, _) in enumerate(ranklist):
                doc_scores[docid] += norm_scores[i]
        else:
            for docid, score in ranklist:
                doc_scores[docid] += score
    
    # Sort by summed score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_docs[:top_k]


def apply_fusion_to_subquery_runs(
    subquery_runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    method: str = "round_robin",
    top_k: int = 100,
    **kwargs
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Apply fusion to runs from multiple sub-queries for each main query.
    
    Args:
        subquery_runs: Nested dictionary {main_qid: {subquery_id: [(docid, score), ...]}}.
        method: Fusion method - "round_robin", "rrf", or "score_sum".
        top_k: Number of documents to return per query.
        **kwargs: Additional arguments for the fusion method.
        
    Returns:
        Fused run dictionary {qid: [(docid, score), ...]}.
    """
    fused_run = {}
    
    for qid, sq_runs in subquery_runs.items():
        if method == "round_robin":
            fused_run[qid] = round_robin_fusion(sq_runs, top_k=top_k)
        elif method == "rrf":
            k = kwargs.get("k", 60)
            fused_run[qid] = reciprocal_rank_fusion(sq_runs, k=k, top_k=top_k)
        elif method == "score_sum":
            normalize = kwargs.get("normalize", True)
            fused_run[qid] = score_summation_fusion(sq_runs, normalize=normalize, top_k=top_k)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
    
    return fused_run
