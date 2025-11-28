"""
Utility functions for diversity retrieval baselines.
Functions for loading embeddings, computing similarities, and data handling.
"""

import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def load_pickle_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a pickle file (Tevatron format).
    
    Args:
        path: Path to the pickle file containing embeddings.
        
    Returns:
        Dictionary mapping document/query IDs to their embeddings.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load a TREC run file.
    
    Args:
        path: Path to the TREC run file.
        
    Returns:
        Dictionary mapping query IDs to list of (doc_id, score) tuples.
    """
    run = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score, _ = parts[:6]
            else:
                qid, _, docid, rank, score = parts[:5]
            
            if qid not in run:
                run[qid] = []
            run[qid].append((docid, float(score)))
    
    # Sort by score descending
    for qid in run:
        run[qid] = sorted(run[qid], key=lambda x: x[1], reverse=True)
    
    return run


def save_run(run: Dict[str, List[Tuple[str, float]]], path: str, run_name: str = "diversity"):
    """
    Save a run to TREC format.
    
    Args:
        run: Dictionary mapping query IDs to list of (doc_id, score) tuples.
        path: Output file path.
        run_name: Name identifier for the run.
    """
    with open(path, 'w') as f:
        for qid in run:
            for rank, (docid, score) in enumerate(run[qid], 1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{run_name}\n")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Cosine similarity score.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple documents.
    
    Args:
        query: Query embedding (1D array).
        docs: Document embeddings (2D array, shape: [n_docs, dim]).
        
    Returns:
        Array of similarity scores.
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(docs.shape[0])
    
    query_normalized = query / query_norm
    doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0, 1, doc_norms)  # Avoid division by zero
    docs_normalized = docs / doc_norms
    
    return docs_normalized @ query_normalized


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization to scores.
    
    Args:
        scores: Array of scores.
        
    Returns:
        Normalized scores in [0, 1] range.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


def load_subtopics(path: str) -> Dict[str, List[str]]:
    """
    Load subtopics (sub-questions) from a file.
    
    Args:
        path: Path to the subtopics file.
        
    Returns:
        Dictionary mapping query IDs to list of subtopic texts.
    """
    subtopics = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                subtopic = parts[1]
                if qid not in subtopics:
                    subtopics[qid] = []
                subtopics[qid].append(subtopic)
    return subtopics


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for a set of embeddings.
    
    Args:
        embeddings: 2D array of embeddings (shape: [n, dim]).
        
    Returns:
        Similarity matrix (shape: [n, n]).
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return normalized @ normalized.T
