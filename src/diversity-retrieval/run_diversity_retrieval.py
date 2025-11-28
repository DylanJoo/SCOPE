#!/usr/bin/env python3
"""
Main script to run diversity retrieval baselines.

Implements the following strategies:
1. MMR (Maximal Marginal Relevance) with lambda = 0.1, 0.5, 1.0
2. Sub-query retrieval with post-hoc fusion
3. Ranklist round-robin (N ground-truth sub-questions)
4. Ranklist fusion / RRF (N ground-truth sub-questions)
5. Score summation with min-max normalization (LANCER-style)

Usage:
    python run_diversity_retrieval.py --method mmr --lambda_param 0.5 \
        --query_emb query_emb.pkl --corpus_emb corpus_emb.pkl \
        --run input.trec --output output.trec
"""

import os
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple

from utils import load_run, save_run, load_pickle_embeddings
from mmr import apply_mmr_to_run
from fusion import apply_fusion_to_subquery_runs
from lancer import apply_lancer_to_runs
from subquery_retrieval import subquery_rerank_candidates


def load_embeddings_dict(path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from Tevatron pickle format.
    Handles both list format [(id, emb), ...] and dict format {id: emb}.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        # Tevatron format: list of (id, embedding) tuples
        return {str(item[0]): np.array(item[1]) for item in data}
    else:
        raise ValueError(f"Unknown embedding format: {type(data)}")


def main():
    parser = argparse.ArgumentParser(description="Diversity Retrieval Baselines")
    
    # Method selection
    parser.add_argument("--method", type=str, required=True,
                        choices=["mmr", "round_robin", "rrf", "score_sum", "lancer"],
                        help="Diversity retrieval method")
    
    # Input/Output
    parser.add_argument("--run", type=str, required=True,
                        help="Path to input TREC run file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output TREC run file")
    
    # Embeddings
    parser.add_argument("--query_emb", type=str,
                        help="Path to query embeddings pickle file")
    parser.add_argument("--corpus_emb", type=str,
                        help="Path to corpus embeddings pickle file")
    parser.add_argument("--subquery_emb", type=str,
                        help="Path to sub-query embeddings pickle file")
    parser.add_argument("--subquery_run", type=str,
                        help="Path to sub-query runs (for fusion methods)")
    
    # Method-specific parameters
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Lambda parameter for MMR (0=diversity, 1=relevance)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha parameter for LANCER (weight for main query)")
    parser.add_argument("--k", type=int, default=60,
                        help="K parameter for RRF")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable min-max normalization (enabled by default)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of documents to return per query")
    
    # Run name
    parser.add_argument("--run_name", type=str, default="diversity",
                        help="Name identifier for the output run")
    
    args = parser.parse_args()
    
    # Load input run
    print(f"Loading run from {args.run}")
    run = load_run(args.run)
    print(f"Loaded {len(run)} queries")
    
    # Execute based on method
    if args.method == "mmr":
        if not args.query_emb or not args.corpus_emb:
            raise ValueError("MMR requires --query_emb and --corpus_emb")
        
        print(f"Loading query embeddings from {args.query_emb}")
        query_embs = load_embeddings_dict(args.query_emb)
        print(f"Loaded {len(query_embs)} query embeddings")
        
        print(f"Loading corpus embeddings from {args.corpus_emb}")
        corpus_embs = load_embeddings_dict(args.corpus_emb)
        print(f"Loaded {len(corpus_embs)} document embeddings")
        
        print(f"Applying MMR with lambda={args.lambda_param}")
        reranked_run = apply_mmr_to_run(
            run=run,
            query_embeddings=query_embs,
            doc_embeddings=corpus_embs,
            lambda_param=args.lambda_param,
            top_k=args.top_k
        )
        
        run_name = f"mmr-l{args.lambda_param}"
    
    elif args.method in ["round_robin", "rrf", "score_sum"]:
        if not args.subquery_run:
            raise ValueError(f"{args.method} requires --subquery_run")
        
        # Load sub-query runs
        # Expected format: nested pickle {qid: {sq_id: [(docid, score), ...]}}
        print(f"Loading sub-query runs from {args.subquery_run}")
        with open(args.subquery_run, 'rb') as f:
            subquery_runs = pickle.load(f)
        
        print(f"Loaded sub-query runs for {len(subquery_runs)} queries")
        
        print(f"Applying {args.method} fusion")
        reranked_run = apply_fusion_to_subquery_runs(
            subquery_runs=subquery_runs,
            method=args.method,
            top_k=args.top_k,
            k=args.k,
            normalize=not args.no_normalize
        )
        
        run_name = args.method
    
    elif args.method == "lancer":
        if not args.subquery_run:
            raise ValueError("LANCER requires --subquery_run")
        
        # Load sub-query runs
        print(f"Loading sub-query runs from {args.subquery_run}")
        with open(args.subquery_run, 'rb') as f:
            subquery_runs = pickle.load(f)
        
        print(f"Applying LANCER with alpha={args.alpha}")
        reranked_run = apply_lancer_to_runs(
            main_run=run,
            subquery_runs=subquery_runs,
            alpha=args.alpha,
            normalize=not args.no_normalize,
            top_k=args.top_k
        )
        
        run_name = f"lancer-a{args.alpha}"
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Use custom run name if provided
    if args.run_name != "diversity":
        run_name = args.run_name
    
    # Save output
    print(f"Saving re-ranked run to {args.output}")
    save_run(reranked_run, args.output, run_name=run_name)
    print(f"Saved {len(reranked_run)} queries")


if __name__ == "__main__":
    main()
