# Diversity Retrieval Baselines

This module implements diversity retrieval baselines for evaluating coverage in retrieval systems (RQ3).

## Implemented Methods

### 1. MMR (Maximal Marginal Relevance)
Re-ranks documents balancing relevance and diversity using:
```
MMR = λ * sim(d, q) - (1-λ) * max(sim(d, d_i)) for d_i in selected set
```

Parameters:
- `lambda_param`: Trade-off between relevance (1.0) and diversity (0.0). Use values 0.1, 0.5, 1.0.

### 2. Sub-query Retrieval with Post-hoc Fusion
Retrieves documents using ground-truth sub-questions and combines results using fusion strategies.

### 3. Ranklist Round-Robin
Interleaves documents from N sub-question ranklists in round-robin fashion.
Takes one document from each ranklist in turn, skipping duplicates.

### 4. Ranklist Fusion (RRF)
Combines ranklists using Reciprocal Rank Fusion:
```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

### 5. Score Summation (LANCER)
Aggregates scores from multiple sub-queries with min-max normalization:
```
Final_score = α * main_score + (1-α) * Σ normalized_subquery_scores
```

## Usage

### MMR Re-ranking
```bash
python run_diversity_retrieval.py \
    --method mmr \
    --lambda_param 0.5 \
    --query_emb path/to/query_emb.pkl \
    --corpus_emb path/to/corpus_emb.pkl \
    --run input.trec \
    --output output.mmr.trec \
    --top_k 100
```

### Round-Robin Fusion
```bash
python run_diversity_retrieval.py \
    --method round_robin \
    --subquery_run path/to/subquery_runs.pkl \
    --run input.trec \
    --output output.round_robin.trec \
    --top_k 100
```

### RRF Fusion
```bash
python run_diversity_retrieval.py \
    --method rrf \
    --subquery_run path/to/subquery_runs.pkl \
    --run input.trec \
    --output output.rrf.trec \
    --k 60 \
    --top_k 100
```

### Score Summation
```bash
python run_diversity_retrieval.py \
    --method score_sum \
    --subquery_run path/to/subquery_runs.pkl \
    --run input.trec \
    --output output.score_sum.trec \
    --top_k 100
```

Note: Min-max normalization is enabled by default. Use `--no_normalize` to disable.

### LANCER
```bash
python run_diversity_retrieval.py \
    --method lancer \
    --alpha 0.5 \
    --subquery_run path/to/subquery_runs.pkl \
    --run input.trec \
    --output output.lancer.trec \
    --top_k 100
```

## Input Formats

### Run File (TREC format)
```
qid Q0 docid rank score run_name
```

### Query/Document Embeddings (Pickle)
Either Tevatron format `[(id, embedding), ...]` or dictionary `{id: embedding}`.

### Sub-query Runs (Pickle)
Nested dictionary format:
```python
{
    "main_qid_1": {
        "subquery_1": [("docid", score), ...],
        "subquery_2": [("docid", score), ...],
    },
    ...
}
```

## Module Structure

- `utils.py`: Utility functions for loading data, computing similarities
- `mmr.py`: MMR re-ranking implementation
- `fusion.py`: Ranklist fusion methods (round-robin, RRF, score sum)
- `lancer.py`: LANCER-style score aggregation
- `subquery_retrieval.py`: Sub-query retrieval functions
- `run_diversity_retrieval.py`: Main runner script

## Integration with Existing Pipeline

This module integrates with the existing Tevatron-based retrieval pipeline:

```bash
# 1. Encode queries and corpus using Tevatron
python -m tevatron.retriever.driver.encode ...

# 2. Run initial retrieval
python -m tevatron.retriever.driver.search ...

# 3. Apply diversity re-ranking
python src/diversity-retrieval/run_diversity_retrieval.py ...

# 4. Evaluate with CRUX metrics
python -m crux.evaluation.rac_eval ...
```
