# Sampling strategies

We explore various negative mining strategies for enhancing model performance. 
The scripts included in this directory implement different strategies for mining negatives:

1. `hn-from-coverage.py`: Mines negatives based on coverage (per-document). We first binarize the multi-asepct judge into binary labels, then select the documents based on the coverage metric:
    - **half-coverage (0.5 <= cov < 0.75)**,
    - **low-coverage (0.0 < cov < 0.25)**, 
    - **zero-coverage (cov == 0.0)**

2. `hn-from-rank-ratio.py`: Mines negatives based on rank ratio: log RR(d)/R(d). 
As the reranked ranking (i.e., RR(d)) is supposed to be more accurate than the initial retrieval ranking (i.e., R(d)). We compute the rank ratio and select documents with high log rank ratio as negatives. It means the document ranked higher (smaller) in the initial retrieval but ranked lower (larger) in the reranked ranking.

3. `hn-mining-from-qwen3-rerank`: Mine negatives using Qwen-3 model reranking. We directly select the documents ranked from 50 - 100 as negative.

4. `hn-mining-from-qwen3-0.6b.py`: Mine negatives using Qwen-3-0.6B model. We re-encode the 3M clueweb documents using Qwen-3-0.6B and use it to mine negatives following the same procedure as in single-view negative mining.
