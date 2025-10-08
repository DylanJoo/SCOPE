# SCOPE: Search for Coverage-based Optimization and Exploration
---
This repository contains the code for the SCOPE project, which focuses on coverage-based optimization and exploration techniques. 

We use the CRUX researchy training queries for training. We compare the different coverage-based training methods with the standard training method.


## Training Setups 

- BERT-based-uncased

| Method            | Dataset                      | Training Steps (eps) | Batch Size | Group Size      | Distillation  | GPU  |
| ------------------|------------------------------|----------------      |------------|-----------------|-------------- |------|
| Relevance-based   | Tevatron/msmarco-passage-aug | 450k (3)             | 32         | 8               | No            | 1    |
| Coverage-based    | CRUX-Researchy               | 100k                 | 32         | 8               | No            | 1    |


- Llama3 Lora

| Method            | Dataset                      | Training Steps (eps) | Batch Size | Group Size      | Distillation  | GPU  |
| ------------------|------------------------------|----------------      |------------|-----------------|-------------- |------|
| Relevance-based   | Tevatron/msmarco-passage-aug | 3k                   | 40         | 16              | No            | 2    |
| Coverage-based    | CRUX-Researchy               | 100k                 | 32         | 8               | No            | 2    |
