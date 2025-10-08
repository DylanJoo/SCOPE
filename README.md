# SCOPE: Search for Coverage-based Optimization and Exploration
---
This repository contains the code for the SCOPE project, which focuses on coverage-based optimization and exploration techniques. 

We use the CRUX researchy training queries for training. We compare the different coverage-based training methods with the standard training method.


## Training Setups 

| Method            | Dataset             | Training Steps | Batch Size | Negative Samples | Distillation  |
| ------------------|---------------------|----------------|------------|------------------|-------------- |
| Relevance-based   | MsMARCO             | 99k            | 256        | 0                | No            |
| Coverage-based    | CRUX-Researchy      | 100k           | 256        | 10               | No            |

