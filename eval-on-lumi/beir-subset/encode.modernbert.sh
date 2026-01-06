#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --array=0-12%13
#SBATCH --mem=64G
#SBATCH --time=01:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

DATASETS=(
"beir.arguana"
"beir.climate_fever"
"beir.dbpedia_entity"
"beir.fever"
"beir.fiqa"
"beir.hotpotqa"
"beir.nfcorpus"
"beir.nq"
"beir.quora"
"beir.scidocs"
"beir.scifact"
"beir.trec_covid"
"beir.webis_touche2020"
)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# for model_dir in ${HOME}/models/ablation.two-stage/modernbert-two-stage-*;do
# for model_dir in ${HOME}/models/*half*covdistil*;do
for model_dir in DylanJHJ/nomic.modernbert-base.msmarco-passage.10k;do
output_dir=${HOME}/indices/beir-subset-corpus/${model_dir##*/}
# model_dir=$model_dir/checkpoint-5000
mkdir -p $output_dir

echo Encoding $DATASET corpus
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 2048 \
    --passage_max_len 512 \
    --exclude_title \
    --pooling mean --normalize \
    --passage_prefix "search_document: " \
    --dataset_name DylanJHJ/beir-subset-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl

echo Encoding $DATASET queries
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --pooling mean --normalize \
    --per_device_eval_batch_size 128 \
    --dataset_name  DylanJHJ/beir-subset \
    --dataset_split $DATASET \
    --query_prefix "search_query: " \
    --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
    --query_max_len 256 \
    --encode_is_query

done
