#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --array=0-3
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.1.orth-0.0001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.1.orth-0.001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.25.orth-0.0001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.25.orth-0.001.request"
)               

model_dir="${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}"
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
model_dir=$model_dir/checkpoint-7500
query_prefix="search_query:[unused0][unused1][unused2][unused3][unused4]"
num_views=5

mkdir -p $output_dir

for split in train test;do
    echo Encoding crus-mds-corpus $split
    singularity exec $SIF  \
        python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 1280 \
        --passage_max_len 512 \
        --pooling mean --normalize --bf16 \
        --passage_prefix "search_document: " \
        --exclude_title \
        --dataset_name DylanJHJ/crux-mds-corpus \
        --dataset_split $split \
        --encode_output_path $output_dir/corpus_emb.$split.pkl 
done

for subset in crux-mds-duc04 crux-mds-multi_news;do
    for topic_path in $CRUX_ROOT/$subset/topic/*jsonl; do
        echo Encoding queries$topic_path
        singularity exec $SIF  \
            python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --query_prefix $query_prefix \
            --view_pooling 'cluster' \
            --num_views $num_views \
            --pooling mean --normalize --bf16 \
            --per_device_eval_batch_size 128 \
            --dataset_path $topic_path \
            --encode_output_path $output_dir/query_emb.${subset}.pkl \
            --query_max_len 128 \
            --encode_is_query
    done
done
