#!/bin/bash -l
#SBATCH --job-name=encode.crux
#SBATCH --output=logs/encode.out
#SBATCH --error=logs/encode.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux

query_prefix="search_query:[unused0][unused1][unused2][unused3][unused4][unused5][unused6][unused7][unused8][unused9]"
num_views=10
view_start_idx=5

for model_dir in ${HOME}/models/main.learning/multiview-max.ce_1.0-wsq_0.0.*;do
    output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
    mkdir -p $output_dir

    for split in train test;do
        echo Encoding crux-mds-corpus $split
        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --per_device_eval_batch_size 2048 \
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
            echo Encoding $subset query 
            python -m tevatron.retriever.driver.encode \
                --output_dir=temp \
                --tokenizer_name answerdotai/ModernBERT-base \
                --model_name_or_path $model_dir \
                --pooling mean --normalize --bf16 \
                --query_prefix $query_prefix \
                --num_views $num_views \
                --view_start_idx $view_start_idx \
                --per_device_eval_batch_size 128 \
                --dataset_path $topic_path \
                --encode_output_path $output_dir/query_emb.${subset}.pkl \
                --query_max_len 128 \
                --encode_is_query
    done
done
