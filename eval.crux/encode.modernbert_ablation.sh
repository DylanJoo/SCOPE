#!/bin/bash -l
#SBATCH --job-name=ablation
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux

# for model_dir in /home/dju/models/main.learning/ce_1.0-selfdistil_0.1;do
# for model_dir in /home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.request;do
# for model_dir in /home/dju/models/ablation.two-stage/modernbert-two-stage-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.crux-researchy.request;do
# for model_dir in nomic-ai/modernbert-embed-base-unsupervised;do
for model_dir in DylanJHJ/nomic.modernbert-base.msmarco-passage.10k;do
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
                --query_prefix "search_query: " \
                --per_device_eval_batch_size 128 \
                --dataset_path $topic_path \
                --encode_output_path $output_dir/query_emb.${subset}.pkl \
                --query_max_len 128 \
                --encode_is_query
        done
    done
done
