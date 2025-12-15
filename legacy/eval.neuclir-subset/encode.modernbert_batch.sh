#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out
#SBATCH --error=logs/encode.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

for model_dir in /home/dju/models/ablation.cov-sampling/modernbert-*;do
    output_dir=${HOME}/indices/neuclir1-subset-corpus/${model_dir##*/}
    mkdir -p $output_dir

    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 256 \
        --passage_max_len 1024 \
        --pooling mean --bf16 --normalize \
        --dataset_name DylanJHJ/neuclir1-subset-corpus  \
        --encode_output_path $output_dir/corpus_emb.pkl

    topic_path=/home/dju/datasets/crux/crux-neuclir/topic/neuclir24-test-request.jsonl
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --pooling mean --bf16 --normalize \
        --per_device_eval_batch_size 64 \
        --dataset_path $topic_path \
        --encode_output_path $output_dir/query_emb.pkl \
        --query_max_len 256 \
        --encode_is_query
done
