#!/bin/bash -l
#SBATCH --job-name=encode
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

model_dir=/home/dju/models/ablation.transfer-learning/modernbert-two-stage.pos_20.neg_51.filtered.b64_n512.1e-4.512.self-prefinetuned
model_dir=/home/dju/models/ablation.transfer-learning/modernbert-two-stage.pos_20.neg_51.filtered.b64_n512.1e-4.512
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for split in train test;do
    echo Encoding crus-mds-corpus $split
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name answerdotai/ModernBERT-base \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 1024 \
        --passage_max_len 384 \
        --pooling mean --normalize \
        --bf16 \
        --exclude_title \
        --dataset_name DylanJHJ/crux-mds-corpus \
        --dataset_split $split \
        --encode_output_path $output_dir/corpus_emb.$split.pkl 
done

for subset in crux-mds-duc04 crux-mds-multi_news;do
    for topic_path in $CRUX_ROOT/$subset/topic/*jsonl; do
        echo "Encoding $topic_path"
        python -m tevatron.retriever.driver.encode \
          --output_dir=temp \
          --tokenizer_name answerdotai/ModernBERT-base \
          --model_name_or_path $model_dir \
          --pooling mean --normalize \
          --bf16 \
          --per_device_eval_batch_size 64 \
          --dataset_path $topic_path \
          --encode_output_path $output_dir/query_emb.${subset}.pkl \
          --query_max_len 128 \
          --encode_is_query
    done
done
