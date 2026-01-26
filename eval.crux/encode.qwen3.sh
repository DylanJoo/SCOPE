#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/qwen3.out
#SBATCH --error=logs/qwen3.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux
model_dir=Qwen/Qwen3-Embedding-0.6B # 160
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for split in train test;do
    echo Encoding crux-mds corpus $split
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --tokenizer_name $model_dir \
        --model_name_or_path $model_dir \
        --bf16 \
        --per_device_eval_batch_size 160 \
        --normalize \
        --pooling last \
        --padding_side left \
        --passage_prefix "" \
        --passage_max_len 512 \
        --dataset_name DylanJHJ/crux-mds-corpus \
        --dataset_split $split \
        --encode_output_path $output_dir/corpus_emb.$split.pkl
done

for subset in crux-mds-duc04 crux-mds-multi_news;do
    for topic_path in $CRUX_ROOT/$subset/topic/*jsonl; do
        echo Encoding crux-mds topic $split 
        python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name $model_dir \
            --model_name_or_path $model_dir \
            --bf16 \
            --per_device_eval_batch_size 128 \
            --normalize \
            --pooling last \
            --padding_side left \
            --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery: " \
            --query_max_len 256 \
            --dataset_path $topic_path \
            --encode_output_path $output_dir/query_emb.${subset}.pkl \
            --encode_is_query
    done
done
