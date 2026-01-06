#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
"nomic-ai/modernbert-embed-base-unsupervised"
"DylanJHJ/nomic.modernbert-base.msmarco-passage.10k"
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
    mkdir -p $output_dir

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        topic_path=${HOME}/SCOPE/src/create-crux-mds-subqueries/${subset}.oracle.subquestions.jsonl
        singularity exec $SIF \
            python -m tevatron.retriever.driver.encode \
            --output_dir=temp \
            --tokenizer_name answerdotai/ModernBERT-base \
            --model_name_or_path $model_dir \
            --pooling mean --normalize --bf16 \
            --query_prefix "search_query: " \
            --per_device_eval_batch_size 256 \
            --dataset_path $topic_path \
            --encode_output_path $output_dir/query_emb.${subset}.mqr.oracle.pkl \
            --query_max_len 128 \
            --encode_is_query
    done

done
