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
#SBATCH --time=00:30:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
"Qwen/Qwen3-Embedding-8B"
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/indices/neuclir1-subset-corpus/${model_dir##*/}
    mkdir -p $output_dir

    singularity exec $SIF  \
        python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --per_device_eval_batch_size 64 \
        --bf16 \
        --normalize \
        --pooling last \
        --padding_side left \
        --passage_prefix "" \
        --passage_max_len 1024 \
        --dataset_name DylanJHJ/neuclir1-subset-corpus  \
        --encode_output_path $output_dir/corpus_emb.pkl

    topic_path=${CRUX_ROOT}/crux-neuclir/topic/neuclir24-test-request.jsonl
    singularity exec $SIF  \
        python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --bf16 \
        --normalize \
        --pooling last \
        --padding_side left \
        --per_device_eval_batch_size 64 \
        --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
        --dataset_path $topic_path \
        --encode_output_path $output_dir/query_emb.pkl \
        --query_max_len 256 \
        --encode_is_query
done
