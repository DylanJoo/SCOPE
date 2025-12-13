#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out
#SBATCH --error=logs/encode.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

for model_dir in ${HOME}/models/modernbert-*kd*;do
checkpoint=checkpoint-10000
output_dir=${HOME}/indices/msmarco-passage/${model_dir##*/}
mkdir -p $output_dir

# Start experiments
for split in dl19 dl20;do
    export CUDA_VISIBLE_DEVICES=0
    export HIP_VISIBLE_DEVICES=0
    singularity exec $SIF \
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name answerdotai/ModernBERT-base \
      --model_name_or_path $model_dir/$checkpoint \
      --per_device_eval_batch_size 1000 \
      --dataset_name Tevatron/msmarco-passage-new \
      --dataset_split $split \
      --pooling mean --normalize --bf16 \
      --encode_output_path $output_dir/query_emb.$split.pkl \
      --query_max_len 32 \
      --encode_is_query
done
done
