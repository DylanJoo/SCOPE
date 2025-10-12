#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-query.out
#SBATCH --error=enc-query.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

cd ${HOME}/SCOPE

## A100x1: msmarco-passage-aug b32 n256 3ep
#dl19/20: 0.6447 0.6567
# model_dir=${HOME}/models/bert-msmarco-psg-aug.b32_n256
# checkpoint=checkpoint-46032

## AMD*4: 
#dl19/20: 
model_dir=${HOME}/models/bert-msmarco-psg.b128_n256-1e-4
checkpoint=checkpoint-10000
output_dir=${HOME}/indices/${model_dir##*/}


echo start running
for split in dl19 dl20;do
    export CUDA_VISIBLE_DEVICES=0
    export HIP_VISIBLE_DEVICES=0
    singularity exec $SIF \
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name bert-base-uncased \
      --model_name_or_path $model_dir/$checkpoint \
      --bf16 --dtype bfloat16 \
      --per_device_eval_batch_size 1024 \
      --dataset_name Tevatron/msmarco-passage-new \
      --corpus_name Tevatron/msmarco-passage-new \
      --dataset_split $split \
      --attn_implementation sdpa \
      --encode_output_path $output_dir/$split.query_emb.pkl \
      --query_max_len 32 \
      --encode_is_query
done
