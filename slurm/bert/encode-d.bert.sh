#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out
#SBATCH --error=enc-doc.err
#SBATCH --partition=dev-g           # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

cd ${HOME}/SCOPE

## A100x1: msmarco-passage-aug b32 n8 3ep 24hr
# model_dir=${HOME}/models/bert-msmarco-psg-aug.b32_n256

## AMD*2
model_dir=${HOME}/models/bert-msmarco-psg.b32_n256.2gpu
checkpoint=checkpoint-50000
# model_dir=${HOME}/models/bert-msmarco-psg.b32_n256.2gpu
# checkpoint=checkpoint-45000
output_dir=${HOME}/indices/${model_dir##*/}

mkdir -p $output_dir

# Start experimentss
for device in {0..7};do
    SHARD=$device
    export CUDA_VISIBLE_DEVICES=$device
    export HIP_VISIBLE_DEVICES=$device
    singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name bert-base-uncased \
      --model_name_or_path $model_dir/$checkpoint \
      --per_device_eval_batch_size 1024 \
      --passage_max_len 256 \
      --bf16 \
      --dataset_name Tevatron/msmarco-passage-corpus-new \
      --corpus_name Tevatron/msmarco-passage-corpus-new \
      --encode_output_path $output_dir/corpus_emb.${SHARD}.pkl \
      --dataset_number_of_shards 8 \
      --attn_implementation sdpa \
      --dataset_shard_index ${SHARD} &
done

wait
