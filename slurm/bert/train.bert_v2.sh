#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/bert-ms.out.%j
#SBATCH --error=logs/bert-ms.err.%j
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

mkdir -p ${HOME}/models/bert-msmarco-psg

cd ${HOME}/SCOPE

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export HIP_VISIBLE_DEVICES=0,1,2,3

model_dir=${HOME}/models/bert-msmarco-psg.b8
GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experimentss
singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --output_dir ${model_dir} \
    --model_name_or_path bert-base-uncased \
    --bf16 --dtype bfloat16 \
    --save_steps 10000 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 8 \
    --train_group_size 8 \
    --dataloader_num_workers 1 \
    --learning_rate 2e-5 \
    --query_max_len 32 \
    --passage_max_len 256 \
    --max_steps 100000 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --attn_implementation sdpa \
    --overwrite_output_dir \
    --run_name bert-base.msmarco-passage.4gpu-b8-acc4.n8-30eps
