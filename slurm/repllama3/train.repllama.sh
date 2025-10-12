#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/repllama3-ms.out.%j
#SBATCH --error=logs/repllama3-ms.err.%j
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

mkdir -p ${HOME}/models/repllama-msmarco-psg

cd ${HOME}/SCOPE

model_dir=${HOME}/models/llama-msmarco-psg.b8
GPUS_PER_NODE=8
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
# Effective batch size: 2 gpus * 16 batch size * 4 accumulation = 128 
# Effective batch size: 8 gpus * 4 batch size * 4 accumulation = 128 
# Efefctive negative contrastive size: 8 gpus * 4 batch size x 16 = 512 negatives 
singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --deepspeed configs/ds_repllama.json \
    --output_dir ${model_dir} \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora \
    --lora_r 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps 500 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --bf16 --dtype bfloat16 \
    --pooling eos \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --per_device_train_batch_size 4 \
    --train_group_size 8 \
    --learning_rate 1e-4 \
    --query_max_len 32 \
    --passage_max_len 196 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --run_name repllama3-lora.msmarco-passage.b32-n8-1ep
