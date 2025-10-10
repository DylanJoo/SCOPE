#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out

# Multi-GPU encoding
MULTIJOBS=${HOME}/multigpu.txt
source ~/.bashrc
enter_conda
conda activate crc


mkdir -p ${HOME}/models/repllama-3-lora-ms-psg
model_dir=${HOME}/models/repllama-3-lora-ms-psg

cd ${HOME}/refined-retrieval-context

# Start experiments
# a100 (2)
# 2 gpus * 16 batch size * 4 accumulation = 128 
# 2 gpus * 16 batch size = 32 negatives 
deepspeed --include localhost:0,1 --master_port 60000 --module \
    tevatron.retriever.driver.train \
    --deepspeed configs/ds_repllama.json \
    --output_dir ${model_dir} \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora \
    --lora_r 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps 250 \
    --dataset_name Tevatron/msmarco-passage-aug \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --bf16 \
    --pooling eos \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --gradient_checkpointing \
    --per_device_train_batch_size 20 \
    --train_group_size 16 \
    --learning_rate 1e-4 \
    --query_max_len 32 \
    --passage_max_len 196 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4
