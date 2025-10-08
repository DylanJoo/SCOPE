#!/bin/sh
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out

# Multi-GPU encoding
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/models/bert-msmarco-psg
model_dir=${HOME}/models/bert-msmarco-psg

cd ${HOME}/refined-retrieval-context

# Start experiments
# a100 (1)
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.train \
    --output_dir ${model_dir} \
    --model_name_or_path bert-base-uncased \
    --save_steps 5000 \
    --dataset_name Tevatron/msmarco-passage-aug \
    --per_device_train_batch_size 32 \
    --train_group_size 8 \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --query_max_len 32 \
    --passage_max_len 256 \
    --num_train_epochs 3 \
    --logging_steps 500 \
    --attn_implementation sdpa \
    --overwrite_output_dir

# deepspeed --include localhost:0,1 --master_port 60000 --module \
