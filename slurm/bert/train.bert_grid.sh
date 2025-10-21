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
module load anaconda3/2024.2
conda activate crc

mkdir -p ${HOME}/models/bert-msmarco-psg
model_dir=${HOME}/models/bert-msmarco-psg

# Start experiments
python -m tevatron.retriever.driver.train \
    --output_dir ${model_dir} \
    --model_name_or_path bert-base-uncased \
    --save_steps 5000 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 32 \
    --train_group_size 8 \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --query_max_len 32 \
    --passage_max_len 256 \
    --max_steps 50000 \
    --logging_steps 10 \
    --warmup_steps 5000 \
    --attn_implementation sdpa \
    --overwrite_output_dir \
    --prediction_loss_only True \
    --eval_strategy steps \
    --do_eval True \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_group_size 100 \
    --per_device_eval_batch_size 16 \
    --eval_steps 100 \
    --run_name NV-bert-base.msmarco-passage.b32_n256.1e-5.5k_50k
