#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=gpu_a100
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00           # Run time (d-hh:mm:ss)

module load 2024
module load CUDA/12.6.0/
source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate ir

bsz=64
nsample=512
lr=1e-4
model_dir=${HOME}/models/modernbert-msmarco-psg.b${bsz}_n${nsample}.${lr}.tau0.01

mkdir -p ${model_dir}

GPUS_PER_NODE=2
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path answerdotai/ModernBERT-base \
    --save_steps 5000 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 32 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --eval_strategy steps \
    --do_eval True \
    --pooling mean --normalize \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_group_size 8 \
    --bf16 \
    --temperature 0.02 \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len 196 \
    --dataloader_num_workers 8 \
    --max_steps 25000 \
    --warmup_steps 2500 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name modernbert-base.msmarco-passage.b${bsz}_n${nsample}.${lr}
