#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --mem=256G
#SBATCH --time=12:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

# bsz=64
# nsample=512
# lr=1e-4
bsz=128
nsample=1024
lr=1e-4
split=pos_20.neg_51.filtered
# split=pos_high.neg_zero # 1
# split=pos_half.neg_zero # 2
# split=pos_low.neg_zero # 3
# split=pos_high.neg_low # a
# split=pos_high.neg_quarter # b
# split=pos_zero.neg_high # b
max_length=512
model_dir=${HOME}/models/modernbert-crux-researchy-${split}.b${bsz}_n${nsample}.${lr}.${max_length}

mkdir -p ${model_dir}

GPUS_PER_NODE=8
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path answerdotai/ModernBERT-base \
    --save_steps 2500 \
    --dataset_name DylanJHJ/crux-researchy \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --dataset_split $split \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --pooling mean --normalize \
    --temperature 0.02 \
    --prediction_loss_only True \
    --do_eval True \
    --eval_strategy steps \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_corpus_name Tevatron/msmarco-passage-corpus-new \
    --eval_group_size 8 \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --bf16 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len $max_length \
    --dataloader_num_workers 4 \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --logging_steps 10 \
    --overwrite_output_dir \
    --run_name modernbert-base.crux-researchy-${split}.b${bsz}_n${nsample}.${lr}.${max_length}
