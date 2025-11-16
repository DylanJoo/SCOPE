#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/modernbert.out.%j
#SBATCH --error=logs/modernbert.err.%j
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

bsz=64
nsample=256
lr=1e-5
split=flatten
model_dir=${HOME}/models/modernbert-crux-researchy.${bsz}_n${nsample}.${lr}.${split}
pretrained=answerdotai/ModernBERT-base

mkdir -p ${model_dir}

GPUS_PER_NODE=2
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --exclude_title \
    --output_dir ${model_dir} \
    --model_name_or_path $pretrained \
    --save_steps 10000 \
    --dataset_name DylanJHJ/crux-researchy \
    --dataset_split $split \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --per_device_train_batch_size 16 \
    --train_group_size 8 \
    --prediction_loss_only True \
    --eval_strategy steps \
    --do_eval True \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_corpus_name Tevatron/msmarco-passage-corpus-new \
    --eval_group_size 8 \
    --pooling mean \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataloader_num_workers 2 \
    --max_steps 50000 \
    --warmup_steps 5000 \
    --logging_steps 10 \
    --gradient_accumulation_steps 2 \
    --overwrite_output_dir \
    --run_name modernbert-base.crux-researchy.b${bsz}_n${nsample}.${lr}.${split}
