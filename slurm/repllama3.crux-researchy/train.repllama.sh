#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/repllama3.%j.out
#SBATCH --error=logs/repllama3.%j.err
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

bsz=128
nsample=512
lr=1e-4
model_dir=${HOME}/models/repllama-crux-researchy.b${bsz}_n${nsample}.${lr}

mkdir -p ${model_dir}

GPUS_PER_NODE=4
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experiments
srun singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --exclude_title \
    --deepspeed ${HOME}/SCOPE/configs/ds_repllama.json \
    --output_dir ${model_dir} \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora \
    --lora_r 32 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
    --save_steps 1000 \
    --dataset_name DylanJHJ/crux-researchy \
    --corpus_name DylanJHJ/crux-researchy-corpus \
    --per_device_train_batch_size 8 \
    --train_group_size 16 \
    --prediction_loss_only True \
    --query_prefix "query: " \
    --passage_prefix "passage: " \
    --bf16 \
    --pooling eos \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --eval_strategy steps \
    --do_eval True \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_corpus_name Tevatron/msmarco-passage-corpus-new \
    --eval_group_size 8 \
    --per_device_eval_batch_size 16 \
    --eval_steps 100 \
    --learning_rate $lr \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataloader_num_workers 4 \
    --max_steps 4000 \
    --warmup_steps 400 \
    --logging_steps 10 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir \
    --run_name repllama3-lora.crux-researchy.b${bsz}_n${nsample}.${lr}
