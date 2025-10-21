#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=logs/bert-ms.out.%j
#SBATCH --error=logs/bert-ms.err.%j
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

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

bsz=64
nsample=512
model_dir=${HOME}/models/bert-msmarco-psg.b${bsz}_n${nsample}.100k
GPUS_PER_NODE=2
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

mkdir -p ${model_dir}

# Start experimentss
singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --output_dir ${model_dir} \
    --model_name_or_path bert-base-uncased \
    --save_steps 5000 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 32 \
    --prediction_loss_only True \
    --eval_strategy steps \
    --do_eval True \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_group_size 8 \
    --per_device_eval_batch_size 64 \
    --eval_steps 100 \
    --train_group_size 8 \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --query_max_len 32 \
    --passage_max_len 196 \
    --max_steps 100000 \
    --logging_steps 10 \
    --attn_implementation sdpa \
    --overwrite_output_dir \
    --warmup_steps 10000 \
    --run_name bert-base.msmarco-passage.b${bsz}_n${nsample}.1e-5.10k_100k
