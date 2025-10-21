#!/bin/bash -l
#SBATCH --job-name=train
#SBATCH --output=dev.out
#SBATCH --error=dev.err
#SBATCH --partition=dev-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2           # Allocate one gpu per MPI rank
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

mkdir -p ${HOME}/models/bert-msmarco-psg.b32_n256

cd ${HOME}/SCOPE

model_dir=${HOME}/models/dev
GPUS_PER_NODE=2
NUM_NODES=1
NUM_PROCESSES=$(expr $NUM_NODES \* $GPUS_PER_NODE)

# Start experimentss
echo start
singularity exec $SIF \
    accelerate launch -m \
    --multi_gpu \
    --num_processes $NUM_PROCESSES  --num_machines $NUM_NODES \
    tevatron.retriever.driver.train \
    --output_dir ${model_dir} \
    --model_name_or_path bert-base-uncased \
    --save_steps 100 \
    --dataset_name Tevatron/msmarco-passage-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --per_device_train_batch_size 16 \
    --prediction_loss_only True \
    --eval_strategy steps \
    --do_eval True \
    --eval_dataset_name DylanJHJ/Qrels \
    --eval_dataset_split msmarco_passage.trec_dl_2019 \
    --eval_group_size 32 \
    --per_device_eval_batch_size 16 \
    --eval_steps 10 \
    --train_group_size 8 \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --query_max_len 32 \
    --passage_max_len 256 \
    --max_steps 100 \
    --logging_steps 1 \
    --attn_implementation sdpa \
    --overwrite_output_dir \
    --warmup_steps 10 \
    --run_name dev

echo done
