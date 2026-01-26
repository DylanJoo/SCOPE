#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out
#SBATCH --error=logs/encode.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux

model_dir=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
# model_dir=${HOME}/models/main.learning/ce_1.0-selfdistil_0.1
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
mkdir -p $output_dir

topic_path=/home/dju/SCOPE/src/create-crux-mds-subqueries/crux-neuclir.qwen2.5-7b-instruct.subquestions.jsonl
# topic_path=/home/dju/SCOPE/src/create-crux-mds-subqueries/crux-neuclir.oracle.subquestions.jsonl
python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_dir \
  --pooling mean --normalize --bf16 \
  --per_device_eval_batch_size 256 \
  --dataset_path $topic_path \
  --encode_output_path $output_dir/query_emb.mqr.pkl \
  --query_max_len 128 \
  --encode_is_query
