#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/enc-query/out.%a
#SBATCH --error=logs/enc-query.err.%a
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-12%1
#SBATCH --mem=32G
#SBATCH --time=0-00:10:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference 

model_dir=nomic-ai/modernbert-embed-base
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}
mkdir -p $output_dir

for split in dl19 dl20;do
    python -m tevatron.retriever.driver.encode \
      --output_dir=temp \
      --tokenizer_name answerdotai/ModernBERT-base \
      --model_name_or_path $model_dir \
      --pooling mean --normalize \
      --bf16 \
      --per_device_eval_batch_size 1024 \
      --dataset_name Tevatron/msmarco-passage-new \
      --dataset_split $split \
      --encode_output_path $output_dir/$split.query_emb.pkl \
      --query_max_len 32 \
      --encode_is_query
done
