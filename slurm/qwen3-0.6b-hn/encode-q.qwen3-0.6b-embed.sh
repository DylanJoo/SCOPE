#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-query.out
#SBATCH --error=enc-query.err
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=05:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/crux-researchy-corpus/${model_dir##*/}
mkdir -p $output_dir

# Original queqry
split=train
python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path $model_dir \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling last \
  --padding_side left \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that provide context to the query.\nQuery:" \
  --append_eos_token \
  --query_max_len 64 \
  --dataset_name DylanJHJ/crux-researchy \
  --dataset_split pos_20.neg_51 \
  --encode_output_path $output_dir/query_emb.$split.pkl \
  --encode_is_query
  # --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery:" \
