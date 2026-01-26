#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/enc-doc.out.%a
#SBATCH --error=logs/enc-doc.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-7%2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference 

model_dir=nomic-ai/modernbert-embed-base
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}
mkdir -p $output_dir

SHARD_ID=$SLURM_ARRAY_TASK_ID
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 128 \
    --passage_max_len 256 \
    --pooling mean --normalize \
    --bf16 \
    --exclude_title \
    --dataset_name Tevatron/msmarco-passage-corpus-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --encode_output_path $output_dir/corpus_emb.${SHARD_ID}.pkl \
    --dataset_number_of_shards 8 \
    --dataset_shard_index ${SHARD_ID}
