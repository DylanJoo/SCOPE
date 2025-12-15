#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out.%j
#SBATCH --error=enc-doc.err.%j
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-15%4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/crux-researchy-corpus/${model_dir##*/}
mkdir -p $output_dir

# Start experimentss
N_SHARD=16
SHARD_ID=$SLURM_ARRAY_TASK_ID
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path $model_dir \
    --bf16 \
    --exclude_title \
    --per_device_eval_batch_size 40 \
    --normalize \
    --pooling last \
    --padding_side left \
    --passage_prefix "" \
    --passage_max_len 4096 \
    --dataset_name DylanJHJ/crux-researchy-corpus \
    --dataset_number_of_shards ${N_SHARD} \
    --encode_output_path ${output_dir}/corpus_emb.${SHARD_ID}.pkl \
    --dataset_shard_index ${SHARD_ID}
