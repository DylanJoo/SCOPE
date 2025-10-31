#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out.%j
#SBATCH --error=enc-doc.err.%j
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-7%2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

# model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage-titled.25k
model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/crux-researchy-corpus/${model_dir##*/}

mkdir -p $output_dir

EXCLUDE_TITLE="--exclude_title "
if [[ $model_dir == *"title"* ]]; then
    EXCLUDE_TITLE=""
    echo 'title is included.'
fi

# Start experimentss
SHARD_ID=$SLURM_ARRAY_TASK_ID
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 2048 \
    --passage_max_len 256 \
    --bf16 \
    ${EXCLUDE_TITLE} \
    --dataset_name Tevatron/msmarco-passage-corpus-new \
    --corpus_name Tevatron/msmarco-passage-corpus-new \
    --encode_output_path $output_dir/corpus_emb.${SHARD_ID}.pkl \
    --dataset_number_of_shards 8 \
    --attn_implementation sdpa \
    --dataset_shard_index ${SHARD_ID}


python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path Qwen/Qwen3-Embedding-4B \
    --bf16 \
    --per_device_eval_batch_size 128 \
    --normalize \
    --pooling last \
    --padding_side left \
    --passage_prefix "" \
    --passage_max_len 512 \
    --dataset_name DylanJHJ/crux-researchy-corpus \
    --dataset_number_of_shards ${N_SHARD} \
    --encode_output_path /home/hltcoe/jhueiju/temp/output_qwen3/corpus_neuclir_${lang}.mt.pkl$each \
    --dataset_shard_index $each
