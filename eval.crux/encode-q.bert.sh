#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-query.out.%j
#SBATCH --error=enc-query.err.%j
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate ir

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
CRUX_ROOT=/home/dju/datasets/crux
mkdir -p $output_dir

for dataset in crux-mds-duc04 crux-mds-multi-news;do
    for topic_path in $CRUX_ROOT/$dataset/*jsonl; do
        echo Encoding $topic_file

        python -m tevatron.retriever.driver.encode \
          --output_dir=temp \
          --tokenizer_name bert-base-uncased \
          --model_name_or_path $model_dir \
          --bf16 \
          --per_device_eval_batch_size 50 \
          --dataset_path $topic_path \
          --attn_implementation sdpa \
          --encode_output_path $output_dir/query_emb.${dataset}.pkl \
          --query_max_len 128 \
          --encode_is_query
    done
done
