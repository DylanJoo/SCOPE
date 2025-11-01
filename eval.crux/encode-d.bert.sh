#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out
#SBATCH --error=enc-doc.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for split in train test;do
echo Encoding crus-mds-corpus $split
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 128 \
    --passage_max_len 384 \
    --bf16 \
    --exclude_title \
    --dataset_name DylanJHJ/crux-mds-corpus \
    --dataset_split $split \
    --encode_output_path $output_dir/corpus_emb.$split.pkl \
    --attn_implementation sdpa
done
