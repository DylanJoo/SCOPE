#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-12%1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference 

model_dir=${HOME}/models/crux-research-train-series/modernbert-msmarco-psg.b64_n512.1e-4
output_dir=${HOME}/indices/nano-beir-corpus/${model_dir##*/}
model_dir=${model_dir}/checkpoint-20000
mkdir -p $output_dir

DATASETS=(
"nano_beir.arguana"
"nano_beir.climate_fever"
"nano_beir.dbpedia_entity"
"nano_beir.fever"
"nano_beir.fiqa"
"nano_beir.hotpotqa"
"nano_beir.nfcorpus"
"nano_beir.nq"
"nano_beir.quora"
"nano_beir.scidocs"
"nano_beir.scifact"
"nano_beir.webis_touche2020"
)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo Encoding $DATASET corpus
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 64 \
    --passage_max_len 512 \
    --pooling mean --normalize \
    --bf16 \
    --exclude_title \
    --dataset_name DylanJHJ/nano-beir-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl 

echo Encoding $DATASET queries
python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --tokenizer_name answerdotai/ModernBERT-base \
  --model_name_or_path $model_dir \
  --pooling mean --normalize --bf16 \
  --per_device_eval_batch_size 64 \
  --dataset_name  DylanJHJ/nano-beir \
  --dataset_split $DATASET \
  --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
  --query_max_len 256 \
  --encode_is_query 
