#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-12%2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

model_dir=DylanJHJ/repllama-3.1-8b.msmarco-passage.4k
output_dir=${HOME}/indices/nano-beir-corpus/${model_dir##*/}
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
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_name_or_path $model_dir \
    --lora \
    --bf16 \
    --per_device_eval_batch_size 32 \
    --normalize \
    --pooling last \
    --passage_prefix "passage: " \
    --append_eos_token \
    --passage_max_len 512 \
    --exclude_title \
    --dataset_name DylanJHJ/nano-beir-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl

echo Encoding $DATASET queries
python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --lora_name_or_path $model_dir \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --normalize \
  --pooling last  \
  --query_prefix "query: " \
  --append_eos_token \
  --query_max_len 256 \
  --dataset_name DylanJHJ/nano-beir \
  --dataset_split ${DATASET} \
  --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
  --encode_is_query
