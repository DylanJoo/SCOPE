#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-11%1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference 

model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
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
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 64 \
    --passage_max_len 512 \
    --exclude_title \
    --dataset_name DylanJHJ/nano-beir-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl \
    --attn_implementation sdpa

echo Encoding $DATASET queries
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 64 \
    --dataset_name  DylanJHJ/nano-beir \
    --dataset_split $DATASET \
    --attn_implementation sdpa \
    --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
    --query_max_len 256 \
    --encode_is_query
