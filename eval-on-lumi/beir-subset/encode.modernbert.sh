#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/encode.out.%a
#SBATCH --error=logs/encode.err.%a
#SBATCH --partition=small-g         # partition name
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --array=0-12%13
#SBATCH --mem=64G
#SBATCH --time=00:30:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465002438 # Project for billing

# ENV
# source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
# conda activate inference 
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

pooling=mean
model_dir=${HOME}/models/modernbert-crux-researchy-pos_20.neg_51.filtered.b32_n256.1e-4.1024
# model_dir=${HOME}/models/modernbert-crux-researchy-pos_high.neg_zero.b64_n512.1e-4.512
output_dir=${HOME}/indices/beir-subset-corpus/${model_dir##*/}
model_dir=${model_dir}/checkpoint-10000
mkdir -p $output_dir

DATASETS=(
"beir.arguana"
"beir.climate_fever"
"beir.dbpedia_entity"
"beir.fever"
"beir.fiqa"
"beir.hotpotqa"
"beir.nfcorpus"
"beir.nq"
"beir.quora"
"beir.scidocs"
"beir.scifact"
"beir.trec_covid"
"beir.webis_touche2020"
)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo Encoding $DATASET corpus
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 768 \
    --passage_max_len 512 \
    --pooling $pooling --normalize --bf16 \
    --exclude_title \
    --dataset_name DylanJHJ/beir-subset-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl

echo Encoding $DATASET queries
singularity exec $SIF  \
    python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name answerdotai/ModernBERT-base \
    --model_name_or_path $model_dir \
    --pooling $pooling --normalize --bf16 \
    --per_device_eval_batch_size 128 \
    --dataset_name  DylanJHJ/beir-subset \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
    --query_max_len 256 \
    --encode_is_query
