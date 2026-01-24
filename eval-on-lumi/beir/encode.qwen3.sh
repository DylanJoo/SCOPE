#!/bin/bash -l
#SBATCH --job-name=qwen3
#SBATCH --output=logs/qwen3.out.%a
#SBATCH --error=logs/qwen3.err.%a
#SBATCH --partition=small-g
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --array=1,3,5
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/beir-corpus/${model_dir##*/}
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

for SHARD_ID in {0..7};do
    echo Encoding $DATASET corpus $SHARD_ID
    export CUDA_VISIBLE_DEVICES=$SHARD_ID
    export HIP_VISIBLE_DEVICES=$SHARD_ID
    singularity exec $SIF \
        python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --bf16 \
        --per_device_eval_batch_size 384 \
        --normalize \
        --pooling last \
        --padding_side left \
        --passage_prefix "" \
        --passage_max_len 384 \
        --dataset_name DylanJHJ/beir-corpus \
        --dataset_split $DATASET \
        --encode_output_path $output_dir/corpus_emb.${DATASET}-${SHARD_ID}.pkl \
        --dataset_number_of_shards 8 \
        --dataset_shard_index ${SHARD_ID} &
done
wait

# echo Encoding $DATASET queries
# singularity exec $SIF \
#     python -m tevatron.retriever.driver.encode \
#     --output_dir=temp \
#     --model_name_or_path $model_dir \
#     --bf16 \
#     --per_device_eval_batch_size 128 \
#     --normalize \
#     --pooling last \
#     --padding_side left \
#     --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
#     --dataset_name  DylanJHJ/beir-subset \
#     --dataset_split $DATASET \
#     --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
#     --query_max_len 128 \
#     --encode_is_query
