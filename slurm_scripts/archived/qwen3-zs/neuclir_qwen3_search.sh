#!/bin/sh
#SBATCH --job-name=qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Encode queries
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_qwen3/runs
lang=zho
output_run_file=${HOME}/temp/output_qwen3/runs/run.qwen3_neuclir_${lang}.mt.txt
qrels_file=/expscratch/eyang/collections/neuclir/qrels/2024.${lang}.qrels
topic_file=/expscratch/eyang/collections/neuclir/topics/2024.jsonl

python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path Qwen/Qwen3-Embedding-4B \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last \
    --padding_side left \
    --query_prefix "Instruct: Given the query title and the description, retrieve documents that fulfil user's information need.\nQuery:" \
    --query_type topic \
    --query_max_len 512 \
    --dataset_name scale/neuclir \
    --dataset_path ${topic_file} \
    --dataset_config neuclir \
    --dataset_split none \
    --encode_output_path /home/hltcoe/jhueiju/temp/output_qwen3/queries_neuclir.pkl \
    --encode_is_query

python -m tevatron.retriever.driver.search \
    --query_reps ${HOME}/temp/output_qwen3/queries_neuclir.pkl \
    --passage_reps "${HOME}/temp/output_qwen3/corpus_neuclir_${lang}.mt.pkl*" \
    --depth 100 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${output_run_file}

# python -m tevatron.utils.format.convert_result_to_trec \
#     --input ${output_run_file} \
#     --output ${output_run_file/txt/trec} \
#     --remove_query
#
# ${HOME}/temp/trec_eval \
#     -m ndcg_cut.20 -m recall.1000 \
#     ${qrels_file} ${output_run_file/txt/trec} 
