#!/bin/sh
#SBATCH --job-name=encode-q
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Encode queries
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_repllama/repllama2/runs

# for lang in zho rus fas; do
for lang in zho; do
for year in 2022 2023 2024;do
topic_file=/expscratch/eyang/collections/neuclir/topics/${year}.jsonl
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_name_or_path castorini/repllama-v1-7b-lora-passage \
    --lora \
    --bf16 \
    --per_device_eval_batch_size 16 \
    --normalize \
    --pooling last  \
    --query_prefix "query: " \
    --query_type topic \
    --append_eos_token \
    --query_max_len 512 \
    --dataset_name scale/neuclir \
    --dataset_path ${topic_file} \
    --dataset_config neuclir \
    --dataset_split none \
    --attn_implementation sdpa \
    --encode_output_path /home/hltcoe/jhueiju/temp/output_repllama/repllama2/queries_neuclir_${year}.pkl \
    --encode_is_query
done
done
