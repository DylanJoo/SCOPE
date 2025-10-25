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

mkdir -p ${HOME}/temp/output_repllama/repllama3/runs

for year in 2022 2023 2024;do
topic_file=/expscratch/eyang/collections/neuclir/topics/${year}.jsonl
python -m tevatron.retriever.driver.encode  \
    --output_dir=temp \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_name_or_path /home/hltcoe/jhueiju/models/repllama-3-lora-ms-psg/checkpoint-3069 \
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
    --encode_output_path /home/hltcoe/jhueiju/temp/output_repllama/repllama3/queries_neuclir_${year}.pkl \
    --encode_is_query
done
