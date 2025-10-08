#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40
#SBATCH --nodes=1
#SBATCH --array=1-10%2
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Multi-GPU encoding
MULTIJOBS=${HOME}/multigpu.txt
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_repllama
N_SHARD=10

each=$(head -$SLURM_ARRAY_TASK_ID $MULTIJOBS | tail -1)
echo It is running on $each task

# batch size 32 x max_length 512 | batch size 8 x max_length 3072
for lang in rus fas; do
python -m tevatron.retriever.driver.encode  \
  --output_dir=temp \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --lora_name_or_path /home/hltcoe/jhueiju/models/repllama-3-lora-ms-psg/checkpoint-3069 \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 128 \
  --normalize \
  --pooling last  \
  --passage_prefix "passage: " \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_config neuclir \
  --dataset_name scale/neuclir \
  --dataset_split none \
  --dataset_path /exp/scale25/neuclir/docs/${lang}.mt.jsonl \
  --dataset_number_of_shards ${N_SHARD} \
  --encode_output_path /home/hltcoe/jhueiju/temp/output_repllama/repllama3/corpus_neuclir_${lang}.mt.pkl$each \
  --dataset_shard_index $each
done
