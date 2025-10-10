#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-10%2
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

model_dir=${HOME}/models/bert-msmarco-psg/checkpoint-45000
shard=${SLURM_ARRAY_TASK_ID}

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encode_output_path /home/hltcoe/jhueiju/temp/output_bert/bert-msmarco-psg/corpus.bert-msmarco-psg.pkl$shard \
  --dataset_number_of_shards 10 \
  --dataset_shard_index $shard
