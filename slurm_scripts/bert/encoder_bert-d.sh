#!/bin/sh
#SBATCH --job-name=encode-d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=%x-%j.out

module load anaconda3/2024.2
conda activate crc

model_dir=${HOME}/models/bert-msmarco-psg/checkpoint-45000

for s in $(seq -f "%02g" 0 19)
do
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${model_dir} \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --passage_max_len 128 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encode_output_path /home/hltcoe/jhueiju/temp/output_bert/bert-msmarco-psg/corpus_msmarco-psg.pkl$each \
  --dataset_number_of_shards 20 \
  --dataset_shard_index ${s}
done
