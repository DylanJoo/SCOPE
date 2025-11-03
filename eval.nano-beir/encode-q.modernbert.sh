#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-query.out.%a
#SBATCH --error=enc-query.err.%a
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-12%1
#SBATCH --mem=32G
#SBATCH --time=0-00:10:00

# ENV
# source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
# conda activate pyserini
module load anaconda3/2024.2 # grid
conda activate crux

# model_dir=DylanJHJ/dpr.bert-base-uncased.msmarco-passage.25k
# model_dir=/home/hltcoe/jhueiju/models/crux-research-train-series/bert-crux-researchy.b32_n256.1e-6.train
# model_dir=/home/hltcoe/jhueiju/models/crux-research-train-series/bert-crux-researchy.b32_n256.1e-5.25k.train
model_dir=DylanJHJ/dpr.modernbert-base.msmarco-passage.25k
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
# "nano_beir.msmarco"
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo Encoding $DATASET ...
python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --tokenizer_name bert-base-uncased \
  --model_name_or_path $model_dir \
  --bf16 \
  --per_device_eval_batch_size 50 \
  --dataset_name  DylanJHJ/nano-beir \
  --dataset_split $DATASET \
  --encode_output_path $output_dir/query_emb.${DATASET}.pkl \
  --query_max_len 32 \
  --encode_is_query
