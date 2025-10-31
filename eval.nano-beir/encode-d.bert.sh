#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=enc-doc.out.%j
#SBATCH --error=enc-doc.err.%j
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --nodes=1                
#SBATCH --array=0-13%1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate pyserini

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
"nano_beir.msmarco"
"nano_beir.nfcorpus"
"nano_beir.nq"
"nano_beir.quora"
"nano_beir.scidocs"
"nano_beir.scifact"
"nano_beir.webis_touche2020"
)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo Encoding $DATASET ...
python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --tokenizer_name bert-base-uncased \
    --model_name_or_path $model_dir \
    --per_device_eval_batch_size 64 \
    --passage_max_len 512 \
    --bf16 \
    --exclude_title \
    --dataset_name DylanJHJ/nano-beir-corpus \
    --corpus_name DylanJHJ/nano-beir-corpus \
    --dataset_split $DATASET \
    --encode_output_path $output_dir/corpus_emb.${DATASET}.pkl \
    --attn_implementation sdpa
