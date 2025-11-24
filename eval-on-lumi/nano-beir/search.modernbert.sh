#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.%a
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-11%12
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=project_465001640 # Project for billing

# ENV
# source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
# conda activate inference
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

pooling=cls
# model_dir=${HOME}/models/modernbert-msmarco-psg.b64_n512.1e-4.$pooling
# model_dir=${HOME}/models/modernbert-crux-researchy-flatten.64_n512.1e-4.512.cls
# model_dir=${HOME}/models/crux-research-train-series/modernbert-crux-researchy-flatten.64_n512.1e-4.512
model_dir=${HOME}/models/crux-research-train-series/modernbert-crux-researchy-flatten.pos_5.neg_1.64_n512.1e-4.512
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
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo Encoding $DATASET ...
singularity exec $SIF  \
    python -m tevatron.retriever.driver.search \
    --query_reps $output_dir/query_emb.${DATASET}.pkl \
    --passage_reps $output_dir/corpus_emb.${DATASET}.pkl \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $output_dir/${DATASET}.run

singularity exec $SIF  \
    python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/${DATASET}.run \
    --output $output_dir/${DATASET}.trec

irds_tag=$(echo "$DATASET" | sed 's/_/-/g; s/\./\//g')
result=$(singularity exec $SIF  python -m ir_measures $irds_tag $output_dir/${DATASET}.trec nDCG@10)

short_name=$(basename "$irds_tag" | cut -c1-3)
echo "${short_name} | $result"
