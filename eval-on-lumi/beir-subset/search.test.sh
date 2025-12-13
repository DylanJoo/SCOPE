#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.%a
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-12%13
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=project_465001640 # Project for billing

# ENV
# source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
# conda activate inference
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

model_dir=DylanJHJ/dpr.modernbert-base.msmarco-passage.25k
output_dir=${HOME}/indices/beir-subset-corpus/${model_dir##*/}
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

QRELS=(
"beir/arguana"
"beir/climate-fever"
"beir/dbpedia-entity/test"
"beir/fever/test"
"beir/fiqa/test"
"beir/hotpotqa/test"
"beir/nfcorpus/test"
"beir/nq"
"beir/quora/test"
"beir/scidocs"
"beir/scifact/test"
"beir/trec-covid"
"beir/webis-touche2020/v2"
)
for lambda in 0.95 0.90 0.8 0.7 0.6 0.5 0.25 0.0;do

    singularity exec $SIF  \
        python -m tevatron.retriever.driver.search_mmr \
        --query_reps $output_dir/query_emb.${DATASET}.pkl \
        --passage_reps $output_dir/corpus_emb.${DATASET}.pkl \
        --depth 100 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to $output_dir/${DATASET}.run \
        --lambda_param $lambda 

    singularity exec $SIF  \
        python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/${DATASET}.run \
        --output $output_dir/${DATASET}.trec

    irds_tag=${QRELS[$SLURM_ARRAY_TASK_ID]}
    result=$(singularity exec $SIF  python -m ir_measures $irds_tag $output_dir/${DATASET}.trec nDCG@10)

    short_name=$(basename "$DATASET" | cut -c6-8)
    echo "${short_name} | $lambda |$result"

done
