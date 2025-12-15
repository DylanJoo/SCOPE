#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=project_465001640 # Project for billing

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source "/users/judylan1/temp/miniconda3/etc/profile.d/conda.sh"
conda activate gpt

CRUX_ROOT=${HOME}/datasets/crux
model_dir="nomic-ai/modernbert-embed-base-unsupervised"
# model_dir="DylanJHJ/nomic.modernbert-base.msmarco-passage.10k"
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for subset in crux-mds-duc04 crux-mds-multi_news;do
    for fusion in ranklist-round-robin score-sum ranklist-reciprocal-fusion;do
        singularity exec $SIF  \
            python -m tevatron.retriever.driver.diversify.mqr \
            --query_reps $output_dir/query_emb.$subset.mqr.pkl \
            --passage_reps $output_dir/'corpus_emb.*.pkl' \
            --depth 100 \
            --batch_size -1 \
            --save_text \
            --save_ranking_to $output_dir/$subset.mqr.run \
            --fusion_method $fusion

        singularity exec $SIF  \
            python -m tevatron.utils.format.convert_result_to_trec \
            --input $output_dir/$subset.mqr.run \
            --output $output_dir/$subset.mqr.trec.$fusion

        python -m crux.evaluation.rac_eval \
            --run $output_dir/$subset.mqr.trec.$fusion \
            --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/$subset/judge 
    done
done
