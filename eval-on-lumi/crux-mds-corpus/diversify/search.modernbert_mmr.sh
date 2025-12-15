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

# model_dir=nomic-ai/modernbert-embed-base-unsupervised
model_dir=DylanJHJ/DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

for lambda in 1.0 0.99 0.98 0.97 0.96 0.95 0.90 0.89 0.88 0.87 0.86 0.85 0.84 0.83 0.82 0.81 0.80;do

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        echo model: $model_dir $subset
        singularity exec $SIF  \
            python -m tevatron.retriever.driver.diversify.mmr \
            --query_reps $output_dir/query_emb.$subset.pkl \
            --passage_reps $output_dir/'corpus_emb.*.pkl' \
            --depth 100 \
            --batch_size -1 \
            --save_text \
            --save_ranking_to $output_dir/$subset.mmr.run \
            --lambda_param $lambda

        singularity exec $SIF  \
            python -m tevatron.utils.format.convert_result_to_trec \
            --input $output_dir/$subset.mmr.run \
            --output $output_dir/$subset.mmr-$lambda.trec
    done

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        python -m crux.evaluation.rac_eval \
            --run $output_dir/$subset.mmr-$lambda.trec \
            --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/$subset/judge 
    done
done
