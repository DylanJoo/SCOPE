#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --partition=small
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
source "/users/judylan1/temp/miniconda3/etc/profile.d/conda.sh"
conda activate gpt

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.1.orth-0.0001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.1.orth-0.001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.25.orth-0.0001.request"
"${HOME}/models/msmarco-passage-pft.multiview-mean.kld-0.25.sq-0.25.orth-0.001.request"
)

for subset in crux-mds-duc04 crux-mds-multi_news;do

    for model_dir in "${MODEL_DIRS[@]}"; do
        output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
        mkdir -p $output_dir
        echo $output_dir
        singularity exec $SIF  \
            python -m tevatron.retriever.driver.search \
            --query_reps $output_dir/query_emb.$subset.pkl \
            --passage_reps $output_dir/'corpus_emb.*.pkl' \
            --depth 100 \
            --batch_size 32 \
            --save_text \
            --aggregation_strategy rrf \
            --save_ranking_to $output_dir/$subset.run
        
        singularity exec $SIF  \
            python -m tevatron.utils.format.convert_result_to_trec \
            --input $output_dir/$subset.run \
            --output $output_dir/$subset.trec

        python -m crux.evaluation.rac_eval \
            --run $output_dir/$subset.trec \
            --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/$subset/judge 
    done
done

