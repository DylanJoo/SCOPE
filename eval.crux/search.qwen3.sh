#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=result.out

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux
model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
mkdir -p $output_dir

# duc04
for subset in crux-mds-duc04 crux-mds-multi_news;do
    python -m tevatron.retriever.driver.search \
        --query_reps $output_dir/query_emb.${subset}.pkl \
        --passage_reps "$output_dir/corpus_emb.*.pkl" \
        --depth 100 \
        --batch_size 64 \
        --save_text \
        --save_ranking_to $output_dir/$subset.run

    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/$subset.run \
        --output $output_dir/$subset.trec

    # search
    python -m crux.evaluation.rac_eval \
        --run $output_dir/$subset.trec \
        --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
        --filter_by_oracle \
        --judge $CRUX_ROOT/$subset/judge 
done
