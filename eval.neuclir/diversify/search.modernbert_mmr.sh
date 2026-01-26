#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux

model_dir=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
# model_dir=${HOME}/models/main.learning/ce_1.0-selfdistil_0.1
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
mkdir -p $output_dir

for lambda in 1.00 0.99 0.98 0.8 0.7;do
    echo model: $model_dir
    python -m tevatron.retriever.driver.diversify.mmr \
        --query_reps $output_dir/query_emb.pkl \
        --passage_reps $output_dir/'corpus_emb.*.pkl' \
        --depth 100 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to $output_dir/neuclir24-test.run \
        --lambda_param $lambda

    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/neuclir24-test.run \
        --output $output_dir/neuclir24-test.mmr-$lambda.trec

    python -m crux.evaluation.rac_eval \
        --run $output_dir/neuclir24-test.mmr-$lambda.trec \
        --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
        --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
done

