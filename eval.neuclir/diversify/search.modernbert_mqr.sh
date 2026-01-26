#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result.out.o
#SBATCH --error=result.err.o
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

echo model: $model_dir
query_reps=$output_dir/query_emb.mqr.pkl
query_reps=$output_dir/query_emb.mqr-oracle.pkl
for fusion in ranklist-round-robin score-sum ranklist-reciprocal-fusion;do
    python -m tevatron.retriever.driver.diversify.mqr \
        --query_reps $query_reps \
        --passage_reps $output_dir/'corpus_emb.*.pkl' \
        --depth 100 \
        --batch_size -1 \
        --save_text \
        --save_ranking_to $output_dir/neuclir24-test.run \
        --fusion_method $fusion 

    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/neuclir24-test.run \
        --output $output_dir/neuclir24-test.mqr-oracle.trec.$fusion

    python -m crux.evaluation.rac_eval \
        --run $output_dir/neuclir24-test.mqr-oracle.trec.$fusion \
        --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
        --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
done
