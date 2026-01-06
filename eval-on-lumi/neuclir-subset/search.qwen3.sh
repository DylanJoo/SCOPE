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
source ${HOME}/temp/venv/crux/bin/activate

CRUX_ROOT=${HOME}/datasets/crux
model_dir=Qwen/Qwen3-Embedding-8B
output_dir=${HOME}/indices/neuclir1-subset-corpus/${model_dir##*/}
mkdir -p $output_dir

singularity exec $SIF  \
    python -m tevatron.retriever.driver.search \
    --query_reps $output_dir/query_emb.pkl \
    --passage_reps $output_dir/'corpus_emb*pkl' \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $output_dir/neuclir24-test.run

singularity exec $SIF  \
    python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/neuclir24-test.run \
    --output $output_dir/neuclir24-test.trec

echo $output_dir/neuclir24-test.trec
python -m crux.evaluation.rac_eval \
    --run $output_dir/neuclir24-test.trec \
    --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
    --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl

