#!/bin/bash -l
#SBATCH --job-name=result
#SBATCH --cpus-per-task=32
#SBATCH --partition cpu
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=%x.out

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
Qwen/Qwen3-Embedding-0.6B
)

model_dir=${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}

python -m tevatron.retriever.driver.search \
    --query_reps $output_dir/query_emb.pkl \
    --passage_reps $output_dir/'corpus_emb*pkl' \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $output_dir/neuclir24-test.run

python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/neuclir24-test.run \
    --output $output_dir/neuclir24-test.trec

echo $output_dir/neuclir24-test.trec
python -m crux.evaluation.rac_eval \
    --run $output_dir/neuclir24-test.trec \
    --qrel $CRUX_ROOT/crux-neuclir/qrels/neuclir24-test-request.qrel \
    --judge $CRUX_ROOT/crux-neuclir/judge/ratings.human.jsonl
