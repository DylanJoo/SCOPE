#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=search.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00         # Run time (d-hh:mm:ss)

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate ir

model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/crux-researchy-corpus/${model_dir##*/}
mkdir -p $output_dir

# Start experimentss
split=train
python -m tevatron.retriever.driver.search \
    --query_reps $output_dir/query_emb.$split.pkl \
    --passage_reps $output_dir/'corpus_emb.*.pkl' \
    --depth 100 \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $output_dir/$split.run

python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/$split.run \
    --output $output_dir/$split.trec
