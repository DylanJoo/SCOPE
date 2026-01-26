#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=result
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh # ilps
conda activate inference
# module load anaconda3/2024.2 # grid
# conda activate crc

model_dir=nomic-ai/modernbert-embed-base
output_dir=${HOME}/indices/msmarco-passage-corpus-new/${model_dir##*/}
mkdir -p $output_dir

# trec DL
for split in dl19 dl20;do
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/$split.query_emb.pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 100 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$split.run

    python -m tevatron.utils.format.convert_result_to_trec \
    --input $output_dir/$split.run \
    --output $output_dir/$split.trec
done

python -m ir_measures msmarco-passage/trec-dl-2019 $output_dir/dl19.trec nDCG@10
python -m ir_measures msmarco-passage/trec-dl-2020 $output_dir/dl20.trec nDCG@10
