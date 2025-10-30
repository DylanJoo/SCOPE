#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=search.o
#SBATCH --error=search.e
#SBATCH --partition=debug           # partition name
#SBATCH --ntasks-per-node=1       # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                 # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-00:30:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

cd ${HOME}/SCOPE
mkdir -p runs/msmarco-passage

# dl19/20:
# model_dir=${HOME}/models/dpr-bert-base-uncased.b32_n256.msmarco-passage.3ep
# A100 45K: 0.6548/0.6443/0.3449

model_dir=${HOME}/models/bert-msmarco-psg.b64_n512.1e-5
checkpoint=checkpoint-25000
# AMD  25K: 0.6548 0.6400
# AMD  25K (title) 0.6677 0.6291
# AMD  25K (no title) 0.6487 0.6710

# train without title/ eval with title: 0.6195/0.6110
# train with title/ eval with title: 

output_dir=${HOME}/indices/${model_dir##*/}


echo model: $model_dir, checkpoint: $checkpoint
# trec DL
for split in dl19 dl20;do
    singularity exec $SIF \
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/$split.query_emb.pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 100 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$split.run

    singularity exec $SIF \
    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/$split.run \
        --output $output_dir/$split.trec
done

singularity exec $SIF \
    python -m ir_measures msmarco-passage/trec-dl-2019 $output_dir/dl19.trec nDCG@10
singularity exec $SIF \
    python -m ir_measures msmarco-passage/trec-dl-2020 $output_dir/dl20.trec nDCG@10

# Dev
# split=dev
# singularity exec $SIF \
# python -m tevatron.retriever.driver.search \
#     --query_reps $output_dir/$split.query_emb.pkl \
#     --passage_reps $output_dir/'corpus_emb.*.pkl' \
#     --depth 100 \
#     --batch_size -1 \
#     --save_text \
#     --save_ranking_to $output_dir/$split.run
#
# singularity exec $SIF \
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input $output_dir/$split.run \
#     --output $output_dir/$split.trec
#
# singularity exec $SIF \
#     python -m ir_measures msmarco-passage/dev/small $output_dir/dev.trec RR@10
