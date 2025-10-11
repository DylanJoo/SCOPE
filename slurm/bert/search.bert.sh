#!/bin/bash -l
#SBATCH --job-name=search
#SBATCH --output=logs/search.o
#SBATCH --error=logs/search.e
#SBATCH --partition=debug           # partition name
#SBATCH --ntasks-per-node=1       # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --nodes=1                 # Total number of nodes 
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=0-00:30:00         # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001640 # Project for billing

# module use /appl/local/csc/modulefiles/
# module use /appl/local/training/modules/AI-20241126/

cd ${HOME}/SCOPE

model_dir=${HOME}/models/bert-msmarco-psg.train-b8
output_dir=${HOME}/indices/${model_dir##*/}

for split in dl19 dl20;do
    singularity exec $SIF \
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/$split.query_emb.pkl \
	--passage_reps $output_dir/'corpus_emb.*.pkl' \
	--depth 200 \
	--batch_size -1 \
	--save_text \
	--save_ranking_to $output_dir/$split.run

    singularity exec $SIF \
    python -m tevatron.utils.format.convert_result_to_trec \
        --input $output_dir/$split.run \
        --output runs/msmarco-passage/$split.trec
done

