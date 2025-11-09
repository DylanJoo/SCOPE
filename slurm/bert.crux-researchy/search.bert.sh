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

model_dir=${HOME}/models/bert-crux-researchy.b32_n256.1e-5.25k.train
checkpoint=checkpoint-20000
output_dir=${HOME}/indices/${model_dir##*/}
# msmarco-passage 25K: 0.6487 0.6710
# crux-researchy: 10k: 0.4279 0.3719
# crux-researchy: 20k: 
# crux-researchy: 25k: 0.4289 0.3964
# further-pretrained 10k: 0.6157 0.6118

echo model: $model_dir
for split in dl19 dl20;do
    singularity exec $SIF \
    python -m tevatron.retriever.driver.search \
	--query_reps $output_dir/query_emb.$split.pkl \
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
