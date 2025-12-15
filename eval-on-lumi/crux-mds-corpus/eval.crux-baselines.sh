#!/bin/bash -l
#SBATCH --job-name=baseline
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
source "/users/judylan1/temp/miniconda3/etc/profile.d/conda.sh"
conda activate gpt

CRUX_ROOT=${HOME}/datasets/crux
for subset in crux-mds-duc04 crux-mds-multi_news;do
    for baseline in ${HOME}/crux/*${subset}*.trec;do
        echo $baseline
        python -m crux.evaluation.rac_eval \
            --run $baseline \
            --qrel $CRUX_ROOT/${subset}/qrels/div_qrels-tau3.txt \
            --filter_by_oracle \
            --judge $CRUX_ROOT/${subset}/judge 
    done
done
