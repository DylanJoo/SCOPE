#!/bin/bash -l
#SBATCH --job-name=ttest
#SBATCH --output=ttest.out
#SBATCH --error=ttest.err
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=${HOME}/datasets/crux

model_dir=${HOME}/models/main.learning/ce_1.0-selfdistil_0.1
model_dir=/home/dju/models/ablation.cov-sampling/modernbert-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.request
# model_dir=/home/dju/models/ablation.two-stage/modernbert-two-stage-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.crux-researchy.request
output_dira=${HOME}/indices/crux-mds-corpus/${model_dir##*/}

model_dir=nomic-ai/modernbert-embed-base-unsupervised
# model_dir=DylanJHJ/nomic.modernbert-base.msmarco-passage.10k
output_dirb=${HOME}/indices/crux-mds-corpus/${model_dir##*/}

subset=crux-mds-duc04
python run_ttest_from_runs-mds.py \
    --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
    --filter_by_oracle \
    --judge $CRUX_ROOT/$subset/judge  \
    --run_a $output_dira/$subset.trec --run_b $output_dirb/$subset.trec \

subset=crux-mds-multi_news
python run_ttest_from_runs-mds.py \
    --qrel $CRUX_ROOT/$subset/qrels/div_qrels-tau3.txt \
    --filter_by_oracle \
    --judge $CRUX_ROOT/$subset/judge  \
    --run_a $output_dira/$subset.trec --run_b $output_dirb/$subset.trec \

