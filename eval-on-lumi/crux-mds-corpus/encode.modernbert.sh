#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/modernbert.out
#SBATCH --error=logs/modernbert.err
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=project_465002438

# ENV
module use /appl/local/csc/modulefiles/
module use /appl/local/training/modules/AI-20241126/

CRUX_ROOT=${HOME}/datasets/crux
MODEL_DIRS=(
# "nomic-ai/modernbert-embed-base-unsupervised"
# "DylanJHJ/nomic.modernbert-base.msmarco-passage.10k"
# "DylanJHJ/nomic.modernbert-base.crux-researchy-flatten.10k"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_20.neg_51.filtered.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_quarter.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_zero.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_half.neg_zero.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_low.neg_zero.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_high.neg_low.b64_n512.1e-4"
# "${HOME}/models/ablation.cov-sampling/modernbert-crux-researchy-pos_zero.neg_high.b64_n512.1e-4"
"${HOME}/models/ablation.two-stage/modernbert-two-stage-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.msmarco"
"${HOME}/models/ablation.two-stage/modernbert-two-stage-crux-researchy-pos_half.neg_zero.b64_n512.1e-4.crux-researchy"
)

for model_dir in "${MODEL_DIRS[@]}"; do
    output_dir=${HOME}/indices/crux-mds-corpus/${model_dir##*/}
    mkdir -p $output_dir

    for split in train test;do
	echo Encoding crus-mds-corpus $split
	singularity exec $SIF  \
	    python -m tevatron.retriever.driver.encode \
	    --output_dir=temp \
	    --tokenizer_name answerdotai/ModernBERT-base \
	    --model_name_or_path $model_dir \
	    --per_device_eval_batch_size 1280 \
	    --passage_max_len 512 \
	    --pooling mean --normalize --bf16 \
	    --passage_prefix "search_document: " \
	    --exclude_title \
	    --dataset_name DylanJHJ/crux-mds-corpus \
	    --dataset_split $split \
	    --encode_output_path $output_dir/corpus_emb.$split.pkl 
    done

    for subset in crux-mds-duc04 crux-mds-multi_news;do
        for topic_path in $CRUX_ROOT/$subset/topic/*jsonl; do
            echo Encoding queries$topic_path
            singularity exec $SIF  \
                python -m tevatron.retriever.driver.encode \
                --output_dir=temp \
                --tokenizer_name answerdotai/ModernBERT-base \
                --model_name_or_path $model_dir \
                --pooling mean --normalize --bf16 \
                --query_prefix "search_query: " \
                --per_device_eval_batch_size 128 \
                --dataset_path $topic_path \
                --encode_output_path $output_dir/query_emb.${subset}.pkl \
                --query_max_len 128 \
                --encode_is_query
        done
    done

done
