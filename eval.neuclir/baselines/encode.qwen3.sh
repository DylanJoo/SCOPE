#!/bin/bash -l
#SBATCH --job-name=encode
#SBATCH --output=logs/qwen3.out.%a
#SBATCH --error=logs/qwen3.err.%a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=1        
#SBATCH --array=0-2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# ENV
source /ivi/ilps/personal/dju/miniconda3/etc/profile.d/conda.sh
conda activate inference

CRUX_ROOT=/home/dju/datasets/crux
model_dir=Qwen/Qwen3-Embedding-0.6B
output_dir=${HOME}/indices/neuclir1/${model_dir##*/}
mkdir -p $output_dir

LANGS=(
"rus"
"fas"
"zho"
)
LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}

for shard_idx in 0;do
    echo Encoding $LANG $shard_idx
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --bf16 \
        --per_device_eval_batch_size 128 \
        --normalize \
        --pooling last \
        --padding_side left \
        --passage_prefix "" \
        --passage_max_len 1024 \
        --dataset_path ${HOME}/datasets/neuclir1/${LANG}.processed_output.jsonl.gz \
        --encode_output_path $output_dir/corpus_emb.${LANG}-${shard_idx}.pkl \
        --dataset_number_of_shards 2 \
        --dataset_shard_index $shard_idx
done

topic_path=$CRUX_ROOT/crux-neuclir/topic/neuclir24-test-request.jsonl
if [[ $LANG -eq "rus" ]]; then
    echo "Encoding $topic_path"
    python -m tevatron.retriever.driver.encode \
        --output_dir=temp \
        --model_name_or_path $model_dir \
        --bf16 \
        --per_device_eval_batch_size 32 \
        --normalize \
        --pooling last  \
        --padding_side left \
        --query_prefix "Instruct: Given a report request, retrieve relevant passages that provide context to the report.\nQuery: " \
        --query_max_len 256 \
        --dataset_path $topic_path \
        --encode_output_path $output_dir/query_emb.pkl \
        --encode_is_query
fi
