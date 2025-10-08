# Encode queries
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_qwen3/runs
lang=zho
output_run_file=${HOME}/temp/output_qwen3/runs/run.qwen3_neuclir_${lang}.mt.txt
qrels_file=/expscratch/eyang/collections/neuclir/qrels/2024.${lang}.qrels
topic_file=/expscratch/eyang/collections/neuclir/topics/2024.jsonl

python -m tevatron.utils.format.convert_result_to_trec \
    --input ${output_run_file} \
    --output ${output_run_file/txt/trec} \
    --remove_query

${HOME}/temp/trec_eval \
    -m ndcg_cut.20 -m recall.1000 \
    ${qrels_file} ${output_run_file/txt/trec} 
