# Encode queries
source ~/.bashrc
enter_conda
conda activate crc

mkdir -p ${HOME}/temp/output_repllama/repllama3/runs

# for lang in zho rus fas;do
# for year in 2022 2023 2024;do
# output_run_file=${HOME}/temp/output_repllama/repllama3/runs/run.repllama_neuclir_${year}_${lang}.mt.txt 
# python -m tevatron.retriever.driver.search \
#     --query_reps ${HOME}/temp/output_repllama/repllama3/queries_neuclir_${year}.pkl \
#     --passage_reps "${HOME}/temp/output_repllama/repllama3/corpus_neuclir_${lang}.mt.pkl*" \
#     --depth 100 \
#     --batch_size 64 \
#     --save_text \
#     --save_ranking_to ${output_run_file}
#
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input ${output_run_file} \
#     --output ${output_run_file/txt/trec} \
#     --remove_query
# done
# done

# multilingual
# lang=mlir
# for year in 2023 2024;do
# output_run_file=${HOME}/temp/output_repllama/repllama3/runs/run.repllama_neuclir_${year}_${lang}.mt.txt 
# python -m tevatron.retriever.driver.search \
#     --query_reps ${HOME}/temp/output_repllama/repllama3/queries_neuclir_${year}.pkl \
#     --passage_reps "${HOME}/temp/output_repllama/repllama3/corpus_neuclir_*.mt.pkl*" \
#     --depth 100 \
#     --batch_size 64 \
#     --save_text \
#     --save_ranking_to ${output_run_file}
#
# python -m tevatron.utils.format.convert_result_to_trec \
#     --input ${output_run_file} \
#     --output ${output_run_file/txt/trec} \
#     --remove_query
# done

for lang in zho rus fas mlir;do
for year in 2022 2023 2024;do
echo "Evaluating ${year} ${lang}..."
qrels_file=/expscratch/eyang/collections/neuclir/qrels/${year}.${lang}.qrels
output_run_file=${HOME}/temp/output_repllama/repllama3/runs/run.repllama_neuclir_${year}_${lang}.mt.txt 
${HOME}/temp/trec_eval \
    -M 1000 -c -m ndcg_cut.20 ${qrels_file} ${output_run_file/txt/trec} 
done
done
