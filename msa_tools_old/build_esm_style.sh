set -u
set -x

for task in train test;
do
    /opt/conda/bin/python ../tools/preprocess_data.py --input /dataset/ee84df8b/release/ProteinLM/pretrain/data/corpus/$task.json \
            --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
            --output-prefix /dataset/ee84df8b/release/ProteinLM/pretrain/data/corpus/$task --dataset-impl mmap --workers 72
done