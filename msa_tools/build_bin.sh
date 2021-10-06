set -ux

/opt/conda/bin/python ../tools/preprocess_data.py --input /workspace/data/TOTAL.jsonl \
        --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
        --output-prefix /workspace/DATA/TOTAL --dataset-impl mmap --workers 72

exit

# for f in UniRef50-xa-a2m-2017 UniRef50-xb-a2m-2018 UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018;
for f in UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018; # UniRef50-xa-a2m-2017; # UniRef50-xb-a2m-2018;
do
    mkdir -p /workspace/DATA/$f/$f
    /opt/conda/bin/python ../tools/preprocess_data.py --input /workspace/data/$f/$f.json \
        --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
        --output-prefix /workspace/DATA/$f/$f --dataset-impl mmap --workers 72
done

# /opt/conda/bin/python ../tools/preprocess_data.py --input /workspace/corpus/Tiny.json --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt --output-prefix /workspace/corpus/Tiny --dataset-impl mmap --workers 72

# /opt/conda/bin/python ../../tools/preprocess_data.py --input xc.json \
#         --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
#         --output-prefix xc --dataset-impl mmap --workers 72


# for f in UniRef50-xa-a2m-2017 UniRef50-xb-a2m-2018 UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018;
# do
#  echo /workspace/data/$f/$f.json
# #  /opt/conda/bin/python multi_process_preprocess.py $f;
# done
