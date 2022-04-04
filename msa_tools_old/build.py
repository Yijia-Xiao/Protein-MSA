job_num = 50

def process_split(split):
    # print(f'start process {split}')
    cmd = f"""/opt/conda/bin/python ../tools/preprocess_data.py --input /dataset/ee84df8b/data/JSON/MSA_AB384BL512_0_{split}.json \
            --tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
            --output-prefix /dataset/ee84df8b/data/BIN/MSA_AB384BL512_0_{split} --dataset-impl mmap --workers {job_num}"""
    print(cmd)


for i in range(4):
    process_split(i)

