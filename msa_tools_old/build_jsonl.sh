set -u
#set -x

# for name in UniRef50-xa-a2m-2017 UniRef50-xb-a2m-2018 UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018;
for name in UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018; # UniRef50-xb-a2m-2018;
do
	/opt/conda/bin/python multi_process_preprocess.py $name
done
exit


/opt/conda/bin/python ../tools/preprocess_data.py --input /workspace/contact/a2m.jsonl \
	--tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
	--output-prefix /workspace/contact/a2m_contact --dataset-impl mmap --workers 72

exit

cat /workspace/data/UniRef50-xa-a2m-2017/UniRef50-xa-a2m-2017.json /workspace/data/UniRef50-xb-a2m-2018/UniRef50-xb-a2m-2018.json /workspace/data/UniRef50-xc-a2m-2017/UniRef50-xc-a2m-2017.json /workspace/data/UniRef50-xd-a2m-2018/UniRef50-xd-a2m-2018.json /workspace/data/UniRef50-xe-a2m-2017/UniRef50-xe-a2m-2017.json /workspace/data/UniRef50-xf-a2m-2018/UniRef50-xf-a2m-2018.json > /workspace/data/Uniref.json

/opt/conda/bin/python ../tools/preprocess_data.py --input /workspace/data/Uniref.json \
	--tokenizer-type BertWordPieceCase --vocab-file ./msa_vocab.txt \
	--output-prefix /workspace/data/Uniref --dataset-impl mmap --workers 72


#for f in UniRef50-xa-a2m-2017 UniRef50-xb-a2m-2018 UniRef50-xc-a2m-2017 UniRef50-xd-a2m-2018 UniRef50-xe-a2m-2017 UniRef50-xf-a2m-2018;
#do
#  echo /workspace/data/$f/$f.json
##  /opt/conda/bin/python multi_process_preprocess.py $f;
#done
