# Pretrain Protein Language Model with Megatron-LM

# Setup

## Docker Environment
We recommend you use Docker for environment setup. Download [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) version 20.12, which uses python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3.


To use this repository, please install the latest supported versions of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We strongly recommend using one of [NGC's recent PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) (the latest compatible version at time of publication can be pulled with `docker pull nvcr.io/nvidia/pytorch:20.12-py3`). Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.


## Downloading Pretrained Models
We have provided pretrained protein model. 

### Protein-MSA (1B) 
For the pretrained model with 1 billion parameters (950 million), you can download model checkpoint from [here](https://resource.wudaoai.cn/).


The models require vocabulary files to run, we use 20 standard amino acids tokens, and 2 special tokens: `[MASK], [-]`. The vocabulary is provided in text format [msa_vocab.txt](./msa_tools/msa_vocab.txt)

# Usage

There are two stages as follows:
1. Data preprocessing
2. Pretraining


We've provided scripts for pretraining MSA transformer model: [pretrain_tape.sh](./examples/pretrain_tape.sh), and [pretrain_tape_distributed.sh](./examples/pretrain_tape_distributed.sh) for multi-node training.


## Data Preprocessing
### Datasets
Our pretraining is carried out on [PFAM](http://s3.amazonaws.com/proteindata/data_pytorch/pfam.tar.gz) dataset. After downloading and extracting the data, you will find the following three folders and one text file `pfam_train.lmdb, pfam_valid.lmdb, pfam_holdout.lmdb, pfam_strings.txt`. What we will use is the text file, which contains 32M animo acids entries. With `pfam_strings.txt` ready, next steps are preprocessing and pretraining.

Scripts and guidance are available in [protein_tools](./protein_tools/).

### Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:

<pre>
{"text": "GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ"}
{"text": "RTIKVRILHAIGFEGGLMLLTIPMVAYAMDMTLFQAILLDLSMTTCILVYTFIFQWCYDILENR"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-tape-corpus.json \
       --output-prefix my-tape \
       --vocab iupac-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-tape_text_sentence.bin` and `my-tape_text_sentence.idx`. The `--data-path` specified in later TAPE training is the full path and new filename, but without the file extension.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).


## Pretraining

### Protein Model Training
Scripts and guidance are available in [pretrain_tools](./pretrain_tools/).

`bash ./examples/pretrain_tape.sh`

This script runs single GPU protein model pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min-lr` over `--lr-decay-iters` iterations. The fraction of training iterations used for warmup is set by `--lr-warmup-fraction`. While this is single GPU training, the batch size specified by `--micro-batch-size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global-batch-size` whcih is the batch size per iteration. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train-iters` as the training iterations requested. Alternatively, one can provide `--train-samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr-decay-iters`, one will need to provide `--lr-decay-samples`.

The logging, checkpoint-saving, and evaluation intervals are specified. Checkpointing the activations facilitates the training of larger models and/or batches. Note that the `--data-path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).


### Distributed Protein Model Training

[bash examples/pretrain_tape_distributed.sh](./examples/pretrain_tape_distributed.sh)

These scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables and using `init_method='env://'` in the launcher. See the official PyTorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the Python flag `-m torch.distributed.launch`, detailed below, are the only additional requirements to adopt distributed training.


**Note**
If you encounter `timeout` problem when running `pretrain_tape_distributed.sh`, you can set `'timeout'` parameter of `torch.distributed.init_process_group()` to a longer interval.


# Citation
Please cite our paper if you find our work useful for your research. Our paper is can be accessed [here](https://arxiv.org/abs/2108.07435).
```
@article{DBLP:journals/corr/abs-2108-07435,
  author    = {Yijia Xiao and
               Jiezhong Qiu and
               Ziang Li and
               Chang{-}Yu Hsieh and
               Jie Tang},
  title     = {Modeling Protein Using Large-scale Pretrain Language Model},
  journal   = {CoRR},
  volume    = {abs/2108.07435},
  year      = {2021},
  url       = {https://arxiv.org/abs/2108.07435},
  eprinttype = {arXiv},
  eprint    = {2108.07435},
  timestamp = {Fri, 20 Aug 2021 13:55:54 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2108-07435.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


# Reference

Our work is based on the following papers. And part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

[__Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism__](https://arxiv.org/abs/1909.08053v4)
```
@article{DBLP:journals/corr/abs-1909-08053,
  author    = {Mohammad Shoeybi and
               Mostofa Patwary and
               Raul Puri and
               Patrick LeGresley and
               Jared Casper and
               Bryan Catanzaro},
  title     = {Megatron-LM: Training Multi-Billion Parameter Language Models Using
               Model Parallelism},
  journal   = {CoRR},
  volume    = {abs/1909.08053},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.08053},
  archivePrefix = {arXiv},
  eprint    = {1909.08053},
  timestamp = {Tue, 24 Sep 2019 11:33:51 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-08053.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

[__MSA Transformer__](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1))
```
@article {Rao2021.02.12.430858,
	author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
	title = {MSA Transformer},
	elocation-id = {2021.02.12.430858},
	year = {2021},
	doi = {10.1101/2021.02.12.430858},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Unsupervised protein language models trained across millions of diverse sequences learn structure and function of proteins. Protein language models studied to date have been trained to perform inference from individual sequences. The longstanding approach in computational biology has been to make inferences from a family of evolutionarily related sequences by fitting a model to each family independently. In this work we combine the two paradigms. We introduce a protein language model which takes as input a set of sequences in the form of a multiple sequence alignment. The model interleaves row and column attention across the input sequences and is trained with a variant of the masked language modeling objective across many protein families. The performance of the model surpasses current state-of-the-art unsupervised structure learning methods by a wide margin, with far greater parameter efficiency than prior state-of-the-art protein language models.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2021/02/13/2021.02.12.430858},
	eprint = {https://www.biorxiv.org/content/early/2021/02/13/2021.02.12.430858.full.pdf},
	journal = {bioRxiv}
}

```
