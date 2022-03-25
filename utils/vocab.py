import sys
esm_vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}

esm_to_megatron = {
    '<cls>': "[CLS]",
    '<pad>': "[PAD]",
    '<mask>': "[MASK]",
}

megatron_tok_to_idx = dict()
with open('./msa_tools/msa_vocab.txt', 'r') as f:
    for idx, c in enumerate(f.readlines()):
        megatron_tok_to_idx[c.strip()] = idx

with open('./msa_tools/msa_vocab.txt', 'r') as f:
    for idx, c in enumerate(f.readlines()[5: 31]):
        esm_to_megatron[c.strip()] = c.strip()

megatron_to_esm = dict()

m_e = dict()
for e, m in esm_to_megatron.items():
    m_e[megatron_tok_to_idx[m]] = esm_vocab[e]
print(m_e)
