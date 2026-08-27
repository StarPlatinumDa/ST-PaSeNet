import gzip

merges=gzip.open("bpe_simple_vocab_16e6.txt.gz").read().decode("utf-8").split('\n')

print(merges)