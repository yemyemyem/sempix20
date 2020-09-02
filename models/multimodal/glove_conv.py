#!/usr/bin/env python3
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os

root = './'
files = ['glove.6B.100d.txt']
for f in files:
    print(f)
    print("Converting glove txt -> word2vec txt...")

    glove_input_file = os.path.join(root, f)
    word2vec_output_file = glove_input_file + '.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)
    print("Converting word2vec txt -> word2vec bin...")
    w2v = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    outpath = os.path.splitext(glove_input_file)[0] + ".bin.word2vec"
    w2v.save_word2vec_format(outpath, binary=True)
