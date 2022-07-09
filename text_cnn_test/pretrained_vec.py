from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import os
import nltk
from collections import defaultdict
import numpy as np
# nltk.download('all')
nltk.tokenize.word_tokenize



def load_pretrained_vectors(word2idx: Dict[str, int], frame:str):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print('Loading pretrained vectors...')
    # ファイルを開いて...
    fin = open(frame, encoding='utf-8', newline='\n', errors='ignore')
    # intで行数とか(?)を取得
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(
        low=-0.25, high=0.25,
        size=(len(word2idx), d)
        )
    embeddings[word2idx['<pad>']] = np.zeros(shape=(d,))

    # Load pretrained vector
    count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count+=1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)
        
    print(f'There are {count} / {len(word2idx)} pretrained vector found.')

    return embeddings