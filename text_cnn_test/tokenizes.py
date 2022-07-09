
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import os
import nltk
from collections import defaultdict
import numpy as np
# nltk.download('all')
nltk.tokenize.word_tokenize

def conduct_tokenize(texts: List[str]):
    """Tokenize texts, build vocabulary and find maximum sentence length.
    
    Args:
        texts (List[str]): List of text data
    
    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    # 全てのDocumentの中で最大の長さを記録
    max_len = 0
    # 結果格納用のList
    tokenized_texts:List[List[str]] = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0 # 長さの短いSentenceに対して、長さをmax_lenにそろえるために使う?
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sentence in texts:
        tokenized_sent = nltk.tokenize.word_tokenize(text=sentence)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            # word2idxに登録されていないtoken(単語?)があれば...
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len

def encode(tokenized_text:List[List[str]], word2idx:Dict[str, int], max_len:int):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids:List[List[int]] = []
    for tokenized_sent in tokenized_text:
        # Pad sentences to max_len tokenized_sentの長さをmax_lenまで補完する
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)

    # 最後は配列としてReturn
    return np.array(input_ids)



