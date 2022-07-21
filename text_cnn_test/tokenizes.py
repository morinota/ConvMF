
from typing import Dict, List
from tqdm import tqdm
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import gensim
# nltk.download('all')


def conduct_tokenize(texts: List[str]):
    """文章を単語をtokenとしてtokenizeする。
    全文章に使われている単語を確認しvocabularyを生成すると共に、文章の最大長さを記録する。
    Tokenize texts, build vocabulary and find maximum sentence length.

    Args:
        texts (List[str]): List of text data

    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    # 結果格納用の変数をInitialize
    tokenized_texts: List[List[str]] = []
    word2idx: Dict[str, int] = {}
    max_len = 0

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0  # 長さの短いSentenceに対して、長さをmax_lenにそろえるために使う?
    word2idx['<unk>'] = 1  # 未知のtokenに対する通し番号

    # Building our vocab from the corpus starting from index 2
    idx = 2

    # 各文章に対して繰り返し処理
    for text in texts:
        # tokenize
        # tokenized_text = nltk.tokenize.word_tokenize(text=text)
        tokenized_text = gensim.utils.tokenize(text=text)
        tokenized_text = list(tokenized_text)

        # Add `tokenized_text` to `tokenized_texts`
        tokenized_texts.append(tokenized_text)

        # Add new token to `word2idx`
        # text内の各tokenをチェックしていく...
        for token in tokenized_text:
            # word2idxに登録されていないtoken(単語?)があれば、通し番号を登録!
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_text))

    return tokenized_texts, word2idx, max_len


def encode(tokenized_texts: List[List[str]], word2idx: Dict[str, int], max_len: int):
    """tokenizeされた各テキストを、the maximum sentence lengthに合わせてゼロパディングする。
    加えて、tokenizeされたテキスト内の各tokenを、vocabularyの通し番号にencode(符号化)する.
    Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids: List[List[int]] = []
    for tokenized_text in tokenized_texts:
        # tokenized_textの長さがmax_lenと一致するように、最後尾に<pad>を追加する。
        # Pad sentences to max_len
        tokenized_text += ['<pad>'] * (max_len - len(tokenized_text))

        # tokenized_text内の各tokenを通し番号へ符号化
        # Encode tokens to input_ids
        input_id: List[int] = [word2idx.get(token) for token in tokenized_text]
        input_ids.append(input_id)

    # 最後は配列としてReturn
    # (R^{n \times max_len}の行列。各要素はtokenの通し番号)
    return np.array(input_ids)
