import os

# import nltk
from collections import defaultdict
from typing import Dict, List

# nltk.download('all')
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_pretrained_vectors(word2idx: Dict[str, int], frame: str):
    """学習済みの単語埋め込み(embedding)ベクトルのデータを読み込んで、
    学習データのvocabularyに登録された各tokenに対応する、単語埋め込み(embedding)ベクトルを作成する。
    Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
            配列の行indexが、word2idxの通し番号に対応。
    """

    print("Loading pretrained vectors...")
    # ファイルを開いて...
    fin = open(frame, encoding="utf-8", newline="\n", errors="ignore")
    # intで行数とか(?)を取得
    n, d = map(int, fin.readline().split())  # 登録されてる単語数, 埋め込みベクトルの次元数

    # Initilize random embeddings
    embeddings: np.ndarray = np.random.uniform(
        low=-0.25, high=0.25, size=(len(word2idx), d)  # (Vocabularyに登録された単語数, 埋め込みベクトルの次元数)
    )
    # <pad>の埋め込みベクトルは0ベクトル
    embeddings[word2idx["<pad>"]] = np.zeros(shape=(d,))

    # Load pretrained vector
    count = 0
    for line in tqdm(fin):
        # 学習済みモデルに登録されている単語と、対応する埋め込みベクトルを取得。
        tokens = line.rstrip().split(" ")
        word = tokens[0]
        # 今回のVocabularyにある単語の場合
        if word in word2idx:
            count += 1
            # 配列の行index = word2idxの通し番号として、埋込ベクトルを保存
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vector found.")

    return embeddings
