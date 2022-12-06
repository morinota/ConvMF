from io import TextIOWrapper
import os

# import nltk
from collections import defaultdict
from typing import Dict, List

import torch

from torch import Tensor

# nltk.download('all')
import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class WordEmbeddingVector:
    word2idx_mapping: Dict[str, int]
    vectors: np.ndarray
    word_num: int
    vec_dim: int

    @classmethod
    def load_pretrained_vectors(
        cls,
        word2idx: Dict[str, int],
        pretrained_vectors_path: str,
        padding_word: str = "<pad>",
    ) -> "WordEmbeddingVector":
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
        pretrained_file = open(
            pretrained_vectors_path, encoding="utf-8", newline="\n", errors="ignore"
        )
        # intで行数とか(?)を取得
        word_num, vec_dim = map(
            int, pretrained_file.readline().split()
        )  # 登録されてる単語数, 埋め込みベクトルの次元数

        embeddings_initialized = cls._initialize_vectors(
            word2idx, vec_dim, padding_word
        )

        pretrained_vectors = cls._load_pretrained(
            embeddings_initialized, word2idx, pretrained_file
        )

        pretrained_file.close()

        return WordEmbeddingVector(
            word2idx_mapping=word2idx,
            vectors=pretrained_vectors,
            word_num=word_num,
            vec_dim=vec_dim,
        )

    @classmethod
    def _initialize_vectors(
        cls, word2idx: Dict[str, int], vec_dim: int, padding_word: str
    ) -> np.ndarray:
        # Initilize random embeddings(学習済みベクトルを入れる器を作る)
        embedding_vectors: np.ndarray = np.random.uniform(
            low=-0.25,
            high=0.25,
            size=(len(word2idx), vec_dim),  # (Vocabularyに登録された単語数, 埋め込みベクトルの次元数)
        )
        # <pad>の埋め込みベクトルは0ベクトル
        embedding_vectors[word2idx[padding_word]] = np.zeros(shape=(vec_dim,))

        return embedding_vectors

    @classmethod
    def _load_pretrained(
        cls,
        embedding_vectors: np.ndarray,
        word2idx: Dict[str, int],
        pretrained_file: TextIOWrapper,
    ) -> np.ndarray:
        # Load pretrained vector
        word_count = 0
        for line in tqdm(pretrained_file):
            # 学習済みモデルに登録されている単語と、対応する埋め込みベクトルを取得。
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            # 今回のVocabularyにある単語の場合
            if word in word2idx:
                word_count += 1
                # 配列の行index = word2idxの通し番号として、埋込ベクトルを保存
                embedding_vectors[word2idx[word]] = np.array(
                    tokens[1:], dtype=np.float32
                )
        print(f"[LOG]There are {word_count} / {len(word2idx)} pretrained vector found.")

        return embedding_vectors

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.vectors)  # np.ndarray => torch.Tensor
