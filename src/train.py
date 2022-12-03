import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary

from src.config import MyConfig
from src.dataclasses.rating_data import RatingData
from src.model.matrix_factorization import MatrixFactrization


def make_rating_data() -> List[RatingData]:
    """評価値のcsvファイルから、ConvMFに入力するRatings情報(Rating Matrix)を作成する関数。
    Returns:
        List[RatingData]: Rating MatrixをCOO形式で。
    """
    ratings = pd.read_csv(MyConfig.ratings_path).rename(columns={"movie": "item"})
    ratings["user"] = ratings["user"].astype(np.int32)
    ratings["item"] = ratings["item"].astype(np.int32)
    ratings["rating"] = ratings["rating"].astype(np.float32)

    print("=" * 10)
    n_item = len(ratings["item"].unique())
    n_user = len(ratings["user"].unique())
    print(f"num of unique items is ...{n_item:,}")
    print(f"num of unique users is ...{n_user:,}")
    print(f"num of observed rating is ...{len(ratings):,}")
    print(f"num of values of rating matrix is ...{n_user*n_item:,}")
    print(f"So, density is {len(ratings)/(n_user*n_item) * 100 : .2f} %")
    # DataFrameからRatingDataに型変換してReturn。
    return [RatingData(*t) for t in ratings.itertuples(index=False)]


def make_item_description(max_sentence_length=None) -> Tuple[np.ndarray, np.ndarray, int]:
    """CovMFに入力するDocument情報(X_j)を作成する関数。

    Args:
        max_sentence_length (_type_, optional):
        Document情報の最大長さ(Tokenの数)
        Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
        - アイテムid のndarray
        - アイテムidに対応するdocument contextのndarray(wordをindex化してる)
        -
    """
    descriptions = pd.read_csv(MyConfig.descriptions_path).rename(columns={"movie": "item"})
    texts = descriptions["description"]
    # 英文なので半角スペースでtokenize
    texts = texts.apply(lambda x: x.strip().split())
    # str.strip()：stringの両端の指定した文字を削除する.
    # defaultは空白文字(改行\nや全角スペース\u3000やタブ\tなどが空白文字とみなされ削除)

    # 単語(token)をDictionaryオブジェクトに登録?
    dictionary = Dictionary(texts.values)
    dictionary.filter_extremes()
    eos_id = len(dictionary.keys())
    print(dictionary)

    # to index list
    # 各description の各単語を通し番号化(encoding)
    texts = texts.apply(lambda x: dictionary.doc2idx(x, unknown_word_index=eos_id))
    # List[word_index]をndarray化(unknown_word_indexを含めない)
    texts = texts.apply(lambda x: np.array([a for a in x if a != eos_id]))
    # descriptionの最大長さを取得
    max_sentence_length = max(texts.apply(len)) if max_sentence_length is None else min(max(texts.apply(len)), max)

    # padding(descriptionの長さをそろえる?)
    texts = texts.apply(lambda x: x[:max_sentence_length])
    texts = texts.apply(
        lambda x: np.pad(x, pad_width=(0, max_sentence_length - len(x)), mode="constant", constant_values=(0, eos_id))
    )

    # change types
    texts = texts.apply(lambda x: x.astype(np.int32))
    descriptions["id"] = descriptions["id"].astype(np.int32)

    return descriptions["id"].values, texts.values, len(dictionary.keys()) + 1


def train_convmf(batch_size: int, n_epoch: int, n_sub_epoch: int, n_out_channel: int):
    """_summary_

    Parameters
    ----------
    batch_size : int
        _description_
    n_epoch : int
        _description_
    n_sub_epoch : int
        _description_
    n_out_channel : int
        _description_
    """

    ratings = make_rating_data()
    filter_windows = [3, 4, 5]  # 窓関数の設定
    max_sentence_length = 300  # 300 token(word)
    movie_ids, item_descriptions, n_word = make_item_description(max_sentence_length)
    n_factor = 300
    dropout_ratio = 0.5
    user_lambda = 10
    item_lambda = 100


if __name__ == "__main__":

    ratings = make_rating_data()
    print(len(ratings))

    # mf = MatrixFactrization(ratings=ratings, n_factor=10)
    # print(type(mf.user_item_list))
    # print(len(mf.user_item_list))
    # print(len(mf.item_user_list[1]))
    # print(mf.item_user_list[1].ratings)

    # mf.fit(n_trial=5, additional=None)
