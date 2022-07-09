from typing import List, Tuple
import os
import pandas as pd
import numpy as np
from model.matrix_factorization import RatingData, MatrixFactrization
from config import Config
from gensim.corpora.dictionary import Dictionary


def make_rating_data() -> List[RatingData]:
    """ConvMFに入力するRatings情報(Rating Matrix)を作成する関数。

    Returns:
        List[RatingData]: Rating Matrix(実際は、非ゼロ要素のみをListで)
    """
    ratings = pd.read_csv(Config.ratings_path).rename(
        columns={'movie': 'item'})
    ratings['user'] = ratings['user'].astype(np.int32)
    ratings['item'] = ratings['item'].astype(np.int32)
    ratings['rating'] = ratings['rating'].astype(np.float32)
    # DataFrameからRatingDataに型変換して返す。
    return [RatingData(*t) for t in ratings.itertuples(index=False)]


def make_item_description(max_sentence_length=None) -> Tuple[np.ndarray, np.ndarray, int]:
    """CovMFに入力するDocument情報(X_j)を作成する関数。

    Args:
        max_sentence_length (_type_, optional): _description_. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: 
        - アイテムid のndarray
        - アイテムidに対応するdocument contextのndarray(wordをindex化してる)
        - 
    """
    descriptions = pd.read_csv(Config.descriptions_path).rename(
        columns={'movie': 'item'})
    texts = descriptions['description']
    texts = texts.apply(lambda x: x.strip().split())
    # str.strip()：stringの両端の指定した文字を削除する.
    # defaultは空白文字(改行\nや全角スペース\u3000やタブ\tなどが空白文字とみなされ削除)

    # 単語(token)をDictionaryオブジェクトに登録?
    dictionary = Dictionary(texts.values)
    dictionary.filter_extremes()
    eos_id = len(dictionary.keys())
    print(dictionary)

    # to index list
    # 各description の各単語をindex化()
    texts = texts.apply(lambda x: dictionary.doc2idx(
        x, unknown_word_index=eos_id))
    # List[word_index]をndarray化(unknown_word_indexを含めない)
    texts = texts.apply(lambda x: np.array([a for a in x if a != eos_id]))
    # descriptionの最大長さを取得
    max_sentence_length = max(texts.apply(
        len)) if max_sentence_length is None else min(max(texts.apply(len)), max)

    # padding(descriptionの長さをそろえる?)
    texts = texts.apply(lambda x: x[:max_sentence_length])
    texts = texts.apply(
        lambda x: np.pad(
            x,
            pad_width=(0, max_sentence_length-len(x)),
            mode='constant',
            constant_values=(0, eos_id)
        )
    )

    # change types
    texts = texts.apply(lambda x: x.astype(np.int32))
    descriptions['id'] = descriptions['id'].astype(np.int32)

    return descriptions['id'].values, texts.values, len(dictionary.keys())+1


def train_convmf(batch_size: int, n_epoch: int,
                 n_sub_epoch: int, n_out_channel: int):
    
    ratings = make_rating_data()
    filter_windows = [3,4,5] # 窓関数の設定
    max_sentence_length = 300 # 300 token(word)
    movie_ids, item_descriptions, n_word = make_item_description(max_sentence_length)
    n_factor = 300 
    dropout_ratio = 0.5
    user_lambda = 10
    item_lambda = 100




if __name__ == '__main__':
    make_item_description()
