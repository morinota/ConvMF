import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary

from src.config import MyConfig
from src.dataclasses.rating_data import RatingLog
from src.model.matrix_factorization import MatrixFactrization
from src.utils.rating_log_loader import RatingLogReader


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

    rating_log_reader = RatingLogReader(ratings_csv_path=MyConfig.ratings_path)
    rating_logs = rating_log_reader.load()

    filter_windows = [3, 4, 5]  # 窓関数の設定
    max_sentence_length = 300  # 300 token(word)
    movie_ids, item_descriptions, n_word = make_item_description(max_sentence_length)
    n_factor = 300
    dropout_ratio = 0.5
    user_lambda = 10
    item_lambda = 100


if __name__ == "__main__":

    rating_log_reader = RatingLogReader(ratings_csv_path=MyConfig.ratings_path)
    rating_logs = rating_log_reader.load()
    print(len(rating_logs))

    # mf = MatrixFactrization(ratings=ratings, n_factor=10)
    # print(type(mf.user_item_list))
    # print(len(mf.user_item_list))
    # print(len(mf.item_user_list))
    # print(mf.item_user_list[1].ratings)

    # mf.fit(n_trial=5, document_vectors=None)
