import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary

from src.config import MyConfig
from src.dataclasses.rating_data import RatingLog
from src.model.matrix_factorization import MatrixFactrization


class RatingLogReader:
    def __init__(self, ratings_csv_path: str) -> None:
        self.ratings_csv_path = ratings_csv_path

    def load(self) -> List[RatingLog]:
        ratings_df = pd.read_csv(self.ratings_csv_path).rename(
            columns={"user": "user_id", "movie": "item_id", "rating": "rating"},
        )
        ratings_df["user_id"] = ratings_df["user_id"].astype(np.int32)
        ratings_df["item_id"] = ratings_df["item_id"].astype(np.int32)
        ratings_df["rating"] = ratings_df["rating"].astype(np.float32)

        self._print_rating_logs_info(ratings_df)

        # DataFrameからRatingDataに型変換してReturn。
        return [RatingLog.from_named_tuple(rating_log) for rating_log in ratings_df.itertuples()]

    def _print_rating_logs_info(self, ratings_df: pd.DataFrame) -> None:
        print("=" * 10)
        print(ratings_df.head())
        n_item = len(ratings_df["item_id"].unique())
        n_user = len(ratings_df["user_id"].unique())
        print(f"num of unique items is ...{n_item:,}")
        print(f"num of unique users is ...{n_user:,}")
        print(f"num of observed rating is ...{len(ratings_df):,}")
        print(f"num of values of rating matrix is ...{n_user*n_item:,}")
        print(f"So, density or rating matrix is {len(ratings_df)/(n_user*n_item) * 100 : .2f} %")
