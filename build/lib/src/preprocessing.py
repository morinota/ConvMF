import os
import re
from cgi import print_arguments

import numpy as np
import pandas as pd

from config import MyConfig


def make_descriptions() -> pd.DataFrame:
    """アイテムのDescription documentのファイルを整形する関数.

    Returns:
        pd.DataFrame: 整形されたDescription document
    """
    # Movielensデータを読み込み
    title_id = pd.read_csv(
        os.path.join(MyConfig.movie_len_dir, "movies.dat"), sep="::", engine="python", names=["id", "title", "tag"]
    )
    title_id = title_id[["id", "title"]]
    # titleカラムの前処理:title(year)=>titleのみに変換
    title_id["title"] = title_id["title"].apply(lambda x: re.sub(r"\(\d+\)", "", x).rstrip())
    print(title_id.head())

    # アイテムの説明文書のデータを読み込み
    movie_df = pd.read_csv(MyConfig.tmdb_movies_path)[["title", "overview"]]
    print(movie_df.head())

    # title_idとmovie_dfにおいて、titleカラムの表記をそろえる(全て小文字に)
    movie_df["title"] = movie_df["title"].apply(lambda x: x.lower())
    title_id["title"] = title_id["title"].apply(lambda x: x.lower())

    # merge
    merged = pd.merge(movie_df, title_id, on="title", how="inner")

    merged = merged.rename(columns={"overview": "description"})
    merged["id"] = merged["id"].astype(np.int32)
    print(merged.head())
    return merged


def make_ratings() -> pd.DataFrame:
    """MovieLensのデータから、評価行列を整形する関数

    Returns:
        pd.DataFrame: _description_
    """
    ratings = pd.read_csv(
        os.path.join(MyConfig.movie_len_dir, "ratings.dat"),
        sep="::",
        engine="python",
        names=["user", "movie", "rating", "timestamp"],
    )
    ratings = ratings[["user", "movie", "rating"]]
    # 型変換
    ratings["user"] = ratings["user"].astype(np.int32)
    ratings["movie"] = ratings["movie"].astype(np.int32)
    ratings["rating"] = ratings["rating"].astype(np.float32)
    print(ratings.head())
    return ratings


def preprocess():
    descriptions = make_descriptions()
    ratings = make_ratings()

    # 欠損値があるレコードを除去
    ratings = ratings.dropna()
    descriptions = descriptions.dropna()

    # re-indexing
    users = ratings["user"].unique()  # user idのユニーク値のList
    user_map = dict(zip(users, range(len(users))))  # user id =通し番号 対応表の作成
    movies = descriptions["id"].unique()  # item idのユニーク値のList
    movie_map = dict(zip(movies, range(len(movies))))  # item id =通し番号対応表の作成

    # user id=>user通し番号、item id=>item通し番号に変換.
    ratings["user"] = ratings["user"].apply(lambda x: user_map.get(x, None))
    ratings["movie"] = ratings["movie"].apply(lambda x: movie_map.get(x, None))
    descriptions["id"] = descriptions["id"].apply(lambda x: movie_map.get(x, None))

    # 欠損値があるレコードを除去
    ratings = ratings.dropna()
    descriptions = descriptions.dropna()

    # export
    ratings.to_csv(os.path.join(MyConfig.data_dir, "ratings.csv"), index=False)
    descriptions.to_csv(os.path.join(MyConfig.data_dir, "descriptions.csv"), index=False)


if __name__ == "__main__":
    preprocess()
