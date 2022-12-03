import os


class MyConfig:
    data_dir = r"data"
    # 元データ
    movie_len_dir = os.path.join(data_dir, "ml-10M100K")
    # movie_len_dir = os.path.join(data_dir, 'ml-1m')
    tmdb_movies_path = os.path.join(data_dir, "tmdb_5000_movies.csv")
    tmdb_credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")

    # 整形後のRating matrixとDescripiton documents
    ratings_path = os.path.join(data_dir, "ratings.csv")
    descriptions_path = os.path.join(data_dir, "descriptions.csv")
