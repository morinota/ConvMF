import os


class MyConfig:
    data_dir = r"data"
    # movie_lensの元データ
    movie_len_dir = os.path.join(data_dir, "ml-10M100K")
    # movie_len_dir = os.path.join(data_dir, 'ml-1m')
    tmdb_movies_path = os.path.join(data_dir, "tmdb_5000_movies.csv")
    tmdb_credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")

    # 整形後のRating matrixとDescripiton documents
    ratings_path = os.path.join(data_dir, "ratings.csv")
    descriptions_path = os.path.join(data_dir, "descriptions.csv")

    fast_text_path = os.path.join(data_dir, r"fastText\crawl-300d-2M.vec")

    # UCI news aggregator dataset
    uci_news_data_path = os.path.join(data_dir, "uci_news/uci_news.snappy.parquet")
