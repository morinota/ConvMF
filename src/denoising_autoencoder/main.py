from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import MyConfig
from src.denoising_autoencoder.count_vector import get_count_vectors, get_tfidf_vectors
from src.denoising_autoencoder.preprocessing import (
    TEXT_COL,
    get_valid_category_label,
    get_valid_story_label,
    read_parquet_articles,
)


def main():
    article_df = read_parquet_articles(MyConfig.uci_news_data_path)

    article_df = article_df.sort_values(by="id")  # article idの昇順でソート

    article_df, label_idx_story_mapping = get_valid_story_label(article_df)

    article_df, label_idx_category_mapping = get_valid_category_label(article_df)

    text_count_vectors, count_vectorizer = get_count_vectors(
        corpus=article_df[TEXT_COL].tolist(),
        # 以下、CountVectorizerのオプション
        min_df=0.01,
        max_df=0.09,
        max_features=1000,
        binary=False,
    )
    print(f"[LOG]text_count_vectors.shape:{text_count_vectors.shape}")
    print(text_count_vectors.nnz)  # number of stored values

    text_tfidf_vectors, tfidf_vectorizer = get_tfidf_vectors(
        count_vectors=text_count_vectors,
    )
    print(f"[LOG]text_tfidf_vectors.shape:{text_tfidf_vectors.shape}")
    print(text_tfidf_vectors.nnz)  # number of stored values

    # denoizing autoencoder による学習(anchorのテキストとラベルを渡せば良い)


if __name__ == "__main__":
    main()
