from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer

from src.config import MyConfig
from src.denoising_autoencoder.count_vector import get_count_vectors, get_tfidf_vectors
from src.denoising_autoencoder.denoising_autoencoder import AutoEncoder, train
from src.denoising_autoencoder.loss_function import DenoisingAutoEncoderLoss
from src.denoising_autoencoder.preprocessing import (
    TEXT_COL,
    get_valid_category_label,
    get_valid_story_label,
    read_parquet_articles,
)
from src.text_cnn_test.utils.dataloader import create_dataloader


def main():
    article_df = read_parquet_articles(MyConfig.uci_news_data_path)

    article_df = article_df.sort_values(by="id")  # article idの昇順でソート

    story_labels, label_idx_story_mapping = get_valid_story_label(article_df)

    category_labels, label_idx_category_mapping = get_valid_category_label(article_df)

    text_count_vectors, count_vectorizer = get_count_vectors(
        corpus=article_df[TEXT_COL].tolist(),
        # 以下、CountVectorizerのオプション
        min_df=0.01,
        max_df=0.09,
        max_features=1000,
        binary=False,
    )
    input_dim = text_count_vectors.shape[1]
    print(f"[LOG]text_count_vectors.shape:{text_count_vectors.shape}")
    print(text_count_vectors.nnz)  # number of stored values

    text_tfidf_vectors, tfidf_vectorizer = get_tfidf_vectors(
        count_vectors=text_count_vectors,
    )
    print(f"[LOG]text_tfidf_vectors.shape:{text_tfidf_vectors.shape}")
    print(text_tfidf_vectors.nnz)  # number of stored values

    # dataloaderを作る
    train_dataloader_category = create_dataloader(
        inputs=text_count_vectors.toarray(),
        outputs=np.array(category_labels.values),
        batch_size=32,
    )

    autoencoder = AutoEncoder(input_dim=input_dim, embedding_dim=50)
    optimizer = torch.optim.Adadelta(
        params=autoencoder.parameters(),  # 最適化対象
        lr=0.00001,  # parameter更新の学習率
        rho=0.95,  # 移動指数平均の係数(? きっとハイパーパラメータ)
    )
    loss_function = DenoisingAutoEncoderLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # denoizing autoencoder による学習
    autencoder_trained = train(
        model=autoencoder,
        train_dataloader=train_dataloader_category,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=1,
    )


if __name__ == "__main__":
    main()
