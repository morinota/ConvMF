from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import Tensor

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
from src.denoising_autoencoder.vec_similarity import CategorySimilarityEvaluator
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
        # max_features=1000,
        max_features=10000,
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

    # The dataset is splitted into two: 5000 for training and 5348 for testing.
    train_text_count_vectors, valid_text_count_vectors, train_labels, valid_labels = train_test_split(
        text_count_vectors.toarray(),
        np.array(category_labels.values),
        train_size=5000,  # 5000 for training
        random_state=42,
        stratify=np.array(category_labels.values),
    )

    train_dataloader_category = create_dataloader(
        inputs=np.array(train_text_count_vectors),
        outputs=np.array(train_labels),
        batch_size=32,
    )
    valid_dataloader_category = create_dataloader(
        inputs=np.array(valid_text_count_vectors),S
        outputs=np.array(valid_labels),
        batch_size=32,
    )

    autoencoder = AutoEncoder(input_dim=input_dim, embedding_dim=50)
    optimizer = torch.optim.Adadelta(
        params=autoencoder.parameters(),  # 最適化対象
        lr=0.001,  # parameter更新の学習率
        rho=0.95,  # 移動指数平均の係数(ハイパーパラメータ)
    )
    loss_function = DenoisingAutoEncoderLoss(
        alpha=10,
        margin=10,
        mining_storategy="batch_hard",
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # denoizing autoencoder による学習
    autencoder_trained, valid_loss_list = train(
        model=autoencoder,
        train_dataloader=train_dataloader_category,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=100,
        # valid_dataloader=valid_dataloader_category,
    )
    # torch.save(
    #     obj=autencoder_trained.state_dict(),
    #     f=r"src\denoising_autoencoder\autoencoder.pt",
    # )

    valid_embeddings, _ = autencoder_trained.forward(Tensor(np.array(valid_text_count_vectors)))

    evaluator_obj = CategorySimilarityEvaluator()
    auroc_result_DAE = evaluator_obj.eval_embeddings(
        embeddings=Tensor(text_count_vectors.toarray()),
        labels=Tensor(np.array(category_labels.values)),
        n=10000,
    )
    auroc_result_tfidf = evaluator_obj.eval_embeddings(
        embeddings=Tensor(text_tfidf_vectors.toarray()),
        labels=Tensor(np.array(category_labels.values)),
        n=10000,
    )

    fig, ax = plt.subplots()
    ax = auroc_result_DAE.visualize_roc(
        ax,
        legend_name="embeddings by DAE with triplet loss",
        title_name="ROC of category similarity",
    )
    ax = auroc_result_tfidf.visualize_roc(
        ax,
        legend_name="tf-idf vectors",
    )
    fig.savefig(r"src\denoising_autoencoder\embeddings_with_category_similarity.png")


if __name__ == "__main__":
    main()
