from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from src.dataclasses.item_description import ItemDescription
from src.dataclasses.rating_data import RatingLog
from src.model.cnn_nlp_model import CnnNlpModel, initilize_cnn_nlp_model
from src.model.matrix_factorization import MatrixFactrization
from src.text_cnn_test.cnn_nlp_model.train_nlp_cnn import train
from src.text_cnn_test.utils.dataloader import create_dataloader
from src.utils.word_vector_preparer import WordEmbeddingVector


class ConvMF(object):
    def __init__(
        self,
        rating_logs: List[RatingLog],
        item_descriptions: List[ItemDescription],
        embedding_vectors: WordEmbeddingVector,
        num_filters: List[int] = [100, 100, 100],
        learning_rate: float = 0.01,
        filter_sizes: List[int] = [3, 4, 5],  # 窓関数の設定,
        n_factor: int = 300,
        dropout_ratio: float = 0.5,
        user_lambda: float = 10.0,
        item_lambda: float = 100.0,
    ) -> None:
        # check the device (GPU|CPU)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.rating_logs = rating_logs
        self.item_descriptions = item_descriptions
        self.embedding_vectors = embedding_vectors

        self.n_factor = n_factor
        self.n_item = len(item_descriptions)

        # model architecture
        self.mf_obj = MatrixFactrization(
            rating_logs=rating_logs,
            n_factor=self.n_factor,
            n_item=self.n_item,  # 登録されているアイテム数を追加しておく(rating_logsに含まれてない可能性がある)
            user_lambda=user_lambda,
            item_lambda=item_lambda,
        )
        self.cnn_nlp_obj, self.optimizer = initilize_cnn_nlp_model(
            pretrained_embedding=self.embedding_vectors.to_tensor(),
            freeze_embedding=True,  # fastText で事前学習された単語ベクトルが使われ、学習中は凍結される。
            dropout=dropout_ratio,
            device=self.device,
            output_dimension=self.mf_obj.n_factor,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            learning_rate=learning_rate,
        )

    def __call__(self, x: Tensor, y=None, train=True) -> Tensor:
        if train:
            loss = self.cnn_nlp_obj(x, y)
            return loss
        else:
            return self.cnn_nlp_obj(x)

    def fit(self, batch_size: int, n_epoch_convmf: int) -> None:
        """ConvMFの学習を実行する"""
        token_indices_array = ItemDescription.merge_token_indices_of_descriptions(
            self.item_descriptions,
        )

        self.mf_obj.fit(
            n_trial=5,
            document_vectors=None,
        )  # まずmfを一回学習させる
        # ALS実行
        for epoch_idx in range(n_epoch_convmf):
            print(f"[LOG] epoch: {epoch_idx+1}/{n_epoch_convmf}")

            train_dataloader = create_dataloader(
                inputs=token_indices_array,
                outputs=self.mf_obj.item_latent_factor,
                batch_size=batch_size,
            )
            self.cnn_nlp_obj = train(
                model=self.cnn_nlp_obj,
                optimizer=self.optimizer,
                train_dataloader=train_dataloader,
                epochs=20,
                device=self.device,
            )
            document_vectors = self.cnn_nlp_obj.predict(
                token_indices_arrays=[item_d.token_indices for item_d in self.item_descriptions],
            )
            self.mf_obj.fit(
                n_trial=5,
                document_vectors=document_vectors,
            )

        return

    def predict(self, user_ids: List[int], item_ids: List[int]) -> List[np.ndarray]:
        """指定されたitem_id, user_idの類似度(ベクトルの内積)を返す"""
        # item factor Vを取得
        # (item descriptionを使う。新アイテムのベクトルも算出できるので)
        # といいつつ、item_idsを指定してpredictする場合には、
        # token_indicesがself.item_descriptionsに登録されてる必要がある。
        item_vectors = self.get_document_latent_vectors(item_ids)
        # user factor Uを所得
        user_vectors = self.get_user_latent_vectors(user_ids)
        # r_hatを計算
        predictions = []

        return predictions

    def get_document_latent_vectors(self, item_ids: List[int]) -> Dict[int, np.ndarray]:
        """item_id: document_latent_vectorのmapを返す"""
        token_indices_arrays = [self.item_descriptions[id].token_indices for id in item_ids]
        item_factors = self.cnn_nlp_obj.predict(token_indices_arrays)
        return {item_id: item_factors[idx] for idx, item_id in enumerate(item_ids)}

    def get_user_latent_vectors(self, user_ids: List[int]) -> Dict[int, np.ndarray]:
        """user_id: user_latent_vectorのmapを返す"""
        return self.mf_obj.get_user_latent_vectors(user_ids)

    def get_item_latent_vectors(self, item_ids: List[int]) -> Dict[int, np.ndarray]:
        """item_id: item_latent_vectorのmapを返す"""
        return self.mf_obj.get_user_latent_vectors(item_ids)
