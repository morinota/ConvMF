from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from src.triplet_mining.pairwise_distances import calc_pairwise_distances
from src.triplet_mining.valid_triplet import TripletValidetor


class BatchHardStrategy:
    def __init__(
        self,
        margin: float,
        squared: bool = False,
    ) -> None:
        self.margin = margin
        self.squared = squared
        self.triplet_validetor = TripletValidetor()

    def calc_triplet_loss(
        self,
        labels: Tensor,
        embeddings: Tensor,
    ) -> Tensor:
        """Build the triplet loss over a batch of embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_distance_matrix = calc_pairwise_distances(embeddings, is_squared=self.squared)

        hardest_positive_idxs = self._extract_hardest_positives(pairwise_distance_matrix, labels)

        hardest_negative_idxs = self._extract_hardest_negatives(pairwise_distance_matrix, labels)

        init_triplet_loss = (
            pairwise_distance_matrix[hardest_positive_idxs]
            - pairwise_distance_matrix[hardest_negative_idxs]
            + self.margin
        )

        triplet_loss = torch.max(
            input=init_triplet_loss,
            other=torch.zeros(size=init_triplet_loss.shape),
        )  # easy tripletを取り除く.

        # Get final mean triplet loss
        triplet_loss_mean = torch.mean(triplet_loss)
        return triplet_loss_mean

    def _extract_hardest_positives(
        self,
        pairwise_distance_matrix: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """各anchorに対して、hardest positiveを見つける
        For each anchor, get the hardest positive.
        1. 有効なペア(anchor,positive)の2次元マスクを取得する
        2. 有効なペアのみ考慮された、距離行列の各行に対する最大距離を取る
        返り値は、各行(anchor)に対するhardest positiveのindex
            Tensor with shape (batch_size, 1)
        """
        is_anchor_positive_matrix = self.triplet_validetor.get_anchor_positive_mask(
            labels,
        )
        is_anchor_positive_matrix_binary = is_anchor_positive_matrix.float()  # boolを0 or 1に

        pairwise_dist_matrix_masked = torch.mul(
            pairwise_distance_matrix,
            is_anchor_positive_matrix_binary,
        )  # アダマール積(要素毎の積)

        return pairwise_dist_matrix_masked.argmax(dim=1)

    def _extract_hardest_negatives(
        self,
        pairwise_distance_matrix: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """各anchorに対して、hardest negativeを見つける.
        For each anchor, get the hardest negative.
        1. 有効なペア(anchor, negative)の2次元マスクを取得する.
        2. 無効なペアを考慮から取り除く為に、無効なペアのdistanceに各行の最大値を足す.
        3. 距離行列の各行に対する最小距離を取る
        返り値は、各行(anchor)に対するhardest negativeのindex
            Tensor with shape (batch_size, 1)
        """
        is_anchor_negative_matrix = self.triplet_validetor.get_anchor_negative_mask(
            labels,
        )
        is_anchor_negative_matrix_binary = is_anchor_negative_matrix.float()

        max_dist_each_rows, _ = pairwise_distance_matrix.max(
            dim=1,
            keepdim=True,
        )  # 各行の最大値を取得
        pairwise_dist_matrix_masked = pairwise_distance_matrix + (
            max_dist_each_rows * (1.0 - is_anchor_negative_matrix_binary)
        )  # is_anchor_negative=Falseの要素にmax_distを足す

        # hardest_negative_dists, _ = pairwise_dist_matrix_masked.min(dim=1, keepdim=True)

        return pairwise_dist_matrix_masked.argmin(dim=1)

    def mining(
        self,
        labels: Tensor,
        embeddings: Tensor,
    ) -> Dict[str, Tensor]:
        """損失の計算は行わず、miningしたtripletのDict[str, Tensor(embeddingのindex)]を返す"""
        pairwise_distance_matrix = calc_pairwise_distances(embeddings, is_squared=self.squared)

        anchor_idxs = torch.arange(len(labels))  # build-inのarange()みたいな

        hardest_positive_idxs = self._extract_hardest_positives(pairwise_distance_matrix, labels)

        hardest_negative_idxs = self._extract_hardest_negatives(pairwise_distance_matrix, labels)

        return {
            "anchor_ids": anchor_idxs,
            "positive_ids": hardest_positive_idxs,
            "negative_ids": hardest_negative_idxs,
        }


if __name__ == "__main__":
    num_data = 5
    feat_dim = 6
    margin = 0.2
    num_classes = 5
    is_squared = False

    embeddings = Tensor(np.random.rand(num_data, feat_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))

    print(embeddings)
    print(labels)

    batch_hard_obj = BatchHardStrategy(
        margin=margin,
        squared=is_squared,
    )
    triplet_embeddings_dict = batch_hard_obj.mining(
        labels,
        embeddings,
    )
    print(triplet_embeddings_dict)
