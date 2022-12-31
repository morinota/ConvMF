from typing import Tuple

import numpy as np
import torch
from pairwise_distances import calc_pairwise_distances
from torch import Tensor
from valid_triplet import TripletValidetor


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
        pairwise_distance_matrix = calc_pairwise_distances(embeddings, squared=self.squared)

        hardest_positive_dists = self._extract_hardest_positives(pairwise_distance_matrix, labels)

        hardest_negative_dists = self._extract_hardest_negatives(pairwise_distance_matrix, labels)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        init_triplet_loss = hardest_positive_dists - hardest_negative_dists + self.margin
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
        """各anchorに対して、hardest positiveを見つける.
        For each anchor, get the hardest positive.
        1. 有効なペア(anchor,positive)の2次元マスクを取得する
        2. 修正(有効なペアのみ考慮)された、距離行列の各行に対する最大距離を取る
        返り値は、Tensor with shape (batch_size, 1)
        """
        is_anchor_positive_matrix = self.triplet_validetor.get_anchor_positive_mask(
            labels,
        )
        is_anchor_positive_matrix_binary = is_anchor_positive_matrix.float()

        pairwise_dist_matrix_masked = torch.mul(
            pairwise_distance_matrix,
            is_anchor_positive_matrix_binary,
        )  # アダマール積(要素毎の積)

        hardest_positive_dists, _ = pairwise_dist_matrix_masked.max(
            dim=1,  # dim番目の軸に沿って最大値を取得
            keepdim=True,  # 2次元Tensorを保つ
        )  # ->Tensor with shape (batch_size, 1)

        return hardest_positive_dists

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
        返り値は、Tensor with shape (batch_size, 1)
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

        hardest_negative_dists, _ = pairwise_dist_matrix_masked.min(dim=1, keepdim=True)

        return hardest_negative_dists
