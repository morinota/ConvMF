from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from src.triplet_mining.pairwise_distances import calc_pairwise_distances
from src.triplet_mining.valid_triplet import TripletValidetor


class BatchAllStrategy:
    def __init__(
        self,
        margin: float,
        squared: bool = False,
    ) -> None:
        """
        - margin : float
            margin for triplet loss
        - squared : bool, optional
            If true, output is the pairwise squared euclidean distance matrix.
            If false, output is the pairwise euclidean distance matrix.,
            by default False
        """
        self.margin = margin
        self.squared = squared
        self.triplet_validetor = TripletValidetor()

    def mining(
        self,
        labels: Tensor,
        embeddings: Tensor,
    ) -> Dict[str, Tensor]:
        """損失の計算は行わず、miningしたtripletのDict[str, Tensor(embeddingのindex)]を返す"""
        pairwise_distance_matrix = calc_pairwise_distances(embeddings, is_squared=self.squared)

        valid_triplet_mask = self.triplet_validetor.get_valid_mask(labels)

        anchor_ids, positive_ids, negative_ids = torch.nonzero(valid_triplet_mask).unbind(dim=1)  # Trueの座標を取り出す
        return {
            "anchor_ids": anchor_ids,
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
        }

    def calc_triplet_loss(
        self,
        labels: Tensor,
        embeddings: Tensor,
    ) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.

        Parameters
        ----------
        labels : Tensor
            labels of the batch, of size (batch_size,)
        embeddings : Tensor
            tensor of shape (batch_size, embed_dim)


        Returns
        -------
        Tuple[Tensor, Tensor]
            triplet_loss: scalar tensor containing the triplet loss
            fraction_positive_triplets: scalar tensor containing 有効なtripletに対するpositive(i.e. not easy) tripletsの割合
        """
        pairwise_distance_matrix = calc_pairwise_distances(embeddings, is_squared=self.squared)
        triplet_loss = self._initialize_triplet_loss(pairwise_distance_matrix)

        valid_triplet_mask = self.triplet_validetor.get_valid_mask(labels)

        triplet_loss = self._remove_invalid_triplets(triplet_loss, valid_triplet_mask)

        triplet_loss = self._remove_negative_loss(triplet_loss)

        num_positive_triplets = self._count_positive_triplet(triplet_loss)

        num_valid_triplets = torch.sum(valid_triplet_mask)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        # -> 有効なtripletに対するnot easy tripletsの割合

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss_mean = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss_mean

    def _initialize_triplet_loss(self, pairwise_distance_matrix: Tensor) -> Tensor:
        """triplet_loss(batch_size*batch_size*batch_sizeの形のTensor)の初期値を作る.
        各要素がtriplet_loss(i,j,k),
        一旦、全てのi,j,kの組み合わせでtriplet_lossを計算する
        """
        # 指定されたidxにサイズ1の次元をinsertする
        anchor_positive_dist = pairwise_distance_matrix.unsqueeze(dim=2)
        # -> (batch_size, batch_size, 1)
        anchor_negative_dist = pairwise_distance_matrix.unsqueeze(dim=1)
        # -> (batch_size, 1, batch_size)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        return anchor_positive_dist - anchor_negative_dist + self.margin

    def _remove_invalid_triplets(self, triplet_loss: Tensor, valid_triplet_mask: Tensor) -> Tensor:
        """triplet lossのTensorから、有効なtripletのlossのみ残し、無効なtripletのlossをゼロにする"""
        masks_float = valid_triplet_mask.float()  # True->1.0, False->0.0
        return triplet_loss * masks_float  # アダマール積(要素積)を取る

    def _remove_negative_loss(self, triplet_loss: Tensor) -> Tensor:
        """triplet lossのTensorから、negative(easy) triplet lossをゼロにし、positive(hard)なlossの要素のみ残す.
        negative(easy)なtriplet loss= triplet lossが0未満の要素.
        Remove negative losses (i.e. the easy triplets).
        """
        return torch.max(
            input=triplet_loss,
            other=torch.zeros(size=triplet_loss.shape),
        )

    def _count_positive_triplet(self, triplet_loss: Tensor) -> Tensor:
        """triplet_lossのTensorの中で、positive(i.e. not easy) triplet lossの要素数をカウントして返す
        Count number of positive triplets (where triplet_loss > 0)
        """
        valid_triplets = torch.gt(input=triplet_loss, other=1e-16)
        valid_triplets = valid_triplets.float()  # positive triplet->1.0, negative triplet->0.0
        return torch.sum(valid_triplets)


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

    batch_all_obj = BatchAllStrategy(
        margin=margin,
        squared=is_squared,
    )
    triplet_embeddings_dict = batch_all_obj.mining(
        labels,
        embeddings,
    )
    print(triplet_embeddings_dict)
