import numpy as np
import torch
from torch import BoolTensor, Tensor


class TripletValidetor:
    """tripletが有効か無効かを判定する為のクラス"""

    def __init__(self) -> None:
        pass

    def get_valid_mask(self, labels: Tensor) -> Tensor:
        """有効な(valid) triplet(i,j,k)->True, 無効な(invalid) triplet(i,j,k)->Falseとなるような
        Tensor(batch_size*batch_size*batch_size)を作成して返す.
        Return a 3D mask where mask[i, j, k]
            is True iff the triplet (i, j, k) is valid.

        A triplet (i, j, k) is valid if:有効なtripletである条件は以下の2つ:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]

        Parameters
        ----------
        labels : Tensor
            int32 `Tensor` with shape [batch_size]

        return:Tensor
            shape = (batch_size, batch_size, batch_size)
            mask[i, j, k] は $(i,j,k)$ が有効なトリプレットであれば真
        """
        # 条件1:Check that i, j and k are distinct  独立したindicesか否か
        is_not_distinct_matrix = torch.eye(n=labels.size(0)).bool()  # labelsのサイズに応じた単位行列を生成し、bool型にキャスト
        is_distinct_matrix = ~is_not_distinct_matrix  # boolを反転する
        i_not_equal_j = is_distinct_matrix.unsqueeze(dim=2)
        i_not_equal_k = is_distinct_matrix.unsqueeze(dim=1)
        j_not_equal_k = is_distinct_matrix.unsqueeze(dim=0)
        is_distinct_triplet_tensor = i_not_equal_j & i_not_equal_k & j_not_equal_k

        # 条件2: Check if labels[i] == labels[j] and labels[i] != labels[k]
        is_label_equal_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = is_label_equal_matrix.unsqueeze(2)
        i_equal_k = is_label_equal_matrix.unsqueeze(1)
        is_valid_labels_triplet_tensor = i_equal_j & (~i_equal_k)

        return is_distinct_triplet_tensor & is_valid_labels_triplet_tensor

    def get_anchor_positive_mask(self, labels: Tensor) -> Tensor:
        """各要素がboolの2次元のTensorを返す.
        Return a 2D mask where mask[a, p] is True,
        if a and p are distinct and have same label.

        Parameters
        ----------
        labels : Tensor
            with shape [batch_size]

        Returns
        -------
        Tensor
            bool Tensor with shape [batch_size, batch_size]
        """
        # 条件1: iとjがdistinct(独立か)を確認する
        is_not_distinct_matrix = torch.eye(n=labels.size(0)).bool()
        is_distinct_matrix = ~is_not_distinct_matrix  # boolを反転する

        # 条件2: labels[i]とlabels[j]が一致しているか否かを確認する
        is_label_equal_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 条件1と条件2をcombineして返す
        return is_distinct_matrix & is_label_equal_matrix

    def get_anchor_negative_mask(self, labels: Tensor) -> Tensor:
        """各要素がboolの2次元のTensorを返す.
        Return a 2D mask where mask[a, n] is True,
        if a and n have distinct labels.

        Parameters
        ----------
        labels : Tensor
            with shape [batch_size]

        Returns
        -------
        Tensor
            bool Tensor with shape [batch_size, batch_size]
        """
        # 条件1: iとjがdistinct(独立か)を確認する
        is_not_distinct_matrix = torch.eye(n=labels.size(0)).bool()
        is_distinct_matrix = ~is_not_distinct_matrix  # boolを反転する

        # 条件2: labels[i]とlabels[j]が一致していないか否かを確認する
        is_not_label_equal_matrix = labels.unsqueeze(0) != labels.unsqueeze(1)

        # 条件1と条件2をcombineして返す
        return is_distinct_matrix & is_not_label_equal_matrix


def test_get_valid_mask() -> None:
    pass


def test_get_anchor_positive_mask() -> None:
    """get_anchor_positive_maskが想定通りにmaskを生成するか否かのテスト"""
    num_data = 64
    num_classes = 10
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    # expectedをnumpyで簡単に作る.
    mask_expected = np.zeros((num_data, num_data))
    for data_idx_i in range(num_data):
        for data_idx_j in range(num_data):
            is_distinct = data_idx_i != data_idx_j
            is_valid_label = labels[data_idx_i] == labels[data_idx_j]
            mask_expected[data_idx_i, data_idx_j] = is_distinct and is_valid_label
            # ndarrayは数値型のみなので0(False) or 1(True)として入る

    triplet_validator = TripletValidetor()
    mask_actual = triplet_validator.get_anchor_positive_mask(
        labels=Tensor(labels),
    )

    assert torch.equal(mask_actual, Tensor(mask_expected).bool())


def test_get_anchor_negative_mask() -> None:
    """get_anchor_negative_maskが想定通りにmaskを生成するか否かのテスト"""
    num_data = 64
    num_classes = 10
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    # expectedをnumpyで簡単に作る.
    mask_expected = np.zeros((num_data, num_data))
    for data_idx_i in range(num_data):
        for data_idx_j in range(num_data):
            is_distinct = data_idx_i != data_idx_j
            is_valid_label = labels[data_idx_i] != labels[data_idx_j]
            mask_expected[data_idx_i, data_idx_j] = is_distinct and is_valid_label
            # ndarrayは数値型のみなので0(False) or 1(True)として入る

    triplet_validator = TripletValidetor()
    mask_actual = triplet_validator.get_anchor_negative_mask(
        labels=Tensor(labels),
    )

    assert torch.equal(mask_actual, Tensor(mask_expected).bool())
