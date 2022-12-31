import numpy as np
import torch
from torch import Tensor

from src.triplet_mining.valid_triplet import TripletValidetor


def test_get_valid_mask() -> None:
    num_data = 64
    num_classes = 10
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    # expectedをnumpyで簡単に作る.
    mask_expected = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                is_distinct = i != j and i != k and j != k
                is_valid_label = labels[i] == labels[j] and labels[i] != labels[k]
                mask_expected[i, j, k] = is_distinct and is_valid_label

    triplet_validator = TripletValidetor()
    mask_actual = triplet_validator.get_valid_mask(
        labels=Tensor(labels),
    )

    assert torch.equal(mask_actual, Tensor(mask_expected).bool())


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
