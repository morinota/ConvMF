from typing import Tuple

import numpy as np
import pytest
from torch import Tensor

from src.triplet_mining.batch_all_strategy import BatchAllStrategy


def calc_pairwise_distance_np(embeddings: np.ndarray, is_squared: bool = False) -> np.ndarray:
    """Computes the pairwise distance matrix in numpy.
    Args:
        embeddings: 2-D numpy array of size [number of data, feature dimension]
        is_squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(
        n=embeddings.shape[0],  # data num
        k=1,
    )
    upper_tri_pdists = np.linalg.norm(embeddings[triu[1]] - embeddings[triu[0]], axis=1)
    if is_squared:
        upper_tri_pdists **= 2.0
    num_data = embeddings.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(pairwise_distances.diagonal())
    return pairwise_distances


def calc_triplet_loss_with_batch_all_strategy_by_numpy(
    num_data: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    margin: float,
    is_squared: bool,
) -> Tuple[float, float]:
    """triplet_loss_expectedとfraction_expectedをnumpyで作成する関数."""
    pdist_matrix = calc_pairwise_distance_np(embeddings, is_squared=is_squared)

    triplet_loss_expected = 0.0
    num_positives_expected = 0.0
    num_valid_expected = 0.0
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                is_distinct = i != j and i != k and j != k
                is_valid = (labels[i] == labels[j]) and (labels[i] != labels[k])

                if not is_distinct or not is_valid:
                    continue  # 無効なtripletのケース

                num_valid_expected += 1.0
                a_p_distance = pdist_matrix[i][j]
                a_n_distance = pdist_matrix[i][k]

                one_triplet_loss = np.maximum(
                    0.0,
                    a_p_distance - a_n_distance + margin,
                )
                triplet_loss_expected += one_triplet_loss

                if one_triplet_loss > 0:
                    num_positives_expected += 1  # not easyなtripletの数をカウント

    triplet_loss_expected = triplet_loss_expected / (num_positives_expected + 1e-16)
    fraction_expected = num_positives_expected / (num_valid_expected + 1e-16)

    return triplet_loss_expected, fraction_expected


def test_batch_all_strategy_with_only_one_class() -> None:
    """Test the triplet loss with batch all triplet mining in a simple case.
    There is just 1 class in this super simple edge case, and we want to make sure that
    the loss is 0.
    """
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1
    is_squared = False

    embeddings = Tensor(np.random.rand(num_data, feat_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))

    triplet_loss_expected, fraction_expected = calc_triplet_loss_with_batch_all_strategy_by_numpy(
        num_data,
        embeddings.numpy(),
        labels.numpy(),
        margin,
        is_squared,
    )

    batch_all_obj = BatchAllStrategy(
        margin=margin,
        squared=is_squared,
    )
    triplet_loss, fraction = batch_all_obj.calc_triplet_loss(
        labels,
        embeddings,
    )

    assert np.allclose(triplet_loss.item(), triplet_loss_expected)
    assert np.allclose(fraction.item(), fraction_expected)


def test_batch_all_strategy_with_some_classes() -> None:
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5
    is_squared = False

    embeddings = Tensor(np.random.rand(num_data, feat_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))

    triplet_loss_expected, fraction_expected = calc_triplet_loss_with_batch_all_strategy_by_numpy(
        num_data,
        embeddings.numpy(),
        labels.numpy(),
        margin,
        is_squared,
    )

    batch_all_obj = BatchAllStrategy(
        margin=margin,
        squared=is_squared,
    )
    triplet_loss, fraction = batch_all_obj.calc_triplet_loss(
        labels,
        embeddings,
    )
    # np.allclose()で、近似的に等しいか判定.
    assert np.allclose(triplet_loss.item(), triplet_loss_expected)
    assert np.allclose(fraction.item(), fraction_expected)
