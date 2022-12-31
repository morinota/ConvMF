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
    triu = np.triu_indices(embeddings.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(embeddings[triu[1]] - embeddings[triu[0]], axis=1)
    if is_squared:
        upper_tri_pdists **= 2.0
    num_data = embeddings.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(pairwise_distances.diagonal())
    return pairwise_distances


def test_batch_all_strategy() -> None:
    """Test the triplet loss with batch all triplet mining in a simple case.
    There is just 1 class in this super simple edge case, and we want to make sure that
    the loss is 0.
    """
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1
    squared = False

    embeddings = Tensor(np.random.rand(num_data, feat_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))
    batch_all_obj = BatchAllStrategy(
        margin=margin,
        squared=squared,
    )
    triplet_loss, fraction = batch_all_obj.calc_triplet_loss(
        labels,
        embeddings,
    )
    triplet_loss_expected, fraction_expected = 0.0, 0.0

    print(f"[LOG]triplet_loss:{triplet_loss}")
    assert triplet_loss.item() == triplet_loss_expected
    assert fraction.item() == fraction_expected
