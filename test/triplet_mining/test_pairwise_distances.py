import torch
from torch import Tensor

from src.triplet_mining.pairwise_distances import calc_pairwise_distances


def test_pairwise_distances() -> None:
    embeddings = Tensor([[1, 2], [3, 4], [5, 6]])  # ベクトル数:3, 次元数:2
    print(embeddings.shape)
    distances_actual = calc_pairwise_distances(embeddings, is_squared=True)
    print(distances_actual)
    distance_expected = Tensor(
        [
            [0.0, 8.0, 32.0],
            [8.0, 0.0, 8.0],
            [32.0, 8.0, 0.0],
        ]  # -> ベクトル数 * ベクトル数のfloat tensor
    )

    assert torch.equal(distances_actual, distance_expected)
