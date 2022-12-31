import numpy as np
import pytest
from torch import Tensor

from src.triplet_mining.batch_hard_strategy import BatchHardStrategy


def test_batch_hard_strategy() -> None:
    """Test the triplet loss with batch hard triplet mining"""
    num_data = 50
    feat_dim = 6
    margin = 0.2
    num_classes = 5
    squared = False

    embeddings = Tensor(np.random.rand(num_data, feat_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))

    batch_hard_obj = BatchHardStrategy(
        margin=margin,
        squared=squared,
    )
    triplet_loss = batch_hard_obj.calc_triplet_loss(
        labels,
        embeddings,
    )
    print(f"[LOG]triplet_loss:{triplet_loss}")
