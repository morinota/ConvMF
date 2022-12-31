import torch
from torch import Tensor


def calc_pairwise_distances(embeddings: Tensor, squared: bool = False) -> Tensor:
    """compute distances between all the embeddings.

    Parameters
    ----------
    embeddings : Tensor
        tensor of shape (batch_size, embed_dim)
    squared : bool, optional
        If true, output is the pairwise squared euclidean distance matrix.
        If false, output is the pairwise euclidean distance matrix.,
        by default False

    Returns
    -------
    Tensor
        pairwise_distances: tensor of shape (batch_size, batch_size)
        行列の各要素に、2つのembedding vector間の距離が入っている.
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(
        input=embeddings,
        other=embeddings.t(),
    )  # ->各ベクトル間の内積を要素とした行列
    squared_norm = dot_product.diag().unsqueeze(dim=1)  # 対角要素(=各ベクトルの長さの二乗)を取り出す

    # euclidean distance(p, q) = \sqrt{|p|^2 + |q|^2 - 2 p*q}
    euclidean_distances = squared_norm + squared_norm.t() - 2 * dot_product  # ユークリッド距離を算出

    if not squared:
        return torch.sqrt(euclidean_distances)

    return euclidean_distances
