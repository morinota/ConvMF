import torch
from torch import Tensor


def calc_pairwise_distances(embeddings: Tensor, is_squared: bool = False) -> Tensor:
    """compute distances between all the embeddings.

    Parameters
    ----------
    embeddings : Tensor
        tensor of shape (batch_size, embed_dim)
    is_squared : bool, optional
        If true, output is the pairwise squared euclidean distance matrix.
        If false, output is the pairwise euclidean distance matrix.,
        by default False

    Returns
    -------
    Tensor
        pairwise_distances: tensor of shape (batch_size, batch_size)
        行列の各要素に、2つのembedding vector間の距離が入っている.
    """
    dot_product_matrix = torch.matmul(
        input=embeddings,
        other=embeddings.t(),
    )  # ->各ベクトル間の内積を要素とした行列
    squared_embedding_norms = dot_product_matrix.diag().unsqueeze(dim=1)  # 対角要素(=各ベクトルの長さの二乗)を取り出す

    # euclidean distance(p, q) = \sqrt{|p|^2 + |q|^2 - 2 p*q}
    euclidean_distances = squared_embedding_norms + squared_embedding_norms.t() - 2 * dot_product_matrix  # ユークリッド距離を算出

    if not is_squared:
        return torch.sqrt(euclidean_distances)

    return euclidean_distances
