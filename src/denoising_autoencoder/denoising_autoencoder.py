from typing import Tuple

import torch
from torch import Tensor, nn


def add_noise(
    X_input: Tensor,
    noise_type: str = "masking",
    noise_rate: float = 0.3,
) -> Tensor:
    """入力データにノイズを加える関数
    元論文では、noise_type == "masking", noise_rate: float = 0.3を適用.
    (入力ベクトルの各要素を確率0.3でランダムにマスク（0に設定）する)
    """
    if noise_type not in ["gaussian", "masking"]:
        print(f"[WARN]please set the arg: noise_type is in [gaussian, masking]. So the noises was not added.")
        return X_input
    if noise_type == "gaussian":
        noises_mask = torch.randn(X_input.size()) * noise_rate
        return X_input + noises_mask
    elif noise_type == "masking":
        noises_mask = torch.rand(X_input.size())
        noises_mask[noises_mask < noise_rate] = 0
        noises_mask[noises_mask >= noise_rate] = 1
        return X_input * noises_mask


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()

        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, input_dim)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """出力はembeddeed(encodeされた後のTensor)、decodeされた後のTensorのtuple"""
        X_embedded = self.encoder(X)
        X_output = self.decoder(X_embedded)
        return X_embedded, X_output
