from cProfile import label
from operator import mod
from statistics import mode
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.denoising_autoencoder.loss_function import DenoisingAutoEncoderLoss
from src.triplet_mining.batch_hard_strategy import BatchHardStrategy


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
        """モデルを通して入力値を出力値に再構成して返す.
        返り値はembeddeed(encodeされた後のTensor)、decodeされた後のTensorのtuple"""
        X_embedded = self.encoder(X)
        X_output = self.decoder(X_embedded)
        return X_embedded, X_output


def train(
    model: AutoEncoder,
    train_dataloader: DataLoader,
    loss_function: DenoisingAutoEncoderLoss,
    optimizer: torch.optim.Adam,
    device: torch.device,
    epochs: int = 20,
) -> AutoEncoder:
    """Train the AutoEncoder model. 学習を終えたAutoEncoderオブジェクトを返す。"""

    model.to(device)

    for epoch_idx in range(epochs):
        model.train()

        for batch_idx, batch_dataset in enumerate(train_dataloader):
            print(f"=====epoch_idx:{epoch_idx}, batch_idx:{batch_idx}=====")
            input_vectors: Tensor
            labels: Tensor
            input_vectors, labels = tuple(tensors for tensors in batch_dataset)
            labels = labels.type(dtype=torch.LongTensor)
            input_vectors, labels = input_vectors.to(device), labels.to(device)

            input_vectors_noised = add_noise(input_vectors).to(device)

            # 勾配が累積してく仕組みなので,1バッチ毎に勾配の値を初期化しておく.
            model.zero_grad()

            # embedded_vectors, output_vecotrs = model.forward(input_vectors_noised)
            embedded_vectors, output_vecotrs = model(input_vectors_noised)

            loss = loss_function.forward(
                inputs=input_vectors,
                embeddings=embedded_vectors,
                outputs=output_vecotrs,
                labels=labels,
            )
            print(f"the loss: {loss}")
            loss.backward()
            optimizer.step()

    return model
