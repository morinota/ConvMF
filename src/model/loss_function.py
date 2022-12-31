from typing import Iterator, List

import torch
import torch.nn as nn
from torch import Tensor

from src.model.cnn_nlp_model import CnnNlpModel


class ConvMFLossFunc(nn.Module):
    def __init__(self, lambda_v: float, lambda_w: float) -> None:
        super().__init__()
        # 初期化処理
        # self.param = ...
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w

    def forward(
        self,
        outputs: Tensor,
        targets: Tensor,
        parameters: Iterator[nn.Parameter],
    ) -> torch.Tensor:
        """
        outputs: 予測結果(ネットワークの出力)
        targets: 正解
        parameters: CNNモデルのパラメータ
        """
        # 右辺第一項(Wに関する二乗誤差関数)
        loss = (self.lambda_v / 2) * ((targets - outputs) ** 2).sum()
        # 右辺第二項(WにとってのL2正則化項)を損失関数に加える
        l2 = torch.tensor(0.0, requires_grad=True)
        for w in parameters:
            l2 = l2 + torch.norm(w) ** 2
        loss = loss + (self.lambda_w / 2) * l2
        return loss


if __name__ == "__main__":
    cnn_nlp_model = CnnNlpModel(
        output_dimension=10,
        vocab_size=100,
        embed_dimension=15,
    )
    loss_function = ConvMFLossFunc(lambda_v=0.01, lambda_w=0.1)

    # 実際の値(item latent vector)
    y_true = torch.Tensor(
        [
            [2.0, 1.3, 0.1],
            [4.0, 1.2, 4.1],
        ]
    )
    # 予測値(NlpCnnModelの出力)
    y_pred = torch.Tensor(
        [
            [1.5, 1.2, 0.4],
            [3.0, 2.0, 3.1],
        ]
    )

    # 実際の値と予測値の差を計算する
    loss = loss_function(y_true, y_pred, cnn_nlp_model.parameters())
    print(loss)
    print(type(loss))
