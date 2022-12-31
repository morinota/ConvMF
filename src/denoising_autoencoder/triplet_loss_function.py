from turtle import forward

import torch
import torch.nn as nn
from torch import Tensor


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, h_anchor: Tensor, h_positive: Tensor, h_negative: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        h_anchor : Tensor
            anchorの埋め込みベクトル
        h_positive : Tensor
            positiveの埋め込みベクトル
        h_negative : Tensor
            negativeの埋め込みベクトル

        Returns
        -------
        Tensor
            _description_
        """
        distance_positive = (h_anchor - h_positive).pow(exponent=2).sum(dim=1)
        distance_negative = (h_anchor - h_negative).pow(exponent=2).sum(dim=1)
        loss = torch.clamp(
            input=distance_positive - distance_negative + self.margin,
            min=0.0,
        )
        return loss.mean()


if __name__ == "__main__":

    triplet_loss_function = TripletLoss(margin=1.0)

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
    loss = triplet_loss_function(y_true, y_pred)
    print(loss)
    print(type(loss))
