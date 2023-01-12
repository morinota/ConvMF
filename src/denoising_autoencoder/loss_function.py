from ast import Raise
from distutils.log import error
from turtle import forward
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.triplet_mining.batch_all_strategy import BatchAllStrategy
from src.triplet_mining.batch_hard_strategy import BatchHardStrategy


class DenoisingAutoEncoderLoss(nn.Module):
    MINING_STORATEGIES = ["batch_all", "batch_hard"]

    def __init__(
        self,
        alpha: float = 10.0,
        margin: float = 1.0,
        mining_storategy: str = "batch_all",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.margin = margin

        self.mse_loss_func = nn.MSELoss(reduction="sum")

        if mining_storategy not in self.MINING_STORATEGIES:
            raise ValueError(
                "Unexpected storategy name is inputted. Please choose mining_storategy in [batch_all, batch_hard]"
            )
        self.triplet_mining_obj = (
            BatchAllStrategy(self.margin) if mining_storategy == "batch_all" else BatchHardStrategy(self.margin)
        )

        self.triplet_loss_func = nn.TripletMarginLoss(
            margin=self.margin,
            reduction="sum",
        )

    def forward(
        self,
        inputs: Tensor,
        embeddings: Tensor,
        outputs: Tensor,
        labels: Tensor,
    ) -> Tensor:

        triplet_indices_dict = self.triplet_mining_obj.mining(labels, embeddings)

        squared_error_term = 0
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["anchor_ids"]],
            outputs[triplet_indices_dict["anchor_ids"]],
        )
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["positive_ids"]],
            outputs[triplet_indices_dict["positive_ids"]],
        )
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["negative_ids"]],
            outputs[triplet_indices_dict["negative_ids"]],
        )

        triplet_loss = self.triplet_loss_func.forward(
            anchor=embeddings[triplet_indices_dict["anchor_ids"]],
            positive=embeddings[triplet_indices_dict["positive_ids"]],
            negative=embeddings[triplet_indices_dict["negative_ids"]],
        )
        print(f"squared_error:{squared_error_term}, triplet_loss:{triplet_loss}")
        loss = squared_error_term + self.alpha * triplet_loss
        loss = loss.to(torch.float32)
        return loss
