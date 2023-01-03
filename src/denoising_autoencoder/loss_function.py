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
        alpha: float = 0.00001,
        margin: float = 1.0,
        mining_storategy: str = "batch_all",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.margin = margin

        self.mse_loss_func = nn.MSELoss(reduction="mean")
        self.triplet_mining_obj = BatchAllStrategy(self.margin)
        self.triplet_loss_func = nn.TripletMarginLoss(
            margin=self.margin,
            reduction="mean",
        )

    def forward(
        self,
        inputs: Tensor,
        embeddings: Tensor,
        outputs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        triplet_indices_dict = self.triplet_mining_obj.mining(labels, embeddings)

        first_term_loss: Tensor = self.mse_loss_func(inputs, outputs)
        # triplet_embeddings = self._extract_triplet_embeddings(embeddings, is_triplet_tensor)
        triplet_loss = self.triplet_loss_func.forward(
            anchor=embeddings[triplet_indices_dict["anchor_ids"]],
            positive=embeddings[triplet_indices_dict["positive_ids"]],
            negative=embeddings[triplet_indices_dict["negative_ids"]],
        )

        loss = first_term_loss + self.alpha * triplet_loss
        loss = loss.to(torch.float32)
        return loss
