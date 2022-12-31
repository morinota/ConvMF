import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.model.cnn_nlp_model import CnnNlpModel
from src.model.loss_function import ConvMFLossFunc


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(
    model: CnnNlpModel,
    optimizer: optim.Adadelta,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    epochs: int = 10,
):
    """Train the CNN_NLP model. 学習を終えたCNN_NLPオブジェクトを返す。

    Parameters
    ----------
    model : nn.Module
        CNN_NLPオブジェクト。
    optimizer : optim.Adadelta
        Optimizer
    device : torch.device
        'cuda' or 'cpu'
    train_dataloader : DataLoader
        学習用のDataLoader
    val_dataloader : DataLoader, optional
        検証用のDataLoader, by default None
    epochs : int, optional
        epoch数, by default 10

    Returns
    -------
    学習を終えたCNN_NLPオブジェクト
        CnnNlpModel
    """
    model.to(device)  # modelをdeviceに渡す

    # 損失関数の定義ConvMFLossFunc
    loss_function = ConvMFLossFunc(lambda_v=0.01, lambda_w=0.1)

    # Tracking best validation accuracy
    print("Start training...\n")

    for epoch_idx in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        # バッチ学習
        for batch_idx, batch_dataset in enumerate(train_dataloader):
            batch_X, batch_y = tuple(tensors for tensors in batch_dataset)

            # データをGPUにわたす。
            batch_X: Tensor = batch_X.to(device)
            batch_y: Tensor = batch_y.to(device)

            # 1バッチ毎に勾配の値を初期化(累積してく仕組みだから...)
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            y_predicted = model(batch_X)
            # 損失関数の値を計算
            loss = loss_function(y_predicted, batch_y, parameters=model.parameters())

            # 1 epoch全体の損失関数の値を評価する為に、1 batch毎の値を累積していく.
            total_loss += loss.item()

            # Update parameters(パラメータを更新)
            loss.backward()  # 誤差逆伝播で勾配を取得
            optimizer.step()  # 勾配を使ってパラメータ更新

        # 1 epoch全体の損失関数の平均値を計算
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        # 1 epochの学習が終わる毎にEvaluation
        if val_dataloader is None:
            continue

        # After the completion of each training epoch, measure the model's
        # performance on our validation set.
        val_loss = _evaluate(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
        )

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        print(f"the validation result of epoch {epoch_idx + 1:^7} is below.")
        print(f"the values of loss function : train(average)={avg_train_loss:.6f}, valid={val_loss:.6f}")

    print("\n")
    print(f"Training complete!")

    return model  # 学習済みのモデルを返す


def _evaluate(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> float:
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # 損失関数の定義
    loss_fn = ConvMFLossFunc(lambda_v=0.01, lambda_w=0.1)

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_loss_list = []

    # For each batch in our validation set...
    for batch_datasets in val_dataloader:
        batch_X, batch_y = tuple(tensors for tensors in batch_datasets)
        # Load batch to GPU
        batch_X: Tensor = batch_X.to(device)
        batch_y: Tensor = batch_y.to(device)

        # Compute logits
        with torch.no_grad():
            y_predicted = model(batch_X)

        # Compute loss
        loss: Tensor = loss_fn(y_predicted, batch_y, model.parameters())
        val_loss_list.append(loss.item())

    # Compute the average accuracy and loss over the validation set.
    val_loss_mean = np.mean(val_loss_list)

    return val_loss_mean


if __name__ == "__main__":
    # 入力データ
    x = torch.Tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 1, 3, 4, 5, 0],
        ]
    ).long()  # LongTensor型に変換する(default はFloatTensor?)
    y_true = torch.Tensor(
        [
            [0.5, 1.0, 2.0, 1.5, 1.8, 1.9, 1.0],
            [0.8, 1.5, 2.1, 1.0, 1.0, 1.2, 1.8],
        ]
    ).float()
    dataset = TensorDataset(x, y_true)
    train_dataloader = DataLoader(dataset)
    valid_dataloader = DataLoader(TensorDataset(x, y_true))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cnn_nlp_model = CnnNlpModel(
        output_dimension=y_true.shape[1],
        vocab_size=100,
        embed_dimension=15,
    )
    optimizer = optim.Adadelta(
        params=cnn_nlp_model.parameters(),  # 最適化対象
        lr=0.01,  # parameter更新の学習率
        rho=0.95,  # 移動指数平均の係数
    )

    cnn_nlp_model_trained = train(
        model=cnn_nlp_model,
        optimizer=optimizer,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
    )
