import random
import time
from operator import mod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from src.model.cnn_nlp_model import CnnNlpModel

# Specify loss function
loss_fn = nn.MSELoss()


def set_seed(seed_value: int = 42) -> None:
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
) -> CnnNlpModel:
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
        nn.Module
    """
    # modelをdeviceに渡す
    model.to(device)

    # Tracking best validation accuracy
    best_accuracy = 0

    print("Start training...\n")
    print("-" * 60)

    # エポック毎に繰り返し
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        # バッチ学習
        for step_idx, batch_dataset in enumerate(train_dataloader):
            # inputデータとoutputデータを分割
            batch_input_ids, batch_outputs = tuple(tensors for tensors in batch_dataset)

            # ラベル側をキャストする
            # batch_outputs: Tensor = batch_outputs.type(torch.LongTensor)
            # データをGPUにわたす。
            batch_input_ids: Tensor = batch_input_ids.to(device)
            batch_outputs: Tensor = batch_outputs.to(device)

            # Zero out any previously calculated gradients
            # 1バッチ毎に勾配の値を初期化(累積してく仕組みだから...)
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            # モデルにinputデータを入力して、出力値を得る。
            output_pred = model(batch_input_ids)
            # Compute loss and accumulate the loss values
            # 損失関数の値を計算
            loss = loss_fn(output_pred, batch_outputs)

            # 1 epoch全体の損失関数の値を評価する為に、1 batch毎の値を累積していく.
            total_loss += loss.item()

            # Update parameters(パラメータを更新)
            loss.backward()  # 誤差逆伝播で勾配を取得
            optimizer.step()  # 勾配を使ってパラメータ更新

        # Calculate the average loss over the entire training data
        # 1 epoch全体の損失関数の平均値を計算
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        # 1 epochの学習が終わる毎に、検証用データを使って汎化性能評価。
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(model=model, val_dataloader=val_dataloader, device=device)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"the validation result of epoch {epoch_i + 1:^7} is below.")
            print(f"the values of loss function : train(average)={avg_train_loss:.6f}, valid={val_loss:.6f}")
            print(f"accuracy of valid data: {val_accuracy:.2f}, time: {time_elapsed:.2f}")

        print("-" * 20)

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    # 学習済みのモデルを返す
    return model


def evaluate(model: nn.Module, val_dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray]:
    """各epochの学習が完了した後、検証用データを使ってモデルの汎化性能を評価する。
    After the completion of each training epoch, measure the model's
    performance on our validation set.

    Parameters
    ----------
    model : nn.Module
        CNN_NLPオブジェクト。
    val_dataloader : DataLoader
        検証用のDataLoader
    device : torch.device
        'cuda' or 'cpu'

    Returns
    -------
    Tuple[np.ndarray]
        検証用データセットに対する、モデルの損失関数とAccuracyの値。
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        b_input_ids, b_labels = tuple(t for t in batch)
        # ラベル側を浮動小数点型から整数型にキャストする(分類問題の場合)
        b_labels: Tensor = b_labels.type(torch.LongTensor)
        # Load batch to GPU
        b_input_ids: Tensor = b_input_ids.to(device)
        b_labels: Tensor = b_labels.to(device)

        # モデルにinputデータを入力して、出力値を得る。
        with torch.no_grad():
            output_pred = model(b_input_ids)

        # Compute loss
        # 損失関数の値を計算
        loss: Tensor = loss_fn(output_pred, b_labels)
        # 得られたbacth毎の損失関数の値を保存
        val_loss.append(loss.item())

        # Get the predictions
        # 分類問題の予測結果を取得
        preds = torch.argmax(output_pred, dim=1).flatten()

        # Calculate the accuracy rate(正解率)
        preds: Tensor
        b_labels: Tensor
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy
