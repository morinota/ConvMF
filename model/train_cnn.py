
from typing import List, Iterator
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import time
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)

# Specify loss function
# loss_fn = nn.CrossEntropyLoss()

class CustomLoss(nn.Module):
    def __init__(self, lambda_v:float, lambda_w:float) -> None:
        super().__init__()
        # 初期化処理
        # self.param = ... 
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w

    def forward(self, outputs:Tensor, targets:Tensor, parameters:Iterator[nn.Parameter]):
        '''
        outputs: 予測結果(ネットワークの出力)
　　　　 targets: 正解
        '''
        # ロスの計算を何かしら書く
        # loss = torch.mean(outputs - targets)
        loss = (self.lambda_v/2) * ((targets - outputs)**2).sum()
        # L2ノルムの２乗を損失関数に足す.
        l2 = torch.tensor(0., requires_grad=True)
        for w in parameters:
            l2 = l2 + torch.norm(w)**2
        loss = loss + (self.lambda_w/2) * l2
        # ロスの計算を返す
        return loss

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model: nn.Module, optimizer: optim.Adadelta, device: torch.device,
          train_dataloader: DataLoader, val_dataloader: DataLoader = None,
          epochs: int = 10
          ):
    """Train the CNN model."""
    
    # 損失関数の定義
    loss_fn = CustomLoss(lambda_v=0.01, lambda_w=0.1)


    # Tracking best validation accuracy
    best_accuracy = 0
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

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
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t for t in batch)

            # ラベル側をキャストする(そのままだと何故かエラーが出るから)
            b_labels: Tensor = b_labels.type(torch.LongTensor)
            # Load batch to GPU
            b_input_ids: Tensor = b_input_ids.to(device)
            b_labels: Tensor = b_labels.to(device)

            # Zero out any previously calculated gradients
            model.zero_grad()  # 勾配の値を初期化(累積してく仕組みだから...)

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)
            # Compute loss and accumulate the loss values
            loss = loss_fn(input=logits, target=b_labels)
            model.parameters()

            # 1エポックのloss関数の値を保存するために追加
            total_loss += loss.item()

            # Update parameters(パラメータを更新)
            loss.backward()
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        # 1 epochの学習が終わる毎にEvaluation
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(
                model=model, val_dataloader=val_dataloader, device=device)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    # 学習したモデルを返す
    return model


def evaluate(model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # 損失関数の定義
    loss_fn = CustomLoss(lambda_v=0.01, lambda_w=0.1)

    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        b_input_ids, b_labels = tuple(t for t in batch)
        # ラベル側をキャストする(そのままだと何故かエラーが出るから)
        b_labels: Tensor = b_labels.type(torch.LongTensor)
        # Load batch to GPU
        b_input_ids: Tensor = b_input_ids.to(device)
        b_labels: Tensor = b_labels.to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss: Tensor = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        preds: Tensor
        b_labels: Tensor
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy
