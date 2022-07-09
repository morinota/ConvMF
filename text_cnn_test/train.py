

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
loss_fn = nn.CrossEntropyLoss()

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
