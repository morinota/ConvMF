import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def create_data_loaders(
    train_inputs: np.ndarray,
    val_inputs: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int = 50,
):
    """Convert train and validation sets to torch.Tensors and load them to DataLoader.

    Parameters
    ----------
    train_inputs : np.ndarray
        学習用データ(tokenize & encode された文章データ)
    val_inputs : np.ndarray
        検証用データ(tokenize & encode された文章データ)
    train_labels : np.ndarray
        学習用データ(ラベル)
    val_labels : np.ndarray
        検証用データ(ラベル)
    batch_size : int, optional
        バッチサイズ, by default 50

    Returns
    -------
    Tuple[DataLoader]
        学習用と検証用のDataLoaderをそれぞれ返す。
    """
    print(train_labels[:10, :])
    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels = tuple(
        torch.tensor(data) for data in [train_inputs, val_inputs, train_labels, val_labels]
    )

    # Create DataLoader for training data
    # DatasetオブジェクトのInitialize
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    # DataLoaderオブジェクトのInitialize
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    # DatasetオブジェクトのInitialize
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    # DataLoaderオブジェクトのInitialize
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def create_dataloader(inputs: np.ndarray, outputs: np.ndarray, batch_size: int) -> DataLoader:
    # Convert data type to torch.Tensor
    inputs_tensor = torch.tensor(inputs)
    outputs_tensor = torch.tensor(outputs)
    # Initialize Dataset object
    dataset_obj = TensorDataset(inputs_tensor, outputs_tensor)
    sampler_obj = RandomSampler(dataset_obj)
    # Initialize DataLoader object
    return DataLoader(dataset_obj, sampler=sampler_obj, batch_size=batch_size)
