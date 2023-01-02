import numpy as np
import torch
from torch import Tensor

from src.denoising_autoencoder.denoising_autoencoder import AutoEncoder, add_noise


def test_add_noise() -> None:
    num_data = 10
    input_vec_length = 6

    X_input = Tensor(np.random.rand(num_data, input_vec_length).astype(np.float32))
    X_input_noised = add_noise(X_input=X_input)

    assert not torch.equal(X_input, X_input_noised)


def test_denoising_autoencoder() -> None:
    num_data = 10
    input_vec_length = 6

    X_input = Tensor(np.random.rand(num_data, input_vec_length).astype(np.float32))
    X_input_noised = add_noise(X_input=X_input)

    auto_encoder = AutoEncoder(input_dim=input_vec_length, embedding_dim=3)
    X_embedded, X_output = auto_encoder.forward(X_input_noised)
    print(f"=" * 10)
    print(X_input_noised)
    print(f"=" * 10)
    print(X_embedded)
    print(f"=" * 10)
    print(X_output)
