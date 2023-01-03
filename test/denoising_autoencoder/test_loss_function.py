import numpy as np
from torch import Tensor

from src.denoising_autoencoder.loss_function import DenoisingAutoEncoderLoss


def test_denoising_autoencoder_loss() -> None:
    num_data = 10
    input_dim = 100
    embedding_dim = 6
    margin = 0.2
    num_classes = 4
    is_squared = False

    inputs = Tensor(np.random.rand(num_data, input_dim).astype(np.float32))
    embeddings = Tensor(np.random.rand(num_data, embedding_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))
    outputs = Tensor(np.random.rand(num_data, input_dim).astype(np.float32))

    loss_func = DenoisingAutoEncoderLoss()
    loss = loss_func.forward(
        inputs=inputs,
        embeddings=embeddings,
        outputs=outputs,
        labels=labels,
    )

    print(loss)
