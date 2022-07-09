from modulefinder import Module
from typing import List, Iterator

from torch import Tensor
import torch.optim as optim
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CNN_NLP(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self, pretrained_embedding: torch.Tensor = None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5
                 ) -> None:
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """
        super(CNN_NLP, self).__init__()

        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=freeze_embedding
            )
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0
                                          )

        # Conv Network
        modules = []
        for i in range(len(filter_sizes)):
            conv_layer = nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=num_filters[i],
                kernel_size=filter_sizes[i]
            )
            modules.append(conv_layer)
        self.conv1d_list = nn.ModuleList(modules=modules)

        # Fully-connected layer and Dropout 全結層
        # 入力d_f(次元数はR^{n_c}), 出力はs_j(次元数はR^{n_factor})
        # バイアスあり。活性化関数は2回ともtanh.
        f = num_filters * 2
        n_factor = 300
        self.fc1 = nn.Linear(in_features=np.sum(
            num_filters), out_features=f)
        self.fc2 = nn.Linear(in_features=f, out_features=n_factor)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        # ここでbは、input_idsの長さ=itemの数
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # # Permute `x_embed` to match input shape requirement of `nn.Conv1d`. shapeをcnnの入力用に整形する。
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        # ここで_L_outは各c_iの次元数?(各アイテムの文書の長さに依存?)
        x_conv_list = [F.relu(conv1d(x_reshaped))
                       for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        # 各c_iの代表値を取得。
        x_pool_list:List[Tensor] = [F.max_pool1d(
            x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer(全結合層).
        # Output shape: (b, n_factor)
        x_fc1_list:List[Tensor] = [F.tanh(self.fc1(x_pool)) for x_pool in x_pool_list]
        x_fc2_list = [F.tanh(self.fc2(x_fc1)) for x_fc1 in x_fc1_list]
        # x_fc:Tensor = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        x_fc2 = torch.cat(tensors=x_fc2_list, dim=1)
        return x_fc2


"""To train Deep Learning models, we need to define a loss function 
and minimize this loss. 
We’ll use back-propagation to compute gradients and use an optimization algorithm (ie. Gradient Descent) to minimize the loss. The original paper used the Adadelta optimizer.Deep Learningのモデルを学習させるためには、損失関数を定義し、この損失を最小化する必要があります。バックプロパゲーションを用いて勾配を計算し、最適化アルゴリズム（gradient Descentなど）を用いて損失を最小化することになります。元の論文ではAdadeltaオプティマイザを使用しています。
"""


def initilize_model(pretrained_embedding=None,
                    freeze_embedding=False,
                    vocab_size=None,
                    embed_dim=300,
                    filter_sizes=[3, 4, 5],
                    num_filters=[100, 100, 100],
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.01,
                    device:torch.device=None
                    ):
    """Instantiate a CNN model and an optimizer."""
    assert (len(filter_sizes) == len(num_filters)), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        vocab_size=vocab_size,
                        embed_dim=embed_dim,
                        filter_sizes=filter_sizes,
                        num_filters=num_filters,
                        num_classes=num_classes,
                        dropout=dropout
                        )

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(cnn_model.parameters(),
                               lr=learning_rate,
                               rho=0.95
                               )

    return cnn_model, optimizer
