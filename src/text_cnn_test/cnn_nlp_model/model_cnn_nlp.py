from turtle import forward
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


class CNN_NLP(nn.Module):
    """
    文章分類の為の一次元CNN
    An 1D Convulational Neural Network for Sentence Classification.
    """

    def __init__(
        self,
        pretrained_embedding: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
        vocab_size: Optional[int] = None,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        dim_output: int = 2,
        dropout: float = 0.5,
    ) -> None:
        """
        CNN_NLPクラスのコンストラクタ
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)。学習済みの単語埋め込みベクトル。
                Default: None
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. 学習済みの単語埋め込みベクトルをfine-tuningするか否か。
                Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
                Default: None
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used.
                学習済みの単語埋め込みベクトルが渡されない場合、指定する必要がある。
                Default: 300
            filter_sizes (List[int]): List of filter sizes.
            畳み込み層のスライド窓関数のwindow sizeを指定する。
            Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. 畳み込み層のスライド窓関数(Shared weihgt)の数
                Default: [100, 100, 100]
            dim_output (int): Number of classes. 最終的なCNNの出力次元数。
            Default: 2

            dropout (float): Dropout rate. 中間層のいくつかのニューロンを一定確率でランダムに選択し非活性化する。
            Default: 0.5
        """
        super(CNN_NLP, self).__init__()

        # Embedding layerの定義
        # 学習済みの単語埋め込みベクトルの配列が渡されていれば...
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding)
        # 渡されていなければ...
        else:
            self.embed_dim = embed_dim
            # 単語埋め込みベクトルを初期化
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,  # 語彙サイズ
                embedding_dim=self.embed_dim,  # 埋め込みベクトルの次元数
                padding_idx=0,  # 文章データ(系列データ)の長さの統一：ゼロパディング
                # 単語埋め込みベクトルのnorm(長さ?)の最大値の指定。これを超える単語ベクトルはnorm=max_normとなるように正規化される?
                max_norm=5.0,
            )

        # Conv Networkの定義
        modules = []
        # スライド窓関数のwindow size(resign size)の種類分、繰り返し処理
        for i in range(len(filter_sizes)):
            # 畳み込み層の定義
            conv_layer = nn.Conv1d(
                # 入力チャネル数:埋め込みベクトルの次元数
                in_channels=self.embed_dim,
                # 出力チャネル数(pooling後、resign size毎に代表値を縦にくっつける)
                out_channels=num_filters[i],
                # window size(resign size)(Conv1dなので高さのみ指定)
                kernel_size=filter_sizes[i],
                padding=0,  # ゼロパディング
                stride=1,  # ストライド
            )
            # 保存
            modules.append(conv_layer)
        # 一次元の畳み込み層として保存
        self.conv1d_list = nn.ModuleList(modules=modules)

        # 全結合層(中間層なし)とDropoutの定義
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(in_features=np.sum(num_filters), out_features=dim_output)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                dim_output)
        """
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        #
        x_embed: Tensor = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        #
        x_reshaped = x_embed.permute(0, 2, 1)
        # Output shape:(batch_size, embed_dim, max_len)

        # Apply CNN and ReLU.
        # Output shape: (batch_size, num_filters[i], L_out(convolutionの出力数))
        x_conv_list: List[Tensor] = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling.
        # 各convolutionの出力値にmax poolingを適用して、一つの代表値に。
        # Output shape: (batch_size, num_filters[i], 1)
        # kernel_size引数はx_convの次元数に！=>poolingの出力は1次元!
        x_pool_list: List[Tensor] = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer(全結合層).
        # x_pool_listを連結して、fully connected layerに投入する為のshapeに返還
        # Output shape: (batch_size, sum(num_filters))
        x_fc: Tensor = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Compute logits. Output shape: (batch_size, dim_output)
        logits = self.fc(self.dropout(x_fc))

        return logits


"""To train Deep Learning models, we need to define a loss function and minimize this loss. We’ll use back-propagation to compute gradients and use an optimization algorithm (ie. Gradient Descent) to minimize the loss. 
The original paper used the Adadelta optimizer.
Deep Learningのモデルを学習させるためには、損失関数を定義し、この損失を最小化する必要があります。バックプロパゲーションを用いて勾配を計算し、最適化アルゴリズム（gradient Descentなど）を用いて損失を最小化することになります。
元の論文ではAdadeltaオプティマイザを使用しています。
"""


def initilize_model(
    pretrained_embedding: torch.Tensor = None,
    freeze_embedding=False,
    vocab_size=None,
    embed_dim=300,
    filter_sizes=[3, 4, 5],
    num_filters=[100, 100, 100],
    num_classes=2,
    dropout=0.5,
    learning_rate=0.01,
    device: torch.device = None,
) -> Tuple[CNN_NLP, optim.Adadelta]:
    """Instantiate a CNN model and an optimizer.

    Parameters
    ----------
    pretrained_embedding (torch.Tensor):
        Pretrained embeddings with
        shape (vocab_size, embed_dim)。学習済みの単語埋め込みベクトル。
        Default: None
    freeze_embedding (bool):
        Set to False to fine-tune pretraiend
        vectors. 学習済みの単語埋め込みベクトルをfine-tuningするか否か。
        Default: False
    vocab_size (int):
        Need to be specified when not pretrained word
        embeddings are not used. 学習済みの単語埋め込みベクトルが渡されない場合、指定する必要がある。
        Default: None
    embed_dim (int):
        Dimension of word vectors. Need to be specified
        when pretrained word embeddings are not used.
        学習済みの単語埋め込みベクトルが渡されない場合、指定する必要がある。
        Default: 300
    filter_sizes (List[int]):
        List of filter sizes.
        畳み込み層のスライド窓関数のwindow sizeを指定する。
        Default: [3, 4, 5]
    num_filters (List[int]):
        List of number of filters, has the same
        length as `filter_sizes`. 畳み込み層のスライド窓関数(Shared weihgt)の数
        Default: [100, 100, 100]
    dim_output (int):
        Number of classes. 最終的なCNNの出力次元数。
        Default: 2
    dropout (float):
        Dropout rate. 中間層のいくつかのニューロンを一定確率でランダムに選択し非活性化する。
        Default: 0.5
    learning_rate : float, optional
        Optimizerの学習率, by default 0.01
    device : torch.device, optional
        学習時に使用する計算機(CPUかGPUか), by default None

    Returns
    -------
    Tuple(CNN_NLP, optim.Adadelta)
        Initializeしたモデルと、Optimizerオブジェクトを返す。
    """
    assert len(filter_sizes) == len(
        num_filters
    ), "filter_sizes and \
    num_filters need to be of the same length."

    # Instantiate CNN model
    cnn_model = CNN_NLP(
        pretrained_embedding=pretrained_embedding,
        freeze_embedding=freeze_embedding,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dim_output=num_classes,
        dropout=dropout,
    )

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(
        params=cnn_model.parameters(),  # 最適化対象
        lr=learning_rate,  # parameter更新の学習率
        rho=0.95,  # 移動指数平均の係数(? きっとハイパーパラメータ)
    )

    return cnn_model, optimizer
