from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

# torch.set_default_tensor_type("torch.cuda.FloatTensor")


class CnnNlpModel(nn.Module):
    """ConvMFにおける CNNパート ($s_j = CNN(W, X_j)$)を実装する為のクラス"""

    def __init__(
        self,
        output_dimension: int,
        pretrained_embedding: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
        vocab_size: Optional[int] = None,
        embed_dimension: int = 300,
        kernel_sizes: List[int] = [3, 4, 5],
        num_kernels: List[int] = [100, 100, 100],
        dropout_ratio: float = 0.5,
    ) -> None:
        """
        Args:
            output_dimension (int): Number of classes.
                最終的なCNNの出力次元数。
                目的変数の次元数(ConvMFの場合はこれがItem Latent Vectorの次元数になる)
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim). 学習済みの単語埋め込みベクトル。
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. 学習済みの単語埋め込みベクトルをfine-tuningするか否か。
                Default: False
            vocab_size (int): vocabrary size. Need to be specified when not pretrained word
                embeddings are not used. 学習済みの単語埋め込みベクトルが渡されない場合、指定する必要がある。
            embed_dim (int): Dimension of word vectors. Word Vectorの次元数。
                Need to be specified when pretrained word embeddings are not used.
                学習済みの単語埋め込みベクトルが渡されない場合、指定する必要がある。
                Default: 300
            kernel_sizes (List[int]): List of filter sizes.
                畳み込み層のスライド窓関数(カーネル)のwindow size(高さのみ。幅はw_iの埋め込み次元の大きさ)を指定する。
                Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`.
                畳み込み層のスライド窓関数(Shared weihgt)の数
                Default: [100, 100, 100]
            dropout_ratio (float): Dropout rate.
                中間層のいくつかのニューロンを一定確率でランダムに選択し非活性化する。
                Default: 0.5
        """

        super(CnnNlpModel, self).__init__()

        self.embedding_layer, self.embed_dim = self._define_embedding_layer(
            pretrained_embedding,
            freeze_embedding,
            embed_dimension,
            vocab_size,
        )

        self.conv1d_layers = self._define_conv_layers(
            input_dim=self.embed_dim,
            num_kernels=num_kernels,
            kernel_sizes=kernel_sizes,
        )
        self.fc_layer1, self.fc_layer2 = self._define_fc_layers(
            input_dim=np.sum(num_kernels),
            output_dim=output_dimension,
        )
        self.tanh = torch.nn.Tanh()

        self.dropout = nn.Dropout(p=dropout_ratio)

    def _define_embedding_layer(
        self,
        pretrained_embedding: Optional[torch.Tensor] = None,
        freeze_embedding: bool = False,
        embed_dimension: Optional[int] = None,
        vocab_size: Optional[int] = None,
    ) -> Tuple[nn.Embedding, int]:
        """Embedding layer(埋め込み層)の定義
        - 学習済みの単語埋め込みベクトルの配列が指定されない場合、embed_dimensionとvocab_sizeに基づき単語埋め込みベクトルを初期化
        - 指定された場合、pretrained_embedding_vectorsに基づき、nn.Embeddingを生成
        Return
        Tuple[embedding_layer,embed_dimension,vocab_size]
        """

        if pretrained_embedding is None:
            embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,  # 語彙サイズ
                embedding_dim=embed_dimension,  # 埋め込みベクトルの次元数
                padding_idx=0,  # 文章データ(系列データ)の長さの統一：ゼロパディング
                max_norm=5.0,  # 単語埋め込みベクトルのnorm(長さ?)の最大値の指定。これを超える単語ベクトルはnorm=max_normとなるように正規化される?
            )
        else:
            vocab_size, embed_dimension = pretrained_embedding.shape
            embedding_layer = nn.Embedding.from_pretrained(
                pretrained_embedding,
                freeze=freeze_embedding,
            )

        return embedding_layer, embed_dimension

    def _define_conv_layers(
        self,
        input_dim: int,
        num_kernels: List[int],
        kernel_sizes: List[int],
    ) -> nn.ModuleList:
        """Conv Layersの定義
        スライド窓関数のwindow size(resign size)の種類分の繰り返し処理で,
        一次元の畳み込み層を定義していく.
        """
        modules = []
        for kernel_size, num_kernel in zip(kernel_sizes, num_kernels):
            conv_layer = nn.Conv1d(
                in_channels=input_dim,  # 入力チャネル数:埋め込みベクトルの次元数
                out_channels=num_kernel,  # 出力チャネル数(pooling後、resign size毎に代表値を縦にくっつける)
                kernel_size=kernel_size,  # window size(resign size)(Conv1dなので高さのみ指定)
                padding=0,  # ゼロパディング
                stride=1,  # ストライド
            )
            modules.append(conv_layer)
        return nn.ModuleList(modules=modules)

    def _define_fc_layers(
        self,
        input_dim: int,
        output_dim: int,
    ) -> Tuple[nn.Linear, nn.Linear]:
        """Fully-connected layer and Dropout 全結層を定義する
        - 入力d_f(次元数はR^{n_c}), 出力はs_j(次元数はR^{n_factor})
        - バイアス項あり。活性化関数は2回ともtanh.
        """
        hidden_dim = input_dim * 2
        fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        return fc1, fc2

    def forward(self, input_token_indices: Tensor):
        """Perform a forward pass through the network.
        - $batch_size$は、input_idsの長さ=モデルに投入するitemの数
        - まずEmbedding層にtokenizedされたテキスト(符号化済み)を渡して、文書行列を取得する
        - 続いて、Tensorの軸の順番を入れ替えて、shapeをcnnの入力用に整形する。:(batch_size, max_len, embed_dim)=>(batch_size, embed_dim, max_len)
        - 続いて、CNN及び活性化関数(ReLU)を適用する.
        - 続いて、各convolutionの出力値にmax poolingを適用して、代表値を取得
        - 続いて、x_pool_listを連結して、fully connected layerに投入する為のshapeに変換
        - 最後に、全結合層に入力。batch_size個のdim_output次元ベクトルを得る
        Args:
            input_ids (torch.Tensor): A tensor of token ids
            with shape (batch_size, length_of_sentence)
        Returns:
            torch.Tensor: Output tensors with shape (batch_size, dim_output)
        """
        x_embed: Tensor = self.embedding_layer(input_token_indices).float()
        # -> Output shape: (batch_size, length_of_sentence, embed_dim) ex) (32, 100, 300)
        x_reshaped = x_embed.permute(0, 2, 1)
        # -> Output shape: (batch_size, embed_dim, max_len) ex) (32, 300, 100)

        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_layers]
        # ->Output shape: (batch_size, num_filters[i], L_out(convolutionの出力数))
        # ex) (32, (100 or 100 or 100), 100)

        x_pool_list: List[Tensor] = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]
        # ->Output shape: (batch_size, num_filters[i], 1) kernel_size引数はx_convの次元数に！=>poolingの出力は1次元!
        # ex) (32, 300, 100)

        x_squeezed_for_fc = torch.cat(
            [x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1
        )
        # ->Output shape: (batch_size, sum(num_filters))

        x_fc1 = torch.tanh(self.fc_layer1(x_squeezed_for_fc))
        # -> Output shape: (batch_size, dim_hidden)

        x_fc2 = torch.tanh(self.fc_layer2(x_fc1))
        # -> Output shape: (batch_size, dim_output)

        return x_fc2

    def predict(self, token_indices_arrays: List[np.ndarray]) -> List[np.ndarray]:
        self.eval()
        token_indices_tensors = [
            torch.from_numpy(token_indices) for token_indices in token_indices_arrays
        ]
        batch_size = len(token_indices_tensors)

        # 計算グラフの構築を抑制します。これにより、推論時に不要な計算が省略されるようになります。
        with torch.no_grad():
            token_indices_tensors_batch = torch.stack(token_indices_tensors, dim=0)

            outputs: Tensor = self(token_indices_tensors_batch)

        return [item_description_vector.numpy() for item_description_vector in outputs]


"""To train Deep Learning models, we need to define a loss function 
and minimize this loss. 
We’ll use back-propagation to compute gradients and use an optimization algorithm (ie. Gradient Descent) to minimize the loss. 
The original paper used the Adadelta optimizer.
Deep Learningのモデルを学習させるためには、損失関数を定義し、この損失を最小化する必要があります。
バックプロパゲーションを用いて勾配を計算し、最適化アルゴリズム（gradient Descentなど）を用いて損失を最小化することになります。
元の論文ではAdadeltaオプティマイザを使用しています。
"""


def initilize_cnn_nlp_model(
    pretrained_embedding: Optional[torch.Tensor] = None,
    freeze_embedding: bool = False,
    vocab_size: int = 100,
    embed_dim: int = 300,
    filter_sizes: List[int] = [3, 4, 5],
    num_filters: List[int] = [100, 100, 100],
    output_dimension: int = 2,
    dropout: float = 0.5,
    learning_rate: float = 0.01,
    device: Optional[torch.device] = None,
) -> Tuple[CnnNlpModel, optim.Adadelta]:
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
    assert len(filter_sizes) == len(num_filters)

    # Instantiate CNN model
    cnn_model = CnnNlpModel(
        pretrained_embedding=pretrained_embedding,
        freeze_embedding=freeze_embedding,
        vocab_size=vocab_size,
        embed_dimension=embed_dim,
        kernel_sizes=filter_sizes,
        num_kernels=num_filters,
        output_dimension=output_dimension,
        dropout_ratio=dropout,
    )

    # Send model to `device` (GPU/CPU)
    cnn_model.to(device)

    # Instantiate Adadelta optimizer
    optimizer = optim.Adadelta(
        params=cnn_model.parameters(),  # 最適化対象
        lr=learning_rate,  # parameter更新の学習率
        rho=0.95,  # 移動指数平均の係数(?きっとハイパーパラメータ)
    )

    return cnn_model, optimizer


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cnn_nlp_model = CnnNlpModel(
        output_dimension=10,
        vocab_size=100,
        embed_dimension=15,
    )
    # 入力データ
    x = torch.Tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 1, 3, 4, 5, 0],
        ]
    ).long()  # LongTensor型に変換する(default はFloatTensor?)
    print(x)
    y = cnn_nlp_model(x)
    print(y)
