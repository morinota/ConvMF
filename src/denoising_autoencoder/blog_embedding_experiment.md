<!-- title:  -->

## 参考

- [Compute Document Similarity Using Autoencoder With Triplet Loss](https://medium.com/deep-learning-hk/compute-document-similarity-using-autoencoder-with-triplet-loss-eb7eb132eb38)
- [論文:Article De-duplication Using Distributed Representations](http://gdac.uqam.ca/WWW2016-Proceedings/companion/p87.pdf)
  - 分散表現を用いた記事の重複排除

# はじめに

本記事は、[Compute Document Similarity Using Autoencoder With Triplet Loss](https://medium.com/deep-learning-hk/compute-document-similarity-using-autoencoder-with-triplet-loss-eb7eb132eb38)を読んで、**テキスト埋め込みベクトルの生成時にカテゴリ間の類似性を試みた方法論**を知るとともに、提案された手法を実装、ローカル環境で実験をしてみたものです:)

もし気になった点や理解が誤っている箇所がありましたら、ぜひコメントいただけると喜びます:)

# 手法の振り返り

テキストの類似度を計算する最も一般的な方法は、文書をTFIDFベクトルに変換し、これらのベクトルにコサイン類似度などの任意の類似度指標を適用することです.

しかし、ニュースアイテムのPersonalized Recommendationに適用する為には、以下の２つの課題があります.

- ニュースアイテムは鮮度が高く時間制約が厳しい為、高次元のベクトルだと計算量的に厳しい.低次元のベクトルでテキストの特徴を表現する必要がある.
- **同じカテゴリの記事で異なる出来事について話している記事同士**にも類似性を持たせたい.

論文ではこの2つの課題を解決する手法として、"Triplet Lossを損失関数に含めたDAE(Denoising Autoencoder)"を用いてテキスト埋め込みベクトルを生成する手法を提案しています。

- DAEを用いることで、高次元のTFIDFベクトルを低次元の埋め込みに圧縮することができる
- 損失関数にTriplet Lossを用いることで、テキスト間のカテゴリカルな類似性を保持することができる

## denoising Autoencoder

オートエンコーダーは、基本的に教師なし学習を行うニューラルネットワークで、圧縮された**埋め込みデータ（デコーダー）から入力ベクトル（エンコーダー）を再構成**しようとするものです.

denoising Autoencoderは、Autoencoderの学習段階で**確率的に入力データを破損**させる方法です.
入力データの小さな変化に対して得られる埋め込みベクトルをより頑健にすることがモチベーションになっています.

以下が、DAEの定義式になります.

$$
\tilde{x} \sim C(\tilde{x}|x) \\
h = f(W\tilde{x} + b) \\
y = f(W'h + b') \\
\theta = \argmin_{W, W', b, b'} \sum_{x \in X} L(y, x)
$$

## Triplet Lossを含めた損失関数

論文の手法における損失関数は以下で定義されます.
二乗誤差の項にTriplet Lossの項が追加されています.

$$
h_n = f(W \tilde{x}_n + b) - f(b) \tag{1}
$$

$$
\phi(h_1,h_2, h_3) = \log(1 + \exp (h_1^T h_3 - h_1^T h_2)) \tag{2}
\\
\theta = \argmin_{W, W', b, b'} \sum_{(x_1,x_2,x_3)\in T} \sum_{n=1}^3 L(y_n,x_n) + \alpha \phi(h_1, h_2, h_3)
\\
\text{Triplet Loss function of autoencoder}
$$

# 実装してみる

[前回記事]()でTriplet Mining(Triplet Loss計算の為の３つ組をMiningする手法)を実装しました。
今回はそれを用いてDenoising Autoencoderを学習させ、テキストの埋め込みベクトルを作成してみます。
その後、作成したテキスト埋め込みベクトルがカテゴリ間の類似性を保持しているかどうか、技術記事と同様の方法を用いて実験してみます。

## Autoencoderクラスの実装

まずは`Autoencoder`クラスの実装です. denoising Autoencoderでは学習時にノイズを付与する以外はAutoencoderと同じなので、`Autoencoder`クラスとして実装しています.

`nn.Module`クラスを継承して、自作モデルクラス`AutoEncoder`を作ります.
コンストラクタでencoderとdecoderの形状を定義し、`forward()`メソッドに"入力値をモデルに通して出力値を返す"処理の中身を記述します. 今回のケースでは、損失関数の値の計算にembedding(=encoderの出力)とoutput(decoderの出力)の両方を用いる為、返り値はタプルにしています.

```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()

        self.encoder = nn.Linear(input_dim, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, input_dim)

    def forward(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """モデルを通して入力値を出力値に再構成して返す.
        返り値はembeddeed(encodeされた後のTensor)、decodeされた後のTensorのtuple"""
        X_embedded = self.encoder(X)
        X_output = self.decoder(X_embedded)
        return X_embedded, X_output
```

## noiseを付与する処理の実装

続いてdenoising Autoencoderにおける"学習時にノイズを付与する"処理を`add_noise`関数として実装します.
引数としてAutoencoderの入力データ、及びノイズ付与方法に関するoption(`noise_type`, `noise_rate`)を受け取り、返り値としてノイズ付与された入力データを返します.
なお、ノイズ付与方法のoptionに関して、元論文では"入力ベクトルの各要素を確率0.3でランダムにマスク（0に設定）"している為、`noise_type == "masking", noise_rate: float = 0.3`に該当します.

```python
def add_noise(
    X_input: Tensor,
    noise_type: str = "masking",
    noise_rate: float = 0.3,
) -> Tensor:
    """入力データにノイズを加える関数
    元論文では、noise_type == "masking", noise_rate: float = 0.3を適用.
    (入力ベクトルの各要素を確率0.3でランダムにマスク（0に設定）する)
    """
    if noise_type not in ["gaussian", "masking"]:
        print(f"[WARN]please set the arg: noise_type is in [gaussian, masking]. So the noises was not added.")
        return X_input
    if noise_type == "gaussian":
        noises_mask = torch.randn(X_input.size()) * noise_rate
        return X_input + noises_mask
    elif noise_type == "masking":
        noises_mask = torch.rand(X_input.size())
        noises_mask[noises_mask < noise_rate] = 0
        noises_mask[noises_mask >= noise_rate] = 1
        return X_input * noises_mask
```

## Triplet Lossを含んだ損失関数の実装

続いて、論文の手法のキモであるTriplet Lossを含んだ損失関数`DenoisingAutoEncoderLoss`を定義していきます.

損失関数の定義式を再度、記述しておきます.

$$
h_n = f(W \tilde{x}_n + b) - f(b) \tag{1}
$$

$$
\phi(h_1,h_2, h_3) = \log(1 + \exp (h_1^T h_3 - h_1^T h_2)) \tag{2}
\\
\theta = \argmin_{W, W', b, b'} \sum_{(x_1,x_2,x_3)\in T} \sum_{n=1}^3 L(y_n,x_n) + \alpha \phi(h_1, h_2, h_3)
\\
\text{Triplet Loss function of autoencoder}
$$

自作の損失関数を作るときも、自作のモデルクラスを作る際と同様に`nn.Module`クラスを継承し、`forward`を定義します.

コンストラクタ`__init__`では、損失関数の第一項を計算する`nn.MSELoss`オブジェクト,第二項を計算する`nn.TripletMarginLoss`オブジェクトを初期化し、インスタンス変数に格納しておきます.
ここで元論文の損失関数の仕様に合わせる為に、`reduction`オプションを`mean`ではなく`sum`で指定しておきます.

また、[前回記事](https://qiita.com/morinota/items/d732d77a598da30948bc)で作成したTriplet Miningする為の`BatchHogehogeStrategy`クラスも、インスタンス変数に格納しています.
Miningの戦略に関しては、コンストラクタの引数`mining_storategy`で、batch allかbatch hardか指定するようにしています.

`forward()`メソッドでは、入力ベクトル$x$, 埋め込みベクトル$h$, 出力ベクトル$y$, カテゴリラベル(それぞれbatch_size\*1の`Tensor`)を引数として受け取り、損失関数の値(スカラーの`Tensor`)を返します.
まずカテゴリラベルを用いて、Triplet(３つ組)$T$をMiningで取得します.
`mining()`メソッドの返り値`triplet_indices_dict`はDict型で、３つのkey(`anchor_ids`, `positive_ids`, `negative_ids`)を持ち、それぞれのvalueには、anchor, positive, negativeに対応する`batch_size * 1のTensorにおけるindexのList`が入っています.
(`inputs[triplet_indices_dict["anchor_ids"]]`のi番目の要素が、i番目のTripletにおけるanchorの入力ベクトル$x_1$になります.)

続いて第一項を計算していきます. 全ての`(x_1,x_2,x_3)\in T`について二乗誤差を計算して足し合わせるので、MiningしたTriplet達における全てのanchor($x_1$), 全てのpositive($x_2$), 全てのnegative($x_3$)において、二乗誤差をそれぞれ計算して足し合わせています.

第二項の計算では単に、`nn.TripletMarginLoss`にTripletのEmbeddings($h_1$, $h_2$, $h_3$)を渡してtriplet lossの値を計算しています.

最後に第一項と第二項を足し合わせて、(念のため`torch.float32`に型変換して、)損失関数の値を返しています.

```python
class DenoisingAutoEncoderLoss(nn.Module):
    MINING_STORATEGIES = ["batch_all", "batch_hard"]

    def __init__(
        self,
        alpha: float = 10.0,
        margin: float = 1.0,
        mining_storategy: str = "batch_all",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.margin = margin

        self.mse_loss_func = nn.MSELoss(reduction="sum")

        self.triplet_loss_func = nn.TripletMarginLoss(
            margin=self.margin,
            reduction="sum",
        )

        if mining_storategy not in self.MINING_STORATEGIES:
            raise ValueError(
                "Unexpected storategy name is inputted. Please choose mining_storategy in [batch_all, batch_hard]"
            )
        self.triplet_mining_obj = (
            BatchAllStrategy(self.margin) if mining_storategy == "batch_all" else BatchHardStrategy(self.margin)
        )


    def forward(
        self,
        inputs: Tensor,
        embeddings: Tensor,
        outputs: Tensor,
        labels: Tensor,
    ) -> Tensor:

        triplet_indices_dict = self.triplet_mining_obj.mining(labels, embeddings)

        squared_error_term = 0
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["anchor_ids"]],
            outputs[triplet_indices_dict["anchor_ids"]],
        )
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["positive_ids"]],
            outputs[triplet_indices_dict["positive_ids"]],
        )
        squared_error_term += self.mse_loss_func(
            inputs[triplet_indices_dict["negative_ids"]],
            outputs[triplet_indices_dict["negative_ids"]],
        )

        triplet_loss = self.triplet_loss_func.forward(
            anchor=embeddings[triplet_indices_dict["anchor_ids"]],
            positive=embeddings[triplet_indices_dict["positive_ids"]],
            negative=embeddings[triplet_indices_dict["negative_ids"]],
        )

        loss = squared_error_term + self.alpha * triplet_loss
        loss = loss.to(torch.float32)
        return loss
```

## 一連の学習処理をまとめたtrain関数を実装

Denoising Autoencoderの学習の一連の処理をまとめた、`train`関数を実装します.
引数にモデルオブジェクト、学習用データ(`Dataloader`)、損失関数オブジェクト、オプティマイザー、deviceオブジェクト(cpuかcudaか)、epoch数の指定, 検証用データを渡して、学習処理を実行します.
返り値として学習後(パラメータ更新後)のモデルオブジェクトと、検証用データに対する損失関数の値の推移のlistを返します.

```python
def train(
    model: AutoEncoder,
    train_dataloader: DataLoader,
    loss_function: DenoisingAutoEncoderLoss,
    optimizer: torch.optim.Adam,
    device: torch.device,
    epochs: int = 20,
    valid_dataloader: Optional[DataLoader] = None,
) -> Tuple[AutoEncoder, List[Tensor]]:
    """Train the AutoEncoder model. 学習を終えたAutoEncoderオブジェクトを返す。"""

    model.to(device)
    valid_loss_list = []

    for epoch_idx in range(epochs):
        # =================training=====================
        model.train()

        for batch_idx, batch_dataset in enumerate(train_dataloader):
            print(f"=====epoch_idx:{epoch_idx}, batch_idx:{batch_idx}=====")
            input_vectors: Tensor
            labels: Tensor
            input_vectors, labels = tuple(tensors for tensors in batch_dataset)
            labels = labels.type(dtype=torch.LongTensor)
            input_vectors, labels = input_vectors.to(device), labels.to(device)

            input_vectors_noised = add_noise(input_vectors).to(device)

            # 勾配が累積してく仕組みなので,1バッチ毎に勾配の値を初期化しておく.
            model.zero_grad()

            embedded_vectors, output_vecotrs = model.forward(input_vectors_noised)

            loss = loss_function.forward(
                inputs=input_vectors,
                embeddings=embedded_vectors,
                outputs=output_vecotrs,
                labels=labels,
            )
            print(f"the loss(train): {loss}")
            loss.backward()
            optimizer.step()

        if valid_dataloader is None:
            continue
        # =================validation=====================
        model.eval()
        valid_loss_in_epoch = 0.0
        len_valid_dataset = len(valid_dataloader.dataset)

        for batch_idx, batch_dataset in enumerate(valid_dataloader):
            input_vectors: Tensor
            labels: Tensor
            input_vectors, labels = tuple(tensors for tensors in batch_dataset)

            labels = labels.type(dtype=torch.LongTensor)
            input_vectors = input_vectors.type(dtype=torch.FloatTensor)

            input_vectors, labels = input_vectors.to(device), labels.to(device)
            embedded_vectors, output_vecotrs = model(input_vectors)

            valid_loss = loss_function.forward(
                inputs=input_vectors,
                embeddings=embedded_vectors,
                outputs=output_vecotrs,
                labels=labels,
            )
            valid_loss_in_epoch += valid_loss / len_valid_dataset

        print(f"the valid_loss_in_epoch: {valid_loss_in_epoch}")
        valid_loss_list.append(valid_loss_in_epoch)

    return model, valid_loss_list
```

## 埋め込みベクトルがカテゴリ間の類似性を保持しているかの実験

# おわりに
