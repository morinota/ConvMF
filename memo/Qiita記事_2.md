title: 評価行列とアイテムの説明文書を活用した推薦システム「ConvMF」を何とか実装していきたい!①MFパートの実装

# 参考

- [元論文](https://dl.acm.org/doi/10.1145/2959100.2959165)
- nishiba様のConvMF実装例(Chainerをお使いになってました！)
  - [エムスリー様のテックブログ](https://www.m3tech.blog/entry/2018/03/07/122353)
  - [nishiba様のgithub](https://github.com/nishiba/convmf)

# はじめに

KaggleのPersonalized Recommendationコンペに参加して以降、推薦システムが自分の中で熱くなっております。以前、Implicit Feedbackに対するモデルベースの協調フィルタリング(Matrix Factorization)の論文を読んで実装してみて、今度は更に実用的(?)で発展的な手法を触ってみたいと思い、「Convolutional Matrix Factorization for Document Context-Aware Recommendation」を読みました。この論文では、Matrix Factorizationによるモデルベース協調フィルタリングに、CNNを用いてアイテムの説明文書の情報を組み合わせる ConvMF(Convolutional Matrix Factorization)を提案しています。

今実装中ですが、なかなかPytorchと仲良くなれず、苦戦しております...。(ちなみに元論文はKerasで実装しておりました!)

パート2とした本記事では、**ConvMFの理論**と実装に向けた評価行列データ(MovieLens)と対応するアイテム説明文書データ(TMDb)の前処理についてまとめています。

本記事以前のパートは、以下のリンクを御覧ください。

- [評価行列とアイテムの説明文書を活用した推薦システム「ConvMF」を何とか実装していきたい!①MFパートの実装](https://qiita.com/morinota/items/d84269b7b4bf55d157d8)

# 前回のリマインド

## ConvMF（畳み込み行列分解）とは？

Convolutional Matrix Factorization(通称ConvMF)は、モデルベース協調フィルタリングにおいて**評価行列のスパース性の上昇問題やコールドスタート問題**に対応する為に提案された、Explicit FeedbackやImplicit Feedbackの評価情報に加えて**アイテムの説明文書(ex. ニュース記事の中身、動画のタイトル、etc.)の情報を考慮した**推薦手法の一つです。
その為に、ConvMFではモデルベース協調フィルタリングであるPMF(Probabilistic Matrix Factorization)にCNN(convolutional neural network)を統合しています。
その結果、ConvMFは最終的に協調情報と文脈情報の両方を効果的に利用することができ、評価データが極めて疎な場合でも、ConvMFは未知の評価を正確に予測することができる、らしいです...。

## ConvMFの確率モデル

以下の図は、NLPに対するCNNモデルをPMF(確率的行列分解)モデルに統合した、ConvMFの確率モデルの概要を示したモノになります。

![](https://d3i71xaburhd42.cloudfront.net/9e91c370a8fae365c731947ad9178fb3788f6593/500px/3-Figure1-1.png))

ちなみに上図において、各記号の意味合いは以下です。(順次まとめていきます)

- $U$: user latent model
- $V$: item latent model
- $R$: Rating Matrix
- $X$: アイテムのDescription(説明文)
- $W$: CNNのパラメータ達。
- $i, j$: それぞれ、各ユーザと各アイテムを表す添字。
- $k$:CNN内の各パラメータを表す添字(kに関しては、潜在ファクターの次元数の記号と混在してるかもしれません...?)

また問題設定として、N人のユーザとM個のアイテムがあり、観測された評価行列は$R\in \mathbb{R}^{N\times M}$行列で表現されるとします。
そして、その積（$U^T \cdot V$）が評価行列 $R$を再構成するような、ユーザとアイテムの潜在モデル（$U\in \mathbb{R}^{k\times N}$ と $V \in \mathbb{R}^{k\times M}$）を見つけることが目的になります。

確率的な観点からは、観測評価に関する条件付き分布(=**すなわち尤度関数！！**)は次式で与えられます。

$$
p(R|U, V, \sigma^2)
= \prod_{i}^{N} \prod_{j}^{M}
 N(r_{ij}|u_i^T v_j, \sigma^2)^{I_{ij}}
 \tag{1}
$$

ここで、

- $N (x|\mu, \sigma2)$は、一次元正規分布(ハイパラは$\mu, \sigma^2$)に従うxの確率密度関数。
  - PMFでは「各$r_{ij}$間は独立」と仮定しているって事ですかね...!
- $I_{ij}$は「Ratingが観測されていれば1, 未観測であれば0を返す」ような指標関数。
  - 要するに、$I_{ij}$を使って、観測済みのRatingのみを尤度関数に含める事を表現しているっぽいですね...!

また、user latent model$U$(各列ベクトルがユーザ因子ベクトル$u_i$)のPriori(事前分布)として、Spherical Gaussian prior(=>共分散行列が対角成分のみ。且つ全ての対角成分の値が等しい??)を選びます。
(要するに、$U_i$間は独立、各$U_i$はk次元正規分布に従って生成されると仮定...!)

$$
u_i = \epsilon_i \\
\epsilon_i \sim N(\mathbf{0}, \sigma^2_U I)\\

p(U|\sigma^2_U) = \prod_{i}^{N} N(u_i|\mathbf{0}, \sigma_U^2 I)
\tag{2}
$$

続いてitem latent modelですが、**従来のPMF(確率的行列分解)におけるアイテム特徴行列の確率モデルとは異なり**、ConvMFのアイテム因子行列は以下の３変数から生成されます：

1. CNNにおけるパラメータ達$W$
2. アイテムjのDescriptionを表す$X_j$
3. 正規分布に従うと仮定された、撹乱項$\epsilon$

数式で書くと...v_jの生成仮定は...

$$
v_j = cnn(W, X_j) + \epsilon_j \\
\epsilon_j \sim N(\mathbf{0}, \sigma^2_V I)
$$

また、CNNのパラメータ$W$の各重み$w_k$に対しても、事前分布として、最も一般的に用いられるゼロ平均のSpherical Gaussian prior（=>つまり平均ベクトル0, 共分散行列が対角成分のみ、且つ全ての非ゼロ要素の値が等しい）を設定します。

$$
p(W|\sigma^2_W) = \prod_k N(w_k|0, \sigma_W^2) \tag{3}
$$

したがって、アイテム特徴行列に対する条件付き分布は次式で与えられます。

$$
p(V|W, X, \sigma^2_V) = \prod_j^{M}
N(v_j|cnn(W,X_j),\sigma_V^2I) \tag{4}
$$

ここで、

- $X$: アイテムに関する記述文書の集合
  - $X_j$はアイテムjの特徴量
- $cnn(W,X_j)$:
  - CNNモデルから得られるDocument latent vactor($cnn(W,X_j)$)。
  - $v_i$の事前分布の平均$\mu$として用いられている。
  - このDocument Latent Vectorが**CNNとPMFの橋渡し**になる。
  - これが説明文書(すなわち、アイテムが持つ特徴量、コンテキスト)と評価(Explicit or Implicit feedback)の両方を考慮するために重要な役割を果たすらしい...!

## $cnn(W,X_j)$について

ここに関しては追記するかもしれないし、しないかもしれません。
NLPへのCNNパートの実装の際に記述するかもしれません！
ここでは、アイテムjに関する説明文$X_j$を入力として、document latent vector $s_j = cnn(W,X_j) \in \mathbf{R}^{factorの数}$を出力する関数、とだけ記述しておきます。

## ConvMFにおけるパラメータ推定法

ConvMFでは、パラメータ($U, V, W$)を最適化する為に、MAP推定(maximum a posteriori estimation)を行います。

$$
\max_{U, V, W} p(U, V, W|R, X, \sigma^2, \sigma_U^2, \sigma_V^2, \sigma_W^2) \\
= \max_{U, V, W}[尤度関数 \times 事前分布] \\
= \max_{U, V, W}[
  p(R|U, V, \sigma^2)
  \cdot p(U|\sigma_U^2)
  \cdot p(V|W, X, \sigma_V^2)
  \cdot p(W|\sigma_W^2)
  ]
  \tag{5}
$$

式(5)の3行目に、上述した式(1)~(4)を代入する事で、目的関数の値を計算できますね。
更にここから式(5)を対数化してマイナスを掛け、いい感じに変形($\sigma^2$で割る!)と、以下のようになります。

$$
L(U,V,W|R, X, \lambda_U, \lambda_V, \lambda_W)
= \frac{1}{2} \sum_{i}^N \sum_{j}^M I_{ij}(r_{ij} - u_{i}^T v_j)^2 \\
  + \frac{\lambda_U}{2} \sum_{i}^N||u_i||^2 \\
  + \frac{\lambda_V}{2} \sum_{j}^M ||v_j - cnn(W,X_j)||^2 \\
  + \frac{\lambda_W}{2} \sum_{k}^{|W_k|}||w_k||^2
\tag{6}

\\
(
\lambda_U = \frac{\sigma^2}{\sigma_U^2},
\lambda_V = \frac{\sigma^2}{\sigma_V^2},
\lambda_W = \frac{\sigma}{\sigma_W^2}
)
$$

この式(6)を最小化するような$U, V, W$を求める事を目指します。
ではどのように最適化していくのでしょうか?
どうやら、UとVとWの内２つを固定して、一つずつ最適化していく、Alternating Least Square(ALS)的なアプローチを取っていくようです！

## UとVの推定方法

UとVの推定方法に関しては、ALSと同じような印象です！
(今回は推定法がLeast SquareではなくMAPなので、AMAPとでもいうのでしょうか?)
user latent model $U$に関して、具体的には、W と Vを一時的に一定とみなすと、式（6）は **Uに関して二次関数**となります。そして、Uの最適解は、最適化関数$L$を$u_i$に関して以下のように微分するだけで、**閉形式(closed-form, 要するに解析的に解ける式？)で解析的に計算**することができます。

$$
u_i \leftarrow (VI_i V^T + \lambda_U I_K)^{-1}VR_i \tag{7}
$$

item latent model $V$も同様に、以下のような更新式を導出する事ができます。
($V$に関しては、$\lambda_V \cdot cnn(W, X_j)$が含まれているのが、通常のMFとの大きな違い！)

$$
v_j \leftarrow (U I_j U^T + \lambda_V I_K)^{-1}(UR_j + \lambda_V \cdot cnn(W, X_j)) \tag{8}
$$

ここで

- ユーザiについて
  - $I_i$ は$I_{ij} , (j=1, \cdots, M)$を対角要素とする対角行列。
  - $R_i$ はユーザiについて$(r_{ij})_{j=1}^M$とするベクトル。
    - つまり、ユーザiの各アイテムjに対する評価値が入ったベクトル!
- アイテムjについて
  - $I_j$と$R_j$の定義は、$I_i$と$R_i$のものと同様。
  - 式(8)はアイテム潜在ベクトル$v_j$を生成する際のCNNのDocument潜在ベクトル$s_j = cnn(W, X_j)$の効果を示している。
  - $\lambda_V$はバランシングパラメータ(要は重み付け平均みたいな?, 意味合いとしては正則化項のハイパラ?)になる。

## Wの推定方法

CNN内のパラメータWの推定方法に関しても、UとVを定数と仮定してWを推定する方針は同じです。

しかし、Wはmax pooling層や非線形活性化関数などCNNアーキテクチャの特徴と密接に関係している為、UやVのように解析的に解く事はできません。

それでも、上述したようにUとVが一時的に定数と仮定する事で、$L$は以下のように「$W$に関してL2正則化項を持つ二乗誤差関数」として解釈する事ができます。

$$
式(6)よりUとVが定数と仮定して\dots \\

\varepsilon(W) = \frac{\lambda_V}{2} \sum_{j}^M ||v_j - cnn(W,X_j)||^2 \\
+ \frac{\lambda_W}{2} \sum_{k}^{|W_k|}||w_k||^2 + constant
\tag{9}
$$

よって上式を最小化すべき損失関数(目的関数)として勾配降下法やら逆誤差伝搬やらを用いて、問題なく$W$も最適化する事ができそうです。

## Uの最適化→Vの最適化→Wの最適化→...を繰り返す...!

パラメータ全体の最適化処理($U, V, W$は交互に更新される)は収束されるまで(エポック数分)繰り返されます。

最終的には、最適化された$U, V, W$により、「アイテム$j$に対するユーザ$i$の未知の評価 $r_{ij}$」を推定する事ができます。

$$
r_{ij} \approx E[r_{ij}|u_i^T v_j, \sigma^2] \\
= u_i^T v_j = u_i^T \cdot (cnn(W, X_j) + \epsilon_j)
$$

ここまでで、ConvMFの理論はざっくり完了ですね。
やっぱり数式は世界共通なので、英語論文読んでいる最中に出てきてくれると救われた気分になります...！砂漠地帯の中のオアシス的な存在ですね：）

# MF部分の実装

以下、MFクラス定義の全文になります。

```python:matrix_factrization.py

from typing import NamedTuple, List, Optional
import pandas as pd
import numpy as np


class RatingData(NamedTuple):
  
    user: int
    item: int
    rating: float


class IndexRatingSet(NamedTuple):
    indices: List[int]
    ratings: List[float]


class MatrixFactrization(object):
    def __init__(self, ratings: List[RatingData], n_factor=300,
                 user_lambda=0.001, item_lambda=0.001, n_item: int = None):

        data = pd.DataFrame(ratings)
        self.n_user = max(data['user'].unique()) + 1
        self.n_item = n_item if n_item is not None else max(
            data['item'].unique()) + 1
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda
        # user factor の初期値
        self.user_factor = np.random.normal(
            size=(self.n_factor, self.n_user)
        ).astype(np.float32)
        # item factor の初期値
        self.item_factor = np.random.normal(
            size=(self.n_factor, self.n_item)
        ).astype(np.float32)
        # user factor 推定用(Rating Matrixにおける各userの非ゼロ要素)
        self.user_item_list = {user_i: v for user_i, v in data.groupby('user').apply(
            lambda x: IndexRatingSet(indices=x.item.values, ratings=x.rating.values)).items()}
        # item factor 推定用(Rating Matrixにおける各itemの非ゼロ要素)
        self.item_user_list = {item_i: v for item_i, v in data.groupby('item').apply(
            lambda x: IndexRatingSet(indices=x.user.values, ratings=x.rating.values)).items()}

    def fit(self, n_trial=5, additional: Optional[List[np.ndarray]] = None):
        """ UとVを推定するメソッド。

        Args:
            n_trial (int, optional): _description_. Defaults to 5.
            additional (Optional[List[np.ndarray]], optional):
            document factor vector =s_j = Conv(W, X_j)のリスト. Defaults to None.
        """
        # ALSをn_trial周していく！
        # (実際には、AMAP? = Alternating Maximum a Posteriori)
        for n in range(n_trial):
            self.update_user_factors()
            self.update_item_factors(additional)
            pass

    def predict(self, users: List[int], items: List[int]) -> np.ndarray:
        """user factor vectorとitem factor vectorの内積をとって、r\hatを推定

        Args:
            users (List[int]): _description_
            items (List[int]): _description_
        """
        ratings_hat = []
        for user_i, item_i in zip(users, items):
            # ベクトルの内積を計算
            r_hat = np.inner(
                self.user_factor[:, user_i],
                self.item_factor[:, item_i]
            )
            ratings_hat.append(r_hat)
        # ndarrayで返す。
        return np.array(ratings_hat)

    def update_user_factors(self):
        # 各user_id毎に繰り返し処理
        for i in self.user_item_list.keys():
            # rating matrix内のuser_id行の非ゼロ要素のindexとrating
            indices = self.user_item_list[i].indices
            ratings = self.user_item_list[i].ratings
            # item factor vector(ここでは定数)を取得
            v = self.item_factor[:, indices]
            # 以下、更新式の計算(aが左側の項, bが右側の項)
            a = np.dot(v, v.T)
            # aの対角成分にlambda_uを追加?
            a[np.diag_indices_from(a)] += self.user_lambda
            b = np.dot(v, ratings)  # V R_i

            # u_{i}の値を更新 a^{-1} * b
            self.user_factor[:, i] = np.linalg.solve(a, b)
            # 逆行列と何かの積を取る場合，
            # numpy.linalg.inv()じゃなくてnumpy.linalg.solve()の方が速いらしい...！

    def update_item_factors(self, additional: Optional[List[np.ndarray]] = None):
        # 各item_id毎に繰り返し処理
        for j in self.item_user_list.keys():
            # rating matrix内のitem_id列の非ゼロ要素のindexとrating
            indices = self.item_user_list[j].indices
            ratings = self.item_user_list[j].ratings
            # user factor vector(ここでは定数)を取得
            u = self.user_factor[:, indices]
            # 以下、更新式の計算(aが左側の項, bが右側の項)
            a = np.dot(u, u.T)
            a[np.diag_indices_from(a)] += self.item_lambda
            b = np.dot(u, ratings)
            # \lambda_V・cnn(W, X_j)の項を追加
            if additional is not None:
                b += self.item_lambda * additional[j]

            # v_{j}の値を更新 a^{-1} * b
            self.item_factor[:, j] = np.linalg.solve(a, b)

```

# 終わりに

今回は「Convolutional Matrix Factorization for Document Context-Aware Recommendation」の理解と実装のパート2として、ConvMFのMatrix Factorization部分の実装をまとめました。

次回は、ConvMFの特徴である、CNNのパートを実装し、記事にまとめていきます。
そしてこの一連のConvMFの実装経験を通じて、"Ratingデータ"＋"アイテムの説明文書"を活用した推薦システムについて実現イメージを得ると共に、"非常に疎な評価行列問題"や"コールドスタート問題"に対応し得る"頑健"な推薦システムについて理解を深めていきたいです。

間違っている点や気になる点があれば、ぜひコメントにてアドバイスいただけますと嬉しいです：）
