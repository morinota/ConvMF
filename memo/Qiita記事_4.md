title: 評価行列とアイテムの説明文書を活用した推薦システム「ConvMF」を何とか実装していきたい!③CNNパートの実装

# 参考

- [元論文](https://dl.acm.org/doi/10.1145/2959100.2959165)
  - [figure and table](https://www.semanticscholar.org/paper/Convolutional-Matrix-Factorization-for-Document-Kim-Park/af9c4dda90e807246a2f6fa0a922bbf8029767cf)
- nishiba様のConvMF実装例(Chainerをお使いになってました！)
  - [エムスリー様のテックブログ](https://www.m3tech.blog/entry/2018/03/07/122353)
  - [nishiba様のgithub](https://github.com/nishiba/convmf)
- [自然言語処理におけるEmbeddingの方法一覧とサンプルコード](https://yukoishizaki.hatenablog.com/entry/2020/01/03/175156)
- けんごのお屋敷 様の記事
  - [自然言語処理における畳み込みニューラルネットワークを理解する](https://tkengo.github.io/blog/2016/03/11/understanding-convolutional-neural-networks-for-nlp/)

# はじめに

KaggleのPersonalized Recommendationコンペに参加して以降、推薦システムが自分の中で熱くなっております。以前、Implicit Feedbackに対するモデルベースの協調フィルタリング(Matrix Factorization)の論文を読んで実装してみて、今度は更に実用的(?)で発展的な手法を触ってみたいと思い、「Convolutional Matrix Factorization for Document Context-Aware Recommendation」を読みました。この論文では、Matrix Factorizationによるモデルベース協調フィルタリングに、CNNを用いてアイテムの説明文書の情報を組み合わせる ConvMF(Convolutional Matrix Factorization)を提案しています。

今実装中ですが、なかなかPytorchと仲良くなれず、苦戦しております...。(ちなみに元論文はKerasで実装しておりました!)

パート3とした本記事では、ConvMFにおけるCNNパートの実装についてまとめています。**アイテムjの説明文書$X_j$を受け取って、document latent vector $s_j$を出力する$CNN(W, X_j)$**の事ですね：）

本記事以前のパートは、以下のリンクを御覧ください。

- [評価行列とアイテムの説明文書を活用した推薦システム「ConvMF」を何とか実装していきたい!①MFパートの実装](https://qiita.com/morinota/items/d84269b7b4bf55d157d8)
- [評価行列とアイテムの説明文書を活用した推薦システム「ConvMF」を何とか実装していきたい!②MFパートの実装](https://qiita.com/morinota/items/6bcad7dfe9f406364bfd)

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
特にConvMFでは、アイテムjの説明文書ベクトル$X_j$を考慮して$V_j$をを推定する点が大きな特徴になります。

## ConvMFにおけるパラメータ推定法

ConvMFでは、パラメータ($U, V, W$)を最適化する為に、MAP推定(maximum a posteriori estimation)を行います。

事後分布の式を対数化してマイナスを掛け、いい感じに変形($\sigma^2$で割る!)と、以下のようになりますね([前パート参照](https://qiita.com/morinota/items/d84269b7b4bf55d157d8))。

$$
L(U,V,W|R, X, \lambda_U, \lambda_V, \lambda_W)
= \frac{1}{2} \sum_{i}^N \sum_{j}^M I_{ij}(r_{ij} - u_{i}^T v_j)^2 \\
  + \frac{\lambda_U}{2} \sum_{i}^N||u_i||^2 \\
  + \frac{\lambda_V}{2} \sum_{j}^M ||v_j - cnn(W,X_j)||^2 \\
  + \frac{\lambda_W}{2} \sum_{k}^{|W_k|}||w_k||^2

\\
(
\lambda_U = \frac{\sigma^2}{\sigma_U^2},
\lambda_V = \frac{\sigma^2}{\sigma_V^2},
\lambda_W = \frac{\sigma}{\sigma_W^2}
)
$$

この式を最小化するような$U, V, W$を求めるわけです。

ConvMFの学習では、UとVとWの内２つを固定して、一つずつ最適化していく、Alternating Least Square(ALS)的なアプローチを取っていきます。

user latent matrix $U$とitem latent matrix $V$ の推定方法に関しては、ALS同様に、**閉形式(closed-form, 要するに解析的に解ける式？)で解析的に計算**することができます。

$$
u_i \leftarrow (V I_i V^T + \lambda_U I_K)^{-1}VR_i \tag{7}
$$

$$
v_j \leftarrow (U I_j U^T + \lambda_V I_K)^{-1}(UR_j + \lambda_V \cdot cnn(W, X_j)) \tag{8}
$$

$V$に関しては、$\lambda_V \cdot cnn(W, X_j)$が含まれているのが、通常のMFとの大きな違いであり、ConvMFの特徴ですね。

ここで

- ユーザiについて
  - $I_i$ は$I_{ij} , (j=1, \cdots, M)$を対角要素とする対角行列。
  - $R_i$ はユーザiについて$(r_{ij})_{j=1}^M$とするベクトル。
    - つまり、ユーザiの各アイテムjに対する評価値が入ったベクトル!
- アイテムjについて
  - $I_j$と$R_j$の定義は、$I_i$と$R_i$のものと同様。
  - 式(8)はアイテム潜在ベクトル$v_j$を生成する際のCNNのDocument潜在ベクトル$s_j = cnn(W, X_j)$の効果を示している。
  - $\lambda_V$はバランシングパラメータ(要は重み付け平均みたいな?, 意味合いとしては正則化項のハイパラ?)になる。

CNN内のパラメータWの推定方法に関しても、UとVを定数と仮定してWを推定する方針は同じです。以下の損失関数を最小化するようなパラメータ$W$を求めていきます。CNNに関しては次回のパートで実装をまとめていきます。

$$
\varepsilon(W) = \frac{\lambda_V}{2} \sum_{j}^M ||v_j - cnn(W,X_j)||^2 \\
+ \frac{\lambda_W}{2} \sum_{k}^{|W_k|}||w_k||^2 + constant \tag{9}
$$

最終的には、最適化された$U, V, W$により、「アイテム$j$に対するユーザ$i$の未知の評価 $r_{ij}$」を推定する事ができます。

$$
r_{ij} \approx E[r_{ij}|u_i^T v_j, \sigma^2] \\
= u_i^T v_j = u_i^T \cdot (cnn(W, X_j) + \epsilon_j)
\tag{10}
$$

ここまでで、簡単なConvMFの理論の復習は完了です。

# $s_j = CNN(W, X_j)$についての詳細な説明

ConvMFにおけるCNNアーキテクチャの目的は、アイテムのdocuments(文書)からdocument latent vectors(文書潜在ベクトル)を生成し、それと撹乱項($\epsilon$)を用いてitem latent models(アイテム特徴行列)を構成する事です。

![](https://d3i71xaburhd42.cloudfront.net/9e91c370a8fae365c731947ad9178fb3788f6593/500px/3-Figure2-1.png)

上の図はConvMFで用いられるCNNアーキテクチャを示しており、

- 1)Embedding layer
- 2)Convolution layer
- 3)Pooling layer
- 4)Output layer

の4層から構成されています。

## Embedding Layer(埋め込み層)

Embedding layerでは、生のDocumentを、次のConvolution layerへ入力する準備として、**Documentを表す密な数値行列**に変換します。

- 具体的には、Documentを$l$個の単語の列と見なし、Document中の各単語のベクトルを連結して行列として表現する。(場合によっては、各"文字"を最小単位として扱う事もありうる??)
- 単語ベクトルはランダムに初期化されるか、事前に学習された単語埋め込みモデルで初期化される。(ex. Word2Vec??)
- この単語ベクトルは、さらに最適化処理によって学習される。

そしてこの場合、ある一つのDocumentのDocument Matrix$D \in \mathbb{R}^{p \times l}$は次のようになります。

$$
D_j = [
\begin{array}{cc}
   & | & | & | & \\
  \cdots & w_{i-1} & w_{i} & w_{i+1} & \cdots \\
   & | & | & | & \\
\end{array}
] = \text{embedding layer}(X_j)
$$

ここで

- $l$：Documentの長さ(最小単位が"単語"の場合、単語数)
- $p$：各単語ベクトル$w_i$の埋め込み次元の大きさ
- $D_j$：アイテムjのDocument $X_j$を変換して得られたDocument Matrix

## Convolution Layer 畳み込み層

Convolution LayerはDocument Matrix $D_j$から文書の特徴を抽出します。
一般にCNNはコンピュータビジョンの分野で多く活用されますが、自然言語に対しても使われます。

画像データに対するCNNとは異なり、自然言語に対するCNNでは、Kernel(i.e. スライド窓関数, filter, feature detector)の幅が、Document Matrixの幅と合致します。
つまり本記事においては、Kernelの幅は「各単語ベクトル$w_i$の埋め込み次元の大きさ$p$」と合致する、という事ですね。

Kernelの高さに関しては、元論文では[3, 4, 5]の三種類を使用しており、本記事でもそれに従って実装します。

k番目のConvolution layerから得られる document feature $c_{i}^{k}$は、k番目のShared Weight$W_{c}^{k} \in \mathbb{R}^{p\times ws}$(=これがKernel！) によって特徴抽出されます。

$$
c_i^k \in \mathbb{R}= f(W_c^k * D_{(:, i:(i+ws -1))} + b_c^k)
$$

ここで、各記号は以下の意味です。

- $i$はDocument Matrix中の各単語ベクトルを表す添字。
- $k$はk番目のShared Weight（＝カーネル??）を表す添字。
- $c_i^k$: ある任意のDocument Matrix $D_j$において、i番目の単語ベクトル周辺をk番目のShared Weight（＝カーネル??）によって畳み込んで得られた実数値。
- $*$は畳み込み演算子(Convolution operator)
  - Document Matrixの一部にKernelが要素毎に掛け合わされ、それらの和が計算されたのが $W_c^k * D_{(:, i:(i+ws -1))}$
  - つまり、**アダマール積の和をとってる**って事ですね！
- $b_c^k$は$W_c^k$のバイアス。
  - つまり、アダマール積の和に足される定数項ですね！
- $f$は非線形活性化関数(non-linear activation function)
  - シグモイド、tanh、rectified linear unit (ReLU)などの非線形活性化関数のうち、最適化の収束が遅く、局所最小値が悪くなる可能性のあるvanish gradientの問題を避けるためにReLUを用いる
- 

# 終わりに

今回の記事では「Convolutional Matrix Factorization for Document Context-Aware Recommendation」の理解と実装のパート2として、ConvMFのMatrix Factorization部分の実装をまとめました。
今回の実装を経て、Matrix Factorizationの学習処理に対して、何らかの方法(ex. 処理を並列化, Cython?, etc.)で高速化を図る必要があるのかなと感じました。

次回は、ConvMFの特徴である、CNNのパートを実装し、記事にまとめていきます。
そしてこの一連のConvMFの実装経験を通じて、"Ratingデータ"＋"アイテムの説明文書"を活用した推薦システムについて実現イメージを得ると共に、"非常に疎な評価行列問題"や"コールドスタート問題"に対応し得る"頑健"な推薦システムについて理解を深めていきたいです。

理論や実装において、間違っている点や気になる点があれば、ぜひコメントにてアドバイスいただけますと嬉しいです：）
