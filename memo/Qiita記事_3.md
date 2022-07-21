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
- [PyTorchを使ってCNNで文章分類を実装してみた](https://qiita.com/m__k/items/6c39cfe7dfa99102fa8e)
- [A Complete Guide to CNN for Sentence Classification with PyTorch](https://chriskhanhtran.github.io/posts/cnn-sentence-classification/)

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

# $s_j = CNN(W, X_j)$についてまとめる前に...NLPのCNNについて確認

$s_j = CNN(W, X_j)$の実装の前に、自然言語処理における畳み込みニューラルネットワークを理解していきます。

## 畳み込みとは？

- 畳み込みについては、行列に適用されるスライド窓関数 (sliding window function) として考えるとわかりやすいらしい...。
  - (けんごのお屋敷 様のgifを貼り付けさせていただきました)
  - ![](https://tkengo.github.io/assets/img/understanding-convolutional-neural-networks-for-nlp/convolution_schematic.gif)
  - スライド窓は**カーネル(kernel)**や**フィルタ(filter)**または**特徴検出器(Feature Detector)**等と呼ばれる。
- - 上の例では3×3のスライド窓関数を使っており、そのスライド窓関数の値と行列の値を**要素毎にかけ合わせ**、それらの値を**合計したもの**を、Convoloved Featureの**一つの要素とする**。
  - ＝＞つまり、**「スライド窓関数 & 畳み込み対象の行列の、ウィンドウサイズと合致する一部分」のアダマール積の和が**、Convolved Featureの要素の一つになる。
- この操作を、行列全体をカバーするように、スライド窓関数をスライドさせながら行い、全体の畳み込み特徴(Covolved Feature)を取得する。

## 畳み込みニューラルネットワークとは？

- CNN は、ReLU や tanh のような非線形な活性化関数を通した、いくつかの畳み込みの層のこと
- 伝統的な順伝搬型ニューラルネットワークでは、**それぞれの入力ニューロンは次の層のニューロンにそれぞれ接続**されており、これは**全結合層**や**アフィン層**とも呼ばれる。
- しかし CNN ではそのようなことはせずに、ニューロンの出力を計算するのに畳み込みを使う。これによって、**入力となるニューロンのある領域が、それぞれ対応する出力のニューロンに接続**されているような、**局所的な接続**ができることになる。
- 各層は別々の異なるフィルタを適用し、(これは一般的には 100 〜 1000 程度の数になるが) それらを結合する。(この結合する層を**プーリング層(subsampling)**と呼ぶ。)

- CNNの学習フェーズでは、解決したいタスクに適応できるように、**フィルタの値(=スライド窓行列=カーネルの各要素！)を自動的に学習していく**。
- 例えば画像分類の話でいうと、
  - CNNは最初の層で生のピクセルデータからエッジを検出する為の学習を進め、
  - そのエッジを使って今度は次の層で単純な形状を検出し、
  - 更により深い層ではその形状を使ってより高レベルな特徴、つまり顔の形状等の特徴を検出するようになる。
  - そして最後の層は、高レベルな特徴を使った(=入力とした)分類器になる。

## これをどうやってNLPへ適用するのか？

- 画像分類では入力は画像のピクセル行列になるが、**ほとんどの NLP タスクではピクセル行列の代わりに、行列で表現された文章または文書が入力**となる。
  - 行列の**各行は 1 つのトークンに対応**しており、一般的には単語がトークンになることが多いが、文字がトークンのケースもある。
  - ＝＞すなわち、**各行は単語を表現するベクトル**。
  - 普通、これらのベクトルは word2vec や GloVe のような低次元な単語埋め込み表現 (word embeddings) を使う。one-hot ベクトルのケースもある。
  - 例えば、**100 次元の単語埋め込みを使った 10 単語の文章があった場合、10x100 の行列となる**。これがNLPにおける"画像"になる。
- コンピュータビジョンでは、フィルタは画像のある区画上をスライドしていくが、NLP では一般的に**行列の行全体 (つまり単語毎) をスライドする**フィルタを使う。
  - つまり、**フィルタの幅(横幅)は入力となる行列の幅と同じ**になる!
  - つまり、NLPの場合のフィルタ(スライド窓, カーネル, 特徴検出器)は横方向にはスライドせず、縦方向にのみスライドしていく！
- フィルタの高さ(縦幅)は様々だが、一般的には2~5くらい?

これらのことを加味すると NLP の畳み込みニューラルネットワークはこんな感じになる。(けんごのお屋敷 様のgifを使用させていただきました!)

![](https://tkengo.github.io/assets/img/understanding-convolutional-neural-networks-for-nlp/convolutional-neural-network-for-nlp-without-pooling.png)

上図の解釈

- 文章分類のための畳み込みニューラルネットワーク (CNN) のアーキテクチャを説明した図.
- 畳み込み層
  - この図には 2、3、4 の高さをもったフィルタが、それぞれ 2 つずつ(計6)ある。
  - 各フィルタは文章の行列上で**畳み込み**を行い、特徴マップ(〇行1列のやつ！)を生成する。
- プーリング層
  - 各特徴マップに対して最大プーリングをかけていき、**各特徴マップの中で一番大きい値を記録していく**。(=Max pooling)
  - そして、全6つの特徴マップから単変量な特徴 (**univariate feature**) が生成され、それら 6 つの特徴は結合されて、それが最後から 2 番目の層になる。
- 全結合層(アフィン層)(一層？出力層？)
  - 一番最後の softmax 層では先程の特徴を入力として受け取り、文章を分類する。
  - ここでは二値分類を前提としているので、最終的には 2 つの出力がある。

## CNN のハイパーパラメータ

- スライド窓関数のサイズ(畳み込み幅のサイズ)
- wide convolution か narrow convolution か
- ストライドのサイズ
- プーリング層の選択(メジャーなのがmax pooling?)
- チャンネル数

### 畳み込み幅のサイズ

- 最初に畳み込みの説明をした時、フィルタ(スライド窓、カーネル、特徴検出器)を適用する際の詳細について説明を飛ばしたものがある。
  - 行列の真ん中辺りに 3x3 のフィルタを適用するのは問題ないが、それでは**フチの辺りに適用する場合はどうなんだろうか**??
  - 行列の左側にも上側にも隣接した要素がないような、たとえば行列の最初の要素にはどうやってフィルタを適用すればよいのだろうか？
- そういった場合には、**ゼロパディング**が使える！
  - 行列の外側にはみ出してしまう要素は全て 0 で埋めるのである。
  - こうすることで、入力となる行列の全要素にわたってフィルタを適用することができる。
  - **ゼロパディングを行うことは wide convolution**とも呼ばれ、逆に**ゼロパディングをしない場合は narrow convolution**と呼ばれる。
- 以下は1次元での例：
  (Narrow Convolution と Wid Convolution。フィルタのサイズは 5 で、入力データのサイズは 7。)
  (けんごのお屋敷 様の画像を使用させていただきました!)

  - ![](https://tkengo.github.io/assets/img/understanding-convolutional-neural-networks-for-nlp/narrow_vs_wide_convolution.png)

- **入力データのサイズに対してフィルタサイズが大きい時**には wide convolution が有用。
  - 上記の場合、narrow convolution は出力されるサイズが $(7 - 5) + 1 = 3$ になり、wide convolutin は $(7 + 2 * 4 - 5) + 1 = 11$ になる。
  - 一般化すると、**wide convolutionの場合の出力サイズ(畳み込み層の出力=特徴マップの大きさ?)**は $n_{out}=(n_{in} + 2*n_{padding} - n_{filter}) + 1$

### ストライド

- フィルタを順に適用していく際に、**フィルタをどれくらいシフトするのか**という値。
  - これまでに示してきた例は全てストライド=1 で、フィルタは重複しながら連続的に適用されている。
- ストライドを大きくするとフィルタの適用回数は少なくなって、出力のサイズも小さくなる。
- 以下のような図が Stanford cs231 にあるが、これは 1 次元の入力に対して、ストライドのサイズが 1 または 2 のフィルタを適用している様子。
  (畳み込みのストライドのサイズ。左側のストライドは 1。右側のストライドは 2)
- ![](https://tkengo.github.io/assets/img/understanding-convolutional-neural-networks-for-nlp/stride.png)
- **普通、文書においてはストライドのサイズは 1**だが、ストライドのサイズを大きくすることで、例えばツリーのような 再帰型ニューラルネットワーク と似た挙動を示すモデルを作れるかもしれない...!

### プーリング層

- 畳み込みニューラルネットワークの鍵は、畳み込み層の後に適用されるプーリング層
  - プーリング層は、入力をサブサンプリングする。
- 最も良く使われるプーリングは、各フィルタの結果(=各畳み込み層の出力=特徴マップ)の中から最大値を得る操作。＝＞**Max Pooling**
  - ただ、畳み込み結果の行列全体にわたってプーリングする必要はなく、指定サイズのウィンドウ上でプーリングすることもできる。
  - たとえば、以下の図は 2x2 のサイズのウィンドウ上で最大プーリングを実行した様子。
  - ![](https://tkengo.github.io/assets/img/understanding-convolutional-neural-networks-for-nlp/max-pooling.png)
  - (**NLP では一般的に出力全体にわたってプーリングを適用する**。つまり各フィルタ(=>特徴マップ)からは **1 つの数値**が出力されることになる。)

### チャンネル数

- チャンネルとは、**入力データを異なる視点から見たもの**と言える。
  - 画像認識での例を挙げると、普通は画像は RGB (red, green, blue) の 3 チャンネルを持っている。
  - 畳み込みはこれらのチャンネル全体に適用でき、その時のフィルタは各チャンネル毎に別々に用意してもいいし、同じものを使っても問題ない。
- NLP では、**異なる単語埋め込み表現 (word2vec や GloVe など) でチャンネルを分けたり**、同じ文章を**異なる言語で表現**してみたり、また異なるフレーズで表現してみたり、という風にして**複数チャンネルを持たせる**ことができそう...!

# NLPタスクにおけるCNNを実装してみる(CNNによるDocumentの２クラス分類)

さてここから、CNNによるDocumentの２クラス分類をPytorchで実装していきます。
[A Complete Guide to CNN for Sentence Classification with PyTorch](https://chriskhanhtran.github.io/posts/cnn-sentence-classification/)を参考に（ほぼ写経でコメントアウトをはさみまくりながら）実装します。

ConvMFのCNNパート$s_j = CNN(W, X_j)$に関しても、出力次元数と損失関数の形以外は、この実装と変わらないので、今回実装するスクリプトを調整すれば、すぐにできるはずです...!


## データの準備

今回は、パート1⃣で用意したデータセットの内、`descriptions.csv`のみを使用します。
また、文章をtokenizeする為に、fastTextをダウンロードしておきます。

```python:main.py
TEXT_FILE = r'data\descriptions.csv'
FAST_TEXT_PATH = r'fastText\crawl-300d-2M.vec'


def load_data():
    texts_df = pd.read_csv(TEXT_FILE)
    return texts_df


def load_word_vector():
    URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
    FILE = "fastText"

    if os.path.isdir(FILE):
        print("fastText exists.")
    else:
        print('please download fastText.')
```

## tokenizeの処理
tokenizeとは、文章を何らかの単位に区切る事を意味します。

今回は映画の説明文に対して、「単語」をtokenとしてtokenizeします。


# 終わりに

NLPにおけるCNNだけで長くなってしまったので、次回は今回実装したCNNをConvMF用にアレンジしていきます。

今回の記事では「Convolutional Matrix Factorization for Document Context-Aware Recommendation」の理解と実装のパート3として、ConvMFのCNN部分の実装をまとめました。

次回は、ConvMFの特徴である、CNNのパートを実装し、記事にまとめていきます。
そしてこの一連のConvMFの実装経験を通じて、"Ratingデータ"＋"アイテムの説明文書"を活用した推薦システムについて実現イメージを得ると共に、"非常に疎な評価行列問題"や"コールドスタート問題"に対応し得る"頑健"な推薦システムについて理解を深めていきたいです。

理論や実装において、間違っている点や気になる点があれば、ぜひコメントにてアドバイスいただけますと嬉しいです：）
