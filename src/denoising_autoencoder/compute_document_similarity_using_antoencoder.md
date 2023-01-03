## 参考

- https://medium.com/deep-learning-hk/compute-document-similarity-using-autoencoder-with-triplet-loss-eb7eb132eb38
- [論文:Article De-duplication Using Distributed Representations](http://gdac.uqam.ca/WWW2016-Proceedings/companion/p87.pdf)
  - 分散表現を用いた記事の重複排除

## title & overview

Compute Document Similarity Using Autoencoder With Triplet Loss

In this blog post, we will examine how to compute document similarity for news articles using Denoising Autoencoder (DAE) combined with a Triplet Loss function. This approach is presented in Article De-duplication Using Distributed Representations published by Yahoo! JAPAN and my Tensoflow implementation could be find here.このブログでは、**Denoising Autoencoder (DAE)** と**Triplet Loss関数**を組み合わせて、ニュース記事の類似性を計算する方法について説明します。この手法は、Yahoo! JAPANが発行したArticle De-duplication Using Distributed Representationsで紹介されており、私のTensoflowでの実装はこちらでご覧いただけます。

# Introduciton

The most common way of computing document similarity is to transform documents into TFIDF vectors and then apply any similarity measure e.g. cosine similarity to these vectors.
文書の類似度を計算する最も一般的な方法は、文書をTFIDFベクトルに変換し、これらのベクトルにコサイン類似度などの任意の類似度指標を適用することである。

However this approach has 2 disadvantages / limitations:

- This requires transforming every document into a vector of large dimensions (~10k) and it may not be ideal to compute similarity between these high-dimensional vectors under a tight time constraints which is the case in news domain even when these vectors are sparse. この手法は、**すべての文書を大きな次元（〜10k）のベクトルに変換**する必要があり、これらのベクトルが疎である場合でも、**ニュース領域で見られるような厳しい時間制約**の中で、これらの高次元ベクトル間の類似性を計算することは理想的ではないかもしれない。
- For news articles talking about similar or exact same event, this approach gives high similarity value. **類似した、あるいは全く同じ出来事について書かれたニュース記事**に対しては、このアプローチは高い類似度値を与える。But for articles of the same category talking about different event may end up with zero similarity value as there are no common keywords. しかし、**同じカテゴリの記事で異なる出来事について**話している場合は、**共通のキーワードが存在しない**ため、類似度がゼロになる可能性がある。 In other words, this approach does not preserve categorical similarity. 言い換えれば、このアプローチはカテゴリ的な類似性を保持しない。

And this is how Yahoo! JAPAN solves the limitations:

- By using DAE, we can compress the high-dimension TFIDF vector into low-dimension embeddings. DAEを用いることで、高次元のTFIDFベクトルを低次元の埋め込みに圧縮することができる。
- By using a triplet loss function, we can preserve categorical similarity between documents. 三重項損失関数を用いることで、文書間のカテゴリカルな類似性を保持することができる。

Now let go through DAE and Triplet loss separately. それでは、DAEとTriplet lossを別々に見ていきましょう。

# Denoisining Autoencoder

Autoencoder is basically a neural network doing unsupervised learning that tries to reconstruct input vector (encoder) from compressed embeddings (decoder).
オートエンコーダーは、基本的に教師なし学習を行うニューラルネットワークで、圧縮された**埋め込みデータ（デコーダー）から入力ベクトル（エンコーダー）を再構成**しようとするものである。

![](https://miro.medium.com/max/1100/1*pPzjE7tO0rizacSs0Bqk7w.webp)

In the graph, Layer L1 is the input vector, hidden layer L2 is the embeddings, Layer L3 is the reconstructed input. Usually L1 and L2 are called encoder, L2 and L3 are called decoder.
グラフでは、L1層が入力ベクトル、隠れ層L2が埋め込み、L3層が再構成された入力となります。通常、L1とL2はエンコーダ、L2とL3はデコーダと呼ばれます。

> The idea is that autoencoder will filter out noise and preserve only useful information during encoding so that decoder could use the compressed information to reconstruct the input entirely.
> オートエンコーダーは、**エンコード時にノイズを除去して有用な情報だけを残し**、**デコーダーが圧縮された情報を使って入力を完全に復元することができる**という考え方です。

And **denoising** autoencoder corrupts the input stochastically by setting some percentage of input elements to zero during training phase. This is motivated by making embeddings more robust to small irrelevant changes in input.
そして、**ノイズ除去**オートエンコーダは、**学習段階で入力要素のある割合をゼロにすることで、確率的に入力を破損**する。これは，**入力の小さな変化に対して埋め込みをより頑健にする**ことが動機となっている．

Below is the formulation of a traditional DAE:

$$
\tilde{x} \sim C(\tilde{x}|x) \\
h = f(W\tilde{x} + b) \\
y = f(W'h + b') \\
\theta = \argmin_{W, W', b, b'} \sum_{x \in X} L(y, x)
$$

where

- $x \in X$ is the original input vector,
- $f$ is the activation function,
- $L$ is the loss function,
- and $C$ is **corrupting distribution**.

The loss function used is usually squared loss or cross-entropy.
損失関数は、通常、二乗損失またはクロスエントロピが使用される。

# Triplet Loss

Triplet loss is first introduced in FaceNet: A Unified Embedding for Face Recognition and Clustering by Google which used to train faces’ embeddings.
トリプレットロスはFaceNetで初めて紹介された。[A Unified Embedding for Face Recognition and Clustering by Google](https://arxiv.org/abs/1503.03832)で初めて紹介され、顔の埋め込みを学習するのに使われた。

With triplet loss, 3 inputs are required during training phase:
トリプレットロスの場合、学習段階で3つの入力が必要です。

1. Anchor (メインの学習対象となるデータの事っぽい...! )
2. Positive (item which has the same label as anchor) (メインの画像と同じラベルを持つデータ)
3. Negative (item which has different label to anchor) (メインの画像と異なるラベルを持つデータ)

![](https://miro.medium.com/max/1100/1*0ABl3GR0T4CUXnKOBBQFuQ.webp)

During training, triplet loss will try to minimise distance (maximise similarity) between anchor and positive, while maximise distance (minimise similarity) between anchor and negative.
学習中、トリプレットロスは**アンカーとポジティブの間の距離を最小化（類似性を最大化）**し、**アンカーとネガティブの間の距離を最大化（類似性を最小化）**しようとする。

![](https://miro.medium.com/max/1100/1*oMUhbHBWoUliuOnnz34rOA.webp)

Above is an example from FaceNet, we can see images of same person have lower euclidean distance between their embeddings. Therefore, it shows that the learned embeddings preserve distance / similarity information of the original images.
上図はFaceNetの例ですが、同一人物の画像は埋め込み間のユークリッド距離が小さくなっていることがわかります。このことから、**学習された埋め込みは、元画像の距離・類似性情報を保持している**ことが分かります。

Yahoo! JAPAN applied the same idea to news articles’ embeddings to preserve categorical similarity, although the loss function is different compared to the one used by FaceNet.
Yahoo! JAPANでは、FaceNetで用いられている損失関数とは異なるものの、**同様の考え方をニュース記事の埋め込みに適用し、カテゴリの類似性を保持する**ようにしました。

# Denoising Autoencoder + Triplet Loss

The approach presented in Article De-duplication Using Distributed Representations used DAE with triplet loss to generate embeddings which preserve categorical similarity and here is how it is done.
Article De-duplication Using Distributed Representationsで紹介したアプローチは、トリプレットロスを伴うDAEを用いて、カテゴリカルな類似性を保持した埋め込みを生成するもので、その方法は以下のとおりです。

**Binary word count vector** is used as input to the DAE, and for every news article (anchor), we need another news article that has the same category label (positive) and another news article that has different category label (negative).
DAEの入力として**2値語数ベクトル**を用い、1つのニュース記事（アンカー）に対して、同じカテゴリラベル（ポジティブ）を持つ別のニュース記事と、異なるカテゴリラベル（ネガティブ）を持つ別のニュース記事が必要である。

### Notation

The input vector of (anchor, positive, negative) denoted as (x1, x2, x3)
(アンカー, 正, 負)の入力ベクトル(x1, x2, x3)
The embeddings / hidden layer of (anchor, positive, negative) denoted as (h1, h2, h3)
(アンカー, 正, 負)の埋め込み/隠れ層は(h1, h2, h3)とする。

### Loss function

To preserve the categorical similarity, we want h1．h2 > h1．h3 since x1 is more similar to x2 than x3. Hence the loss function of DAE is updated as follows:
x1はx3よりもx2に類似しているので、カテゴリカルな類似性を保持するために、h1.h2 > h1.h3としたい。したがって、DAE の損失関数は以下のように更新される。

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

From (1), h satisfies the property x=0 => h = f(b) — f(b) = 0, which means an article has no available information is not similar to any other articles.
(1)より、hはx=0 => h = f(b) - f(b) = 0の性質を満たすので、ある記事が他の記事と類似していない利用可能な情報を持っていないことを意味します。

![](https://miro.medium.com/max/640/1*PALVCePi4MConToT66fWDg.webp)

From (2), ø is the penalty function for article similarity and from the curve above, minimising ø means making h1．h2 > h1．h3
(2)より、φは記事の類似性に関するペナルティ関数で、上の曲線から、φを最小化することは、h1^T h2 > h1^T h3であることを意味する

> Note that the overall objective is not only to minimise ø but to minimise L and ø where L is elementwise cross entropy loss function. Without L, the network may end up with zero weights so h=0 for all articles. Hence L is necessary to make sure the embeddings h preserve useful information about the articles while ø make sure similar articles have similar embeddings.
> 全体の目的はφを最小化するだけでなく、**Lとφを最小化することであること**に注意してほしい。Lがなければ、ネットワークは重みがゼロになり、全ての記事に対してh=0になる可能性がある。したがって、**L は埋め込み h が記事に関する有用な情報を保持することを確認するために必要**であり、**φは類似の記事が類似の埋め込みを持つことを確認**する。

After training, we could apply cosine similarity on the embeddings to check how similar two articles are.
学習後、埋め込みに余弦類似度を適用することで、2つの記事がどの程度似ているかを確認することができる。

# Triplet Mining

As said before, to train the DAE with triplet loss, we need to prepare a set of triplet (anchor, positive, negative) as training data. So for every anchor, we need to map a positive and a negative. It could be a lot of works if this is done manually.
前述したように、トリプレット損失を用いたDAEを学習するためには、**学習データとしてトリプレット（アンカー、正、負）のセットを用意する必要**がある。つまり、**アンカーごとに、ポジティブとネガティブをマッピングする必要**がある。これを手作業で行うとなると、大変な労力となる。

This could actually be done automatically with **triplet mining**.
実はこれ、**トリプレットマイニング**で自動的にできるんです。

With triplet mining, we only need to provide anchor and label of the anchor to DAE during training. And it will find out all possible triplets by constructing a 3-dimensional matrix. A much more detailed explanation and Tensorflow implementation could be find here.
トリプレットマイニングでは、**学習時にアンカーとアンカーのラベルをDAEに与えるだけでよい**。そして、3次元の行列を構築することで、可能性のあるトリプレットを全て探し出すことができる。より詳細な説明とTensorflowの実装はこちらをご覧ください。

> Note that triplet mining is not mentioned in the paper, but I found this saves a lot of effort to create the triplet manually.
> なお、トリプレットマイニングについては論文では触れられていませんが、トリプレットを手動で作成する手間が大幅に省けることがわかりました。

# Evaluation

Let test the performance of this approach on some open dataset.
この手法の性能を、いくつかのオープンデータセットで検証してみよう。

## dataset

A dataset which contains 10348 news articles with categories labels and story labels is used for evaluation. The dataset is available at https://www.kaggle.com/louislung/uci-news-aggregator-dataset-with-content

評価には、カテゴリラベルとストーリーラベルを持つ10348のニュース記事を含むデータセットを使用した。

## Code

I have implemented this autoencoder using Tensorflow and you may find the implementation [here in my github](https://github.com/louislung/DAE_RNN_News_Recommendation).

## Setting

The embeddings is trained with following autoencoder network’s settings:
埋め込みは、以下のオートエンコーダーネットワークの設定で学習されます。

- Input vector = Binary vector corresponding to each word in the vocabulary ([Scikit learn’s Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) is used to prepare the input vector) 入力ベクトル = 語彙の各単語に対応する2値ベクトル（入力ベクトルの準備には**Scikit learnのCount Vectorizer**を使用します。）
  - ベクトルの長さはコーパス内のトークン数(idfの低いトークンは取り除くかも)
  - ベクトルの要素は、document内における各tokenの出現回数を保持する。
- Dimension of Input layer = 10000, Embeddings layer =50, Output layer = 10000
- Optimization Algorithm = Adam
- Corruption Function = Every element of input vector is masked (set to zero) randomly with probability 0.3. Corruption Function = 入力ベクトルの各要素を確率0.3でランダムにマスク（0に設定）する。
- Alpha = 10 (hyper-parameter of the loss function)
- Loss function of autoencoder = Cross-Entropy
- Batch size = 100
- Number of epochs = 200

The dataset is splitted into two: 5000 for training and 5348 for testing.
データセットは、トレーニング用の5000個とテスト用の5348個に分割されています。

# Result

We can evaluate the performance of the embeddings by checking the AUROC of cosine similarity of similar articles (article of the same category or same story).
埋め込みの性能は、類似記事（同じカテゴリーや同じストーリーの記事）のコサイン類似度のAUROCを確認することで評価することができる。

We use both embeddings and Tfidf vector to calculate the cosine similarity and plot the ROC below.

埋め込みとTfidfベクトルの両方を用いてコサイン類似度を計算し、以下にROCをプロットする。

![](https://miro.medium.com/max/720/1*QpIwdNE__luz9XO3Hr_o7w.webp)

The trained embeddings is able to preserve categorical similarity between articles (higher auroc achieved in both training and testing set).
学習された埋め込みは、**記事間のカテゴリ的な類似性**を保持することができる（学習セットとテストセットの両方で高いオーロクを達成した）。

![](https://miro.medium.com/max/720/1*jcjZ4XU7-NE8a5sj89NEpg.webp)

While Tfidf is still good at identifying similar articles at story level as the embeddings loses some information due to compression.
埋め込みは圧縮によりいくつかの情報が失われる為、**ストーリーレベルでの類似記事の特定にはTfidfの方がまだGoodっぽい**。

We can see that embeddings trained with the approach is highly compressed (dimensions reduced from 10000 to 50) and is able to preserve categorical similarity.
この手法で学習した埋め込みは、高度に圧縮され（**次元は1万から50に減少**）、カテゴリ的な類似性を保持できていることがわかる。

# Wrapping Up

We have seen how to train embeddings which preserve categorical similarity using a variant of denoising autoencoder by introducing triplet loss into the model.
ここまで、三重項損失をモデルに導入したノイズ除去オートエンコーダの変種を用いて、カテゴリ的類似性を保持した埋め込みを学習する方法を見てきました。

This approach of finding similar articles may be preferred in news domain where there is tight time constraint (latest news article should be displayed to users within short time).
時間的制約のあるニュース分野では、この類似記事検索のアプローチが好まれるかもしれない（最新のニュース記事を短時間でユーザに表示する必要がある）。

Moreover, **a lot of tasks could be done after having these embeddings for every news articles**:
さらに、**このエンベッディングをニュース記事ごとに用意することで、さまざまな作業ができるようになります**。

- content based recommendation (Embedding-based News Recommendation for Millions of Users)
- Topic model
- Articles grouping

Thanks you for reading, if you have any feedback please let me know in the comment section!
お読みいただきありがとうございました。もし何かご意見がありましたら、コメント欄でお知らせください。
