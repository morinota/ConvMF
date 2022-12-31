<!-- title: Online Triplet MiningをPytorchで実装してみた-->

# はじめに

本記事は、Triplet Loss and Online Triplet Mining in TensorFlowを読んで**Triplet Miningが何たるか**の理解を深めるとともに、**PytorchでOnline Triplet Miningの実装を試みた**ものです:)
↑の技術記事は、Olivier MoindrotさんによるTriplet Loss, Triplet Miningの解説と、TensorflowによるTriplet Minignの実装例が紹介されています。

# そもそもTriplet Lossとは?

## Triplet Lossの始まり

Triplet Lossという方法論(損失関数)は、2015年にGoogleから出された顔画像認識の論文 [A Unified Embedding for Face Recognition and Clustering]()で初めて紹介されました。
この論文では**Online Triplet Mining**という方法を用いて、顔画像の埋め込みベクトルを学習する為の**新しいアプローチ**について述べています。

通常、教師あり学習による分類タスクでは、**固定数のクラス(ex. Aさん、Bさん、...)**があり、出力層の活性化関数としてソフトマックス関数、損失関数としてクロスエントロピー等を用いてネットワークを学習します。

しかし、場合によっては、**クラスの数を可変にする必要**があります。
例えば、顔認識において2つの未知の顔を比較して、**同一人物の顔かどうか**を判断する必要があります。

![](https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png)

図. Triplet loss on two positive faces (Obama) and one negative face (Macron)

顔画像認識においてTriplet Lossは、各顔の特徴を表現する為のよりよい埋め込みベクトルを学習する為の方法です。
理想的な埋め込み空間では、**同一人物の顔は近く**にあり、異なる人物同士の埋め込みベクトル間では**よく分離したクラスタを形成**しているはずです。

(つまり、**同じ人物の複数の顔写真の埋め込みベクトルは似ている**ように、**違う人物同士の埋め込みベクトル達は似ていない**ように、って事ですね...！
そして、上述した特徴を持った理想的な埋め込みベクトルを作る為に、Triplet Lossを活用できると！)

## Triplet Lossの定義

Triplet Lossは損失関数の一種なので、定義式があります。

上のサブセクションで述べた様に、Triplet Lossを適用する目的は、生成される埋め込みベクトル達が以下の条件を満たすようにすることですよね。

- (条件1)**同じラベル**を持つ2つのサンプルは、埋め込み空間においてその**埋め込みベクトル同士が近接**している.
- (条件2)**ラベルが異なる**2つのサンプルでは、**埋め込みベクトル同士が離れて**いる。

この要件を定式化するために、Triplet Lossという損失関数は、以下のように**埋め込みベクトルのTriplet(i.e. 三つ組ですね!)**に対して定義されます。

- an anchor: アンカー. "メインの埋め込みベクトル"というイメージですかね...!
- a positive of the same class as the anchor: アンカーと同じラベルを持つ、ある一つの埋め込みベクトル
- a negative of a different class: アンカーと異なるラベルを持つ、ある一つの埋め込みベクトル

埋め込み空間(embedding space)における2つのベクトル間の距離を$d()$と表すと、Triplet(3つ組) $(a, p, n)$に関する損失関数$L$は以下の様に定義されます:

$$
L = \max(d(h_a,h_p) - d(h_a, h_n) + \text{margin}, 0)
$$

ここで$h_a, h_p, h_n$はそれぞれ、任意のTripletにおけるanchorの埋め込みベクトル、positiveの埋め込みベクトル、及びnegativeの埋め込みベクトルを表しています。
marginはハイパーパラメータで、**大きく指定するほどクラス間の埋め込みベクトルの距離を広げさせるような定数**だと理解しています。(理解が間違っていたらぜひ指摘してください...!)

上の損失関数を最小化する事で、$d(h_a,h_p)$を0に押し上げ(=条件1)、$d(h_a,h_n)$が$d(h_a,h_p) + \text{margin}$ よりも大きくなるように(=条件2)します。

## Triplet lossの値に基づく3種類のTriplet

前のサブセクションで述べたTriplet lossの定義に基づき、Tripletを**3つのカテゴリー**に分ける事ができます。

- **easy triplets**:
  - $d(h_a,h_p) + \text{margin} < d(h_a,h_n)$により、損失が0になるtriplet.
- **hard triplets**:
  - 埋め込み空間において、negativeの方がpositiveよりもanchorに近いtriplet.
  - i.e $d(h_a,h_n) <d(h_a,h_p)$
- **semi-hard triplets**:
  - 埋め込み空間において、positiveの方がnegativeよりもanchorに近いが、その差がmarginよりも小さい為、損失の値が正になるTriplet.
  - i.e. $d(a,p) < d(a,n) < d(a,p) + \text{margin}$

また、この3つのカテゴリーの定義はそれぞれ、**anchorとpositiveに対するnegativeの相対的な位置関係**によって決まります。
したがって、**Tripletの3つのカテゴリ**を**negativeの位置の3つのカテゴリ**に拡張して、hard negative、semi-hard negative、easy negativeと表す事もできます。
下図は、「anchorとpositiveに対するnegativeの相対的位置」に対応する、埋め込み空間の3つの領域を示しています。下図においてnegativeがどこに置かれるかによって、Tripletのカテゴリが一つに定まるという事ですね...!

![](https://omoindrot.github.io/assets/triplet_loss/triplets.png)

図. anchorとpositiveが与えられた状態でのnegativeの相対的位置と、三種類のTripletカテゴリ(=negativeカテゴリ)の関係

**どのようなTripletで学習させるか**は、測定値に大きく影響します。
Tripletの元祖であるFacenet論文では、**anchorとpositiveのペアごとにランダムにsemi-hardなnegativeを選び**、これらのTripletで学習しています。

# Triplet Mining

上のセクションで、**埋め込みベクトルのTriplet(3つ組)に対する損失関数(i.e. Triplet Loss)**を定義し、定義式を元にTripletを三種類のカテゴリに分類、そしてあるTripletは他のTripletよりも学習に有用であることを見てきました。
問題は、**これらのTripletをどのようにサンプリングする**か、つまり「**マイニング**」するかです。

Triplet MiningはOfflineとOnlineの大きく二種類に分かれます.

## Offline triplet mining

Offline Triplet Miningでは、例えば**各エポック**の最初に、オフラインでトリプレットを見つけます。
**学習セット上のすべての埋め込みベクトルを計算**し、hardまたはsemi-hardなTripletのみを選択します。
そして、これらのTripletを用いて1つのエポックの学習を実行します。

具体的には、Tripletのリスト$(i,j,k)$を作成します。(i.e. **埋め込みベクトルを計算する前に、予め考えられるTripletを取得している**事がオフラインの特徴っぽい...！)

つまり、$B$個のTripletを得るために$3B$個の埋め込みベクトルを計算し、これらの$B$個のTriplet Lossを計算し、モデルを更新させる必要があります。

この手法は、トリプレットを生成するために**トレーニングセットをフルパスする(i.e. パラメータ更新の度に全レコードを一旦モデルに通して埋め込みベクトルを取得するってこと??)必要がある**ため、全体としてあまり効率的ではありません。また、オフラインでマイニングされたTripletを定期的に更新する必要があります。

## Online triplet mining

一方、Online triplet miningはFacenetで導入され、Brandon Amosのブログ記事[OpenFace 0.2.0: Higher accuracy and halved execution time](http://bamos.github.io/2016/01/19/openface-0.2.0/)でよく説明されています。

Online triplet miningのアイデアは、**入力のバッチごとに、有用なTripletをその場で計算すること**です。
B$個の例（例えば顔画像）が与えられたら、先に$B$個の埋め込みベクトルを計算し、最大$B^3$個のTripletを見つけます。もちろん、これらのTripletのほとんどは、**有効(valid)**ではありません.(=anchor & positive & negativeの関係性を満たさない.)

オフライン手法と比べて、オンライン手法では$B$個の埋め込みベクトルを計算したのち、それらの値を元にTripletを選ぶので、**計算量的により効率的なアプローチ**です。

![](https://omoindrot.github.io/assets/triplet_loss/online_triplet_loss.png)

図. Online triplet miningによるTriplet Lossの計算

# Online triplet miningにおける戦略

Online triplet miningでは、$B$個の入力データから$B$個の埋め込みベクトルを一括して計算します。次に重要なのが、これらの**$B$個の埋め込みベクトルを考慮して、どのようにTripletを生成するか**(選ぶか)です。

まず、1 batchの入力データにおいてi,j,kの3つのインデックスを指定します。
$i$と$j$が同じラベルを持ち、$i$と$k$が異なるラベルであれば、$(i,j,k)$は**Valid(有効な) Triplet**であると言えます。
(この時点で$B^3$個からだいぶ削れるはず...!)

ここで残るのは、「**Valid(有効な) Triplet達の中から、どのように実際に損失を計算し学習に活用するTripletを選ぶべきか**」ですよね。

↑に関する戦略のうち2つについては、論文「[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)」の第2節に詳しい説明があるそうです。(私はまだこの技術記事しか読んでいません...!)

入力データとして、$B = PK$個の顔画像のバッチがあり、$P$人の異なる人物からなり、それぞれ$K$枚の画像があると仮定しましょう。
例えば$K=4$です。このとき、**2つの戦略**があります: **batch all storategy**と**batch hard storategy**です。

## batch all storategy

- batch all storategyでは、**有効なトリプレットをすべて選択**し、その上で**hardおよびsemi-hardな**Tripletの損失を**平均化**します。
- ここで重要なのは，**easyなtriplet（損失=0 のTriplet）を考慮しないこと**です。
- なぜなら、これらのTripletを含めて平均化すると、全体の損失が非常に小さくなるからです。
- batch all storategyにより合計 $PK \times (K-1) \times (PK-K)$ 個のTripletが生成されます
  - ($PK$ 個のanchor、1つのanchorあたり $K-1$ 個のpositive、 $PK-K$ 個のnegative)。
  - (実際にはここから更にeasy tripletを取り除くので、更に少なくなる。)

## batch hard storategy

- batch hard storategyでは、$B=PK$個の各anchorについて、バッチの中で**最もhardなpositive**と、**最もhardなnegative**を選択します。
  - 最もhardなpositive = (i.e 距離 $d(h_a,h_p)$が最大)
  - 最もhardなnegative = (i.e. 距離 $d(a,p)$が最小)
  - 即ち↑の方法で選択されたtripletは、**各anchorにおいて最もhardな(損失が大きい)triplet**です。
- batch hard storategyで生成(選択)されるtripletの数は、$B=PK$個です。

なお上述した論文によると、batch hard storategyが最も良い性能を発揮するとの事です。ただこの結論は**データセットに依存**するものであり、開発におけるTriplet Miningの戦略は、**実際のデータセットを用いてパフォーマンスを比較することによって決定されるべき**ものであるとも述べています。

# Online Triplet Mining をPytorchで実装してみた

# 参考

- [Olivier MoindrotさんによるTriplet Loss, Triplet Miningの解説と、Tensorflowによる実装例](https://omoindrot.github.io/triplet-loss)
