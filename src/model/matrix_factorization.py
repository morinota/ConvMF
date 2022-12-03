import os
from typing import Dict, Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from src.dataclasses.index_rating_mappings import IndexRatingSet
from src.dataclasses.rating_data import RatingLog


class MatrixFactrization(object):
    def __init__(
        self,
        ratings: List[RatingLog],
        n_factor: int = 300,
        user_lambda: float = 0.001,
        item_lambda: float = 0.001,
        n_item: Optional[int] = None,
    ):
        """コンストラクタ
        Parameters
        ----------
        ratings : List[RatingData]
            ユーザiのアイテムjに対する評価値 r_{ij}を格納したList
        n_factor : int, optional
            latent factorの次元数, by default 300
        user_lambda : float, optional
            ConvMFのハイパーパラメータ \lambda_U, by default 0.001
        item_lambda : float, optional
            ConvMFのハイパーパラメータ \lambda_V, by default 0.001
        n_item : int, optional
            MFで使用するアイテム数, by default None
        """

        data = pd.DataFrame(ratings)

        # Rating matrixの形状(行数＝user数、列数=item数)を指定。
        self.n_user = max(data["user"].unique()) + 1  # 行数=0始まりのuser_id+1
        self.n_item = n_item if n_item is not None else max(data["item"].unique()) + 1  # 列数=0始まりのitem_id+1
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda

        # user latent matrix をInitialize
        self.user_factor = np.random.normal(size=(self.n_factor, self.n_user)).astype(np.float32)
        # item latent matrix をInitialize
        self.item_factor = np.random.normal(size=(self.n_factor, self.n_item)).astype(np.float32)

        # パラメータ更新時にアクセスしやすいように、Ratingsを整理しておく
        # 各userに対する、Rating Matrixにおける非ゼロ要素のitemsとratingsを取得
        self.user_item_list: Dict[int, IndexRatingSet] = {
            user_i: v
            for user_i, v in data.groupby("user")
            .apply(lambda x: IndexRatingSet(indices=x["item"].values, ratings=x["rating"].values))
            .items()
        }
        # 各itemに対する、Rating Matrixにおける非ゼロ要素のusersとratings
        self.item_user_list: Dict[int, IndexRatingSet] = {
            item_i: v
            for item_i, v in data.groupby("item")
            .apply(lambda x: IndexRatingSet(indices=x["user"].values, ratings=x["rating"].values))
            .items()
        }

    def fit(self, n_trial: int = 5, document_vectors: Optional[List[np.ndarray]] = None) -> None:
        """U:user latent matrix とV:item latent matrixを推定するメソッド。
        Args:
            n_trial (int, optional):
            エポック数。何回ALSするか. Defaults to 5.
            document_vectors (Optional[List[np.ndarray]], None):
            document factor vector =s_j = Conv(W, X_j)のリスト.
        """
        # ALSをn_trial周していく！
        # (今回はPMFだから実際には、AMAP? = Alternating Maximum a Posteriori)
        for _ in tqdm(range(n_trial)):
            # 交互にパラメータ更新
            self._update_user_factors()
            self._update_item_factors(document_vectors)

    def predict(self, users: List[int], items: List[int]) -> np.ndarray:
        """user factor vectorとitem factor vectorの内積をとって、r\hatを推定

        Args:
            users (List[int]): 評価値を予測したいユーザidのlist
            items (List[int]): 評価値を予測したいアイテムidのlist
        """
        ratings_hat = []
        for user_i, item_i in zip(users, items):
            # ベクトルの内積を計算
            r_hat = np.inner(self.user_factor[:, user_i], self.item_factor[:, item_i])
            ratings_hat.append(r_hat)
        # ndarrayで返す。
        return np.array(ratings_hat)

    def _update_user_factors(self):
        """user latent vector (user latent matrixの列ベクトル)を更新する処理"""
        # 各user_id毎(=>各user latent vector毎)に繰り返し処理
        for i in self.user_item_list.keys():
            # rating matrix内のuser_id行のitem indicesとratingsを取得
            indices = self.user_item_list[i].indices
            ratings = self.user_item_list[i].ratings
            # item latent vector(ここでは定数)を取得
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

    def _update_item_factors(self, additional: Optional[List[np.ndarray]] = None):
        """item latent vector (item latent matrixの列ベクトル)を更新する処理

        Parameters
        ----------
        additional : Optional[List[np.ndarray]], optional
            CNN(X_j, W)で出力された各アイテムの説明文書に対応するdocument latent vector
            指定されない場合は、通常のPMF.
            , by default None
        """
        # 各item_id毎(=>各item latent vector毎)に繰り返し処理
        for j in self.item_user_list.keys():
            # rating matrix内のitem_id列のuser indicesとratingsを取得
            indices = self.item_user_list[j].indices
            ratings = self.item_user_list[j].ratings
            # user latent vector(ここでは定数)を取得
            u = self.user_factor[:, indices]
            # 以下、更新式の計算(aが左側の項, bが右側の項)
            a = np.dot(u, u.T)
            # aの対角成分にlambda_Vを追加?
            a[np.diag_indices_from(a)] += self.item_lambda
            b = np.dot(u, ratings)
            # ConvMFの場合は、\lambda_V・cnn(W, X_j)の項を追加
            if additional is not None:
                b += self.item_lambda * additional[j]

            # v_{j}の値を更新 a^{-1} * b
            self.item_factor[:, j] = np.linalg.solve(a, b)


if __name__ == "__main__":
    pass
