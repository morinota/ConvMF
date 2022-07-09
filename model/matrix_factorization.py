from mimetypes import init
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
