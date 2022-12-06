from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataclasses.index_rating_mappings import IndexRatingSet
from src.dataclasses.rating_data import RatingLog


class MatrixFactrization:
    def __init__(
        self,
        rating_logs: List[RatingLog],
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
            rating_logsに含まれているアイテムだけでいい場合はNone。
            ただし、例えば新アイテム(まだlogがない)を含めたい場合は設定する必要がある。
            そしてConvMFはアイテムのコールドスタート問題に対応する事を目的の一つとしてるので、
            設定すべき。
        """
        self.n_factor = n_factor
        self.user_lambda = user_lambda
        self.item_lambda = item_lambda

        # TODO: DataFrameとして扱った方が早いのだろうか...??
        rating_logs_df = pd.DataFrame(rating_logs)

        self.n_user, self.n_item = self._set_rating_matrix_dimension(
            rating_logs_df, n_item
        )

        self.user_factor, self.item_factor = self._initialize_user_and_item_factors()

        self.user_id_ratings_mapping = self._collect_rating_logs_each_user(
            rating_logs_df
        )
        self.item_id_ratings_mapping = self._collect_rating_logs_each_item(
            rating_logs_df
        )

    def _set_rating_matrix_dimension(
        self, rating_logs_df: pd.DataFrame, n_item: int
    ) -> Tuple[int, int]:
        """Rating matrixの形状(行数＝user数、列数=item数)を設定する"""
        n_user = len(rating_logs_df["user_id"].unique())
        # TODO: ↓n_item = 0始まりのitem_idの最後尾 + 1() これは現実的なんだろうか...?
        n_item = (
            n_item
            if n_item is not None
            else max(rating_logs_df["item_id"].unique()) + 1
        )
        return n_user, n_item

    def _initialize_user_and_item_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """ "user_factor(user latent matrix)とitem_factor(item latent matrix)をInitialize"""
        user_factor_initialized = np.random.normal(
            size=(self.n_factor, self.n_user)
        ).astype(np.float32)
        item_factor_initialized = np.random.normal(
            size=(self.n_factor, self.n_item)
        ).astype(np.float32)
        return user_factor_initialized, item_factor_initialized

    def _collect_rating_logs_each_user(
        self, rating_logs_df: pd.DataFrame
    ) -> Dict[int, IndexRatingSet]:
        """パラメータ更新時にアクセスしやすいように、
        各userに対する、Rating Matrixにおける非ゼロ要素のitemsとratingsを取得
        """
        user_id_and_item_ratings_mapping = {
            user_id: item_rating_set
            for user_id, item_rating_set in rating_logs_df.groupby("user_id")
            .apply(
                lambda x: IndexRatingSet(
                    indices=x["item_id"].to_list(), ratings=x["rating"].to_list()
                )
            )
            .items()
        }
        return user_id_and_item_ratings_mapping

    def _collect_rating_logs_each_item(
        self, rating_logs_df: pd.DataFrame
    ) -> Dict[int, IndexRatingSet]:
        """パラメータ更新時にアクセスしやすいように、各itemに関するrating_logsを整理しておく.
        返り値は、「item_id: Rating Matrixにおける非ゼロ要素のuser_idsとratings」のmapping dict
        """
        item_id_and_user_ratings_mapping = {
            item_id: users_ratings_set
            for item_id, users_ratings_set in rating_logs_df.groupby("item_id")
            .apply(
                lambda x: IndexRatingSet(
                    indices=x["user_id"].to_list(), ratings=x["rating"].to_list()
                )
            )
            .items()
        }
        return item_id_and_user_ratings_mapping

    def fit(
        self, n_trial: int = 5, document_vectors: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        return self.user_factor, self.item_factor

    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:
        """user factor vectorとitem factor vectorの内積をとって、r\hatを推定

        Args:
            users (List[int]): 評価値を予測したいユーザidのlist
            items (List[int]): 評価値を予測したいアイテムidのlist
        """
        ratings_hat = []
        for user_id, item_i in zip(user_ids, item_ids):

            # ベクトルの内積を計算
            r_hat = np.inner(self.user_factor[:, user_id], self.item_factor[:, item_i])
            ratings_hat.append(r_hat)
        # ndarrayで返す。
        return np.array(ratings_hat)

    def _update_user_factors(self):
        """user latent vector (user latent matrixの列ベクトル)を更新する処理"""
        # 各user_idx(=今回の場合はuser_id, 数式におけるi)毎(=>各user latent vector毎)に繰り返し処理
        for user_idx in self.user_id_ratings_mapping.keys():
            # rating matrix内のuser_id行のitem indicesとratingsを取得
            indices = self.user_id_ratings_mapping[user_idx].indices
            ratings = self.user_id_ratings_mapping[user_idx].ratings
            # item latent vector(ここでは定数)を取得
            v = self.item_factor[:, indices]
            # 以下、更新式の計算(aが左側の項, bが右側の項)
            a = np.dot(v, v.T)
            # aの対角成分にlambda_uを追加?
            a[np.diag_indices_from(a)] += self.user_lambda
            b = np.dot(v, ratings)  # V R_i

            # u_{i}の値を更新 a^{-1} * b
            self.user_factor[:, user_idx] = np.linalg.solve(a, b)
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
        # 各item_idx(今回の場合は=item_id, 数式におけるj)毎(=>各item latent vector毎)に繰り返し処理
        for item_idx in self.item_id_ratings_mapping.keys():
            # rating matrix内のitem_id列のuser indicesとratingsを取得
            indices = self.item_id_ratings_mapping[item_idx].indices
            ratings = self.item_id_ratings_mapping[item_idx].ratings
            # user latent vector(ここでは定数)を取得
            u = self.user_factor[:, indices]
            # 以下、更新式の計算(aが左側の項, bが右側の項)
            a = np.dot(u, u.T)
            # aの対角成分にlambda_Vを追加?
            a[np.diag_indices_from(a)] += self.item_lambda
            b = np.dot(u, ratings)
            # ConvMFの場合は、\lambda_V・cnn(W, X_j)の項を追加
            if additional is not None:
                b += self.item_lambda * additional[item_idx]

            # v_{j}の値を更新 a^{-1} * b
            self.item_factor[:, item_idx] = np.linalg.solve(a, b)

    @property
    def item_latent_vectors(self) -> List[np.ndarray]:
        item_vector_list = self.item_factor.tolist()
        item_vector_ndarrays = [
            np.array(item_vector) for item_vector in item_vector_list
        ]
        return item_vector_ndarrays

    @property
    def user_latent_vectors(self) -> List[np.ndarray]:
        user_vector_list = self.user_factor.tolist()
        user_vector_ndarrays = [
            np.array(user_vector) for user_vector in user_vector_list
        ]
        return user_vector_ndarrays


if __name__ == "__main__":
    rating_logs_sample = [
        RatingLog(user_id=0, item_id=1, rating=2.0),
        RatingLog(user_id=1, item_id=1, rating=5.0),
        RatingLog(user_id=3, item_id=3, rating=3.0),
        RatingLog(user_id=4, item_id=2, rating=1.0),
    ]
    mf_obj = MatrixFactrization(rating_logs_sample)
