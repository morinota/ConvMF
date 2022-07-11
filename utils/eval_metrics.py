
from typing import Dict, List
from sklearn.metrics import mean_squared_error
import numpy as np


def calc_rmse(tru_ratings: List[float], pred_ratings: List[float]) -> float:
    """RMSEを算出する関数

    Args:
        tru_ratings (List[float]): _description_
        pred_ratings (List[float]): _description_

    Returns:
        _type_: _description_
    """
    rmse = np.sqrt(mean_squared_error(tru_ratings, pred_ratings))
    return rmse


def calc_recall_at_k(
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int = 10) -> float:
    scores = []

    def _recall_at_k(true_items: List[int], pred_items: List[int], k: int) -> float:
        """各ユーザのrecall@kを計算するinnor関数
        """
        if len(true_items) == 0 or k == 0:
            return 0.0
        r_at_k = (len(set(true_items) & set(pred_items[:k])))
        r_at_k /= len(true_items)
        return r_at_k

    # 各ユーザに対して、recall@kを計算
    for user_id in true_user2items.keys():
        r_at_k = _recall_at_k(
            true_items=true_user2items[user_id],
            pred_items=pred_user2items[user_id],
            k=k
        )
        scores.append(r_at_k)

    # 全ユーザの平均値を返す。
    return np.mean(scores)


def calc_precision_at_k(
    true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int = 10) -> float:

    def _precision_at_k(true_items: List[int], pred_items: List[int], k: int):
        """各ユーザのprecision@kを計算するinnor関数
        """
        if k == 0:
            return 0.0
        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    # 各ユーザに対して、recall@kを計算
    scores = []
    for user_id in true_user2items.keys():
        p_at_k = _precision_at_k(
            true_items=true_user2items[user_id],
            pred_items=pred_user2items[user_id],
            k=k
        )
        scores.append(p_at_k)

    # 全ユーザの平均値を返す。
    return np.mean(scores)
