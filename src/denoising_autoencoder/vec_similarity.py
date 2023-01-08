import dataclasses
import random
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch import Tensor


@dataclasses.dataclass
class AUROCResult:
    tpr: np.ndarray
    fpr: np.ndarray
    thresholds: np.ndarray
    auroc_score: float


def get_random_pair_indices(
    num_samples: int,
    num_pairs: int,
) -> List[List[int]]:
    """n対のidxのペアを返す"""
    return [random.sample(range(num_samples), 2) for _ in range(num_pairs)]


def calc_cosine_similarity(vec_1: np.ndarray, vec_2: np.ndarray) -> float:
    """2つのベクトルを受け取り、コサイン類似度を計算する関数
    コサイン類似度(cosine similarity):2つのベクトルの内積を、2つのベクトルのノルム(長さ)の積で除した値
    """
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


def get_is_same_category(label_1: int, label_2: int) -> bool:
    """2つのラベルを受け取り、ラベルが一致してたら1、不一致だったら0を返す"""
    return True if label_1 == label_2 else False


def get_auroc_score(tpr: List[float], fpr: List[float]) -> float:
    """閾値を0~1で変動させた時のTPRとFPRの推移を受け取り、AUROCを算出する.
    np.trapz(y, x): 第1引数に与えられたy座標の値、第2引数に与えられたx座標の値から積分値を求める.
    ここでは、TPRをy座標、FPRをx座標として指定している.
    """
    return np.trapz(tpr, fpr)


class CategorySimilarityEvaluator(object):
    """「埋め込みベクトルがカテゴリ間の類似性を保持しているか」を評価する為の全ての処理を管理するクラス"""

    def __init__(self) -> None:
        pass

    def eval_embeddings(
        self,
        embeddings: Tensor,
        labels: Tensor,
        n: int,
    ) -> AUROCResult:

        pair_indices_list = get_random_pair_indices(
            num_samples=len(embeddings),
            num_pairs=n,
        )  # 任意のn対のレコードのペアを作る.

        cosine_similarities = (
            calc_cosine_similarity(
                embeddings[pair_indices[0]].detach().numpy(),
                embeddings[pair_indices[1]].detach().numpy(),
            )
            for pair_indices in pair_indices_list
        )  # n対のペアに対して、embeddingを用いてそれぞれcosine similarityを算出する.

        is_same_categories = (
            int(
                get_is_same_category(
                    int(labels[pair_indices[0]]),
                    int(labels[pair_indices[1]]),
                )
            )
            for pair_indices in pair_indices_list
        )  # n対のペアに対して、labelを用いてそれぞれbinary値(is_same_category)を作る

        fpr, tpr, thresholds = roc_curve(
            y_true=list(is_same_categories),
            y_score=list(cosine_similarities),
        )  # FPR, TPR, thresholdsを算出する.

        auroc_score = get_auroc_score(list(tpr), list(fpr))  # aucを算出する.

        return AUROCResult(tpr, fpr, thresholds, auroc_score)


if __name__ == "__main__":
    num_data = 10
    embedding_dim = 6
    num_classes = 3

    num_pairs = 10

    embeddings = Tensor(np.random.rand(num_data, embedding_dim).astype(np.float32))
    labels = Tensor(np.random.randint(0, num_classes, size=(num_data)).astype(np.float32))

    evaluator = CategorySimilarityEvaluator()
    auroc_result = evaluator.eval_embeddings(
        embeddings,
        labels,
        num_pairs,
    )
    print(auroc_result.auroc_score)
