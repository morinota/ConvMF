from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MovieLensDataset:
    """推薦システムの学習と評価に使うデータセット
    - 学習用の評価値データセット
    - テスト用の評価値データセット
    - ランキング指標のテストデータセット。
        キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト。
    - アイテムのコンテンツ情報
    """

    train: pd.DataFrame
    test: pd.DataFrame
    test_user2items: Dict[int, List[int]]
    item_content: pd.DataFrame


@dataclass(frozen=True)
class RecommendedResult:
    rating: pd.DataFrame  # metricsの値
    user2items: Dict[int, List[int]]  # key=user_id, value=推薦するitem_idのList


@dataclass(frozen=True)
class Metrics:
    rmse: float
    precision_at_k: float
    recall_at_k: float

    # 評価結果を出力する時に少数は第３桁までにする
    def __repr__(self):
        return f"rmse={self.rmse:.3f}, Precision@K={self.precision_at_k:.3f}, Recall@K={self.recall_at_k:.3f}"
