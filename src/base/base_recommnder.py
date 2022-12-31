from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


class BaseRecommender:
    @abstractmethod
    def recommend(self, dataset) -> RecommendedResult:
        pass

    def run_sample(self) -> None:
        """小量の学習データをアルゴリズムを走らせて
        その結果を確認する"""
        # MovieLensのデータを取得
        movielens = MyDataloader().load()
        # 推薦結果
        recommended_result = self.recommend(movielens)
        # 推薦結果の評価
        metrics = MetricCaluculator().calc()
        print(metrics)
