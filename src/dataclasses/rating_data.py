import os
from typing import Dict, Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RatingLog:
    """ユーザiのアイテムjに対する評価値 r_{ij}を格納するクラス"""

    user_id: int
    item_id: int
    rating: float

    @classmethod
    def from_named_tuple(cls, rating_named_tuple: NamedTuple) -> "RatingLog":
        return RatingLog(
            user_id=rating_named_tuple.user_id,
            item_id=rating_named_tuple.item_id,
            rating=rating_named_tuple.rating,
        )

    @classmethod
    def from_dict(cls, rating_dict: Dict) -> "RatingLog":
        return RatingLog(
            user_id=rating_dict["user_id"],
            item_id=rating_dict["item_id"],
            rating=rating_dict["rating"],
        )

    @staticmethod
    def count_unique_users(rating_logs: List["RatingLog"]) -> int:
        """rating_logsの中のunique_user数をカウントして返す"""
        return len(np.unique([rating_log.user_id for rating_log in rating_logs]))

    @staticmethod
    def count_unique_items(rating_logs: List["RatingLog"]) -> int:
        """rating_logsの中のuniqueなitem数をカウントして返す"""
        return len(np.unique([rating_log.item_id for rating_log in rating_logs]))
