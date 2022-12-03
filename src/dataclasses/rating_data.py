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
    def from_named_tuple(cls, rating_series: NamedTuple) -> "RatingLog":
        return RatingLog(
            user_id=rating_series.user_id,
            item_id=rating_series.item_id,
            rating=rating_series.rating,
        )

    @classmethod
    def from_dict(cls, rating_dict: Dict) -> "RatingLog":
        return RatingLog(
            user_id=rating_dict["user_id"],
            item_id=rating_dict["item_id"],
            rating=rating_dict["rating"],
        )
