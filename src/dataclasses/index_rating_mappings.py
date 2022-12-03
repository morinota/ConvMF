import os
from typing import Dict, Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from tqdm import tqdm


@dataclass
class IndexRatingSet:
    """あるユーザi (or アイテムj)における、任意のアイテムj(or ユーザi)の評価値のリスト"""

    indices: List[int]  # user_id（もしくはitem_id）のList
    ratings: List[float]  # 対応するratingsのList
