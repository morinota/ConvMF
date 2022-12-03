import os
from typing import Dict, Hashable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RatingData:
    """ユーザiのアイテムjに対する評価値 r_{ij}を格納するクラス"""

    user: int
    item: int
    rating: float
