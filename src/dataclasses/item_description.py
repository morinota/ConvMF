
from dataclasses import dataclass
from typing import List,

import numpy as np


@dataclass
class ItemDescription:
    item_id: np.int32
    original_text: str
    tokens: List[str]
    token_indices: np.ndarray
