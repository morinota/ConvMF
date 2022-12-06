from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ItemDescription:
    item_id: np.int32
    original_text: str
    tokens: List[str]
    token_indices: np.ndarray

    @staticmethod
    def merge_token_indices_of_descriptions(item_descriptions: List["ItemDescription"]) -> np.ndarray:
        return np.array([item_description.token_indices for item_description in item_descriptions])
