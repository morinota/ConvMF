from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np


@dataclass
class ItemDescription:
    item_id: np.int32
    original_text: str
    tokens: Optional[List[str]] = None
    token_indices: Optional[np.ndarray] = None

    @staticmethod
    def merge_token_indices_of_descriptions(item_descriptions: List["ItemDescription"]) -> np.ndarray:
        return np.array([item_description.token_indices for item_description in item_descriptions])

    @classmethod
    def from_named_tuple(cls, named_tuple: NamedTuple) -> "ItemDescription":
        return ItemDescription(
            item_id=named_tuple.id,
            original_text=named_tuple.main_content,
        )
