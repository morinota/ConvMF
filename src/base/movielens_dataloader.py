from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.base.dataclasses import MovieLensDataset
from src.utils.rating_log_loader import RatingLogReader


class MovieLensDataloader:
    def __init__(
        self,
        data_path: str,
        num_users: int,
        num_test_items: int,
    ) -> None:
        self.data_path = data_path
        self.num_users = num_users
        self.num_test_items = num_test_items

    def load(self) -> MovieLensDataset:
        rating_logs = RatingLogReader().load()
