from collections import namedtuple

import pandas as pd

from src.utils.rating_log_loader import RatingLogReader


def test__convert_from_df_to_rating_data() -> None:
    ratings_df = pd.DataFrame(data=[{"user_id"}])

    reader = RatingLogReader()
    rating_logs_actual = reader._convert_from_df_to_rating_data(ratings_df=ratings_df)
