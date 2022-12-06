from collections import namedtuple

import pandas as pd


def test__convert_from_df_to_rating_data() -> None:
    base_ratings = []
    ratings_df = pd.DataFrame()
