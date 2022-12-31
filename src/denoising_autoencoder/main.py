from typing import Dict, Tuple

import pandas as pd

from src.config import MyConfig

STORY_COL = "story"
CATEGORY_COL = "category"


class DataLoader:
    def read_parquet_articles(parquet_path: str) -> pd.DataFrame:
        """parquetのパスを受け取ってpd.DataFrameとして返す.
        uciデ ータセットのformatに関しては以下linkを参照.
        https://www.kaggle.com/datasets/louislung/uci-news-aggregator-dataset-with-content?resource=download
        カラムは
        ['id', 'title', 'url', 'publisher', 'category', 'story', 'hostname',
        'timestamp', 'main_content', 'main_content_len']
        - id: article id
        - title:
        - url:
        - publisher:
        - category:ニュース項目のカテゴリ。 次のいずれか.
            (-- b : ビジネス -- t : 科学技術 -- e : エンターテイメント
            -- m : 健康)
        - story: 記事で取り上げるニュース記事の英数字の ID
        - hostname:
        - timestamp:
        - main_content:
        - main_content_len:
        """
        articles_df = pd.read_parquet(parquet_path)
        # articles_df = articles_df[articles_df.main_content.str.strip() != ""]
        # articles_df = articles_df[articles_df.main_content.notna()]

        return articles_df


def get_valid_story_label(article_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """get valid story(テキストのより細かい分類ラベル)"""
    story_value_counts = article_df[STORY_COL].value_counts()
    story_indices = article_df[STORY_COL].isin(
        values=story_value_counts[story_value_counts > 0].index,
    )
    article_df["is_label_story"] = False
    article_df.loc[story_indices, "is_label_story"] = True

    article_df["label_story"], label_list = pd.factorize(article_df[STORY_COL])  # カテゴリをintにencode
    label_idx_mapping = {idx: label for idx, label in enumerate(label_list)}
    return article_df, label_idx_mapping


def get_valid_category_label(article_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """get valid category(テキストのよりラフな分類ラベル)"""
    category_value_counts = article_df[CATEGORY_COL].value_counts()
    category_indices = article_df[CATEGORY_COL].isin(
        values=category_value_counts[category_value_counts > 0].index,
    )
    article_df[f"is_label_category"] = False
    article_df.loc[category_indices, "is_label_category"] = True

    article_df["label_category"], label_list = pd.factorize(article_df[CATEGORY_COL])  # カテゴリをintにencode
    label_idx_mapping = {idx: label for idx, label in enumerate(label_list)}

    return article_df, label_idx_mapping


def main():
    article_df = pd.read_parquet(MyConfig.uci_news_data_path)
    print(article_df.columns)

    article_df = article_df.sort_values(by="id")  # article idの昇順でソート

    article_df, label_idx_story_mapping = get_valid_story_label(article_df)

    article_df, label_idx_category_mapping = get_valid_category_label(article_df)

    print(article_df.head())

    # テキスト(article_df[main_contents])をtokenize

    # Scikit learn’s Count Vectorizerにより、tonenizeされたテキストをbinary ベクトル化

    # denoizing autoencoder による学習(anchorのテキストとラベルを渡せば良い)


if __name__ == "__main__":
    main()
